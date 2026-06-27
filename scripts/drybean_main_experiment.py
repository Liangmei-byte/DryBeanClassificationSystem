#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dry Bean 主体实验：LightGBM、XGBoost、Softmax、KNN
运行：
python drybean_main_experiment.py --data_dir DryBeanProcessed --output_dir DryBeanExperimentResults
先跳过鲁棒性快速跑主实验：
python drybean_main_experiment.py --data_dir DryBeanProcessed --output_dir DryBeanExperimentResults --skip_robustness
依赖：
pip install numpy pandas scikit-learn matplotlib lightgbm xgboost
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

LABELS = ["BARBUNYA","BOMBAY","CALI","DERMASON","HOROZ","SEKER","SIRA"]
LABEL_TO_ID = {c:i for i,c in enumerate(LABELS)}

def read_csv_data(path):
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError(f"{path} 缺少 Class 列")
    features = [c for c in df.columns if c != "Class"]
    X = df[features].to_numpy(np.float32)
    y = df["Class"].map(LABEL_TO_ID).to_numpy(np.int64)
    if np.isnan(X).any():
        raise ValueError(f"{path} 仍有 NaN，请检查预处理")
    return X, y, features

def load_data(data_dir):
    data_dir = Path(data_dir)
    Xtr,ytr,feat = read_csv_data(data_dir/"model_train.csv")
    Xv,yv,_ = read_csv_data(data_dir/"model_val.csv")
    Xte,yte,_ = read_csv_data(data_dir/"model_test.csv")
    Xtrs,ytrs,feats = read_csv_data(data_dir/"model_train_scaled.csv")
    Xvs,yvs,_ = read_csv_data(data_dir/"model_val_scaled.csv")
    Xtes,ytes,_ = read_csv_data(data_dir/"model_test_scaled.csv")
    assert np.array_equal(ytr,ytrs) and np.array_equal(yv,yvs) and np.array_equal(yte,ytes)
    return {
        "raw":{"X_train":Xtr,"y_train":ytr,"X_val":Xv,"y_val":yv,"X_test":Xte,"y_test":yte,"features":feat},
        "scaled":{"X_train":Xtrs,"y_train":ytrs,"X_val":Xvs,"y_val":yvs,"X_test":Xtes,"y_test":ytes,"features":feats}
    }

class SoftmaxRegression:
    def __init__(self, n_classes=7, lr=0.05, reg=1e-4, epochs=250, batch_size=256, seed=42):
        self.n_classes=n_classes; self.lr=lr; self.reg=reg; self.epochs=epochs; self.batch_size=batch_size; self.seed=seed
        self.history_={"train_logloss":[],"val_logloss":[]}
    def _softmax(self,z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    def _onehot(self,y):
        Y = np.zeros((len(y), self.n_classes), dtype=np.float32)
        Y[np.arange(len(y)), y] = 1.0
        return Y
    def predict_proba(self,X):
        return self._softmax(X @ self.W + self.b)
    def predict(self,X):
        return self.predict_proba(X).argmax(axis=1)
    def _loss(self,X,y):
        p = self.predict_proba(X)
        ce = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        return float(ce + 0.5*self.reg*np.sum(self.W*self.W))
    def fit(self,X,y,X_val=None,y_val=None):
        rng = np.random.default_rng(self.seed)
        n,d = X.shape
        scale = np.sqrt(2/(d+self.n_classes))
        self.W = rng.normal(0, scale, (d,self.n_classes)).astype(np.float32)
        self.b = np.zeros(self.n_classes, dtype=np.float32)
        Y = self._onehot(y)
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for st in range(0,n,self.batch_size):
                bi = idx[st:st+self.batch_size]
                xb, yb = X[bi], Y[bi]
                p = self._softmax(xb @ self.W + self.b)
                g = (p-yb)/len(bi)
                self.W -= self.lr*(xb.T @ g + self.reg*self.W)
                self.b -= self.lr*g.sum(axis=0)
            self.history_["train_logloss"].append(self._loss(X,y))
            if X_val is not None:
                self.history_["val_logloss"].append(self._loss(X_val,y_val))
        return self

def train_lightgbm(X,y,Xv,yv,n_estimators=400,seed=42):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("请先安装 LightGBM：pip install lightgbm") from e
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(LABELS), n_estimators=n_estimators,
        learning_rate=0.03, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1
    )
    model.fit(X,y,eval_set=[(X,y),(Xv,yv)],eval_names=["train","val"],
              eval_metric="multi_logloss",callbacks=[lgb.log_evaluation(0)])
    hist = {"train_logloss":model.evals_result_["train"]["multi_logloss"],
            "val_logloss":model.evals_result_["val"]["multi_logloss"]}
    return model,hist

def train_xgboost(X,y,Xv,yv,n_estimators=400,seed=42):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("请先安装 XGBoost：pip install xgboost") from e
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(LABELS), eval_metric="mlogloss",
        n_estimators=n_estimators, learning_rate=0.03, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, tree_method="hist"
    )
    model.fit(X,y,eval_set=[(X,y),(Xv,yv)],verbose=False)
    er = model.evals_result()
    hist = {"train_logloss":er["validation_0"]["mlogloss"],"val_logloss":er["validation_1"]["mlogloss"]}
    return model,hist

def train_softmax(X,y,Xv,yv,seed=42):
    model = SoftmaxRegression(seed=seed)
    model.fit(X,y,Xv,yv)
    return model, model.history_

def train_knn(X,y):
    model = KNeighborsClassifier(n_neighbors=7, weights="distance", p=2, n_jobs=-1)
    model.fit(X,y)
    return model, None

def evaluate(model,Xtr,ytr,Xv,yv,Xte,yte):
    out = {}
    out["train_acc"] = accuracy_score(ytr, model.predict(Xtr))
    out["val_acc"] = accuracy_score(yv, model.predict(Xv))
    out["test_acc"] = accuracy_score(yte, model.predict(Xte))
    out["overfit_gap_train_minus_test"] = out["train_acc"] - out["test_acc"]
    try:
        out["test_logloss"] = log_loss(yte, model.predict_proba(Xte), labels=list(range(len(LABELS))))
    except Exception:
        out["test_logloss"] = np.nan
    return out

def inference_speed(model,X,repeats=30,warmup=5):
    for _ in range(warmup):
        model.predict(X)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        ts.append(time.perf_counter()-t0)
    avg = float(np.mean(ts))
    return {
        "avg_total_ms": avg*1000,
        "avg_per_sample_us": avg/len(X)*1_000_000,
        "samples_per_second": len(X)/avg
    }

def save_detail(model_name,model,X,y,out_dir):
    pred = model.predict(X)
    txt = classification_report(y,pred,target_names=LABELS,digits=4)
    Path(out_dir, f"{model_name}_classification_report.txt").write_text(txt, encoding="utf-8")
    cm = confusion_matrix(y,pred,labels=list(range(len(LABELS))))
    pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(Path(out_dir, f"{model_name}_confusion_matrix.csv"), encoding="utf-8-sig")

def add_noise(X,y,noise_type,strength,rng):
    Xn = X.copy().astype(np.float32)
    yn = y.copy()
    n,d = Xn.shape
    std = Xn.std(axis=0)
    std[std==0] = 1.0
    if noise_type == "gaussian":
        Xn += rng.normal(0, strength*std, Xn.shape).astype(np.float32)
    elif noise_type == "missing":
        mask = rng.random(Xn.shape) < strength
        Xn[mask] = np.nan
        med = np.nanmedian(Xn, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        r,c = np.where(np.isnan(Xn))
        Xn[r,c] = med[c]
    elif noise_type == "outlier":
        mask = rng.random(Xn.shape) < strength
        impulse = rng.choice([-1.0,1.0], Xn.shape) * 8.0 * std
        Xn[mask] += impulse[mask].astype(np.float32)
    elif noise_type == "label_flip":
        k = int(round(strength*n))
        idx = rng.choice(n, k, replace=False)
        for i in idx:
            choices = [c for c in range(len(LABELS)) if c != yn[i]]
            yn[i] = rng.choice(choices)
    else:
        raise ValueError(noise_type)
    return Xn,yn

def train_for_name(name,X,y,Xv,yv,n_estimators,seed):
    if name == "LightGBM":
        return train_lightgbm(X,y,Xv,yv,n_estimators=n_estimators,seed=seed)[0]
    if name == "XGBoost":
        return train_xgboost(X,y,Xv,yv,n_estimators=n_estimators,seed=seed)[0]
    if name == "Softmax":
        return train_softmax(X,y,Xv,yv,seed=seed)[0]
    if name == "KNN":
        return train_knn(X,y)[0]
    raise ValueError(name)

def robustness(data,base_acc,out_dir,seed=42,n_estimators=150):
    noise_types = ["gaussian","missing","outlier","label_flip"]
    strengths = [0.05,0.10,0.20]
    rows = []
    for name in ["LightGBM","XGBoost","Softmax","KNN"]:
        key = "raw" if name in ["LightGBM","XGBoost"] else "scaled"
        d = data[key]
        for nt in noise_types:
            for s in strengths:
                rng = np.random.default_rng(seed)
                Xn,yn = add_noise(d["X_train"],d["y_train"],nt,s,rng)
                m = train_for_name(name,Xn,yn,d["X_val"],d["y_val"],n_estimators,seed)
                acc = accuracy_score(d["y_test"], m.predict(d["X_test"]))
                drop = base_acc[name] - acc
                rows.append({
                    "model":name,"noise_type":nt,"strength":s,
                    "baseline_test_acc":base_acc[name],
                    "noisy_train_test_acc":acc,
                    "accuracy_drop":drop,
                    "relative_drop_percent":drop/base_acc[name]*100
                })
                print(f"[Robustness] {name} {nt} {s:.2f} acc={acc:.4f} drop={drop:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/"robustness_results.csv", index=False, encoding="utf-8-sig")
    return df

def plot_accuracy(df,fig_dir):
    plt.figure(figsize=(8,5))
    plt.bar(df["model"], df["test_acc"])
    plt.xlabel("Model"); plt.ylabel("Test Accuracy"); plt.title("Test Accuracy Comparison")
    plt.ylim(max(0, df["test_acc"].min()-0.05), 1.0)
    plt.tight_layout(); plt.savefig(Path(fig_dir)/"test_accuracy_comparison.png", dpi=300); plt.close()

def plot_speed(df,fig_dir):
    plt.figure(figsize=(8,5))
    plt.bar(df["model"], df["avg_per_sample_us"])
    plt.xlabel("Model"); plt.ylabel("Inference Time per Sample (us)"); plt.title("Inference Speed Comparison")
    plt.tight_layout(); plt.savefig(Path(fig_dir)/"inference_speed_comparison.png", dpi=300); plt.close()

def plot_overfit(df,fig_dir):
    plt.figure(figsize=(8,5))
    plt.bar(df["model"], df["overfit_gap_train_minus_test"])
    plt.xlabel("Model"); plt.ylabel("Train Accuracy - Test Accuracy"); plt.title("Overfitting Gap Comparison")
    plt.tight_layout(); plt.savefig(Path(fig_dir)/"overfit_gap_comparison.png", dpi=300); plt.close()

def plot_loss(histories,fig_dir,out_dir):
    rows = []
    plt.figure(figsize=(10,6))
    for name,h in histories.items():
        tr = h["train_logloss"]; va = h["val_logloss"]
        x = np.arange(1, len(tr)+1)
        plt.plot(x,tr,label=f"{name} Train")
        plt.plot(x,va,linestyle="--",label=f"{name} Val")
        rows += [{"model":name,"round":i+1,"dataset":"train","logloss":v} for i,v in enumerate(tr)]
        rows += [{"model":name,"round":i+1,"dataset":"val","logloss":v} for i,v in enumerate(va)]
    plt.xlabel("Iteration / Epoch"); plt.ylabel("Log Loss"); plt.title("Loss Curves")
    plt.legend(); plt.tight_layout(); plt.savefig(Path(fig_dir)/"loss_curves_comparison.png", dpi=300); plt.close()
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/"loss_curves.csv", index=False, encoding="utf-8-sig")
    return df

def plot_robust(df,fig_dir):
    for nt in df["noise_type"].unique():
        sub = df[df["noise_type"]==nt]
        plt.figure(figsize=(8,5))
        for name in sub["model"].unique():
            tmp = sub[sub["model"]==name].sort_values("strength")
            plt.plot(tmp["strength"], tmp["accuracy_drop"], marker="o", label=name)
        plt.xlabel("Noise Strength"); plt.ylabel("Accuracy Drop"); plt.title(f"Robustness under {nt} Noise")
        plt.legend(); plt.tight_layout(); plt.savefig(Path(fig_dir)/f"robustness_{nt}.png", dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="DryBeanProcessed")
    ap.add_argument("--output_dir", default="DryBeanExperimentResults")
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--robustness_estimators", type=int, default=150)
    ap.add_argument("--skip_robustness", action="store_true")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir/"figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Load data")
    data = load_data(args.data_dir)

    cfg = {
        "labels": LABELS,
        "model_input": {
            "LightGBM_XGBoost": "model_train/val/test.csv",
            "Softmax_KNN": "model_train/val/test_scaled.csv"
        },
        "notes": [
            "KNN is lazy learning, so no loss curve is plotted.",
            "Robustness experiment adds noise only to training data.",
            "Validation and test data remain clean in robustness experiment."
        ]
    }
    (out_dir/"experiment_config.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2), encoding="utf-8")

    models, histories = {}, {}

    print("[2/7] Train LightGBM")
    raw = data["raw"]
    m,h = train_lightgbm(raw["X_train"],raw["y_train"],raw["X_val"],raw["y_val"],
                         n_estimators=args.n_estimators,seed=args.random_state)
    models["LightGBM"] = m; histories["LightGBM"] = h

    print("[3/7] Train XGBoost")
    m,h = train_xgboost(raw["X_train"],raw["y_train"],raw["X_val"],raw["y_val"],
                        n_estimators=args.n_estimators,seed=args.random_state)
    models["XGBoost"] = m; histories["XGBoost"] = h

    print("[4/7] Train Softmax and KNN")
    sc = data["scaled"]
    m,h = train_softmax(sc["X_train"],sc["y_train"],sc["X_val"],sc["y_val"],seed=args.random_state)
    models["Softmax"] = m; histories["Softmax"] = h
    m,_ = train_knn(sc["X_train"],sc["y_train"])
    models["KNN"] = m

    print("[5/7] Evaluate accuracy, speed, and overfitting")
    eval_rows, speed_rows = [], []
    for name,m in models.items():
        key = "raw" if name in ["LightGBM","XGBoost"] else "scaled"
        d = data[key]
        ev = evaluate(m,d["X_train"],d["y_train"],d["X_val"],d["y_val"],d["X_test"],d["y_test"])
        ev["model"] = name
        eval_rows.append(ev)
        sp = inference_speed(m,d["X_test"])
        sp["model"] = name
        speed_rows.append(sp)
        save_detail(name,m,d["X_test"],d["y_test"],out_dir)
        print(f"{name:8s} train={ev['train_acc']:.4f} val={ev['val_acc']:.4f} test={ev['test_acc']:.4f} gap={ev['overfit_gap_train_minus_test']:.4f}")

    eval_df = pd.DataFrame(eval_rows)[["model","train_acc","val_acc","test_acc","test_logloss","overfit_gap_train_minus_test"]]
    speed_df = pd.DataFrame(speed_rows)[["model","avg_total_ms","avg_per_sample_us","samples_per_second"]]
    eval_df.to_csv(out_dir/"accuracy_and_overfit_results.csv", index=False, encoding="utf-8-sig")
    speed_df.to_csv(out_dir/"inference_speed_results.csv", index=False, encoding="utf-8-sig")

    print("[6/7] Plot figures")
    plot_accuracy(eval_df,fig_dir)
    plot_speed(speed_df,fig_dir)
    plot_overfit(eval_df,fig_dir)
    plot_loss(histories,fig_dir,out_dir)

    if not args.skip_robustness:
        print("[7/7] Robustness experiment")
        base_acc = dict(zip(eval_df["model"], eval_df["test_acc"]))
        robust_df = robustness(data,base_acc,out_dir,seed=args.random_state,n_estimators=args.robustness_estimators)
        plot_robust(robust_df,fig_dir)
    else:
        print("[7/7] Skip robustness experiment")

    print("\nDone. Results saved to:", out_dir)
    print("Key files:")
    for f in [
        "accuracy_and_overfit_results.csv",
        "inference_speed_results.csv",
        "loss_curves.csv",
        "robustness_results.csv",
        "figures/test_accuracy_comparison.png",
        "figures/loss_curves_comparison.png",
        "figures/inference_speed_comparison.png",
        "figures/overfit_gap_comparison.png",
    ]:
        p = out_dir/f
        if p.exists():
            print(" -", p)

if __name__ == "__main__":
    main()
