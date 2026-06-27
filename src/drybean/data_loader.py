from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
LABEL_TO_ID = {c: i for i, c in enumerate(LABELS)}

def read_model_csv(path):
    """读取预处理后的模型输入 CSV。"""
    path = Path(path)
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["Class"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    return X, y, feature_cols

def load_train_val_test(data_dir, scaled=False):
    """读取 train/val/test。scaled=True 时读取标准化版本。"""
    data_dir = Path(data_dir)
    suffix = "_scaled" if scaled else ""
    X_train, y_train, feature_cols = read_model_csv(data_dir / f"model_train{suffix}.csv")
    X_val, y_val, _ = read_model_csv(data_dir / f"model_val{suffix}.csv")
    X_test, y_test, _ = read_model_csv(data_dir / f"model_test{suffix}.csv")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
