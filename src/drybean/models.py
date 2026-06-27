import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train_lightgbm(X_train, y_train, X_val, y_val, n_estimators=400, random_state=42):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=7, n_estimators=n_estimators,
        learning_rate=0.03, num_leaves=31, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0,
        random_state=random_state, n_jobs=-1, verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric="multi_logloss",
        callbacks=[lgb.log_evaluation(0)]
    )
    return model

def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=400, random_state=42):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        objective="multi:softprob", num_class=7, eval_metric="mlogloss",
        n_estimators=n_estimators, learning_rate=0.03, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=random_state, n_jobs=-1, tree_method="hist"
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    return model

class SoftmaxRegression:
    """numpy 实现的 Softmax 多分类回归。"""
    def __init__(self, n_classes=7, lr=0.05, reg=1e-4, epochs=250, batch_size=256, seed=42):
        self.n_classes = n_classes
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.history_ = {"train_logloss": [], "val_logloss": []}

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _onehot(self, y):
        Y = np.zeros((len(y), self.n_classes), dtype=np.float32)
        Y[np.arange(len(y)), y] = 1.0
        return Y

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _loss(self, X, y):
        p = self.predict_proba(X)
        ce = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        return float(ce + 0.5 * self.reg * np.sum(self.W * self.W))

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.W = rng.normal(0, np.sqrt(2 / (d + self.n_classes)), (d, self.n_classes)).astype(np.float32)
        self.b = np.zeros(self.n_classes, dtype=np.float32)
        Y = self._onehot(y)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for st in range(0, n, self.batch_size):
                bi = idx[st:st + self.batch_size]
                p = self._softmax(X[bi] @ self.W + self.b)
                g = (p - Y[bi]) / len(bi)
                self.W -= self.lr * (X[bi].T @ g + self.reg * self.W)
                self.b -= self.lr * g.sum(axis=0)
            self.history_["train_logloss"].append(self._loss(X, y))
            if X_val is not None:
                self.history_["val_logloss"].append(self._loss(X_val, y_val))
        return self

def train_softmax(X_train, y_train, X_val, y_val, random_state=42):
    return SoftmaxRegression(seed=random_state).fit(X_train, y_train, X_val, y_val)

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=7, weights="distance", p=2, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
