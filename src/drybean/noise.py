import numpy as np

def add_noise(X, y, noise_type, strength, rng=None):
    """向训练集加入高斯、缺失、异常值或标签翻转噪声。"""
    rng = np.random.default_rng(42) if rng is None else rng
    Xn = X.copy().astype(np.float32)
    yn = y.copy()
    n, d = Xn.shape
    std = Xn.std(axis=0)
    std[std == 0] = 1.0

    if noise_type == "gaussian":
        Xn += rng.normal(0, strength * std, Xn.shape).astype(np.float32)
    elif noise_type == "missing":
        mask = rng.random(Xn.shape) < strength
        Xn[mask] = np.nan
        med = np.nanmedian(Xn, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        rows, cols = np.where(np.isnan(Xn))
        Xn[rows, cols] = med[cols]
    elif noise_type == "outlier":
        mask = rng.random(Xn.shape) < strength
        impulse = rng.choice([-1.0, 1.0], Xn.shape) * 8.0 * std
        Xn[mask] += impulse[mask].astype(np.float32)
    elif noise_type == "label_flip":
        k = int(round(strength * n))
        idx = rng.choice(n, k, replace=False)
        for i in idx:
            choices = [c for c in range(7) if c != yn[i]]
            yn[i] = rng.choice(choices)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return Xn, yn
