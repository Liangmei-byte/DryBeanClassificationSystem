from pathlib import Path
import matplotlib.pyplot as plt

def save_bar(df, x, y, title, ylabel, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(df[x], df[y])
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
