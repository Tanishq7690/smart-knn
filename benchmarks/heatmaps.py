import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASS_METRICS = ["accuracy", "precision", "recall", "f1", "train_time", "pred_time"]
REG_METRICS   = ["mse", "mae", "r2", "train_time", "pred_time"]

OUT_DIR = "heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)


def load_csvs(pattern):
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_src"] = os.path.basename(f)
            dfs.append(df)
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None

def normalize(mat, invert=False):
    arr = mat.copy().astype(float)
    mask = np.isfinite(arr)
    if invert:
        arr[mask] = -arr[mask]
    vals = arr[mask]
    if vals.size == 0:
        return arr
    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)

def plot_heatmap(pivot, original_mat, out_file, metric):
    normed = normalize(original_mat, invert=metric in ["mse", "mae", "train_time", "pred_time"])

    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * pivot.shape[0])))
    im = ax.imshow(normed, aspect="auto")
    
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=7)
    
    ax.set_title(f"Heatmap — {metric}", fontsize=14)
    plt.colorbar(im, ax=ax)
    

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = original_mat[i, j]
            txt = f"{val:.3f}" if np.isfinite(val) else "-"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


class_df = load_csvs("classification_results*.csv")
if class_df is not None:
    class_df["model"] = class_df["model"].astype(str)
    for metric in CLASS_METRICS:
        if metric in class_df.columns:
            pivot = class_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
            mat = pivot.values.astype(float)
            out = f"{OUT_DIR}/class_{metric}.png"
            plot_heatmap(pivot, mat, out, metric)
            print("[Saved]", out)

reg_df = load_csvs("regression_results*.csv")
if reg_df is not None:
    reg_df["model"] = reg_df["model"].astype(str)
    for metric in REG_METRICS:
        if metric in reg_df.columns:
            pivot = reg_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
            mat = pivot.values.astype(float)
            out = f"{OUT_DIR}/reg_{metric}.png"
            plot_heatmap(pivot, mat, out, metric)
            print("[Saved]", out)

print("\n ALL HEATMAPS GENERATED IN 'heatmaps/' FOLDER ")
