# src/eval/h1_separability.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Utilities
# -----------------------------
def load_npz_files(data_dir, patients):
    X_all, U_all, y_all = [], [], []

    for p in patients:
        path = os.path.join(data_dir, f"{p}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing NPZ file: {path}")

        data = np.load(path)
        X_all.append(data["X"])   # raw EEG windows
        U_all.append(data["U"])   # chaos-modulated windows
        y_all.append(data["y"])   # labels

    X = np.concatenate(X_all, axis=0)
    U = np.concatenate(U_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # ===========================
    # Reduce dataset size for H1
    # ===========================
    MAX_SAMPLES = 10000    # recommended
    
    print(f"Original dataset size: {len(y)} windows")
    
    # Stratified sampling: equal seizure and non-seizure
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    n_pos = min(len(pos_idx), MAX_SAMPLES // 2)
    n_neg = min(len(neg_idx), MAX_SAMPLES // 2)
    
    rng = np.random.default_rng(42)
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    
    X = X[sel]
    U = U[sel]
    y = y[sel]
    
    print(f"Sampled {len(y)} windows for H1 separability")

    return X, U, y


def fisher_separability(Z, y):
    z1 = Z[y == 1]
    z0 = Z[y == 0]
    mu1, mu0 = z1.mean(axis=0), z0.mean(axis=0)
    s1, s0 = z1.var(axis=0).mean(), z0.var(axis=0).mean()
    return np.linalg.norm(mu1 - mu0) ** 2 / (s1 + s0 + 1e-8)


def linear_probe_auc(Z, y):
    Zs = StandardScaler().fit_transform(Z)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Zs, y)
    probs = clf.predict_proba(Zs)[:, 1]
    return roc_auc_score(y, probs)


def run_pca(Z, n=2):
    return PCA(n_components=n).fit_transform(Z)

def to_python(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


# -----------------------------
# Main
# -----------------------------
def main(args):
    patients = args.patients.split(",")

    print("🔍 Loading NPZ data...")
    X, U, y = load_npz_files(args.data_dir, patients)

    # Flatten (N, C, T) → (N, D)
    Xf = X.reshape(len(X), -1)
    Uf = U.reshape(len(U), -1)

    print("📐 Computing separability metrics...")

    results = {}

    for name, Z in [("raw", Xf), ("chaos", Uf)]:
        results[name] = {
            "fisher": fisher_separability(Z, y),
            "auc": linear_probe_auc(Z, y),
        }

    # PCA plots
    os.makedirs(args.out, exist_ok=True)

    for name, Z in [("raw", Xf), ("chaos", Uf)]:
        Zp = run_pca(Z)
        plt.figure(figsize=(5, 4))
        plt.scatter(Zp[y == 0, 0], Zp[y == 0, 1], s=5, alpha=0.4, label="non-seizure")
        plt.scatter(Zp[y == 1, 0], Zp[y == 1, 1], s=5, alpha=0.4, label="seizure")
        plt.title(f"PCA – {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"pca_{name}.png"))
        plt.close()

    # Save summary
    summary_path = os.path.join(args.out, "h1_summary.json")
    with open(summary_path, "w") as f:
        import json
        json.dump(to_python(results), f, indent=2)


    print("\n✅ H1 RESULTS")
    for k, v in results.items():
        print(f"{k.upper():>6} | Fisher={v['fisher']:.4f} | AUC={v['auc']:.4f}")

    print("\nSaved to:", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with CHB-MIT NPZ files")
    ap.add_argument("--patients", required=True, help="Comma-separated patient IDs")
    ap.add_argument("--out", default="analysis/h1", help="Output directory")
    args = ap.parse_args()
    main(args)
