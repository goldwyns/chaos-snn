# src/analyze_bonn_full_complexity.py

import os
import json
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map


def build_model(n_channels, fs, mode="chaos_on", device="cpu"):
    """Build a ChaosSNN model with the same config as train_bonn.py."""
    dt = 1.0 / fs

    if mode == "chaos_on":
        alpha = 3.5
        beta = 0.2
        gamma = 0.5
        lam = 0.7
        alpha_rec = 1.0
        print("🧪 Analysis mode: CHAOS ON")
    elif mode == "chaos_off":
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        lam = 1.0
        alpha_rec = 0.8
        print("🧪 Analysis mode: CHAOS OFF")
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    model = ChaosSNNSeizureDetector(
        n_channels=n_channels,
        n_hidden=128,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lam=lam,
        v_th=1.0,
        alpha_rec=alpha_rec,
        dt=dt,
        tau_mem=20e-3,
        tau_syn=10e-3,
    ).to(device)

    return model


def compute_isi_entropy(spikes, eps=1e-8, n_bins=20):
    """
    Improved ISI entropy:
    - spikes: (N, T) binary numpy array
    - For each neuron, compute ISIs, then pool all ISIs.
    - Return entropy normalized to ~[0,1].
    """
    N, T = spikes.shape
    all_isi = []

    for i in range(N):
        ts = np.where(spikes[i] > 0)[0]
        if len(ts) >= 2:
            isi_i = np.diff(ts)
            if len(isi_i) > 0:
                all_isi.append(isi_i)

    if len(all_isi) == 0:
        return 0.0

    all_isi = np.concatenate(all_isi, axis=0)
    if all_isi.size < 2:
        return 0.0

    hist, edges = np.histogram(all_isi, bins=n_bins)
    if hist.sum() == 0:
        return 0.0

    p = hist.astype(float) / (hist.sum() + eps)
    p = p[p > 0]

    H = -np.sum(p * np.log(p + eps))
    H_norm = H / (np.log(len(hist) + eps))
    return float(H_norm)


def analyze_bonn_full_complexity(
    dataset_name="bonn",
    mode="chaos_on",
    save_dir="analysis/bonn_full",
    device=None,
    batch_size=32,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------- DATASET --------------------
    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

    x0, y0 = dataset[0]
    n_channels, T = x0.shape
    print(f"🧠 {dataset_name} sample shape: C={n_channels}, T={T}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # -------------------- MODEL & CHECKPOINT --------------------
    model = build_model(n_channels, fs, mode=mode, device=device)

    ckpt_name = f"chaos_snn_{dataset_name}_{mode}_best.pt"
    ckpt_path = os.path.join("checkpoints", ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # You can ignore the PyTorch FutureWarning since this is your own checkpoint.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"Loaded checkpoint from {ckpt_path}, "
        f"best AUC={ckpt.get('val_auc', 'N/A')}"
    )

    # -------------------- ANALYSIS LOOP (BATCHED) --------------------
    os.makedirs(save_dir, exist_ok=True)
    results = []
    total_segments = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Analyzing segments"):
            batch_size_curr = batch_x.size(0)
            batch_x = batch_x.to(device)  # (B,C,T)
            batch_y = batch_y.to(device)  # (B,)

            p, extra = model(batch_x)       # p: (B,), s_res: (B,H,T)
            s_res = extra["s_res"].cpu().numpy()
            probs = p.cpu().numpy()
            ys = batch_y.cpu().numpy()

            # Process each element in the batch for ISI entropy
            for b in range(batch_size_curr):
                idx = processed + b
                spikes = s_res[b]  # (H,T)
                isi_H = compute_isi_entropy(spikes)

                label_str = "seizure" if ys[b] == 1.0 else "nonseizure"

                results.append(
                    {
                        "idx": int(idx),
                        "label": label_str,
                        "y": int(ys[b]),
                        "prob": float(probs[b]),
                        "isi_entropy": float(isi_H),
                    }
                )

            processed += batch_size_curr

    print(f"✅ Processed all {processed} segments.")

    # -------------------- SAVE RAW RESULTS --------------------
    csv_path = os.path.join(save_dir, f"{dataset_name}_complexity_{mode}.csv")
    json_path = os.path.join(save_dir, f"{dataset_name}_complexity_{mode}.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "label", "y", "prob", "isi_entropy"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Saved CSV to {csv_path}")
    print(f"💾 Saved JSON to {json_path}")

    # -------------------- GROUP STATS & PLOTS --------------------
    probs = np.array([r["prob"] for r in results])
    isi_Hs = np.array([r["isi_entropy"] for r in results])
    labels = np.array([r["y"] for r in results])  # 0=non, 1=seizure

    seizure_mask = labels == 1
    non_mask = labels == 0

    isi_seiz = isi_Hs[seizure_mask]
    isi_non = isi_Hs[non_mask]

    print(
        f"\nISI entropy (chaos={mode}, dataset={dataset_name}): "
        f"mean_seiz={isi_seiz.mean():.6f}, mean_non={isi_non.mean():.6f}"
    )

    # Mann–Whitney U test (non-parametric)
    try:
        stat, pval = mannwhitneyu(isi_seiz, isi_non, alternative="two-sided")
        print(f"Mann-Whitney U test: U={stat:.3f}, p={pval:.3e}")
    except Exception as e:
        print(f"⚠️ Mann-Whitney test failed: {e}")
        stat, pval = np.nan, np.nan

    # Boxplot of ISI entropy
    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [isi_non, isi_seiz],
        labels=["Non-seizure", "Seizure"],
        showfliers=True,
    )
    plt.ylabel("ISI Entropy (normalized)")
    plt.title(f"ISI Entropy by Class ({dataset_name}, {mode})")
    plt.tight_layout()
    boxplot_path = os.path.join(
        save_dir, f"{dataset_name}_complexity_boxplot_{mode}.png"
    )
    plt.savefig(boxplot_path, dpi=200)
    plt.close()
    print(f"📊 Saved boxplot to {boxplot_path}")

    # Scatter: prob vs ISI entropy
    plt.figure(figsize=(5, 4))
    plt.scatter(isi_non, probs[non_mask], alpha=0.7, label="Non-seizure", s=10)
    plt.scatter(isi_seiz, probs[seizure_mask], alpha=0.7, label="Seizure", s=10)
    plt.xlabel("ISI Entropy")
    plt.ylabel("Seizure probability")
    plt.title(f"Complexity vs Probability ({dataset_name}, {mode})")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(
        save_dir, f"{dataset_name}_complexity_scatter_{mode}.png"
    )
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"📊 Saved scatter plot to {scatter_path}")

    print("\n✅ Full complexity analysis finished.")


if __name__ == "__main__":
    analyze_bonn_full_complexity(
        dataset_name="bonn",
        mode="chaos_on",
        save_dir="analysis/bonn_full",
        batch_size=32,   # you can increase if GPU can handle it
    )
