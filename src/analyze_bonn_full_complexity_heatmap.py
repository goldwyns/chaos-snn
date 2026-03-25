# src/analyze_bonn_full_complexity_heatmap.py

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


# ============================================================
# MODEL
# ============================================================

def build_model(n_channels, fs, mode="chaos_on", device="cpu"):
    dt = 1.0 / fs

    if mode == "chaos_on":
        alpha, beta, gamma, lam, alpha_rec = 3.5, 0.2, 0.5, 0.7, 1.0
        print("🧪 Analysis mode: CHAOS ON")
    elif mode == "chaos_off":
        alpha, beta, gamma, lam, alpha_rec = 0.0, 0.0, 0.0, 1.0, 0.8
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


# ============================================================
# ISI ENTROPY (ORIGINAL – KEPT)
# ============================================================

def compute_isi_entropy(spikes, eps=1e-8, n_bins=20):
    N, T = spikes.shape
    all_isi = []

    for i in range(N):
        ts = np.where(spikes[i] > 0)[0]
        if len(ts) >= 2:
            isi = np.diff(ts)
            if len(isi) > 0:
                all_isi.append(isi)

    if len(all_isi) == 0:
        return 0.0

    all_isi = np.concatenate(all_isi)
    hist, _ = np.histogram(all_isi, bins=n_bins)

    if hist.sum() == 0:
        return 0.0

    p = hist.astype(float) / (hist.sum() + eps)
    p = p[p > 0]

    H = -np.sum(p * np.log(p + eps))
    return float(H / (np.log(len(hist) + eps)))


# ============================================================
# NEW: PER-NEURON ISI ENTROPY (FOR HEATMAP)
# ============================================================

def compute_isi_entropy_per_neuron(spikes, eps=1e-8, n_bins=20):
    """
    spikes: (H, T)
    returns: (H,) per-neuron normalized ISI entropy
    """
    H, T = spikes.shape
    entropies = np.zeros(H)

    for i in range(H):
        ts = np.where(spikes[i] > 0)[0]
        if len(ts) >= 2:
            isi = np.diff(ts)
            if len(isi) > 1:
                hist, _ = np.histogram(isi, bins=n_bins)
                if hist.sum() > 0:
                    p = hist.astype(float) / (hist.sum() + eps)
                    p = p[p > 0]
                    Hi = -np.sum(p * np.log(p + eps))
                    entropies[i] = Hi / (np.log(len(hist) + eps))
    return entropies


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_bonn_full_complexity(
    dataset_name="bonn",
    mode="chaos_on",
    save_dir="analysis/bonn_full",
    batch_size=32,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x0, _ = dataset[0]
    n_channels, T = x0.shape

    model = build_model(n_channels, fs, mode=mode, device=device)

    ckpt_path = f"checkpoints/chaos_snn_{dataset_name}_{mode}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    results = []
    neuron_entropy_matrix = []  # NEW

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Analyzing segments"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            probs, extra = model(batch_x)
            s_res = extra["s_res"].cpu().numpy()

            for b in range(batch_x.size(0)):
                spikes = s_res[b]  # (H, T)

                isi_mean = compute_isi_entropy(spikes)
                isi_neurons = compute_isi_entropy_per_neuron(spikes)

                neuron_entropy_matrix.append(isi_neurons)

                results.append({
                    "label": "seizure" if batch_y[b].item() == 1 else "nonseizure",
                    "y": int(batch_y[b].item()),
                    "prob": float(probs[b].item()),
                    "isi_entropy": float(isi_mean),
                    "isi_entropy_neurons": isi_neurons.tolist(),  # NEW
                })

    # ============================================================
    # SAVE
    # ============================================================

    with open(os.path.join(save_dir, f"{dataset_name}_complexity_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, f"{dataset_name}_complexity_{mode}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "y", "prob", "isi_entropy"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # ============================================================
    # HEATMAP (NEURONS × WINDOWS)
    # ============================================================

    entropy_matrix = np.array(neuron_entropy_matrix).T  # (H, W)

    plt.figure(figsize=(8, 4))
    plt.imshow(entropy_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="ISI Entropy")
    plt.xlabel("Window Index")
    plt.ylabel("Reservoir Neuron Index")
    plt.title(f"Reservoir Complexity Heatmap ({dataset_name}, {mode})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_complexity_heatmap_{mode}.png"), dpi=300)
    plt.close()

    print("✅ Complexity heatmap generated.")


if __name__ == "__main__":
    analyze_bonn_full_complexity(
        dataset_name="bonn",
        mode="chaos_on",   # edge-of-chaos
        batch_size=32,
    )
