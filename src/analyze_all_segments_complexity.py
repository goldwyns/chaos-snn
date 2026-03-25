"""
Analyze chaos-related complexity (ISI entropy) for segment datasets:
  - bonn
  - panwar
  - hauz_khas

For each dataset and regime, this script:
  1) Loads the trained Chaos-SNN checkpoint
  2) Runs a forward pass on ALL segments
  3) Computes ISI entropy from reservoir spikes
  4) Saves CSV: analysis/{dataset}_full/{dataset}_complexity_{regime}.csv
  5) After all runs, prints a summary per dataset & regime:
        mean ISI(seiz), mean ISI(non), Mann–Whitney p

Assumptions:
  - Checkpoints are named: checkpoints/chaos_snn_{dataset}_{regime}_best.pt
  - SegmentEEGDataset and ChaosSNNSeizureDetector match your training code
  - extra["s_res"] in model forward contains reservoir spikes of shape (B, H, T)
    If your key is different (e.g. "spikes"), just change EXTRA_SPIKE_KEY below.
"""

import os
import csv
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu

import torch
from torch.utils.data import DataLoader

from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map


# ----------------------------------------------------------
# Config
# ----------------------------------------------------------

DATASETS = ["bonn", "panwar", "hauz_khas"]
REGIMES = ["chaos_off", "edge", "sub", "super"]
BANDPASS = (0.5, 40.0)
BATCH_SIZE = 32
EXTRA_SPIKE_KEY = "s_res"   # <-- change here if different in your model.extra


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def build_model(n_channels, fs, regime, device):
    """
    Build ChaosSNNSeizureDetector with the same hyperparameters used in training
    for a given regime. Adjust if you changed them in run_segment_experiments.
    """
    dt = 1.0 / fs

    # Default: edge-of-chaos settings
    alpha = 3.5
    beta = 0.2
    gamma = 0.5
    lam = 0.7
    alpha_rec = 1.0

    if regime == "chaos_off":
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        lam = 1.0
        alpha_rec = 0.8
    elif regime == "sub":
        alpha_rec = 0.8
    elif regime == "super":
        alpha_rec = 1.2
    # else "edge": use default above

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


def compute_isi_entropy_for_sample(spikes_1d, n_bins=10, eps=1e-8):
    """
    Compute ISI entropy for a single 1D spike train (length T).
    spikes_1d: numpy array of shape (T,) with 0/1 spikes.
    """
    spike_times = np.where(spikes_1d > 0.5)[0]
    if spike_times.size <= 1:
        return 0.0

    dts = np.diff(spike_times)
    if dts.size == 0:
        return 0.0

    # Histogram of inter-spike intervals
    hist, _ = np.histogram(dts, bins=n_bins, density=True)
    p = hist[hist > 0]
    if p.size == 0:
        return 0.0

    H = -np.sum(p * np.log(p + eps))
    return float(H)


def compute_isi_entropy_batch(spikes):
    """
    spikes: numpy array of shape (B, H, T)
    Returns: list of ISI entropy values (length B), averaging across neurons.
    """
    B, H, T = spikes.shape
    entropies = []

    for b in range(B):
        ent_per_neuron = []
        for h in range(H):
            s_1d = spikes[b, h]
            H_isi = compute_isi_entropy_for_sample(s_1d)
            if H_isi > 0.0:
                ent_per_neuron.append(H_isi)

        if len(ent_per_neuron) == 0:
            entropies.append(0.0)
        else:
            entropies.append(float(np.mean(ent_per_neuron)))

    return entropies


# ----------------------------------------------------------
# Main analysis per dataset/regime
# ----------------------------------------------------------

def analyze_dataset_regime(dataset_name, regime, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n===== ANALYZE {dataset_name} | regime={regime} =====")

    if dataset_name not in dataset_paths:
        print(f"❌ Unknown dataset: {dataset_name}")
        return

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    # Output paths
    out_dir = os.path.join("analysis", f"{dataset_name}_full")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset_name}_complexity_{regime}.csv")
    json_path = os.path.join(out_dir, f"{dataset_name}_complexity_{regime}.json")

    # 1. Build dataset & dataloader
    full_dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=BANDPASS,
    )

    print(f"✅ SegmentEEGDataset[{dataset_name}]: {len(full_dataset)} segments, fs={fs} Hz")
    x0, y0 = full_dataset[0]
    n_channels, T = x0.shape
    print(f"🧠 {dataset_name} sample shape: C={n_channels}, T={T}")

    loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # 2. Build model & load checkpoint
    ckpt_name = f"chaos_snn_{dataset_name}_{regime}_best.pt"
    ckpt_path = os.path.join("checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path} -> skipping")
        return

    model = build_model(n_channels, fs, regime, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"Loaded checkpoint from {ckpt_path}, best metric key(s): {', '.join(k for k in ckpt.keys() if k not in ['model_state'])}")

    # 3. Forward pass & complexity computation
    all_rows = []
    idx_global = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Analyzing {dataset_name}/{regime}"):
            x = x.to(device)
            y = y.to(device)

            p, extra = model(x)  # p: (B,), extra: dict
            spikes_res = extra[EXTRA_SPIKE_KEY]  # expected shape: (B, H, T)
            spikes_res = spikes_res.detach().cpu().numpy()
            probs = p.detach().cpu().numpy()
            labels = y.detach().cpu().numpy()

            isi_ent_batch = compute_isi_entropy_batch(spikes_res)

            for i in range(x.size(0)):
                label_int = int(labels[i])
                label_str = "seizure" if label_int == 1 else "nonseizure"
                row = {
                    "idx": idx_global,
                    "label": label_str,
                    "label_int": label_int,
                    "prob": float(probs[i]),
                    "isi_entropy": float(isi_ent_batch[i]),
                }
                all_rows.append(row)
                idx_global += 1

    # 4. Save CSV + JSON
    fieldnames = ["idx", "label", "label_int", "prob", "isi_entropy"]
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    with open(json_path, "w") as f_json:
        json.dump(all_rows, f_json, indent=2)

    print(f"💾 Saved CSV to {csv_path}")
    print(f"💾 Saved JSON to {json_path}")


# ----------------------------------------------------------
# Summary across regimes
# ----------------------------------------------------------

def summarize_dataset_complexity(dataset_name):
    print(f"\n===== FINAL {dataset_name.upper()} REGIME SUMMARY =====\n")

    out_dir = os.path.join("analysis", f"{dataset_name}_full")

    for regime in REGIMES:
        csv_path = os.path.join(out_dir, f"{dataset_name}_complexity_{regime}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV for {dataset_name}/{regime}: {csv_path}")
            continue

        isi_seiz = []
        isi_non = []

        with open(csv_path, "r") as f_csv:
            reader = csv.DictReader(f_csv)
            for row in reader:
                ent = float(row["isi_entropy"])
                label_int = int(row["label_int"])
                if label_int == 1:
                    isi_seiz.append(ent)
                else:
                    isi_non.append(ent)

        if len(isi_seiz) == 0 or len(isi_non) == 0:
            print(f"Regime: {regime}")
            print("  ⚠️ Not enough seizure or nonseizure samples for Mann–Whitney test.")
            print("-" * 50)
            continue

        isi_seiz = np.array(isi_seiz)
        isi_non = np.array(isi_non)

        mean_seiz = float(isi_seiz.mean())
        mean_non = float(isi_non.mean())

        U, pval = mannwhitneyu(isi_seiz, isi_non, alternative="two-sided")

        print(f"Regime: {regime}")
        print(f"  mean ISI (seiz ) = {mean_seiz:.6f}")
        print(f"  mean ISI (non  ) = {mean_non:.6f}")
        print(f"  Mann–Whitney p  = {pval:.3e}")
        print("-" * 50)


# ----------------------------------------------------------
# Run everything
# ----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Run analysis for all dataset/regime combinations
    for dataset_name in DATASETS:
        for regime in REGIMES:
            analyze_dataset_regime(dataset_name, regime, device=device)

    # 2) Summarize per dataset
    for dataset_name in DATASETS:
        summarize_dataset_complexity(dataset_name)


if __name__ == "__main__":
    main()
