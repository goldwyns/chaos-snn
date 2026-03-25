# src/analyze_bonn_spikes.py

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map


def build_model_for_analysis(n_channels, fs, mode="chaos_on", device="cpu"):
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


def plot_spike_raster(spikes, title, out_path):
    """
    spikes: (N, T) binary numpy
    """
    N, T = spikes.shape
    plt.figure(figsize=(8, 4))
    for i in range(N):
        ts = np.where(spikes[i] > 0)[0]
        plt.vlines(ts, i + 0.5, i + 1.5, linewidth=0.5)
    plt.xlabel("Time (samples)")
    plt.ylabel("Neuron index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_isi_entropy(spikes, eps=1e-8, n_bins=20):
    """
    Improved ISI entropy:
    - spikes: (N, T) binary numpy array
    - For each neuron, compute its isi's, then pool all isi's.
    - Return entropy in [0,1], where 0 = very regular, 1 = very complex.
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

    # Histogram of ISIs
    hist, edges = np.histogram(all_isi, bins=n_bins)
    if hist.sum() == 0:
        return 0.0

    p = hist.astype(float) / (hist.sum() + eps)
    p = p[p > 0]

    H = -np.sum(p * np.log(p + eps))                # ≥ 0
    H_norm = H / (np.log(len(hist) + eps))          # normalize to ~[0,1]

    return float(H_norm)



def analyze_bonn_spikes(
    dataset_name="bonn",
    mode="chaos_on",
    n_seizure_examples=3,
    n_nonseizure_examples=3,
    save_dir="analysis/bonn",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    # 1. Dataset
    dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

    # 2. Build model & load checkpoint
    x0, _ = dataset[0]
    n_channels, T = x0.shape
    model = build_model_for_analysis(n_channels, fs, mode=mode, device=device)

    ckpt_name = f"chaos_snn_{dataset_name}_{mode}_best.pt"
    ckpt_path = os.path.join("checkpoints", ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}, best AUC={ckpt.get('val_auc', 'N/A')}")

    os.makedirs(save_dir, exist_ok=True)

    # 3. Select example indices
    seizure_indices = []
    nonseizure_indices = []

    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if y.item() == 1 and len(seizure_indices) < n_seizure_examples:
            seizure_indices.append(idx)
        if y.item() == 0 and len(nonseizure_indices) < n_nonseizure_examples:
            nonseizure_indices.append(idx)
        if (
            len(seizure_indices) >= n_seizure_examples
            and len(nonseizure_indices) >= n_nonseizure_examples
        ):
            break

    print(f"Selected seizure indices: {seizure_indices}")
    print(f"Selected non-seizure indices: {nonseizure_indices}")

    # 4. Run model & save spike rasters and ISI entropy
    results = []

    def process_example(idx, label_str):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)  # (1,C,T)

        with torch.no_grad():
            p, extra = model(x)

        s_res = extra["s_res"].cpu().numpy()  # (1, H, T)
        spikes = s_res[0]  # (H, T)

        # optionally subsample neurons for nicer plots
        H, TT = spikes.shape
        max_neurons_plot = min(64, H)
        spikes_plot = spikes[:max_neurons_plot, :]

        isi_H = compute_isi_entropy(spikes)
        prob = float(p.cpu().item())

        title = f"{label_str} idx={idx}, p={prob:.3f}, ISI_H={isi_H:.3f}"
        out_png = os.path.join(save_dir, f"{label_str}_idx{idx}.png")
        plot_spike_raster(spikes_plot, title, out_png)

        print(f"Saved raster: {out_png}")
        results.append(
            {
                "idx": idx,
                "label": label_str,
                "prob": prob,
                "isi_entropy": isi_H,
            }
        )

    for idx in seizure_indices:
        process_example(idx, "seizure")

    for idx in nonseizure_indices:
        process_example(idx, "nonseizure")

    # 5. Save results as JSON or npy
    import json

    out_json = os.path.join(save_dir, f"spike_analysis_{mode}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis summary to {out_json}")


if __name__ == "__main__":
    # Example usage for chaos ON
    analyze_bonn_spikes(
        dataset_name="bonn",
        mode="chaos_on",
        n_seizure_examples=3,
        n_nonseizure_examples=3,
    )

    # You can also run chaos OFF later:
    # analyze_bonn_spikes(dataset_name="bonn", mode="chaos_off", ...)
