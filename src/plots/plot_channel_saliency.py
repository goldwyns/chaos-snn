# src/plots/plot_channel_saliency.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map


def compute_channel_saliency(
    dataset_name="bonn",
    mode="chaos_on",
    batch_size=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------- Dataset ----------------
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
    n_channels, _ = x0.shape

    # ---------------- Model ----------------
    model = ChaosSNNSeizureDetector(
        n_channels=n_channels,
        n_hidden=128,
        alpha=3.5 if mode == "chaos_on" else 0.0,
        beta=0.2 if mode == "chaos_on" else 0.0,
        gamma=0.5 if mode == "chaos_on" else 0.0,
        lam=0.7 if mode == "chaos_on" else 1.0,
        alpha_rec=1.0 if mode == "chaos_on" else 0.8,
        dt=1.0 / fs,
        tau_mem=20e-3,
        tau_syn=10e-3,
    ).to(device)

    ckpt_path = f"checkpoints/chaos_snn_{dataset_name}_{mode}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Input weights: (hidden, channels)
    W_in = model.reservoir.w_in.weight.detach().cpu().numpy()

    channel_activity = np.zeros(n_channels)
    total_windows = 0

    # ---------------- Forward pass ----------------
    with torch.no_grad():
        for batch_x, _ in tqdm(loader, desc="Computing saliency"):
            batch_x = batch_x.to(device)

            _, extra = model(batch_x)
            s_res = extra["s_res"]  # (B, H, T)

            # Spike counts per neuron
            spike_counts = s_res.sum(dim=(0, 2)).cpu().numpy()  # (H,)

            # Accumulate weighted contribution per channel
            for c in range(n_channels):
                channel_activity[c] += np.sum(
                    np.abs(W_in[:, c]) * spike_counts
                )

            total_windows += batch_x.size(0)

    # Normalize
    saliency = channel_activity / (total_windows + 1e-8)
    saliency = saliency / (saliency.max() + 1e-8)

    return saliency


if __name__ == "__main__":
    saliency = compute_channel_saliency(
        dataset_name="bonn",
        mode="chaos_on",  # edge-of-chaos
    )

    np.save("analysis/bonn_channel_saliency.npy", saliency)
    print("✅ Channel saliency saved.")
