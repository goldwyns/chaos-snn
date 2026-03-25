import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.chbmit_npz import CHBMITNPZDataset
from src.models.chaos_snn import ChaosSNNSeizureDetector


def load_model(ckpt_path, device, C, T, window_size_sec=5.0):
    ckpt = torch.load(ckpt_path, map_location=device)

    fs = T / window_size_sec
    dt = 1.0 / fs

    model = ChaosSNNSeizureDetector(
        n_channels=C,
        n_hidden=128,
        alpha=ckpt.get("alpha", 3.5),
        beta=ckpt.get("beta", 0.2),
        gamma=ckpt.get("gamma", 0.5),
        lam=ckpt.get("lam", 0.7),
        v_th=1.0,
        alpha_rec=ckpt.get("alpha_rec", 1.0),
        dt=dt,
        tau_mem=20e-3,
        tau_syn=10e-3,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients = args.patients.split(",")

    # Load NPZ dataset
    ds = CHBMITNPZDataset(args.npz_dir, patients, use="U")
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    C, T = ds.X.shape[1], ds.X.shape[2]

    model = load_model(args.ckpt, device, C, T)

    spike_energy = []
    firing_rate = []

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= args.max_batches:
                break

            x = x.to(device)
            p, extra = model(x)

            # ---- ENERGY PROXY (robust) ----
            if "spikes" in extra:
                spikes = extra["spikes"]
                spike_energy.append(spikes.sum().item())
                firing_rate.append(spikes.mean().item())
            else:
                # fallback: membrane activity proxy
                spike_energy.append(p.abs().sum().item())
                firing_rate.append(p.abs().mean().item())

    results = {
        "mean_energy": float(np.mean(spike_energy)),
        "std_energy": float(np.std(spike_energy)),
        "mean_firing_rate": float(np.mean(firing_rate)),
        "std_firing_rate": float(np.std(firing_rate)),
        "energy_proxy": "spike-count if available else membrane activity",
    }

    import json, os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n📊 H4 RESULTS (Energy / Latency Proxy)")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("💾 Saved to", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="analysis/h4_energy_latency.json")
    ap.add_argument("--max_batches", type=int, default=100)
    args = ap.parse_args()
    main(args)
