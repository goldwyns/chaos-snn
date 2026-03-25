# src/utils/save_preds_from_ckpt_npz.py

import os
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.chaos_snn import ChaosSNNSeizureDetector


# -----------------------------
# Dataset
# -----------------------------
class CHBMITNPZDataset(Dataset):
    def __init__(self, npz_dir, patients, use="U"):
        self.X = []
        self.y = []
        self.meta = []

        for p in patients:
            path = os.path.join(npz_dir, f"{p}.npz")
            data = np.load(path)

            Xp = data[use]   # "U" or "X"
            yp = data["y"]

            self.X.append(Xp)
            self.y.append(yp)

            for _ in range(len(yp)):
                self.meta.append({"patient": p})

        self.X = np.concatenate(self.X, axis=0).astype(np.float32)
        self.y = np.concatenate(self.y, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.meta[idx],
        )


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients = args.patients.split(",")

    print("🧠 Loading NPZ dataset...")
    ds = CHBMITNPZDataset(
        npz_dir=args.npz_dir,
        patients=patients,
        use=args.use,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,      # 🔥 Windows-safe
        pin_memory=True,
    )

    # Inspect shape
    C, T = ds[0][0].shape
    fs = T / args.window_size_sec
    dt = 1.0 / fs
    print(f"Sample shape: C={C}, T={T}, fs≈{fs:.2f}")

    # Build model
    model = ChaosSNNSeizureDetector(
        n_channels=C,
        n_hidden=128,
        alpha=3.5,
        beta=0.2,
        gamma=0.5,
        lam=0.7,
        v_th=1.0,
        alpha_rec=1.0,
        dt=dt,
        tau_mem=20e-3,
        tau_syn=10e-3,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.out_csv)

    print("💾 Saving predictions to:", out_csv)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prob", "label", "patient"],
        )
        writer.writeheader()

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device=="cuda")):
            for x, y, meta in tqdm(loader, desc="Running inference"):
                x = x.to(device)
                p, _ = model(x)

                probs = p.detach().cpu().numpy().reshape(-1)
                labels = y.numpy()

                patients_batch = meta["patient"]  # list of patient IDs

                for i in range(len(probs)):
                    writer.writerow({
                        "prob": float(probs[i]),
                        "label": int(labels[i]),
                        "patient": patients_batch[i],   # ✅ FIX
                    })


    print("✅ Done.")


# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--out_dir", default="analysis/chb_preds")
    ap.add_argument("--out_csv", default="chb_preds.csv")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--window_size_sec", type=float, default=5.0)
    ap.add_argument("--use", choices=["X", "U"], default="U")

    args = ap.parse_args()
    main(args)
