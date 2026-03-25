# src/precompute/precompute_chbmit_npz.py

import os
import argparse
import numpy as np
from src.datasets.chbmit import CHBMITWindowDataset
from src.utils.chaos_utils import chaos_modulate_numpy


def main(args):
    patients = args.patients.split(",")
    os.makedirs(args.out_dir, exist_ok=True)

    print("📦 Building CHB-MIT dataset (RAW)...")

    ds = CHBMITWindowDataset(
        base_dir=args.base_dir,
        patient_ids=patients,
        ph_minutes=args.ph_minutes,
        window_size_sec=args.window,
        overlap_sec=args.overlap,
        verbose=True,
    )

    X = ds.X.astype(np.float32)   # (N, C, T)
    y = ds.y.astype(np.int64)

    print("⚡ Computing chaos-modulated signals U...")
    U = chaos_modulate_numpy(
        X,
        alpha=3.5,
        beta=0.2,
        gamma=0.5,
        lam=0.7,
        rng_seed=42,
    ).astype(np.float32)

    # Save per-patient (clean & scalable)
    start = 0
    for p in patients:
        idx = [i for i, m in enumerate(ds.meta) if m["patient"] == p]
        if len(idx) == 0:
            continue

        out_path = os.path.join(args.out_dir, f"{p}.npz")
        np.savez_compressed(
            out_path,
            X=X[idx],
            U=U[idx],
            y=y[idx],
        )
        print(f"✅ Saved {out_path}  (N={len(idx)})")

    print("\n🎯 DONE: NPZ files contain X, U, y")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--out_dir", default="precomputed/chbmit_npz")
    ap.add_argument("--ph_minutes", type=int, default=10)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=2.5)
    args = ap.parse_args()
    main(args)
