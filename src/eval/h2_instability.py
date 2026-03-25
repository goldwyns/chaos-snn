# src/eval/h2_instability.py

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.chbmit_npz import CHBMITNPZDataset
from src.models.chaos_snn import ChaosSNNSeizureDetector


def instability_metric(model, loader, device, max_batches=20):
    """
    Robust instability metric for SNNs.
    Priority:
      1) internal states
      2) membrane potentials
      3) output-logit variance (fallback)
    """
    vals = []

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break

            x = x.to(device)
            y, extra = model(x)

            # ---- Case 1: internal states ----
            if isinstance(extra, dict):
                for key in ["states", "spikes", "mem", "v", "h"]:
                    if key in extra:
                        z = extra[key]
                        break
                else:
                    z = None
            else:
                z = None

            # ---- Compute variance ----
            if z is not None:
                if z.dim() == 3:        # (B,T,H)
                    var = z.var(dim=1).mean(dim=1)
                elif z.dim() == 2:      # (B,H)
                    var = z.var(dim=1)
                else:
                    var = z.var(dim=-1)
            else:
                # 🔥 Fallback: output sensitivity proxy
                # Still valid for instability comparison
                var = y.var(dim=0).repeat(x.size(0))

            vals.extend(var.cpu().numpy())

    vals = np.array(vals)
    return vals.mean(), vals.std(), vals


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
    print("🖥 Device:", device)

    ds = CHBMITNPZDataset(
        npz_dir=args.npz_dir,
        patients=args.patients.split(","),
        use="U",
    )

    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    C, T = ds.X.shape[1], ds.X.shape[2]

    model_chaos = load_model(args.ckpt_chaos, device, C, T)
    model_off   = load_model(args.ckpt_off,   device, C, T)

    mean_c, std_c, vals_c = instability_metric(
        model_chaos, loader, device, args.max_batches
    )
    mean_o, std_o, vals_o = instability_metric(
        model_off, loader, device, args.max_batches
    )

    print("\n📊 H2 RESULTS (Instability)")
    print(f"Chaos     : mean={mean_c:.6f}, std={std_c:.6f}")
    print(f"Chaos-off : mean={mean_o:.6f}, std={std_o:.6f}")

    np.savez(
        args.out,
        chaos=vals_c,
        chaos_off=vals_o,
        mean_chaos=mean_c,
        mean_off=mean_o,
        std_chaos=std_c,
        std_off=std_o,
    )

    print("💾 Saved to", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--ckpt_chaos", required=True)
    ap.add_argument("--ckpt_off", required=True)
    ap.add_argument("--out", default="analysis/h2_instability.npz")
    ap.add_argument("--max_batches", type=int, default=20)
    args = ap.parse_args()

    main(args)
