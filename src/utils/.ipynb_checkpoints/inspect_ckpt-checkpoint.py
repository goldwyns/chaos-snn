# src/utils/inspect_ckpt.py
import torch
import argparse
from src.models.chaos_snn import ChaosSNNSeizureDetector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_channels", type=int, default=23)
    ap.add_argument("--fs", type=float, default=256.0)
    args = ap.parse_args()

    # build a model (hyperparams must match training)
    model = ChaosSNNSeizureDetector(n_channels=args.n_channels, n_hidden=128, alpha=3.5, beta=0.2, gamma=0.5, lam=0.7, v_th=1.0, alpha_rec=1.0, dt=1.0/args.fs, tau_mem=20e-3, tau_syn=10e-3)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)

    total_weight_norm = 0.0
    for name, p in model.named_parameters():
        wnorm = p.data.norm(2).item()
        total_weight_norm += wnorm**2
        print(f"{name}: weight_norm={wnorm:.4e}, shape={tuple(p.shape)}")
    total_weight_norm = total_weight_norm**0.5
    print("Total weight norm:", total_weight_norm)

if __name__ == "__main__":
    main()
