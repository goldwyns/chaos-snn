# src/utils/save_preds_from_ckpt.py
"""
Produce CSV of predictions (idx,label,prob,patient,file,start,end) from a saved checkpoint.
Matches CHBMITWindowDataset produced meta.
"""
import os
import argparse
import csv
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.datasets.chbmit import CHBMITWindowDataset
from src.models.chaos_snn import ChaosSNNSeizureDetector

def build_model(ckpt_path, n_channels, dt, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # instantiate with defaults; we stored model_state earlier
    alpha = ckpt.get("alpha", 3.5) if isinstance(ckpt, dict) else 3.5
    model = ChaosSNNSeizureDetector(n_channels=n_channels, n_hidden=128,
                                    alpha=3.5, beta=0.2, gamma=0.5, lam=0.7,
                                    v_th=1.0, alpha_rec=ckpt.get("alpha_rec",1.0) if isinstance(ckpt, dict) else 1.0,
                                    dt=dt, tau_mem=20e-3, tau_syn=10e-3).to(device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        try:
            model.load_state_dict(ckpt)
        except Exception:
            pass
    model.eval()
    return model

def main(args):
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    patients = None
    if args.patients and args.patients.strip() != "":
        patients = [p.strip() for p in args.patients.split(",")]

    ds = CHBMITWindowDataset(base_dir=args.base_dir,
                             patient_ids=patients,
                             ph_minutes=args.ph_minutes,
                             window_size_sec=args.window_size_sec,
                             overlap_sec=args.overlap_sec,
                             downsample_sfreq=None,
                             chaos_precompute=True,
                             chaos_params={"alpha":3.5,"beta":0.2,"gamma":0.5,"lam":0.7,"rng_seed":42},
                             verbose=False)
    N = len(ds)
    if N == 0:
        print("Dataset empty.")
        return
    # infer dt
    x0, y0, m0 = ds[0]
    C, T = x0.shape
    dt = 1.0 / (T / args.window_size_sec)
    model = build_model(args.ckpt, n_channels=C, dt=dt, device=device)
    model.to(device)
    model.eval()

    out_csv = os.path.join(args.out_dir, args.out_csv)
    rows = []
    with torch.no_grad():
        for i in tqdm(range(N), desc="Predicting"):
            x, y, meta = ds[i]
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            p, extra = model(x_t)
            prob = float(p.detach().cpu().numpy().reshape(-1)[0])
            row = {"idx": i, "label": int(y), "prob": prob,
                   "patient": meta.get("patient",""), "file": meta.get("file",""),
                   "start": meta.get("start", ""), "end": meta.get("end","")}
            rows.append(row)
    # write csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx","label","prob","patient","file","start","end"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("Saved predictions to:", out_csv)
    # basic summary
    probs = np.array([r["prob"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = None
    summary = {"n": int(N), "n_pos": int(labels.sum()), "n_neg": int((labels==0).sum()), "auc": auc}
    summary_path = out_csv.replace(".csv","_summary.json")
    import json
    with open(summary_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print("Saved summary to:", summary_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="analysis/chb_preds")
    p.add_argument("--out_csv", default="chb_preds.csv")
    p.add_argument("--patients", default="")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ph_minutes", type=int, default=10)
    p.add_argument("--window_size_sec", type=float, default=5.0)
    p.add_argument("--overlap_sec", type=float, default=2.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    main(args)
