# src/run_segment_experiments.py
#
# Unified experiment driver for segment-level EEG datasets:
#   - bonn
#   - bern
#   - panwar
#   - hauz_khas
#
# It can:
#   - Train Chaos-SNN under multiple regimes (sub, edge, super, chaos_off)
#   - Compute AUC, best threshold, SENS, SPEC
#   - Run full ISI entropy complexity analysis
#   - Save CSV/JSON + boxplot + scatter plots
#
# You can control which datasets & regimes to run using the ACTIVE_* lists
# at the bottom of this file.

import os
import json
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map


# ---------------------------------------------------------------------
# 1. Regime configuration
# ---------------------------------------------------------------------

def get_regime_config(regime: str):
    """
    Returns chaos/SNN hyperparameters for a given regime.
    You can tune these as you like.

    Regimes:
      - "sub":        subcritical (alpha_rec < 1)
      - "edge":       edge-of-chaos (alpha_rec ~ 1)
      - "super":      slightly supercritical (alpha_rec > 1)
      - "chaos_off":  plain SNN baseline (no chaos, subcritical)
    """
    if regime == "sub":
        cfg = dict(
            alpha=3.5,
            beta=0.2,
            gamma=0.5,
            lam=0.7,
            alpha_rec=0.7,
            desc="Subcritical chaos-SNN",
        )
    elif regime == "edge":
        cfg = dict(
            alpha=3.5,
            beta=0.2,
            gamma=0.5,
            lam=0.7,
            alpha_rec=1.0,
            desc="Edge-of-chaos SNN (current default)",
        )
    elif regime == "super":
        cfg = dict(
            alpha=3.5,
            beta=0.2,
            gamma=0.5,
            lam=0.7,
            alpha_rec=1.2,
            desc="Slightly supercritical chaos-SNN",
        )
    elif regime == "chaos_off":
        cfg = dict(
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            lam=1.0,
            alpha_rec=0.8,
            desc="Plain SNN (no chaos)",
        )
    else:
        raise ValueError(f"Unknown regime '{regime}'")

    return cfg


# ---------------------------------------------------------------------
# 2. Model builder
# ---------------------------------------------------------------------

def build_model(n_channels, fs, regime: str, n_hidden=128, device="cpu"):
    dt = 1.0 / fs
    cfg = get_regime_config(regime)

    print(f"🧪 Regime: {regime} — {cfg['desc']}")

    model = ChaosSNNSeizureDetector(
        n_channels=n_channels,
        n_hidden=n_hidden,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        lam=cfg["lam"],
        v_th=1.0,
        alpha_rec=cfg["alpha_rec"],
        dt=dt,
        tau_mem=20e-3,
        tau_syn=10e-3,
    ).to(device)

    return model


# ---------------------------------------------------------------------
# 3. Training for a single (dataset, regime)
# ---------------------------------------------------------------------
def train_segment_model(
    dataset_name: str,
    regime: str,
    batch_size=32,
    n_epochs=50,          # you can set 50 now
    lr=5e-4,
    val_split=0.2,
    device=None,
    seed=123,
    patience=10,          # 🔥 stop if no AUC improvement for 10 epochs
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n================ TRAIN {dataset_name} | regime={regime} ================")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    # Dataset
    full_dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

    print(f"✅ SegmentEEGDataset[{dataset_name}]: {len(full_dataset)} segments, fs={fs} Hz")

    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Inspect one sample
    x0, y0 = full_dataset[0]
    n_channels, T = x0.shape
    print(f"🧠 {dataset_name} sample shape: C={n_channels}, T={T}")

    # Model
    model = build_model(n_channels, fs, regime=regime, n_hidden=128, device=device)

    # Class imbalance: compute pos_weight = (#neg / #pos)
    labels_all = np.array([int(full_dataset[i][1].item()) for i in range(len(full_dataset))])
    n_pos = (labels_all == 1).sum()
    n_neg = (labels_all == 0).sum()
    if n_pos == 0:
        pos_weight = torch.tensor([1.0]).to(device)
        print("⚠️ No positive samples found; pos_weight=1.0")
    else:
        pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)]).to(device)
        print(f"Class counts: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    ckpt_name = f"chaos_snn_{dataset_name}_{regime}_best.pt"

    # 👇 early stopping
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train] {dataset_name}/{regime}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            p, extra = model(x)
            logits = extra["logits"]
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / n_train

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val] {dataset_name}/{regime}"):
                x = x.to(device)
                y = y.to(device)
                p, extra = model(x)
                logits = extra["logits"]
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                y_true_list.append(y.cpu())
                y_pred_list.append(p.cpu())

        val_loss = val_loss / n_val
        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()

        pos_rate = (y_pred > 0.5).astype(float).mean()
        print(f"Epoch {epoch}: predicted positive rate (p>0.5) = {pos_rate:.3f}")

        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = 0.5

        # threshold sweep for best G-mean
        thresholds = np.linspace(0.0, 1.0, 51)
        best_gmean = -1.0
        best_thr = 0.5
        best_sens = 0.0
        best_spec = 0.0

        for thr in thresholds:
            preds_bin = (y_pred > thr).astype(float)
            seizure_mask = (y_true == 1)
            non_mask = (y_true == 0)

            if seizure_mask.sum() == 0 or non_mask.sum() == 0:
                continue

            sens = (preds_bin[seizure_mask] == 1).astype(float).mean()
            spec = (preds_bin[non_mask] == 0).astype(float).mean()
            gmean = np.sqrt(max(sens, 0) * max(spec, 0))

            if gmean > best_gmean:
                best_gmean = gmean
                best_thr = thr
                best_sens = sens
                best_spec = spec

        preds_05 = (y_pred > 0.5).astype(float)
        acc_05 = (preds_05 == y_true).astype(float).mean()

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"AUC={auc:.3f}, ACC@0.5={acc_05:.3f}"
        )
        print(
            f"           best_thr={best_thr:.2f}, "
            f"SENS={best_sens:.3f}, SPEC={best_spec:.3f}, "
            f"G-mean={best_gmean:.3f}"
        )

        # ---- EARLY STOPPING LOGIC (by AUC) ----
        if auc > best_val_auc + 1e-4:  # small margin to ignore noise
            best_val_auc = auc
            best_epoch = epoch
            epochs_no_improve = 0

            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", ckpt_name)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_auc": auc,
                    "val_loss": val_loss,
                    "best_thr": best_thr,
                    "best_sens": best_sens,
                    "best_spec": best_spec,
                    "epoch": epoch,
                    "regime": regime,
                    "dataset": dataset_name,
                },
                ckpt_path,
            )
            print(f"💾 Saved new best model to {ckpt_path} (AUC={auc:.3f})")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch}: no AUC improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(
                f"🛑 Early stopping {dataset_name}/{regime} at epoch {epoch} "
                f"(no AUC improvement for {patience} epochs; best_epoch={best_epoch}, "
                f"best_AUC={best_val_auc:.3f})"
            )
            break

    print(f"✅ Finished training {dataset_name} | regime={regime}. Best AUC={best_val_auc:.3f} (epoch {best_epoch})")



# ---------------------------------------------------------------------
# 4. ISI entropy & complexity analysis for a single (dataset, regime)
# ---------------------------------------------------------------------

def compute_isi_entropy(spikes, eps=1e-8, n_bins=20):
    """
    ISI entropy:
    - spikes: (N, T) binary numpy array
    - compute ISIs per neuron, pool, and compute normalized entropy.
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

    hist, edges = np.histogram(all_isi, bins=n_bins)
    if hist.sum() == 0:
        return 0.0

    p = hist.astype(float) / (hist.sum() + eps)
    p = p[p > 0]

    H = -np.sum(p * np.log(p + eps))
    H_norm = H / (np.log(len(hist) + eps))
    return float(H_norm)


def analyze_segment_complexity(
    dataset_name: str,
    regime: str,
    save_dir_root="analysis",
    device=None,
    batch_size=32,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n================ ANALYZE {dataset_name} | regime={regime} ================")
    print(f"Using device: {device}")

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

    x0, y0 = dataset[0]
    n_channels, T = x0.shape
    print(f"🧠 {dataset_name} sample shape: C={n_channels}, T={T}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Model + checkpoint
    model = build_model(n_channels, fs, regime=regime, n_hidden=128, device=device)

    ckpt_name = f"chaos_snn_{dataset_name}_{regime}_best.pt"
    ckpt_path = os.path.join("checkpoints", ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"Loaded checkpoint from {ckpt_path}, "
        f"best AUC={ckpt.get('val_auc', 'N/A')}"
    )

    save_dir = os.path.join(save_dir_root, f"{dataset_name}_{regime}")
    os.makedirs(save_dir, exist_ok=True)

    # Analysis loop
    results = []
    total_segments = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc=f"Analyzing {dataset_name}/{regime}"):
            batch_size_curr = batch_x.size(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            p, extra = model(batch_x)
            s_res = extra["s_res"].cpu().numpy()  # (B,H,T)
            probs = p.cpu().numpy()
            ys = batch_y.cpu().numpy()

            for b in range(batch_size_curr):
                idx = processed + b
                spikes = s_res[b]  # (H,T)
                isi_H = compute_isi_entropy(spikes)
                label_str = "seizure" if ys[b] == 1.0 else "nonseizure"

                results.append(
                    {
                        "idx": int(idx),
                        "label": label_str,
                        "y": int(ys[b]),
                        "prob": float(probs[b]),
                        "isi_entropy": float(isi_H),
                    }
                )

            processed += batch_size_curr

    print(f"✅ Processed all {processed} segments for {dataset_name}/{regime}.")

    # Save CSV/JSON
    csv_path = os.path.join(save_dir, f"{dataset_name}_complexity_{regime}.csv")
    json_path = os.path.join(save_dir, f"{dataset_name}_complexity_{regime}.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "label", "y", "prob", "isi_entropy"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Saved CSV to {csv_path}")
    print(f"💾 Saved JSON to {json_path}")

    # Group stats & plots
    probs = np.array([r["prob"] for r in results])
    isi_Hs = np.array([r["isi_entropy"] for r in results])
    labels = np.array([r["y"] for r in results])  # 0=non, 1=seizure

    seizure_mask = labels == 1
    non_mask = labels == 0

    isi_seiz = isi_Hs[seizure_mask]
    isi_non = isi_Hs[non_mask]

    print(
        f"ISI entropy ({dataset_name}, regime={regime}): "
        f"mean_seiz={isi_seiz.mean():.6f}, mean_non={isi_non.mean():.6f}"
    )

    try:
        stat, pval = mannwhitneyu(isi_seiz, isi_non, alternative="two-sided")
        print(f"Mann-Whitney U: U={stat:.3f}, p={pval:.3e}")
    except Exception as e:
        print(f"⚠️ Mann-Whitney failed: {e}")
        stat, pval = np.nan, np.nan

    # Boxplot
    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [isi_non, isi_seiz],
        labels=["Non-seizure", "Seizure"],
        showfliers=True,
    )
    plt.ylabel("ISI Entropy (normalized)")
    plt.title(f"ISI Entropy by Class ({dataset_name}, {regime})")
    plt.tight_layout()
    boxplot_path = os.path.join(save_dir, f"{dataset_name}_complexity_boxplot_{regime}.png")
    plt.savefig(boxplot_path, dpi=200)
    plt.close()
    print(f"📊 Saved boxplot to {boxplot_path}")

    # Scatter: prob vs ISI entropy
    plt.figure(figsize=(5, 4))
    plt.scatter(isi_non, probs[non_mask], alpha=0.7, label="Non-seizure", s=10)
    plt.scatter(isi_seiz, probs[seizure_mask], alpha=0.7, label="Seizure", s=10)
    plt.xlabel("ISI Entropy")
    plt.ylabel("Seizure probability")
    plt.title(f"Complexity vs Probability ({dataset_name}, {regime})")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, f"{dataset_name}_complexity_scatter_{regime}.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"📊 Saved scatter plot to {scatter_path}")

    print(f"✅ Finished complexity analysis for {dataset_name} | regime={regime}.")


# ---------------------------------------------------------------------
# 5. MASTER MAIN: choose datasets & regimes here
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # 👇 You can comment/uncomment datasets and regimes as you like
    ACTIVE_DATASETS = [
         #"bonn",
         "panwar",
         "hauz_khas",
         "bern",
    ]

    ACTIVE_REGIMES = [
        "sub",
        "edge",
        "super",
        "chaos_off",
    ]

    for ds in ACTIVE_DATASETS:
        for reg in ACTIVE_REGIMES:
            # 1) Train
            train_segment_model(
                dataset_name=ds,
                regime=reg,
                batch_size=32,
                n_epochs=20,
                lr=5e-4,
            )

            # 2) Analyze complexity
            analyze_segment_complexity(
                dataset_name=ds,
                regime=reg,
                save_dir_root="analysis",
                batch_size=32,
            )
