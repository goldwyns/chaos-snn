# src/train_bonn.py

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map


def build_model(n_channels, fs, mode="chaos_on", n_hidden=128, device="cpu"):
    dt = 1.0 / fs

    if mode == "chaos_on":
        alpha = 3.5
        beta = 0.2
        gamma = 0.5
        lam = 0.7
        alpha_rec = 1.0
        print("🧪 Mode: CHAOS ON")
    elif mode == "chaos_off":
        # Plain SNN: no chaos injection, slightly subcritical recurrence
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        lam = 1.0
        alpha_rec = 0.8
        print("🧪 Mode: CHAOS OFF (plain SNN)")
    else:
        raise ValueError(f"Unknown mode '{mode}' (use 'chaos_on' or 'chaos_off')")

    model = ChaosSNNSeizureDetector(
        n_channels=n_channels,
        n_hidden=n_hidden,
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


def train_bonn(
    dataset_name="bonn",
    batch_size=32,
    n_epochs=20,
    lr=5e-4,
    val_split=0.2,
    device=None,
    mode="chaos_on",
    seed=123,
):
    # -------------------- SETUP --------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    # 1. Dataset
    full_dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

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
    x0, _ = full_dataset[0]
    n_channels, T = x0.shape
    print(f"🧠 {dataset_name} sample shape: C={n_channels}, T={T}")

    # 2. Model
    model = build_model(
        n_channels=n_channels,
        fs=fs,
        mode=mode,
        n_hidden=128,
        device=device,
    )

    # 3. Loss & optimizer
    pos_weight = torch.tensor([2.0]).to(device)  # seizure ~1:4 non-seizure
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    ckpt_name = f"chaos_snn_{dataset_name}_{mode}_best.pt"

    # -------------------- TRAINING LOOP --------------------
    for epoch in range(1, n_epochs + 1):
        # -------- TRAIN --------
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x = x.to(device)  # (B,C,T)
            y = y.to(device)  # (B,)

            optimizer.zero_grad()
            p, extra = model(x)
            logits = extra["logits"]
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / n_train

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
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

        # positive rate at default 0.5 threshold
        pos_rate = (y_pred > 0.5).astype(float).mean()
        print(f"Epoch {epoch}: predicted positive rate (p>0.5) = {pos_rate:.3f}")

        # AUC
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = 0.5

        # Scan thresholds to find best G-mean (balance)
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

        # Also report accuracy at 0.5 just for reference
        preds_05 = (y_pred > 0.5).astype(float)
        acc_05 = (preds_05 == y_true).astype(float).mean()

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"AUC={auc:.3f}, ACC@0.5={acc_05:.3f}"
        )
        print(
            f"           best_thr={best_thr:.2f}, "
            f"SENS={best_sens:.3f}, SPEC={best_spec:.3f}, "
            f"G-mean={best_gmean:.3f}"
        )

        # Save best model by AUC
        if auc > best_val_auc:
            best_val_auc = auc
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
                    "mode": mode,
                    "dataset": dataset_name,
                },
                ckpt_path,
            )
            print(f"💾 Saved new best model to {ckpt_path} (AUC={auc:.3f})")

    print(f"\n✅ Training finished. Best val AUC: {best_val_auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bonn")
    parser.add_argument("--mode", type=str, default="chaos_on", choices=["chaos_on", "chaos_off"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train_bonn(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        mode=args.mode,
    )
