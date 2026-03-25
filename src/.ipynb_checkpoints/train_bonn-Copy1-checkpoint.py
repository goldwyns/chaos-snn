import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


from src.models.chaos_snn import ChaosSNNSeizureDetector
from src.datasets.segments import SegmentEEGDataset
from src.utils.load_dataset import dataset_paths, fs_map 


def train_bonn(
    dataset_name="bonn",
    batch_size=32,
    n_epochs=20,
    lr = 5e-4,
    val_split=0.2,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_path = dataset_paths[dataset_name]
    fs = fs_map[dataset_name]

    # 1. Load dataset via your loader + wrapper
    full_dataset = SegmentEEGDataset(
        dataset_name=dataset_name,
        base_path=base_path,
        fs=fs,
        bandpass=(0.5, 40.0),
    )

    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 2. Inspect one sample
    x0, _ = full_dataset[0]
    n_channels, T = x0.shape
    print(f"🧠 {dataset_name} sample shape: C={n_channels}, T={T}")

    # 3. Build chaos-SNN model
    dt = 1.0 / fs
    model = ChaosSNNSeizureDetector(
        n_channels=n_channels,
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

    # Approx seizure:non-seizure ratio in Bonn = 1:4
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        # -------------------- TRAIN --------------------
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

        # -------------------- VALIDATION --------------------
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x = x.to(device)
                y = y.to(device)
                p, extra = model(x)
                logits = extra["logits"]
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                y_true.append(y.cpu())
                y_pred.append(p.cpu())

        val_loss = val_loss / n_val
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        pos_rate = (y_pred > 0.5).float().mean().item()
        print(f"Epoch {epoch}: predicted positive rate (p>0.5) = {pos_rate:.3f}")


        preds_bin = (y_pred > 0.5).float()
        acc = (preds_bin == y_true).float().mean().item()

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={acc:.4f}"
        )

        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        
        import numpy as np

        # y_true, y_pred are 1D numpy arrays

        # 1. AUC as before
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.5
        
        # 2. Scan thresholds
        thresholds = np.linspace(0.0, 1.0, 51)  # 0.0, 0.02, ..., 1.0
        best_gmean = -1
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
        
            gmean = np.sqrt(sens * spec)  # geometric mean
        
            if gmean > best_gmean:
                best_gmean = gmean
                best_thr = thr
                best_sens = sens
                best_spec = spec
        
        print(
            f"Epoch {epoch}: AUC={auc:.3f}, "
            f"best_thr={best_thr:.2f}, "
            f"SENS={best_sens:.3f}, SPEC={best_spec:.3f}, "
            f"G-mean={best_gmean:.3f}"
        )


        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"chaos_snn_{dataset_name}_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"💾 Saved new best model to {ckpt_path}")


if __name__ == "__main__":
    train_bonn()
