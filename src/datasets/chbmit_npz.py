import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CHBMITNPZDataset(Dataset):
    """
    NPZ-based CHB-MIT dataset.
    Loads precomputed EEG windows only (NO EDF, NO MNE).

    Each .npz file must contain:
      - X : (N, C, T) float32
      - y : (N,) int {0,1}
      - fs : sampling frequency
      - meta : optional metadata
    """

    def __init__(
        self,
        npz_dir,
        patients=None,
        use="X",           # "X" or "U" (future chaos-precomputed)
        return_meta=False,
        verbose=True,
    ):
        self.npz_dir = npz_dir
        self.patients = patients
        self.use = use
        self.return_meta = return_meta
        self.verbose = verbose

        self.X = None
        self.y = None
        self.meta = None
        self.fs = None

        self._load()

    def _load(self):
        files = sorted(f for f in os.listdir(self.npz_dir) if f.endswith(".npz"))

        if self.patients is not None:
            files = [f for f in files if f.replace(".npz", "") in self.patients]

        assert len(files) > 0, "❌ No NPZ files found for given patients"

        X_all, y_all, meta_all = [], [], []

        for f in files:
            path = os.path.join(self.npz_dir, f)
            data = np.load(path, allow_pickle=True)

            key = self.use
            assert key in data, f"❌ Key '{key}' not found in {f}"

            X = data[key].astype(np.float32)
            y = data["y"].astype(np.int64)

            X_all.append(X)
            y_all.append(y)

            if "meta" in data:
                meta_all.extend([(f.replace(".npz", ""), m) for m in data["meta"]])

            if self.fs is None and "fs" in data:
                self.fs = float(data["fs"])

            if self.verbose:
                print(
                    f"Loaded {f}: X={X.shape}, "
                    f"pos={(y==1).sum()}, neg={(y==0).sum()}"
                )

        self.X = np.concatenate(X_all, axis=0)
        self.y = np.concatenate(y_all, axis=0)
        self.meta = meta_all if meta_all else None

        if self.verbose:
            print(
                f"\n✅ CHBMITNPZDataset ready: "
                f"N={len(self.y)}, C={self.X.shape[1]}, T={self.X.shape[2]}"
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # (C, T)
        y = torch.tensor(self.y[idx]).float()  # scalar

        if self.return_meta and self.meta is not None:
            return x, y, self.meta[idx]

        return x, y
