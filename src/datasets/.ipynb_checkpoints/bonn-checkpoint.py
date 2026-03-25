import os
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.signal import butter, filtfilt


def bandpass_filter(x, fs, l_freq=0.5, h_freq=40.0, order=4):
    """
    x: (T,) or (C,T)
    """
    nyq = 0.5 * fs
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = butter(order, [low, high], btype="band")
    if x.ndim == 1:
        return filtfilt(b, a, x)
    else:
        # apply per channel
        out = np.zeros_like(x)
        for c in range(x.shape[0]):
            out[c] = filtfilt(b, a, x[c])
        return out


def zscore_per_channel(x, eps=1e-8):
    """
    x: (C,T)
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return (x - mean) / std


class BonnEEGDataset(Dataset):
    """
    Bonn EEG dataset:
      - root/
          A/, B/, C/, D/, E/
      - each file is a single-channel segment (e.g., 4097 samples)
      - A-D: non-seizure (y=0), E: seizure (y=1)
    """

    def __init__(
        self,
        root,
        fs=173.61,
        bandpass=(0.5, 40.0),
        seizure_sets=("E",),
        nonseizure_sets=("A", "B", "C", "D"),
        extension=".txt",
        device="cpu",
    ):
        super().__init__()
        self.root = root
        self.fs = fs
        self.bandpass = bandpass
        self.seizure_sets = seizure_sets
        self.nonseizure_sets = nonseizure_sets
        self.extension = extension
        self.device = device

        self.samples = []  # list of (filepath, label)

        for s in nonseizure_sets:
            d = os.path.join(root, s)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.endswith(extension):
                    self.samples.append((os.path.join(d, fname), 0))

        for s in seizure_sets:
            d = os.path.join(root, s)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.endswith(extension):
                    self.samples.append((os.path.join(d, fname), 1))

        if len(self.samples) == 0:
            raise RuntimeError(f"No Bonn EEG files found in {root} with extension {extension}")

        print(f"✅ BonnEEGDataset: loaded {len(self.samples)} segments from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # load signal
        sig = np.loadtxt(path)
        # ensure shape (C,T) with C=1
        if sig.ndim == 1:
            sig = sig[None, :]  # (1,T)
        elif sig.ndim > 2:
            raise ValueError(f"Unexpected shape {sig.shape} for file {path}")

        # band-pass
        if self.bandpass is not None:
            l_freq, h_freq = self.bandpass
            sig = bandpass_filter(sig, fs=self.fs, l_freq=l_freq, h_freq=h_freq)

        # z-score per channel
        sig = zscore_per_channel(sig)

        # convert to torch: (C,T)
        x = torch.from_numpy(sig.astype(np.float32))
        y = torch.tensor(label, dtype=torch.float32)

        # We keep shape (C,T); DataLoader will add batch dimension -> (B,C,T)
        return x, y
