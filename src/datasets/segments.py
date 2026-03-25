# src/datasets/segments.py

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

from src.utils.load_dataset import load_dataset  # your existing loader
from src.utils.load_dataset import fs_map   # if fs_map is defined there; otherwise redefine here
# If fs_map is in another file, adjust the import accordingly.

def bandpass_filter(x, fs, l_freq=0.5, h_freq=40.0, order=4):
    """
    Robust band-pass filter:
    - x: np.ndarray, shape (C, T) or (T,)
    - If T <= padlen (needed by filtfilt), skip filtering and just demean.
    """
    # Ensure 2D: (C, T)
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]  # (1, T)

    nyq = 0.5 * fs
    low = l_freq / nyq
    high = h_freq / nyq

    b, a = butter(order, [low, high], btype="band")
    out = np.zeros_like(x)

    # filtfilt's default padlen = 3 * (max(len(a), len(b)) - 1)
    default_padlen = 3 * (max(len(a), len(b)) - 1)
    T = x.shape[1]

    for c in range(x.shape[0]):
        if T <= default_padlen:
            # Too short for filtfilt: just subtract mean as a fallback
            # (prevents crash on tiny segments in Bern, etc.)
            out[c] = x[c] - np.mean(x[c])
            # Optional: print once if you want to see how many are affected
            # print(f"⚠️ Skipping bandpass for short segment (T={T} <= padlen={default_padlen})")
        else:
            out[c] = filtfilt(b, a, x[c])

    return out



def zscore_per_channel(x, eps=1e-8):
    """
    x: (C,T)
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return (x - mean) / std


class SegmentEEGDataset(Dataset):
    """
    Wraps your load_dataset() output into a PyTorch Dataset.

    - Supports: bonn, bern, hauz_khas, panwar
    - Converts each 1D segment to shape (C=1, T)
    - Applies band-pass + z-score
    - Assigns binary label: seizure(1) vs non-seizure(0)
      (with reasonable defaults per dataset)
    """

    def __init__(
        self,
        dataset_name,
        base_path,
        fs=None,
        bandpass=(0.5, 40.0),
        device="cpu",
        hauz_mode="ictal_vs_interictal",
    ):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.base_path = base_path
        self.bandpass = bandpass
        self.device = device

        # Sampling rate
        if fs is None:
            if self.dataset_name in fs_map:
                self.fs = fs_map[self.dataset_name]
            else:
                raise ValueError(f"No fs provided and dataset {self.dataset_name} not in fs_map")
        else:
            self.fs = fs

        # Load all segments using your existing code
        self.data_dict = load_dataset(self.dataset_name, self.base_path)
        # self.data_dict is a dict: label -> list of 1D numpy arrays

        # Build flat list of (signal, label)
        self.samples = []
        if self.dataset_name == "bonn":
            # A,B,C,D: non-seizure (0); E: seizure (1)
            seizure_keys = ["E"]
            nonseizure_keys = ["A", "B", "C", "D"]
            for k in nonseizure_keys:
                if k in self.data_dict:
                    for sig in self.data_dict[k]:
                        self.samples.append((sig, 0))
            for k in seizure_keys:
                if k in self.data_dict:
                    for sig in self.data_dict[k]:
                        self.samples.append((sig, 1))

        elif self.dataset_name == "bern":
            # F: seizure, N: non-seizure (according to your loader)
            seizure_keys = ["F"]
            nonseizure_keys = ["N"]
            for k in nonseizure_keys:
                if k in self.data_dict:
                    for sig in self.data_dict[k]:
                        self.samples.append((sig, 0))
            for k in seizure_keys:
                if k in self.data_dict:
                    for sig in self.data_dict[k]:
                        self.samples.append((sig, 1))

        elif self.dataset_name == "panwar":
            # 'epileptic' vs 'healthy'
            seizure_keys = ["epileptic"]
            nonseizure_keys = ["healthy"]
            for k in nonseizure_keys:
                if k in self.data_dict:
                    for sig in self.data_dict[k]:
                        self.samples.append((sig, 0))
            for k in seizure_keys:
                if k in self.data_dict:
                    for sig in self.data_dict[k]:
                        self.samples.append((sig, 1))

        elif self.dataset_name == "hauz_khas":
            # Many options; simplest: ictal=1, interictal=0, ignore preictal
            # You can adjust this later for 3-class
            if hauz_mode == "ictal_vs_interictal":
                for sig in self.data_dict.get("interictal", []):
                    self.samples.append((sig, 0))
                for sig in self.data_dict.get("ictal", []):
                    self.samples.append((sig, 1))
            else:
                raise NotImplementedError(f"hauz_khas mode {hauz_mode} not implemented yet.")

        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported in SegmentEEGDataset")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples collected from dataset {self.dataset_name} at {self.base_path}"
            )

        print(
            f"✅ SegmentEEGDataset[{self.dataset_name}]: "
            f"{len(self.samples)} segments, fs={self.fs} Hz"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sig, label = self.samples[idx]  # sig: 1D numpy array

        # Ensure shape (C=1, T)
        sig = np.asarray(sig)
        if sig.ndim == 1:
            sig = sig[None, :]  # (1,T)
        elif sig.ndim > 2:
            raise ValueError(f"Unexpected signal shape {sig.shape}")

        # Band-pass
        if self.bandpass is not None:
            l_freq, h_freq = self.bandpass
            sig = bandpass_filter(sig, fs=self.fs, l_freq=l_freq, h_freq=h_freq)

        # Z-score per channel
        sig = zscore_per_channel(sig)

        # To torch: (C,T)
        x = torch.from_numpy(sig.astype(np.float32))
        y = torch.tensor(label, dtype=torch.float32)

        return x, y
