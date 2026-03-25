# src/datasets/chbmit.py

import os
import re
import gc
import numpy as np
import mne

import torch
from torch.utils.data import Dataset
import os, hashlib
from src.utils.chaos_utils import chaos_modulate_numpy


# ----------------------------------------------------------
# 1. File discovery and seizure time parsing
# ----------------------------------------------------------

def load_all_chb_files(base_dir):
    """
    Finds and maps all EEG (.edf) and summary (-summary.txt) files in CHB-MIT.
    Returns: dict[patient_id] -> list of dicts with eeg_file, summary_file, file_name
    """
    all_files = {}
    for root, dirs, files in os.walk(base_dir):
        patient_id = os.path.basename(root)
        if not patient_id.startswith("chb"):
            continue

        patient_files = []
        edf_files = [f for f in files if f.endswith(".edf")]
        summary_files = [f for f in files if f.endswith("-summary.txt")]

        if not summary_files:
            continue

        summary_path = os.path.join(root, summary_files[0])

        for edf_file in edf_files:
            patient_files.append(
                {
                    "eeg_file": os.path.join(root, edf_file),
                    "summary_file": summary_path,
                    "file_name": edf_file,
                }
            )

        if patient_files:
            all_files[patient_id] = sorted(patient_files, key=lambda d: d["file_name"])

    return all_files


def get_seizure_times_from_summary(summary_file_path, file_name):
    """
    Parses a summary file to find seizure (start,end) times in seconds
    for a specific EDF file.
    Returns: list[(start_sec, end_sec)]
    """
    seizure_times = []
    try:
        with open(summary_file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    file_section_found = False
    seizure_start = None

    for line in lines:
        if f"File Name: {file_name}" in line:
            file_section_found = True
            seizure_start = None
            continue

        if file_section_found:
            start_match = re.search(r"Seizure Start Time: (\d+) seconds", line)
            end_match = re.search(r"Seizure End Time: (\d+) seconds", line)

            if start_match:
                seizure_start = int(start_match.group(1))
            elif end_match and seizure_start is not None:
                seizure_end = int(end_match.group(1))
                seizure_times.append((seizure_start, seizure_end))
                seizure_start = None

            # If a new "File Name" appears, stop this section
            if "File Name:" in line and file_name not in line:
                break

    return seizure_times


# ----------------------------------------------------------
# 2. MNE-based loading & preprocessing
# ----------------------------------------------------------

def load_eeg_data(file_path):
    """Loads a single EEG file into an MNE Raw object."""
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def preprocess_eeg(
    raw_data,
    low_freq=0.5,
    high_freq=40.0,
    notch_freq=60.0,
    downsample_sfreq=128.0,
    verbose=True,
):
    """
    Applies:
      - pick EEG channels
      - band-pass filter
      - notch filter
      - optional resampling
    Returns modified MNE Raw.
    """
    if verbose:
        print("Picking EEG channels only...")
    raw_data.pick_types(eeg=True)

    if verbose:
        print(f"Band-pass {low_freq}–{high_freq} Hz...")
    raw_data.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        n_jobs=1,
        verbose="ERROR",
    )

    if notch_freq is not None:
        if verbose:
            print(f"Notch filter at {notch_freq} Hz...")
        raw_data.notch_filter(
            freqs=notch_freq,
            n_jobs=1,
            verbose="ERROR",
        )

    if downsample_sfreq is not None and raw_data.info["sfreq"] != downsample_sfreq:
        if verbose:
            print(f"Resampling to {downsample_sfreq} Hz...")
        raw_data.resample(
            sfreq=downsample_sfreq,
            n_jobs=1,
            verbose="ERROR",
        )

    return raw_data


# ----------------------------------------------------------
# 3. Windowing and labeling
# ----------------------------------------------------------

def create_labeled_windows(
    raw_data,
    seizure_times,
    ph_minutes=10,
    window_size_sec=5.0,
    overlap_sec=2.5,
    mode="detection",
):
    """
    Splits continuous EEG into windows and assigns labels.

    Parameters
    ----------
    raw_data : mne.io.Raw
    seizure_times : list[(start_sec, end_sec)]
    ph_minutes : preictal horizon in minutes
    window_size_sec : window length
    overlap_sec : overlap between windows
    mode : "detection" or "forecast"

    Returns
    -------
    windows : np.ndarray, shape (N, C, T)
    labels  : np.ndarray, shape (N,)
      mode="detection":
        y=1 if window overlaps ictal or preictal
        y=0 otherwise
      mode="forecast":
        y=1 if window overlaps preictal
        y=0 if interictal
        ictal windows are excluded
    """
    sfreq = raw_data.info["sfreq"]
    ph_sec = ph_minutes * 60

    preictal_periods = []
    ictal_periods = []
    for start, end in seizure_times:
        preictal_periods.append((max(0, start - ph_sec), start))
        ictal_periods.append((start, end))

    epochs = []
    labels = []

    start_time = raw_data.times[0]
    end_time = raw_data.times[-1]
    step_sec = window_size_sec - overlap_sec

    current_time = start_time

    while current_time + window_size_sec <= end_time:
        window_start = current_time
        window_end = current_time + window_size_sec

        window_is_preictal = False
        window_is_ictal = False

        for pre_start, pre_end in preictal_periods:
            if max(window_start, pre_start) < min(window_end, pre_end):
                window_is_preictal = True
                break

        for ic_start, ic_end in ictal_periods:
            if max(window_start, ic_start) < min(window_end, ic_end):
                window_is_ictal = True
                break

        # Get data for this window: shape (C, T)
        data, _ = raw_data.get_data(
            start=int(window_start * sfreq),
            stop=int(window_end * sfreq),
            return_times=True,
        )

        if mode == "detection":
            # 1 if preictal OR ictal, 0 otherwise
            label = 1 if (window_is_preictal or window_is_ictal) else 0
            epochs.append(data)
            labels.append(label)

        elif mode == "forecast":
            # skip ictal windows, 1 if preictal, 0 if interictal
            if not window_is_ictal:
                label = 1 if window_is_preictal else 0
                epochs.append(data)
                labels.append(label)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        current_time += step_sec

    if len(epochs) == 0:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    windows = np.stack(epochs, axis=0).astype(np.float32)  # (N, C, T)
    labels = np.array(labels, dtype=np.float32)
    return windows, labels


# ----------------------------------------------------------
# 4. PyTorch Dataset: CHBMITWindowDataset
# ----------------------------------------------------------

class CHBMITWindowDataset(Dataset):
    """
    Window-level dataset for CHB-MIT, returning (C, T) EEG windows and labels.

    Example usage:
        ds = CHBMITWindowDataset(
            base_dir="E:/RESEARCH/DATABASE/CHB-MIT",
            patient_ids=["chb01", "chb02"],
            ph_minutes=10,
            window_size_sec=5.0,
            overlap_sec=2.5,
            mode="detection",
            downsample_sfreq=128.0,
        )
    """

    def __init__(
        self,
        base_dir,
        patient_ids=None,
        ph_minutes=10,
        window_size_sec=5.0,
        overlap_sec=2.5,
        mode="detection",
        low_freq=0.5,
        high_freq=40.0,
        notch_freq=60.0,
        downsample_sfreq=128.0,
        verbose=True,
    ):
        super().__init__()

        self.base_dir = base_dir
        self.patient_ids = patient_ids
        self.ph_minutes = ph_minutes
        self.window_size_sec = window_size_sec
        self.overlap_sec = overlap_sec
        self.mode = mode
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.downsample_sfreq = downsample_sfreq
        self.verbose = verbose

        # Storage
        self.X = None   # (N, C, T)
        self.y = None   # (N,)
        self.meta = []  # list of dicts with patient, file (optional)

        self._build()

    def _build(self):
        all_files_map = load_all_chb_files(self.base_dir)

        if self.patient_ids is None:
            patient_list = sorted(all_files_map.keys())
        else:
            patient_list = [p for p in self.patient_ids if p in all_files_map]

        all_windows = []
        all_labels = []

        for patient_id in patient_list:
            if self.verbose:
                print(f"\n🧑‍⚕️ Processing patient {patient_id}...")

            for f_info in all_files_map[patient_id]:
                eeg_file = f_info["eeg_file"]
                summary_file = f_info["summary_file"]
                fname = f_info["file_name"]

                seizure_times = get_seizure_times_from_summary(summary_file, fname)
                if self.verbose:
                    print(f"  File {fname}: found {len(seizure_times)} seizure interval(s).")

                raw = load_eeg_data(eeg_file)
                if raw is None:
                    continue

                raw = preprocess_eeg(
                    raw,
                    low_freq=self.low_freq,
                    high_freq=self.high_freq,
                    notch_freq=self.notch_freq,
                    downsample_sfreq=self.downsample_sfreq,
                    verbose=self.verbose,
                )

                windows, labels = create_labeled_windows(
                    raw_data=raw,
                    seizure_times=seizure_times,
                    ph_minutes=self.ph_minutes,
                    window_size_sec=self.window_size_sec,
                    overlap_sec=self.overlap_sec,
                    mode=self.mode,
                )

                if windows.shape[0] == 0:
                    continue

                # windows already float32 from create_labeled_windows
                all_windows.append(windows)
                all_labels.append(labels)

                # Meta info per window
                for _ in range(len(labels)):
                    self.meta.append(
                        {
                            "patient": patient_id,
                            "file": fname,
                        }
                    )

        # 🔚 AFTER all patients/files processed, now build X/y once
        if len(all_windows) == 0:
            print("⚠️ No windows found for the given configuration.")
            self.X = np.empty((0, 0, 0), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.float32)
            return

        # Concatenate once
        X = np.concatenate(all_windows, axis=0).astype(np.float32)
        y = np.concatenate(all_labels, axis=0).astype(np.float32)

        # Immediately free the big lists
        del all_windows, all_labels
        gc.collect()

        # 🔥 Class balancing: keep all positives, subsample negatives
        pos_idx = np.where(y == 1.0)[0]
        neg_idx = np.where(y == 0.0)[0]

        max_neg_factor = 5  # at most 5× more negatives than positives
        if len(pos_idx) == 0:
            print("⚠️ No positive windows found – using all data as is.")
            keep_idx = np.arange(len(y))
        else:
            max_neg = min(len(neg_idx), max_neg_factor * len(pos_idx))

            if len(neg_idx) > max_neg:
                rng = np.random.default_rng(seed=42)
                neg_idx_sel = rng.choice(neg_idx, size=max_neg, replace=False)
            else:
                neg_idx_sel = neg_idx

            keep_idx = np.concatenate([pos_idx, neg_idx_sel])

        # Shuffle keep_idx so training is not ordered
        rng = np.random.default_rng(seed=123)
        rng.shuffle(keep_idx)

        # inside CHBMITWindowDataset._build() after keep_idx and self.X/self.y assigned
        # add imports at top: from src.utils.chaos_utils import chaos_modulate_numpy
        # If you want to reuse precomputed file, set a path:
        precomp_dir = os.path.join("precomputed", "chbmit")
        os.makedirs(precomp_dir, exist_ok=True)
        
        # Build an ID for this patient set and window config
        patient_tag = "_".join(patient_list)  # careful if many patients
        precomp_name = f"u_precomputed_{patient_tag}_w{int(self.window_size_sec)}s_ov{int(self.overlap_sec*100)}.npz"
        precomp_path = os.path.join(precomp_dir, precomp_name)
        
        if os.path.exists(precomp_path):
            print("Loading precomputed chaos-drive from:", precomp_path)
            data = np.load(precomp_path)
            U = data["U"].astype(np.float32)
        else:
            print("Precomputing chaos-modulated mixed-drive u (this may take some minutes)...")
            # self.X shape (N, C, T) currently contains preprocessed signals x' (z-scored)
            X_full = self.X.astype(np.float32)
            # Choose chaos hyperparams (match training hyperparams)
            alpha = 3.5
            beta = 0.2
            gamma = 0.5
            lam = 0.7
            # compute (N,C,T) u
            U = chaos_modulate_numpy(X_full, alpha=alpha, beta=beta, gamma=gamma, lam=lam, rng_seed=42)
            # Save for reuse
            try:
                np.savez_compressed(precomp_path, U=U)
                print("Saved precomputed u to:", precomp_path)
            except Exception as e:
                print("Warning: failed to save precomputed file:", e)
        
        # Replace self.X with precomputed U (mixed drive)
        # If your model expects spike encoding inside, keep U; if model expects spikes, consider encoding spikes now.
        self.X = U  # dtype float32

        # Keep only the balanced subset
        self.y = y[keep_idx]

        # Free the big intermediate arrays
        del X, y
        gc.collect()

        if self.verbose and self.X.size > 0:
            print(
                f"\n✅ CHBMITWindowDataset built: {self.X.shape[0]} windows, "
                f"C={self.X.shape[1]}, T={self.X.shape[2]}, "
                f"pos={int((self.y == 1).sum())}, "
                f"neg={int((self.y == 0).sum())}"
            )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        y = self.y[idx]  # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
