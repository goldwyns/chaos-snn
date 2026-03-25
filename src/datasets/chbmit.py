# src/datasets/chbmit.py
"""
Memory-safe CHB-MIT window dataset with optional chaos-precompute.
Mode: detection (preictal OR ictal -> label=1) as requested (D1).
"""

import os
import re
import gc
import numpy as np
import mne
import hashlib
from torch.utils.data import Dataset
from src.utils.chaos_utils import chaos_modulate_numpy  # must exist

# -------------------------
# helpers: file discovery & parsing
# -------------------------
def load_all_chb_files(base_dir):
    all_files = {}
    for root, dirs, files in os.walk(base_dir):
        base = os.path.basename(root)
        if not base.startswith("chb"):
            continue
        edf_files = [f for f in files if f.endswith(".edf")]
        summary_files = [f for f in files if f.endswith("-summary.txt")]
        if not summary_files or len(edf_files) == 0:
            continue
        summary_path = os.path.join(root, summary_files[0])
        lst = []
        for edf in sorted(edf_files):
            lst.append({"eeg_file": os.path.join(root, edf), "summary_file": summary_path, "file_name": edf})
        if lst:
            all_files[base] = lst
    return all_files

def get_seizure_times_from_summary(summary_file_path, file_name):
    seizure_times = []
    try:
        with open(summary_file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    file_section = False
    cur_start = None
    for line in lines:
        if f"File Name: {file_name}" in line:
            file_section = True
            cur_start = None
            continue
        if not file_section:
            continue
        start_m = re.search(r"Seizure Start Time: (\d+) seconds", line)
        end_m = re.search(r"Seizure End Time: (\d+) seconds", line)
        if start_m:
            cur_start = int(start_m.group(1))
        if end_m and cur_start is not None:
            cur_end = int(end_m.group(1))
            seizure_times.append((cur_start, cur_end))
            cur_start = None
        # end when next file section begins
        if "File Name:" in line and file_name not in line:
            break
    return seizure_times

# -------------------------
# mne loading + preprocessing
# -------------------------
def load_eeg_data(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_eeg(raw_data, low_freq=0.5, high_freq=40.0, notch_freq=60.0, downsample_sfreq=None, verbose=False):
    # pick EEG
    raw_data.pick_types(eeg=True)
    # bandpass
    raw_data.filter(l_freq=low_freq, h_freq=high_freq, n_jobs=1, verbose=False)
    # notch
    if notch_freq is not None:
        raw_data.notch_filter(freqs=notch_freq, n_jobs=1, verbose=False)
    # resample only if requested
    if downsample_sfreq is not None and raw_data.info["sfreq"] != downsample_sfreq:
        raw_data.resample(sfreq=downsample_sfreq, n_jobs=1, verbose=False)
    return raw_data

def create_labeled_windows(raw_data, seizure_times, ph_minutes=10, window_size_sec=5.0, overlap_sec=2.5, mode="detection"):
    sfreq = raw_data.info["sfreq"]
    ph_sec = ph_minutes * 60
    preictal_periods = [(max(0, s - ph_sec), s) for s, e in seizure_times]
    ictal_periods = [(s, e) for s, e in seizure_times]
    step_sec = window_size_sec - overlap_sec
    epochs = []
    labels = []
    metas = []
    start_time = raw_data.times[0]
    end_time = raw_data.times[-1]
    t = start_time
    while t + window_size_sec <= end_time:
        ws = t
        we = t + window_size_sec
        is_pre = any(max(ws, ps) < min(we, pe) for ps, pe in preictal_periods)
        is_ict = any(max(ws, is_) < min(we, ie) for is_, ie in ictal_periods)
        data, _ = raw_data.get_data(start=int(ws * sfreq), stop=int(we * sfreq), return_times=True)
        # mode 'detection' : label 1 if preictal OR ictal
        if mode == "detection":
            label = 1 if (is_pre or is_ict) else 0
            epochs.append(data.astype(np.float32))
            labels.append(label)
        elif mode == "forecast":
            # skip ictal windows
            if not is_ict:
                label = 1 if is_pre else 0
                epochs.append(data.astype(np.float32))
                labels.append(label)
        else:
            raise ValueError("mode must be 'detection' or 'forecast'")
        metas.append({"start": ws, "end": we})
        t += step_sec
    if len(epochs) == 0:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    windows = np.stack(epochs, axis=0)
    labels = np.array(labels, dtype=np.float32)
    return windows, labels, metas

# -------------------------
# CHB dataset (memory-safe)
# -------------------------
class CHBMITWindowDataset(Dataset):
    def __init__(self,
                 base_dir,
                 patient_ids=None,
                 ph_minutes=10,
                 window_size_sec=5.0,
                 overlap_sec=2.5,
                 low_freq=0.5,
                 high_freq=40.0,
                 notch_freq=60.0,
                 downsample_sfreq=None,
                 chaos_precompute=True,
                 chaos_params=None,
                 max_neg_factor=5,
                 verbose=True):
        """
        chaos_precompute: if True, compute chaos-modulated mixed-drive U once and save under precomputed/
        chaos_params: dict with keys alpha,beta,gamma,lam and rng_seed
        """
        self.base_dir = base_dir
        self.patient_ids = patient_ids
        self.ph_minutes = ph_minutes
        self.window_size_sec = window_size_sec
        self.overlap_sec = overlap_sec
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.downsample_sfreq = downsample_sfreq
        self.chaos_precompute = chaos_precompute
        self.chaos_params = chaos_params or {"alpha":3.5,"beta":0.2,"gamma":0.5,"lam":0.7,"rng_seed":42}
        self.max_neg_factor = max_neg_factor
        self.verbose = verbose

        # outputs
        self.X = None   # (N, C, T) either raw preprocessed windows or U (mixed drive)
        self.y = None   # (N,)
        self.meta = []  # list of dicts per window (patient,file,start,end)

        # build
        self._build()

    def _build(self):
        all_files = load_all_chb_files(self.base_dir)
        if self.patient_ids is None:
            patients = sorted(all_files.keys())
        else:
            patients = [p for p in self.patient_ids if p in all_files]
        windows_list = []
        labels_list = []
        meta_list = []
        for pid in patients:
            if self.verbose:
                print(f"Processing {pid} ...")
            for finfo in all_files[pid]:
                raw = load_eeg_data(finfo["eeg_file"])
                if raw is None:
                    continue
                raw = preprocess_eeg(raw, low_freq=self.low_freq, high_freq=self.high_freq,
                                     notch_freq=self.notch_freq, downsample_sfreq=self.downsample_sfreq,
                                     verbose=False)
                seizure_times = get_seizure_times_from_summary(finfo["summary_file"], finfo["file_name"])
                windows, labels, metas = create_labeled_windows(raw, seizure_times,
                                                                ph_minutes=self.ph_minutes,
                                                                window_size_sec=self.window_size_sec,
                                                                overlap_sec=self.overlap_sec,
                                                                mode="detection")
                if windows.shape[0] == 0:
                    continue
                # attach patient/file to meta entries
                for m in metas:
                    m.update({"patient": pid, "file": finfo["file_name"]})
                windows_list.append(windows)
                labels_list.append(labels)
                meta_list.extend(metas)
                # free raw
                del raw
                gc.collect()
        if len(windows_list) == 0:
            self.X = np.empty((0,0,0), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.float32)
            self.meta = []
            return
        # concat
        X = np.concatenate(windows_list, axis=0).astype(np.float32)
        y = np.concatenate(labels_list, axis=0).astype(np.float32)
        meta = meta_list  # list length = X.shape[0]
        # balancing: keep all positives, subsample negatives
        pos_idx = np.where(y == 1.0)[0]
        neg_idx = np.where(y == 0.0)[0]
        if len(pos_idx) == 0:
            keep_idx = np.arange(len(y))
        else:
            max_neg = min(len(neg_idx), int(self.max_neg_factor * len(pos_idx)))
            if len(neg_idx) > max_neg:
                rng = np.random.default_rng(seed=42)
                neg_sel = rng.choice(neg_idx, size=max_neg, replace=False)
            else:
                neg_sel = neg_idx
            keep_idx = np.concatenate([pos_idx, neg_sel])
        # shuffle keep_idx deterministically
        rng = np.random.default_rng(seed=123)
        rng.shuffle(keep_idx)
        # keep subset and store meta aligned
        X = X[keep_idx]
        y = y[keep_idx]
        meta = [meta[i] for i in keep_idx]
        # optionally precompute chaos-drive U and replace X
        precomp_dir = os.path.join("precomputed", "chbmit")
        os.makedirs(precomp_dir, exist_ok=True)
        # create unique name from patient list + window config
        pid_hash = hashlib.md5("_".join(patients).encode()).hexdigest()[:8]
        precomp_name = f"U_{pid_hash}_w{int(self.window_size_sec)}_ov{int(self.overlap_sec*100)}.npz"
        precomp_path = os.path.join(precomp_dir, precomp_name)
        if self.chaos_precompute:
            if os.path.exists(precomp_path):
                try:
                    data = np.load(precomp_path)
                    U = data["U"].astype(np.float32)
                    if U.shape[0] != X.shape[0]:
                        print("Precompute mismatch shape; recomputing U.")
                        raise ValueError("shape mismatch")
                    self.X = U
                except Exception:
                    print("Failed to load precomputed file or mismatch — recomputing.")
                    U = chaos_modulate_numpy(X, **self.chaos_params)
                    try:
                        np.savez_compressed(precomp_path, U=U)
                        print("Saved precomputed U to:", precomp_path)
                    except Exception as e:
                        print("Warning: failed to save precomputed U:", e)
                    self.X = U
            else:
                print("Precomputing chaos-modulated mixed-drive U (may take minutes)...")
                U = chaos_modulate_numpy(X, **self.chaos_params)
                try:
                    np.savez_compressed(precomp_path, U=U)
                    print("Saved precomputed U to:", precomp_path)
                except Exception as e:
                    print("Warning: failed to save precomputed U:", e)
                self.X = U
        else:
            # keep raw preprocessed windows
            self.X = X
        self.y = y
        self.meta = meta
        # final log
        if self.verbose:
            print(f"\n✅ CHBMITWindowDataset built: {self.X.shape[0]} windows, C={self.X.shape[1]}, T={self.X.shape[2]}, pos={int((self.y==1).sum())}, neg={int((self.y==0).sum())}")
        # free big intermediates
        del windows_list, labels_list
        gc.collect()

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx]   # (C,T)
        y = self.y[idx]   # scalar float
        meta = self.meta[idx] if len(self.meta) > idx else {}
        return x, float(y), meta
