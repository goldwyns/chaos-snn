# src/utils/load_dataset.py

import os
import numpy as np
import scipy.io

dataset_list = ['bonn','bern','hauz_khas','panwar']
dataset_paths = {
    'bonn': r'E:\RESEARCH\DATABASE\Bonn',
    'bern': r'E:\RESEARCH\DATABASE\Bern-Barcelona',
    'hauz_khas': r'E:\RESEARCH\DATABASE\Neurology_Sleep_Centre_Hauz Khas',
    'panwar': r'E:\RESEARCH\DATABASE\Panwar'
}
fs_map = {'bonn':173.61, 'bern':128, 'hauz_khas':256, 'panwar':256}
domain_map = {name:i for i,name in enumerate(dataset_list)}


def load_dataset(dataset_name, base_path):
    """
    Loads EEG datasets (Bonn, Bern, Hauz Khas, Panwar).
    Handles case variations (.txt/.TXT, folder names) and prints segment counts.
    """
    data = {}
    dataset_name = dataset_name.lower()

    # ------------------------------------
    # 1️⃣ BONN DATASET
    # ------------------------------------
    # --- BONN dataset section (fixed nested structure) ---

    if dataset_name == 'bonn':
        print("Loaded dataset: bonn")
        sets = {
            'A': ('z', 'Z'),
            'B': ('o', 'O'),
            'C': ('n', 'N'),
            'D': ('f', 'F'),
            'E': ('s', 'S')
        }
    
        for label, (lower_folder, upper_folder) in sets.items():
            # Try nested path first: base_path/lower/upper
            nested_path = os.path.join(base_path, lower_folder, upper_folder)
            if not os.path.isdir(nested_path):
                print(f"⚠️ Missing nested folder for {label}: {nested_path}")
                data[label] = []
                continue
    
            signals = []
            for i in range(1, 101):
                loaded = False
                for ext in ['.txt', '.TXT']:
                    filename = f"{upper_folder}{i:03d}{ext}"
                    filepath = os.path.join(nested_path, filename)
                    if os.path.exists(filepath):
                        try:
                            signal = np.loadtxt(filepath)
                            signals.append(signal)
                            loaded = True
                            break
                        except Exception as e:
                            print(f"⚠️ Error reading {filepath}: {e}")
                if not loaded:
                    pass  # Skip silently
    
            data[label] = signals
            print(f"  {label}: {len(signals)} segments")


    # ------------------------------------
    # 2️⃣ BERN-BARCELONA DATASET
    # ------------------------------------
    elif dataset_name == 'bern':
        print("Loaded dataset: bern")

        seizure_folders = [f"Data_F_Ind_{i}_{i+749}" for i in range(1, 3750, 750)]
        nonseizure_folders = [f"Data_N_Ind_{i}_{i+749}" for i in range(1, 3750, 750)]

        def load_bern_group(folders):
            signals = []
            for folder in folders:
                full_folder = os.path.join(base_path, folder)
                if not os.path.isdir(full_folder):
                    continue
                for filename in sorted(os.listdir(full_folder)):
                    if filename.lower().endswith('.txt'):
                        filepath = os.path.join(full_folder, filename)
                        try:
                            signal = np.loadtxt(filepath, delimiter=',')

                            # Ensure shape is (C, T) with T > C (time dimension last)
                            if signal.ndim == 2:
                                C, T = signal.shape
                                # If it looks like (time, channels), flip it
                                if C > T:
                                    signal = signal.T  # now shape (channels, time)
                            
                            elif signal.ndim == 1:
                                # Just in case, treat as single-channel
                                signal = signal[None, :]
                            
                            signals.append(signal)

                        except Exception as e:
                            print(f"⚠️ Error loading {filepath}: {e}")
            return signals

        data['F'] = load_bern_group(seizure_folders)
        data['N'] = load_bern_group(nonseizure_folders)

        print(f"  F: {len(data['F'])} segments")
        print(f"  N: {len(data['N'])} segments")

    # ------------------------------------
    # 3️⃣ HAUZ KHAS DATASET
    # ------------------------------------
    elif dataset_name == 'hauz_khas':
        print("Loaded dataset: hauz_khas")
        folders = ['ictal', 'interictal', 'preictal']

        for label in folders:
            full_folder = os.path.join(base_path, label)
            signals = []
            if not os.path.isdir(full_folder):
                print(f"⚠️ Missing folder: {full_folder}")
                continue

            for filename in sorted(os.listdir(full_folder)):
                if filename.lower().endswith('.mat'):
                    filepath = os.path.join(full_folder, filename)
                    try:
                        mat = scipy.io.loadmat(filepath)
                        signal_key = next((k for k in mat.keys() if not k.startswith('__')), None)
                        if signal_key:
                            signal = mat[signal_key].squeeze()
                            signals.append(signal)
                        else:
                            print(f"⚠️ No valid key found in {filename}")
                    except Exception as e:
                        print(f"⚠️ Error loading {filepath}: {e}")

            data[label] = signals
            print(f"  {label}: {len(signals)} segments")

    # ------------------------------------
    # 4️⃣ PANWAR DATASET
    # ------------------------------------
    elif dataset_name == 'panwar':
        print("Loaded dataset: panwar")
        folders = {'healthy': 'healthy', 'epileptic': 'epileptic'}

        for label, folder_name in folders.items():
            full_folder = os.path.join(base_path, folder_name)
            signals = []
            if not os.path.isdir(full_folder):
                print(f"⚠️ Missing folder: {full_folder}")
                continue

            for filename in sorted(os.listdir(full_folder)):
                if filename.lower().endswith('.txt'):
                    filepath = os.path.join(full_folder, filename)
                    try:
                        signal = np.loadtxt(filepath)
                        signals.append(signal)
                    except Exception as e:
                        print(f"⚠️ Error loading {filepath}: {e}")

            data[label] = signals
            print(f"  {label}: {len(signals)} segments")

    # ------------------------------------
    # UNKNOWN DATASET
    # ------------------------------------
    else:
        print(f"❌ Dataset '{dataset_name}' not yet supported.")
        return None

    # Summary
    print(f"\n✅ Loaded dataset summary: {dataset_name}")
    for label in data:
        print(f"  {label}: {len(data[label])} segments")
    print()

    return data
