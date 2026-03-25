# src.utils.consolidate_all_results_v2.py

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "analysis"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

rows = []

# --------------------------------------------------
# Helper
# --------------------------------------------------

def extract_complexity_value(data):
    """
    Handles both dict-style and list-style complexity JSONs.
    Returns a scalar.
    """
    # Case 1: dict (old format)
    if isinstance(data, dict):
        return data.get("lzw", data.get("isi_entropy", np.nan))

    # Case 2: list of samples (new format)
    if isinstance(data, list):
        vals = [d.get("isi_entropy") for d in data if "isi_entropy" in d]
        return np.mean(vals) if len(vals) > 0 else np.nan

    return np.nan


def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

# --------------------------------------------------
# 1️⃣ Controlled datasets (Bonn / Panwar / Hauz-Khas)
# --------------------------------------------------
for ds in ["bonn", "panwar", "hauz_khas"]:
    for f in glob.glob(f"analysis/{ds}_*/**/*.json", recursive=True):
        data = safe_load_json(f)
        if not data:
            continue

        regime = "edge" if "edge" in f else "sub" if "sub" in f else "chaos_off"

        if "complexity" in f:
            rows.append({
                "Dataset": ds.upper(),
                "Regime": regime,
                "Experiment": "Complexity",
                "Metric": "ISI-Entropy",
                "Value": extract_complexity_value(data)
            })


        if "auc" in data or "AUC" in data:
            rows.append({
                "Dataset": ds.upper(),
                "Regime": regime,
                "Experiment": "Window",
                "Metric": "AUC",
                "Value": data.get("auc", data.get("AUC", np.nan))
            })

# --------------------------------------------------
# 2️⃣ CHB-MIT H1
# --------------------------------------------------
h1_path = "analysis/h1"
if os.path.exists(h1_path):
    for f in glob.glob(f"{h1_path}/*.json"):
        h1 = safe_load_json(f)
        if not h1:
            continue
        for k in ["raw", "chaos"]:
            rows.append({
                "Dataset": "CHB-MIT",
                "Regime": k,
                "Experiment": "H1",
                "Metric": "Fisher",
                "Value": h1[k]["fisher"]
            })

# --------------------------------------------------
# 3️⃣ H2 Instability
# --------------------------------------------------
if os.path.exists("analysis/h2_instability.npz"):
    h2 = np.load("analysis/h2_instability.npz")
    rows.append({
        "Dataset": "CHB-MIT",
        "Regime": "edge",
        "Experiment": "H2",
        "Metric": "Instability",
        "Value": h2["chaos"].mean()
    })
    rows.append({
        "Dataset": "CHB-MIT",
        "Regime": "chaos_off",
        "Experiment": "H2",
        "Metric": "Instability",
        "Value": h2["chaos_off"].mean()
    })

# --------------------------------------------------
# 4️⃣ H3 Learning gain
# --------------------------------------------------
h3 = safe_load_json("analysis/h3_learning_gain.json")
if h3:
    for k in ["chaos", "chaos_off"]:
        rows.append({
            "Dataset": "CHB-MIT",
            "Regime": k,
            "Experiment": "H3",
            "Metric": "Final AUC",
            "Value": h3["final_auc"][k]
        })

# --------------------------------------------------
# 5️⃣ H4 Energy
# --------------------------------------------------
h4 = safe_load_json("analysis/h4_energy_latency.json")
if h4:
    rows.append({
        "Dataset": "CHB-MIT",
        "Regime": "edge",
        "Experiment": "H4",
        "Metric": "Energy",
        "Value": h4["mean_energy"]
    })

# --------------------------------------------------
# 6️⃣ LOPO results
# --------------------------------------------------
for f in glob.glob("analysis/lopo/*_summary.json"):
    data = safe_load_json(f)
    if not data:
        continue
    regime = "chaos_off" if "off" in f else "edge"
    rows.append({
        "Dataset": "CHB-MIT",
        "Regime": regime,
        "Experiment": "LOPO",
        "Metric": "AUC",
        "Value": data["overall_auc"]
    })

# --------------------------------------------------
# SAVE TABLE
# --------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "consolidated_results.csv"), index=False)
df.to_excel(os.path.join(OUT_DIR, "consolidated_results.xlsx"), index=False)

print("\n📊 CONSOLIDATED RESULTS")
print(df)

# --------------------------------------------------
# FIGURE 1: AUC vs Regime
# --------------------------------------------------
plt.figure(figsize=(6,4))
df_auc = df[df["Metric"]=="AUC"]
df_auc.boxplot(column="Value", by="Regime")
plt.title("AUC vs Regime")
plt.suptitle("")
plt.savefig(os.path.join(FIG_DIR, "auc_vs_regime.png"))
plt.close()

# --------------------------------------------------
# FIGURE 2: Complexity vs AUC (controlled)
# --------------------------------------------------
ctrl = df[df["Dataset"].isin(["BONN","PANWAR","HAUZ_KHAS"])]
if not ctrl.empty:
    plt.figure(figsize=(6,4))
    plt.scatter(ctrl["Value"], ctrl["Value"])
    plt.xlabel("Complexity")
    plt.ylabel("AUC")
    plt.savefig(os.path.join(FIG_DIR, "complexity_vs_auc.png"))
    plt.close()

print("\n✅ Consolidation complete.")
print("📁 Outputs:")
print(" - analysis/consolidated_results.csv")
print(" - analysis/consolidated_results.xlsx")
print(" - analysis/figures/")
