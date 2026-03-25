# src.utils.consolidate_all_results.py

import os
import json
import numpy as np
import pandas as pd

ROOT = "analysis"
rows = []

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

def add(dataset, regime, experiment, metric, value):
    rows.append({
        "Dataset": dataset,
        "Regime": regime,
        "Experiment": experiment,
        "Metric": metric,
        "Value": value
    })

# --------------------------------------------------
# 1️⃣ Controlled datasets (Bonn / Panwar / Hauz-Khas)
# --------------------------------------------------
CONTROLLED = ["bonn", "panwar", "hauz_khas"]

for ds in CONTROLLED:
    for regime in ["sub", "edge", "chaos_off"]:
        path = os.path.join(ROOT, ds, f"{regime}_summary.json")
        data = load_json(path)
        if not data:
            continue

        add(ds.upper(), regime, "Classification", "AUC", data.get("overall_auc"))
        add(ds.upper(), regime, "Classification", "Accuracy", data.get("accuracy", None))

# --------------------------------------------------
# 2️⃣ CHB-MIT window-level
# --------------------------------------------------
for f in os.listdir(os.path.join(ROOT, "chb_preds")):
    if not f.endswith("_summary.json"):
        continue
    d = load_json(os.path.join(ROOT, "chb_preds", f))
    regime = "edge" if "edge" in f else "chaos_off"
    add("CHB-MIT", regime, "Window", "AUC", d["overall_auc"])

# --------------------------------------------------
# 3️⃣ H1–H4
# --------------------------------------------------
h1 = load_json(os.path.join(ROOT, "h1", "h1_summary.json"))
if h1:
    add("CHB-MIT", "raw", "H1", "Fisher", h1["raw"]["fisher"])
    add("CHB-MIT", "edge", "H1", "Fisher", h1["chaos"]["fisher"])

h3 = load_json(os.path.join(ROOT, "h3_learning_gain.json"))
if h3:
    add("CHB-MIT", "edge", "H3", "Final AUC", h3["final_auc"]["chaos"])
    add("CHB-MIT", "chaos_off", "H3", "Final AUC", h3["final_auc"]["chaos_off"])

h4 = load_json(os.path.join(ROOT, "h4_energy_latency.json"))
if h4:
    add("CHB-MIT", "edge", "H4", "Energy", h4["mean_energy"])
    add("CHB-MIT", "edge", "H4", "FiringRate", h4["mean_firing_rate"])

# --------------------------------------------------
# 4️⃣ LOPO
# --------------------------------------------------
LOPO_DIR = os.path.join(ROOT, "lopo")
if os.path.exists(LOPO_DIR):
    for f in os.listdir(LOPO_DIR):
        if f.endswith("_summary.json"):
            d = load_json(os.path.join(LOPO_DIR, f))
            patient = f.split("_")[1]
            regime = "edge" if "chaos_off" not in f else "chaos_off"
            add("CHB-MIT", regime, f"LOPO-{patient}", "AUC", d["overall_auc"])

# --------------------------------------------------
# Save
# --------------------------------------------------
df = pd.DataFrame(rows)
os.makedirs("analysis/summary", exist_ok=True)
df.to_csv("analysis/summary/all_results.csv", index=False)

print("\n📊 ALL RESULTS CONSOLIDATED\n")
print(df.head(20))
print("\n💾 Saved → analysis/summary/all_results.csv")
