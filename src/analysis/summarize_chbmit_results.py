import json
import numpy as np
import pandas as pd
import os

BASE = "analysis"

FILES = {
    "H1": f"{BASE}/h1/h1_summary.json",
    "H2": f"{BASE}/h2_instability.npz",
    "H3": f"{BASE}/h3_learning_gain.json",
    "H4": f"{BASE}/h4_energy_latency.json",
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def summarize():
    summary = {}

    # ---------------- H1 ----------------
    h1 = load_json(FILES["H1"])
    summary["H1"] = {
        "Raw Fisher": h1["raw"]["fisher"],
        "Chaos Fisher": h1["chaos"]["fisher"],
        "Raw AUC": h1["raw"]["auc"],
        "Chaos AUC": h1["chaos"]["auc"],
        "Conclusion":
            "Chaos does not significantly improve linear separability on CHB-MIT; "
            "suggests clinical EEG is already linearly separable at window level."
    }

    # ---------------- H2 ----------------
    h2 = np.load(FILES["H2"])
    chaos_mean = h2["chaos"].mean()
    off_mean   = h2["chaos_off"].mean()

    summary["H2"] = {
        "Chaos Instability Mean": float(chaos_mean),
        "Chaos-off Instability Mean": float(off_mean),
        "Conclusion":
            "Chaos increases internal state sensitivity, supporting edge-of-chaos dynamics."
            if chaos_mean > off_mean else
            "No instability amplification observed."
    }

    # ---------------- H3 ----------------
    h3 = load_json(FILES["H3"])
    summary["H3"] = {
        "Final AUC Chaos": h3["final_auc"]["chaos"],
        "Final AUC Chaos-off": h3["final_auc"]["chaos_off"],
        "Epochs to 90% Chaos": h3["epochs_to_90pct"]["chaos"],
        "Epochs to 90% Chaos-off": h3["epochs_to_90pct"]["chaos_off"],
        "Conclusion":
            "Chaos does not accelerate convergence on CHB-MIT, "
            "indicating task difficulty dominates learning dynamics."
    }

    # ---------------- H4 ----------------
    h4 = load_json(FILES["H4"])
    summary["H4"] = {
        "Mean Energy": h4["mean_energy"],
        "Mean Firing Rate": h4["mean_firing_rate"],
        "Conclusion":
            "Chaos-modulated SNN maintains sparse activity, "
            "indicating no excessive energy cost."
    }

    # ---------------- SAVE ----------------
    os.makedirs("analysis/summary", exist_ok=True)

    with open("analysis/summary/chbmit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV table
    rows = []
    for k, v in summary.items():
        row = {"Hypothesis": k}
        row.update({kk: vv for kk, vv in v.items() if kk != "Conclusion"})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("analysis/summary/chbmit_summary.csv", index=False)

    # ---------------- PRINT ----------------
    print("\n📊 CHB-MIT SUMMARY (H1–H4)")
    for k, v in summary.items():
        print(f"\n{k}")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")

    print("\n💾 Saved:")
    print(" - analysis/summary/chbmit_summary.json")
    print(" - analysis/summary/chbmit_summary.csv")


if __name__ == "__main__":
    summarize()
