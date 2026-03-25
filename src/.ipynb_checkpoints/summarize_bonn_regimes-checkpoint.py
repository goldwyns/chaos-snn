import pandas as pd
from glob import glob
import numpy as np
from scipy.stats import mannwhitneyu

paths = glob("analysis/bonn_*/bonn_complexity_*.csv")

print("\n===== FINAL BONN REGIME SUMMARY =====\n")

for p in sorted(paths):
    df = pd.read_csv(p)
    name = p.split("\\")[-1].replace("bonn_complexity_", "").replace(".csv", "")

    seiz = df[df["y"] == 1]["isi_entropy"].values
    non  = df[df["y"] == 0]["isi_entropy"].values

    mean_seiz = seiz.mean()
    mean_non  = non.mean()

    try:
        U, pval = mannwhitneyu(seiz, non)
    except:
        pval = float("nan")

    print(f"Regime: {name}")
    print(f"  mean ISI (seiz ) = {mean_seiz:.6f}")
    print(f"  mean ISI (non  ) = {mean_non:.6f}")
    print(f"  Mann–Whitney p  = {pval:.3e}")
    print("-" * 50)
