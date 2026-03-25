# src.utils.plot.jbhi.results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("analysis/summary/all_results.csv")

sns.set(style="whitegrid", font_scale=1.2)

# --------------------------------------------------
# Figure 1: Chaos regime vs AUC (Controlled datasets)
# --------------------------------------------------
plt.figure(figsize=(8,5))
sns.barplot(
    data=df[df["Metric"]=="AUC"],
    x="Dataset", y="Value", hue="Regime"
)
plt.title("Effect of Chaos Regime on Classification AUC")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig("analysis/summary/fig_regime_vs_auc.png", dpi=300)

# --------------------------------------------------
# Figure 2: LOPO generalization
# --------------------------------------------------
lopo = df[df["Experiment"].str.contains("LOPO", na=False)]
plt.figure(figsize=(8,5))
sns.boxplot(data=lopo, x="Regime", y="Value")
plt.title("LOPO Generalization Performance (CHB-MIT)")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig("analysis/summary/fig_lopo_auc.png", dpi=300)

# --------------------------------------------------
# Figure 3: Learning gain (H3)
# --------------------------------------------------
h3 = df[df["Experiment"]=="H3"]
plt.figure(figsize=(6,5))
sns.barplot(data=h3, x="Regime", y="Value")
plt.title("Learning Gain Comparison (H3)")
plt.ylabel("Final AUC")
plt.tight_layout()
plt.savefig("analysis/summary/fig_h3_learning_gain.png", dpi=300)

# --------------------------------------------------
# Figure 4: Energy vs Performance
# --------------------------------------------------
energy = df[df["Metric"].isin(["Energy","AUC"])]
plt.figure(figsize=(6,5))
sns.scatterplot(
    data=energy,
    x="Metric", y="Value", hue="Regime", s=100
)
plt.title("Energy–Performance Tradeoff")
plt.tight_layout()
plt.savefig("analysis/summary/fig_energy_tradeoff.png", dpi=300)

print("📈 Figures saved in analysis/summary/")
