# plot_lopo_auc.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load LOPO CSV
df = pd.read_csv("table_lopo.csv")

# Ensure consistent labels
df["Regime"] = df["Regime"].replace({
    "Chaos-Off": "Chaos-Off",
    "Edge": "Edge-of-Chaos"
})

# Plot
plt.figure(figsize=(4.5, 4))

sns.violinplot(
    data=df,
    x="Regime",
    y="AUC",
    inner="quartile",
    cut=0
)

sns.stripplot(
    data=df,
    x="Regime",
    y="AUC",
    color="black",
    alpha=0.6,
    jitter=True
)

plt.ylabel("AUC")
plt.xlabel("")
plt.title("LOPO AUC Distribution Across Regimes")

plt.tight_layout()
plt.savefig("fig_lopo_auc.png", dpi=300)
plt.close()
