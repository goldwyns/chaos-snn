# src/plots/plot_h2_instability.py

import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("analysis/h2_instability.npz")
chaos = data["chaos"]
chaos_off = data["chaos_off"]

# Compute statistics
means = [chaos_off.mean(), chaos.mean()]
stds  = [chaos_off.std(), chaos.std()]

labels = ["Chaos-Off", "Edge-of-Chaos"]
x = np.arange(len(labels))

# Plot
plt.figure(figsize=(4.5, 4))

# Mean bar with std
plt.bar(
    x,
    means,
    yerr=stds,
    capsize=5,
    alpha=0.7
)

# Jittered individual points
jitter = 0.08
plt.scatter(
    np.random.normal(x[0], jitter, size=len(chaos_off)),
    chaos_off,
    color="black",
    s=20,
    alpha=0.6,
    zorder=3
)

plt.scatter(
    np.random.normal(x[1], jitter, size=len(chaos)),
    chaos,
    color="black",
    s=20,
    alpha=0.6,
    zorder=3
)

# Labels and formatting
plt.xticks(x, labels)
plt.ylabel("State Instability (Variance)")
plt.title("H2: Dynamical Instability Across Regimes")

plt.tight_layout()
plt.savefig("fig_h2_instability.png", dpi=300)
plt.close()
