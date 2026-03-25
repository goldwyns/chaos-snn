# src.plots.plot_channel_saliency
import numpy as np
import matplotlib.pyplot as plt

sal = np.load("analysis/bonn_channel_saliency.npy")

plt.figure(figsize=(6, 1.5))
plt.imshow(sal[None, :], aspect="auto", cmap="hot")
plt.colorbar(label="Normalized Saliency")
plt.yticks([])
plt.xlabel("EEG Channel Index")
plt.title("Channel Saliency (Bonn, Edge-of-Chaos)")
plt.tight_layout()
plt.savefig("fig_channel_saliency_strip.png", dpi=300)
plt.close()
