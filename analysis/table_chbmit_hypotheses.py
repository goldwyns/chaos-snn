# analysis/table_chbmit_hypotheses.py
import pandas as pd
import json
import numpy as np

rows = []

# H1
# H1
with open("analysis/h1/h1_summary.json") as f:
    d = json.load(f)
rows.append([
    "H1",
    "Fisher",
    d["raw"]["fisher"],
    d["chaos"]["fisher"],
    "No gain"
])

# H2
data = np.load("analysis/h2_instability.npz")
rows.append(["H2", "Instability",
             data["chaos_off"].mean(),
             data["chaos"].mean(),
             "Strong gain"])

# H3
with open("analysis/h3_learning_gain.json") as f:
    d = json.load(f)
rows.append(["H3", "Final AUC",
             d["final_auc"]["chaos_off"],
             d["final_auc"]["chaos"],
             "No gain"])

# H4
with open("analysis/h4_energy_latency.json") as f:
    d = json.load(f)
rows.append(["H4", "Energy",
             "–",
             d["mean_energy"],
             "Bounded"])

df = pd.DataFrame(
    rows,
    columns=["Hypothesis", "Metric", "Chaos-Off", "Chaos / Edge", "Outcome"]
)

df.to_csv("table_chbmit_hypotheses.csv", index=False)
print(df)
