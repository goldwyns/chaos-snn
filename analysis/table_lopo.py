# analysis/table_lopo.py
import glob
import json
import pandas as pd

rows = []

for f in glob.glob("analysis/lopo/*_summary.json"):
    patient = f.split("_")[1]
    regime = "Chaos-Off" if "off" in f else "Edge"

    with open(f) as j:
        d = json.load(j)

    rows.append({
        "Test Patient": patient,
        "Regime": regime,
        "AUC": d["overall_auc"]
    })

df = pd.DataFrame(rows)
df.to_csv("table_lopo.csv", index=False)
print(df)
