# src/utils/diagnose_preds_npz.py

import argparse
import csv
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import warnings


# -----------------------------
# Load predictions
# -----------------------------
def load_preds(path):
    preds = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            preds.append(r)
    return preds


def to_arrays(preds):
    N = len(preds)
    probs = np.zeros(N, dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)
    patients = np.array([""] * N)

    for i, r in enumerate(preds):
        probs[i] = float(r["prob"])
        labels[i] = int(r["label"])
        patients[i] = r.get("patient", "")

    return probs, labels, patients


# -----------------------------
# Metrics
# -----------------------------
def classwise_stats(probs, labels):
    out = {}
    out["p_mean"] = float(np.mean(probs))

    if np.any(labels == 1):
        out["p_pos"] = float(np.mean(probs[labels == 1]))
    else:
        out["p_pos"] = None

    if np.any(labels == 0):
        out["p_neg"] = float(np.mean(probs[labels == 0]))
    else:
        out["p_neg"] = None

    return out


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def per_patient_auc(probs, labels, patients):
    out = {}
    for pid in np.unique(patients):
        idx = patients == pid
        if idx.sum() < 20:
            out[pid] = {"n": int(idx.sum()), "auc": None}
        else:
            out[pid] = {
                "n": int(idx.sum()),
                "auc": safe_auc(labels[idx], probs[idx]),
            }
    return out


def best_threshold_confusion(probs, labels):
    if len(np.unique(labels)) < 2:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr, tpr, thr = roc_curve(labels, probs)

    gmeans = np.sqrt(tpr * (1 - fpr))
    if np.all(np.isnan(gmeans)):
        return None

    ix = int(np.nanargmax(gmeans))
    th = float(thr[ix])

    preds_bin = (probs >= th).astype(int)

    return {
        "best_thr": th,
        "tp": int(((preds_bin == 1) & (labels == 1)).sum()),
        "fp": int(((preds_bin == 1) & (labels == 0)).sum()),
        "fn": int(((preds_bin == 0) & (labels == 1)).sum()),
        "tn": int(((preds_bin == 0) & (labels == 0)).sum()),
        "gmean": float(gmeans[ix]),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV predictions file")
    ap.add_argument("--out", required=True, help="Output JSON")
    args = ap.parse_args()

    preds = load_preds(args.preds)
    probs, labels, patients = to_arrays(preds)

    results = {}

    # Classwise
    results["classwise"] = classwise_stats(probs, labels)

    # Overall AUC
    results["overall_auc"] = safe_auc(labels, probs)

    # Per-patient
    results["per_patient"] = per_patient_auc(probs, labels, patients)

    # Threshold confusion
    results["confusion"] = best_threshold_confusion(probs, labels)

    # Print summary
    print("\n📊 DIAGNOSIS SUMMARY")
    print("p_mean:", results["classwise"]["p_mean"])
    print("p_pos :", results["classwise"]["p_pos"])
    print("p_neg :", results["classwise"]["p_neg"])
    print("AUC   :", results["overall_auc"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("💾 Saved to:", args.out)


if __name__ == "__main__":
    main()
