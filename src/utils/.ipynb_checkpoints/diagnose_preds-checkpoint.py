# src/utils/diagnose_preds.py
"""
Robust diagnosis of predictions CSV (handles all-zero labels gracefully).
"""
import argparse, csv, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def load_preds(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_arrays(rows):
    N = len(rows)
    probs = np.zeros(N)
    labels = np.zeros(N, dtype=int)
    patients = np.array([""]*N)
    files = np.array([""]*N)
    for i,r in enumerate(rows):
        probs[i] = float(r.get("prob", r.get("p",0)))
        lbl = r.get("label","0")
        try:
            labels[i] = int(lbl)
        except:
            labels[i] = 1 if lbl.lower().startswith("seiz") else 0
        patients[i] = r.get("patient","")
        files[i] = r.get("file","")
    return probs, labels, patients, files

def classwise_stats(probs, labels):
    p_mean = float(np.nanmean(probs))
    p_pos = float(np.nanmean(probs[labels==1])) if labels.sum()>0 else float("nan")
    p_neg = float(np.nanmean(probs[labels==0])) if (len(labels)-labels.sum())>0 else float("nan")
    return p_mean, p_pos, p_neg

def per_patient_auc(probs, labels, patients):
    out = {}
    unique = np.unique(patients)
    for u in unique:
        idx = (patients==u)
        if idx.sum() == 0:
            continue
        try:
            auc = roc_auc_score(labels[idx], probs[idx]) if (labels[idx].sum()>0 and (labels[idx].sum()<idx.sum())) else None
        except:
            auc = None
        out[u] = {"n": int(idx.sum()), "auc": auc}
    return out

def best_threshold_confusion(probs, labels):
    # handle edge-cases
    if len(np.unique(labels)) < 2:
        return {"best_thr": None, "tp":0,"fp":0,"fn":0,"tn":int((labels==0).sum()), "gmean": None}
    fpr, tpr, thr = roc_curve(labels, probs)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = int(np.nanargmax(gmeans))
    best_thr = float(thr[ix])
    preds_bin = (probs >= best_thr).astype(int)
    tp = int(((preds_bin==1) & (labels==1)).sum())
    fp = int(((preds_bin==1) & (labels==0)).sum())
    fn = int(((preds_bin==0) & (labels==1)).sum())
    tn = int(((preds_bin==0) & (labels==0)).sum())
    return {"best_thr": best_thr, "tp":tp,"fp":fp,"fn":fn,"tn":tn,"gmean": float(gmeans[ix])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load_preds(args.preds)
    probs, labels, patients, files = to_arrays(rows)
    p_mean, p_pos, p_neg = classwise_stats(probs, labels)
    print(f"p_mean={p_mean:.4f}, p_pos={p_pos}, p_neg={p_neg}")
    try:
        overall_auc = roc_auc_score(labels, probs) if (len(np.unique(labels))>1) else None
    except:
        overall_auc = None
    print("Overall AUC:", overall_auc)
    per_patient = per_patient_auc(probs, labels, patients)
    print("\nPer-patient AUC:")
    for k,v in per_patient.items():
        print(f"  {k}: n={v['n']}, auc={v['auc']}")
    conf = best_threshold_confusion(probs, labels)
    print("\nBest-threshold confusion (by G-mean):")
    print(conf)
    outd = {"p_mean":p_mean,"p_pos":p_pos,"p_neg":p_neg,"overall_auc":overall_auc,"per_patient":per_patient,"conf":conf}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(outd, f, indent=2)
        print("Saved summary to", args.out)

if __name__ == "__main__":
    main()
