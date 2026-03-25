import argparse
import json
import numpy as np

def load_log(path):
    with open(path, "r") as f:
        return json.load(f)

def epoch_to_fraction(auc_curve, frac=0.9):
    max_auc = np.max(auc_curve)
    target = frac * max_auc
    for i, a in enumerate(auc_curve):
        if a >= target:
            return i + 1
    return len(auc_curve)

def main(args):
    chaos = load_log(args.log_chaos)
    off   = load_log(args.log_off)

    auc_c = np.array(chaos["val_auc"])
    auc_o = np.array(off["val_auc"])

    e90_c = epoch_to_fraction(auc_c, frac=0.9)
    e90_o = epoch_to_fraction(auc_o, frac=0.9)

    print("\n📈 H3 — LEARNING GAIN ANALYSIS")
    print(f"Final AUC (chaos)     : {auc_c.max():.4f}")
    print(f"Final AUC (chaos-off) : {auc_o.max():.4f}")
    print(f"Epochs to 90% AUC (chaos)     : {e90_c}")
    print(f"Epochs to 90% AUC (chaos-off) : {e90_o}")

    results = {
        "final_auc": {
            "chaos": float(auc_c.max()),
            "chaos_off": float(auc_o.max())
        },
        "epochs_to_90pct": {
            "chaos": int(e90_c),
            "chaos_off": int(e90_o)
        },
        "auc_curve": {
            "chaos": auc_c.tolist(),
            "chaos_off": auc_o.tolist()
        }
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("💾 Saved to", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_chaos", required=True)
    ap.add_argument("--log_off", required=True)
    ap.add_argument("--out", default="analysis/h3_learning_gain.json")
    args = ap.parse_args()
    main(args)
