#run_chbmit_lopo
import argparse
import subprocess
import sys
import os
from datetime import datetime


# ---------------------------------------------------------
# Safe subprocess runner (Windows / Linux)
# ---------------------------------------------------------
def run(cmd):
    print("\n▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------
# Main LOPO routine
# ---------------------------------------------------------
def main(args):
    npz_dir = args.npz_dir
    patients = args.patients.split(",")
    epochs = str(args.epochs)
    regime = args.regime

    os.makedirs("analysis/chb_preds", exist_ok=True)
    os.makedirs("analysis/lopo", exist_ok=True)

    print("\n================ CHB-MIT LOPO START =================")
    print("Patients:", patients)
    print("Epochs:", epochs)
    print("Regime:", regime)

    for test_patient in patients:
        train_patients = [p for p in patients if p != test_patient]

        print("\n" + "=" * 80)
        print(f"🧪 LOPO TEST PATIENT: {test_patient}")
        print("=" * 80)

        # -------------------------------------------------
        # 1. Train CHAOS model
        # -------------------------------------------------
        print("\n▶ TRAIN CHAOS (train ≠ test)")
        run([
            sys.executable,
            "-m", "src.train_chbmit_npz",
            "--npz_dir", npz_dir,
            "--patients", ",".join(train_patients),
            "--epochs", epochs,
            "--regime", "edge"
        ])

        chaos_ckpt = "checkpoints/chaos_snn_chbmit_edge_best.pt"

        # -------------------------------------------------
        # 2. Train CHAOS-OFF model
        # -------------------------------------------------
        print("\n▶ TRAIN CHAOS-OFF (train ≠ test)")
        run([
            sys.executable,
            "-m", "src.train_chbmit_npz",
            "--npz_dir", npz_dir,
            "--patients", ",".join(train_patients),
            "--epochs", epochs,
            "--regime", "chaos_off"
        ])

        off_ckpt = "checkpoints/chbmit_chaos_off_npz_best.pt"

        # -------------------------------------------------
        # 3. Predict on held-out patient (CHAOS)
        # -------------------------------------------------
        chaos_csv = f"analysis/chb_preds/lopo_{test_patient}_chaos.csv"
        run([
            sys.executable,
            "-m", "src.utils.save_preds_from_ckpt_npz",
            "--npz_dir", npz_dir,
            "--patients", test_patient,
            "--ckpt", chaos_ckpt,
            "--out_dir", "analysis/chb_preds",
            "--out_csv", os.path.basename(chaos_csv),
            "--use", "U"
        ])

        # -------------------------------------------------
        # 4. Predict on held-out patient (CHAOS-OFF)
        # -------------------------------------------------
        off_csv = f"analysis/chb_preds/lopo_{test_patient}_chaos_off.csv"
        run([
            sys.executable,
            "-m", "src.utils.save_preds_from_ckpt_npz",
            "--npz_dir", npz_dir,
            "--patients", test_patient,
            "--ckpt", off_ckpt,
            "--out_dir", "analysis/chb_preds",
            "--out_csv", os.path.basename(off_csv),
            "--use", "U"
        ])

        # -------------------------------------------------
        # 5. Diagnose (CHAOS)
        # -------------------------------------------------
        chaos_json = f"analysis/lopo/{test_patient}_chaos_summary.json"
        run([
            sys.executable,
            "-m", "src.utils.diagnose_preds_npz",
            "--preds", chaos_csv,
            "--out", chaos_json
        ])

        # -------------------------------------------------
        # 6. Diagnose (CHAOS-OFF)
        # -------------------------------------------------
        off_json = f"analysis/lopo/{test_patient}_chaos_off_summary.json"
        run([
            sys.executable,
            "-m", "src.utils.diagnose_preds_npz",
            "--preds", off_csv,
            "--out", off_json
        ])

        print(f"\n✅ LOPO COMPLETED FOR {test_patient}")

    print("\n================ LOPO FINISHED =================")
    print("Results saved in: analysis/lopo")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CHB-MIT Leave-One-Patient-Out Pipeline")
    ap.add_argument("--npz_dir", required=True, help="Directory with CHB-MIT NPZ files")
    ap.add_argument("--patients", required=True, help="Comma-separated patient IDs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--regime", default="edge")

    args = ap.parse_args()
    main(args)
