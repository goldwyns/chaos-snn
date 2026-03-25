# src/run_chbmit_full_pipeline.py
import os
import subprocess
import sys

PYTHON = sys.executable

# ---------------- CONFIG ----------------
CHB_NPZ_DIR = "precomputed/chbmit_npz"
PATIENTS = "chb24,chb01,chb03,chb05"

CKPT_EDGE = "checkpoints/chaos_snn_chbmit_edge_best.pt"
CKPT_OFF  = "checkpoints/chbmit_chaos_off_npz_best.pt"

PRED_DIR = "analysis/chb_preds"
LOG_DIR  = "analysis/logs"
H_DIR    = "analysis"

EPOCHS = 10

# ---------------------------------------

def run(cmd, name):
    print(f"\n{'='*80}")
    print(f"▶ RUNNING: {name}")
    print(f"{'='*80}")
    subprocess.run(cmd, check=True)


# ===================== 1️⃣ TRAIN EDGE =====================
run([
    PYTHON, "-m", "src.train_chbmit_npz",
    "--npz_dir", CHB_NPZ_DIR,
    "--patients", PATIENTS,
    "--epochs", str(EPOCHS),
    "--regime", "edge",
], "TRAIN EDGE-OF-CHAOS")


# ===================== 2️⃣ TRAIN CHAOS-OFF =====================
run([
    PYTHON, "-m", "src.train_chbmit_npz",
    "--npz_dir", CHB_NPZ_DIR,
    "--patients", PATIENTS,
    "--epochs", str(EPOCHS),
    "--regime", "chaos_off",
], "TRAIN CHAOS-OFF")


# ===================== 3️⃣ SAVE PREDICTIONS =====================
run([
    PYTHON, "-m", "src.utils.save_preds_from_ckpt_npz",
    "--npz_dir", CHB_NPZ_DIR,
    "--patients", PATIENTS,
    "--ckpt", CKPT_EDGE,
    "--out_dir", PRED_DIR,
    "--out_csv", "chb_edge_preds.csv",
    "--use", "U",
], "SAVE EDGE PREDICTIONS")

run([
    PYTHON, "-m", "src.utils.save_preds_from_ckpt_npz",
    "--npz_dir", CHB_NPZ_DIR,
    "--patients", PATIENTS,
    "--ckpt", CKPT_OFF,
    "--out_dir", PRED_DIR,
    "--out_csv", "chb_chaos_off_preds.csv",
    "--use", "U",
], "SAVE CHAOS-OFF PREDICTIONS")


# ===================== 4️⃣ DIAGNOSE =====================
run([
    PYTHON, "-m", "src.utils.diagnose_preds_npz",
    "--preds", os.path.join(PRED_DIR, "chb_edge_preds.csv"),
    "--out", os.path.join(PRED_DIR, "chb_edge_summary.json"),
], "DIAGNOSE EDGE")

run([
    PYTHON, "-m", "src.utils.diagnose_preds_npz",
    "--preds", os.path.join(PRED_DIR, "chb_chaos_off_preds.csv"),
    "--out", os.path.join(PRED_DIR, "chb_chaos_off_summary.json"),
], "DIAGNOSE CHAOS-OFF")


# ===================== 5️⃣ H1 SEPARABILITY =====================
run([
    PYTHON, "-m", "src.eval.h1_separability",
    "--data_dir", CHB_NPZ_DIR,
    "--patients", PATIENTS,
    "--out", os.path.join(H_DIR, "h1"),
], "H1 SEPARABILITY")


# ===================== 6️⃣ H2 INSTABILITY =====================
run([
    PYTHON, "-m", "src.eval.h2_instability",
    "--npz_dir", CHB_NPZ_DIR,
    "--patients", PATIENTS,
    "--ckpt_chaos", CKPT_EDGE,
    "--ckpt_off", CKPT_OFF,
    "--out", os.path.join(H_DIR, "h2_instability.npz"),
], "H2 INSTABILITY")


# ===================== 7️⃣ H3 LEARNING GAIN =====================
run([
    PYTHON, "-m", "src.eval.h3_learning_gain",
    "--log_chaos", os.path.join(LOG_DIR, "chbmit_edge_npz.json"),
    "--log_off", os.path.join(LOG_DIR, "chbmit_chaos_off_npz.json"),
    "--out", os.path.join(H_DIR, "h3_learning_gain.json"),
], "H3 LEARNING GAIN")

# ===================== 8️⃣ H4 ENERGY / SPIKES =====================
run([
    PYTHON, "-m", "src.eval.h4_energy_latency",
    "--npz_dir", CHB_NPZ_DIR,
    "--patients", "chb24,chb01,chb03,chb05",
    "--ckpt", CKPT_EDGE,
    "--out", os.path.join(H_DIR, "h4_energy_latency.json"),
], "H4 ENERGY / LATENCY")


print("\n✅ FULL CHB-MIT PIPELINE COMPLETED SUCCESSFULLY")
