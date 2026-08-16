#!/usr/bin/env python3
"""
AD_MTDL_predict.py — Multi-task D-MPNN classifier (AChE / BuChE / BACE1 / MAO-B)
                      for the AD-MTDL pipeline. Two modes:

Prediction (score new compounds):
    python AD_MTDL_predict.py --input compounds.csv --smiles_col smiles --output results.csv

External validation (pubchem CLEAN sets vs. existing, UNCHANGED thresholds):
    python AD_MTDL_predict.py --validate --output validation_results.csv

Validation mode excludes Moderate-label compounds from all metric calculations
(same protocol as Section 2.4: active >=7.0 / inactive <5.0, moderate zone excluded),
reports AUROC, AUPRC, MCC, sensitivity, specificity, accuracy, F1(macro), and the
full confusion matrix per target, and writes ROC curves, PR curves, and bar-plot
comparisons across all four targets to ./validation_plots/.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from chemprop import data, featurizers, models
from chemprop import nn as cpnn

import torch.nn.functional as F_torch
from chemprop.nn import BCELoss as ChempropBCELoss
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score,
)

# ═══════════════════════════════════════════════════════════
# CONFIG — VERIFY AGAINST YOUR TRAINING SCRIPT BEFORE RUNNING
# ═══════════════════════════════════════════════════════════
CKPT_PATH = "AD_MTDL_best.ckpt"

# Task order MUST exactly match the order the multi-task FFN head was trained
# with. A silent mismatch here scores the right probabilities against the
# wrong target with no error raised. Confirm against your training script,
# not against the manuscript text alone.
TARGET_COLS = ["AChE", "BuChE", "BACE1", "MAO_B"]

# MCC-optimized thresholds from internal validation (Section 2.3), derived
# against this exact, uncalibrated probability distribution. Left unchanged
# per the decision to keep the classifier as verified.
OPTIMAL_THRESHOLDS = {
    "AChE":  0.59,
    "BuChE": 0.68,
    "BACE1": 0.60,
    "MAO_B": 0.51,
}

# Curated, ChEMBL-deduplicated pubchem external validation sets
# (output of clean_pubchem.py). Must contain columns: canonical_smiles, label.
EXTERNAL_VALIDATION_DATA = {
    "AChE":  "AChE_pubchem_external_CLEAN.csv",
    "BuChE": "BuChE_pubchem_external_CLEAN.csv",
    "BACE1": "BACE1_pubchem_external_CLEAN.csv",
    "MAO_B": "MAO_B_pubchem_external_CLEAN.csv",
}

PLOT_DIR = "validation_plots"


class WeightedBCELoss(ChempropBCELoss):
    """Must be defined/importable before load_from_checkpoint — matches the
    custom loss class the checkpoint was trained and saved with."""
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.ones(len(TARGET_COLS))
        self.register_buffer("pos_weight", pos_weight)

    def _calc_unreduced_loss(self, preds, targets, *args, **kwargs):
        return F_torch.binary_cross_entropy_with_logits(
            preds, targets, reduction="none",
            pos_weight=self.pos_weight.to(preds.device),
        )


# ═══════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════

def canonicalize(smi):
    """Canonical SMILES string, or None on failure."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def load_model(ckpt_path, device):
    torch.serialization.add_safe_globals([
        WeightedBCELoss,
        cpnn.metrics.BinaryAUPRC,
        cpnn.metrics.BinaryAUROC,
    ])
    try:
        model = models.MPNN.load_from_checkpoint(ckpt_path, map_location=device)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint at '{ckpt_path}': {e}", file=sys.stderr)
        print("        If this checkpoint used a custom loss class (e.g. WeightedBCELoss), "
              "make sure that class is defined/imported before calling load_from_checkpoint.",
              file=sys.stderr)
        sys.exit(1)
    model.eval()
    model = model.to(device)
    return model


def get_probs(model, smiles_list, device, batch_size=256):
    """
    Runs the full forward pass (fingerprint -> predictor), returning
    sigmoid-activated probabilities per task. Deliberately NOT bypassing the
    sigmoid here (unlike a calibration script) since no calibration is applied.
    """
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dummy_targets = np.full((len(smiles_list), len(TARGET_COLS)), np.nan, dtype=np.float32)
    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(smiles_list, dummy_targets)
    ]
    dset = data.MoleculeDataset(datapoints, featurizer)
    loader = data.build_dataloader(dset, shuffle=False, num_workers=2, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            bmg, X_vd, features, *_ = batch
            bmg.V = bmg.V.to(device)
            bmg.E = bmg.E.to(device)
            bmg.edge_index = bmg.edge_index.to(device)
            bmg.batch = bmg.batch.to(device)
            X_vd = X_vd.to(device) if X_vd is not None else None
            features = [f.to(device) for f in features] if features else features

            fp = model.fingerprint(bmg, X_vd, features)
            preds = model.predictor(fp)  # BinaryClassificationFFN.forward applies sigmoid
            all_probs.append(preds.cpu().numpy())

    return np.vstack(all_probs)


def validate_and_dedup(smiles_pairs):
    """Returns (valid_ids, valid_smiles, invalid_records)."""
    valid_ids, valid_smiles, invalid_records = [], [], []
    seen = {}
    for sid, smi in smiles_pairs:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_records.append((sid, smi, "invalid SMILES"))
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon in seen:
            invalid_records.append((sid, smi, f"duplicate of {seen[canon]}"))
            continue
        seen[canon] = sid
        valid_ids.append(sid)
        valid_smiles.append(canon)
    return valid_ids, valid_smiles, invalid_records


# ═══════════════════════════════════════════════════════════
# VALIDATION MODE
# ═══════════════════════════════════════════════════════════

def compute_metrics(labels, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "N": len(labels),
        "N_active": int(labels.sum()),
        "N_inactive": int((1 - labels).sum()),
        "AUROC": roc_auc_score(labels, probs),
        "AUPRC": average_precision_score(labels, probs),
        "MCC": matthews_corrcoef(labels, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "F1_macro": f1_score(labels, y_pred, average="macro"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Threshold": threshold,
    }


def run_validation(device, output_path):
    script_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    print(f"Loading model on device: {device}")
    model = load_model(os.path.join(script_dir, CKPT_PATH), device)

    plot_dir = os.path.join(script_dir, PLOT_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    all_results = {}
    roc_data = {}
    prc_data = {}

    for target, csv_filename in EXTERNAL_VALIDATION_DATA.items():
        csv_path = os.path.join(script_dir, csv_filename)
        print(f"\n{'='*60}\n  Validating: {target}  ({csv_filename})\n{'='*60}")

        if not os.path.exists(csv_path):
            print(f"  [ERROR] {csv_filename} not found — skipping")
            continue

        df = pd.read_csv(csv_path)
        n_total = len(df)

        # Exclude Moderate-label compounds from metric calculation
        # (same active/inactive-only protocol as Section 2.4).
        n_moderate = (df["label"] == "Moderate").sum()
        df = df[df["label"].isin(["Potent", "Inactive"])].copy()
        print(f"  Total rows: {n_total}  |  Excluded (Moderate): {n_moderate}  |  Used: {len(df)}")

        df["canon_smiles"] = df["canonical_smiles"].apply(canonicalize)
        n_invalid = df["canon_smiles"].isna().sum()
        df = df.dropna(subset=["canon_smiles"])
        if n_invalid:
            print(f"  Dropped {n_invalid} rows with invalid SMILES")

        if df.empty:
            print(f"  [ERROR] No evaluable compounds remain for {target} — skipping")
            continue

        df["y"] = (df["label"] == "Potent").astype(int)
        labels = df["y"].values
        smiles_list = df["canon_smiles"].tolist()

        print(f"  Evaluable: {len(smiles_list)}  "
              f"(Potent: {labels.sum()}, Inactive: {(1 - labels).sum()})")

        task_idx = TARGET_COLS.index(target)
        all_probs = get_probs(model, smiles_list, device)
        task_probs = all_probs[:, task_idx]

        threshold = OPTIMAL_THRESHOLDS[target]
        metrics = compute_metrics(labels, task_probs, threshold)
        all_results[target] = metrics

        print(f"  AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  "
              f"MCC={metrics['MCC']:.4f}  (threshold={threshold})")
        print(f"  Sensitivity={metrics['Sensitivity']:.4f}  Specificity={metrics['Specificity']:.4f}  "
              f"F1(macro)={metrics['F1_macro']:.4f}  Accuracy={metrics['Accuracy']:.4f}")
        print(f"  Confusion: TN={metrics['TN']}  FP={metrics['FP']}  "
              f"FN={metrics['FN']}  TP={metrics['TP']}")

        fpr, tpr, _ = roc_curve(labels, task_probs)
        precision, recall, _ = precision_recall_curve(labels, task_probs)
        roc_data[target] = (fpr, tpr, metrics["AUROC"])
        prc_data[target] = (recall, precision, metrics["AUPRC"])

    if not all_results:
        print("\n[ERROR] No targets were validated.", file=sys.stderr)
        sys.exit(1)

    # ── Save results table ──
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Target"
    out_csv = os.path.join(script_dir, output_path)
    results_df.to_csv(out_csv)
    print(f"\nSaved validation metrics to {out_csv}")

    # ── Plots ──
    plot_roc_curves(roc_data, plot_dir)
    plot_prc_curves(prc_data, plot_dir)
    plot_metric_bars(all_results, plot_dir)

    print(f"\n{'='*60}\n  VALIDATION COMPLETE\n{'='*60}")
    print(results_df.round(4).to_string())
    print(f"\nPlots saved to {plot_dir}/")


def plot_roc_curves(roc_data, plot_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    for target, (fpr, tpr, auroc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{target} (AUROC={auroc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("External Validation — ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, "roc_curves.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_prc_curves(prc_data, plot_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    for target, (recall, precision, auprc) in prc_data.items():
        ax.plot(recall, precision, label=f"{target} (AUPRC={auprc:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("External Validation — Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, "prc_curves.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_metric_bars(all_results, plot_dir):
    targets = list(all_results.keys())
    metric_names = ["AUROC", "AUPRC", "MCC", "Sensitivity", "Specificity", "F1_macro", "Accuracy"]

    fig, ax = plt.subplots(figsize=(12, 6))
    n_metrics = len(metric_names)
    n_targets = len(targets)
    bar_width = 0.8 / n_targets
    x = np.arange(n_metrics)

    for i, target in enumerate(targets):
        values = [all_results[target][m] for m in metric_names]
        offset = (i - (n_targets - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=target)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("External Validation — Metric Comparison Across Targets")
    ax.legend(title="Target", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=n_targets)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, "metric_comparison_bars.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")

    # Confusion-matrix count bars, per target
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    conf_labels = ["TN", "FP", "FN", "TP"]
    x2 = np.arange(len(targets))
    bar_width2 = 0.2
    for j, cl in enumerate(conf_labels):
        values = [all_results[t][cl] for t in targets]
        offset = (j - (len(conf_labels) - 1) / 2) * bar_width2
        ax2.bar(x2 + offset, values, bar_width2, label=cl)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(targets)
    ax2.set_ylabel("Compound Count")
    ax2.set_title("External Validation — Confusion Matrix Counts")
    ax2.legend()
    fig2.tight_layout()
    out_path2 = os.path.join(plot_dir, "confusion_counts_bars.png")
    fig2.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"  -> {out_path2}")


# ═══════════════════════════════════════════════════════════
# PREDICTION MODE
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="AD-MTDL multi-task classifier — prediction and external validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Prediction:
    python AD_MTDL_predict.py --input compounds.csv --smiles_col smiles --output results.csv

Validation (pubchem CLEAN sets, Moderate-label excluded from metrics):
    python AD_MTDL_predict.py --validate --output validation_results.csv
""",
    )
    p.add_argument("--validate", action="store_true",
                    help="Run external validation against the four pubchem CLEAN sets "
                         "and produce metrics + plots.")
    p.add_argument("--input", type=str, help="CSV file of compounds to predict.")
    p.add_argument("--smiles_col", type=str, help="Column name containing SMILES in --input.")
    p.add_argument("--output", type=str, required=True,
                    help="Output CSV path (prediction) or metrics CSV path (validation).")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = p.parse_args()

    if not args.validate and not (args.input and args.smiles_col):
        p.error("--input and --smiles_col are required for prediction mode. "
                "Use --validate for external validation mode.")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested but unavailable. Falling back to CPU.", file=sys.stderr)
        device_str = "cpu"
    device = torch.device(device_str)

    if args.validate:
        run_validation(device, args.output)
        return

    # ── Prediction mode ──
    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        print(f"[ERROR] Column '{args.smiles_col}' not found. Available: {list(df.columns)}",
              file=sys.stderr)
        sys.exit(1)

    smiles_pairs = list(enumerate(df[args.smiles_col].astype(str).tolist()))
    valid_ids, valid_smiles, invalid_records = validate_and_dedup(smiles_pairs)
    print(f"Loaded {len(smiles_pairs)} input SMILES. "
          f"Valid & unique: {len(valid_smiles)}  Rejected: {len(invalid_records)}")

    if invalid_records:
        for sid, smi, reason in invalid_records[:10]:
            print(f"    [row {sid}] '{smi}' -> {reason}", file=sys.stderr)
        if len(invalid_records) > 10:
            print(f"    ... and {len(invalid_records) - 10} more", file=sys.stderr)

    if not valid_smiles:
        print("[ERROR] No valid SMILES remain after filtering.", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    print(f"Loading model on device: {device}")
    model = load_model(os.path.join(script_dir, CKPT_PATH), device)

    print("Running inference...")
    probs = get_probs(model, valid_smiles, device)

    out_df = pd.DataFrame({"row_id": valid_ids, "canonical_smiles": valid_smiles})
    for i, col in enumerate(TARGET_COLS):
        out_df[f"{col}_prob"] = probs[:, i]
        out_df[f"{col}_pred"] = np.where(
            probs[:, i] >= OPTIMAL_THRESHOLDS[col], "Active", "Inactive"
        )

    out_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(out_df)} predictions to {args.output}")

    if invalid_records:
        rejected_path = args.output.rsplit(".", 1)[0] + "_rejected.csv"
        pd.DataFrame(invalid_records, columns=["row_id", "smiles", "reason"]).to_csv(
            rejected_path, index=False
        )
        print(f"Wrote {len(invalid_records)} rejected entries to {rejected_path}")


if __name__ == "__main__":
    main()
