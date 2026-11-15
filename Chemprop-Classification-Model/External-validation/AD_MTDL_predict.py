#!/usr/bin/env python3
"""
AD_MTDL_predict.py  v2.3.0
Inference script for the AD multi-task ChemProp classifier.

Changelog vs v2.2.3:
  - FIX: Replaced heuristic sigmoid auto-detection with a deterministic
    model-architecture probe. Reads model.predictor output_transform at
    load time to decide once whether sigmoid is already applied.
    This eliminates the edge-case where all logits happened to fall in
    [0,1] by coincidence, fooling the range check.
  - FIX: Added startup model probe (single dummy SMILES) that prints raw
    model output so you can verify sigmoid handling before running the
    full batch.
  - FIX: Exact 0.0 / 1.0 probability clipping added (1e-6 guard) to
    prevent log(0) in any downstream calibration work.
  - IMPROVEMENT: --debug flag enables per-batch raw output logging.
  - IMPROVEMENT: Version banner and sigmoid-mode clearly logged.

Usage examples:
  # Single SMILES
  python AD_MTDL_predict.py --smiles "CCOc1ccc(NC(=O)c2cccc(Cl)c2)cc1"

  # .txt file (one SMILES per line)
  python AD_MTDL_predict.py --input compounds.txt

  # .csv file — must specify column name
  python AD_MTDL_predict.py --input compounds.csv --smiles_col smiles

  # Optional: specify output CSV path (default: predictions.csv)
  python AD_MTDL_predict.py --input compounds.csv --smiles_col smiles --output results.csv

  # Debug mode: prints raw model output for first batch
  python AD_MTDL_predict.py --input compounds.csv --smiles_col smiles --debug
"""

__version__ = "2.3.0"

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.serialization
import lightning.pytorch as pl
from rdkit import Chem
from rdkit import RDLogger

from chemprop import data, featurizers, models
from chemprop import nn as cpnn
from chemprop.nn import BCELoss as ChempropBCELoss

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# WEIGHTED BCE LOSS — must be defined here for checkpoint loading
# ═══════════════════════════════════════════════════════════
class WeightedBCELoss(ChempropBCELoss):
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)

    def _calc_unreduced_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            preds,
            targets,
            reduction="none",
            pos_weight=self.pos_weight.to(preds.device),
        )


# ═══════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description=f"AD MTDL ChemProp inference  v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str,
                             help="Single SMILES string (quote it)")
    input_group.add_argument("--input",  type=str,
                             help="Path to .txt or .csv file containing SMILES")

    parser.add_argument("--smiles_col",  type=str, default=None,
                        help="Column name for SMILES in CSV (required for CSV input)")
    parser.add_argument("--ckpt",        type=str, default="AD_MTDL_best.ckpt",
                        help="Path to model checkpoint (default: AD_MTDL_best.ckpt)")
    parser.add_argument("--meta",        type=str, default="AD_MTDL_metadata.json",
                        help="Path to metadata JSON (default: AD_MTDL_metadata.json)")
    parser.add_argument("--output",      type=str, default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--no_csv",      action="store_true",
                        help="Skip saving output CSV, print only")
    parser.add_argument("--debug",       action="store_true",
                        help="Print raw model output for first batch")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# SMILES LOADING
# ═══════════════════════════════════════════════════════════
def load_smiles(args) -> list[str]:
    if args.smiles:
        return [args.smiles.strip()]

    path = args.input
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path) as f:
            smiles = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(smiles):,} SMILES from {path}")
        return smiles

    elif ext == ".csv":
        if not args.smiles_col:
            sys.exit("[ERROR] --smiles_col is required for CSV input.")
        df = pd.read_csv(path)
        if args.smiles_col not in df.columns:
            sys.exit(
                f"[ERROR] Column '{args.smiles_col}' not found in CSV.\n"
                f"        Available columns: {list(df.columns)}"
            )
        smiles = df[args.smiles_col].astype(str).tolist()
        print(f"[INFO] Loaded {len(smiles):,} SMILES from "
              f"column '{args.smiles_col}' in {path}")
        return smiles

    else:
        sys.exit(f"[ERROR] Unsupported file type: {ext}. Use .txt or .csv")


# ═══════════════════════════════════════════════════════════
# SMILES VALIDATION
# ═══════════════════════════════════════════════════════════
def validate_smiles(smiles_list: list[str]):
    valid, invalid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append((i, smi))
        else:
            invalid_idx.append((i, smi))

    if invalid_idx:
        print(f"[WARN] {len(invalid_idx)} invalid SMILES will be skipped:")
        for idx, smi in invalid_idx[:10]:
            print(f"       row {idx}: {smi}")
        if len(invalid_idx) > 10:
            print(f"       ... and {len(invalid_idx) - 10} more")

    print(f"[INFO] {len(valid):,} valid SMILES proceeding to inference")
    return valid, invalid_idx


# ═══════════════════════════════════════════════════════════
# MODEL + METADATA LOADING
# ═══════════════════════════════════════════════════════════
def detect_sigmoid_mode(model) -> bool:
    """
    Returns True if the model's predictor already applies sigmoid
    (i.e. model() returns probabilities, NOT logits).

    ChemProp stores the output transform as model.predictor.output_transform.
    A Sigmoid transform means probabilities are already computed.
    A Identity (or no transform) means raw logits are returned.

    Falls back to a name-based heuristic if the attribute is absent.
    """
    try:
        ot = model.predictor.output_transform
        ot_name = type(ot).__name__.lower()
        if "sigmoid" in ot_name:
            return True
        if "identity" in ot_name or "linear" in ot_name:
            return False
        # Some versions wrap it in a Sequential or ModuleList
        for child in ot.children():
            if "sigmoid" in type(child).__name__.lower():
                return True
    except AttributeError:
        pass

    # Hard fallback: inspect the full predictor string representation
    predictor_repr = str(model.predictor).lower()
    if "sigmoid" in predictor_repr:
        return True

    return False  # assume logits if uncertain — safer to apply sigmoid


def load_model_and_meta(ckpt_path: str, meta_path: str, debug: bool = False):
    for path, label in [(ckpt_path, "checkpoint"), (meta_path, "metadata JSON")]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    with open(meta_path) as f:
        meta = json.load(f)

    target_cols        = meta["targets"]
    optimal_thresholds = meta["optimal_thresholds"]

    print(f"[INFO] Targets    : {target_cols}")
    print(f"[INFO] Thresholds : { {k: v for k, v in optimal_thresholds.items()} }")

    torch.serialization.add_safe_globals([
        WeightedBCELoss,
        cpnn.metrics.BinaryAUPRC,
        cpnn.metrics.BinaryAUROC,
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device     : {device}")

    model = models.MPNN.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False
    )
    model.eval()
    model = model.to(device)
    print(f"[INFO] Model loaded from: {ckpt_path}")

    # ── Determine sigmoid handling once at load time ──────────────────────
    already_sigmoid = detect_sigmoid_mode(model)
    print(f"[INFO] Output transform contains sigmoid : {already_sigmoid}")
    print(f"[INFO] Sigmoid will be applied in predict: {not already_sigmoid}")

    # ── Startup probe: run one dummy SMILES to show raw output ────────────
    if debug:
        _run_probe(model, target_cols, device, already_sigmoid)

    return model, target_cols, optimal_thresholds, device, already_sigmoid


def _run_probe(model, target_cols, device, already_sigmoid):
    """Run a single caffeine SMILES through the model and print raw output."""
    probe_smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine
    dummy_y   = np.zeros((1, len(target_cols)), dtype=np.float32)
    dp        = [data.MoleculeDatapoint.from_smi(probe_smi, dummy_y[0])]
    feat      = featurizers.SimpleMoleculeMolGraphFeaturizer()
    ds        = data.MoleculeDataset(dp, feat)
    loader    = data.build_dataloader(ds, shuffle=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            bmg, X_vd, features, *_ = batch
            bmg.V = bmg.V.to(device); bmg.E = bmg.E.to(device)
            bmg.edge_index = bmg.edge_index.to(device)
            bmg.batch = bmg.batch.to(device)
            raw = model(bmg, X_vd, features).cpu().numpy()[0]

    print(f"\n[DEBUG] Probe SMILES : {probe_smi}")
    print(f"[DEBUG] Raw output   : {[round(float(v),6) for v in raw]}")
    print(f"[DEBUG] Interpretation: "
          f"{'already probabilities' if already_sigmoid else 'logits → sigmoid will be applied'}")

    if already_sigmoid:
        if any(v < 0 or v > 1 for v in raw):
            print(f"[WARN] Output detected as probabilities but values outside [0,1] found! "
                  f"Check model architecture.")
    print()


# ═══════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════
def run_inference(
    model,
    valid_smiles:    list[tuple[int, str]],
    target_cols:     list[str],
    thresholds:      dict,
    device:          torch.device,
    batch_size:      int,
    already_sigmoid: bool,
    debug:           bool = False,
) -> pd.DataFrame:

    smi_list = [smi for _, smi in valid_smiles]
    dummy_y  = np.zeros((len(smi_list), len(target_cols)), dtype=np.float32)

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(smi_list, dummy_y)
    ]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset    = data.MoleculeDataset(datapoints, featurizer)
    loader     = data.build_dataloader(
        dataset,
        shuffle=False,
        batch_size=min(batch_size, len(dataset)),
        num_workers=0,
        drop_last=False,
    )

    all_probs = []
    first_batch = True

    with torch.no_grad():
        for batch in loader:
            bmg, X_vd, features, targets, weights, lt_mask, gt_mask = batch
            bmg.V          = bmg.V.to(device)
            bmg.E          = bmg.E.to(device)
            bmg.edge_index = bmg.edge_index.to(device)
            bmg.batch      = bmg.batch.to(device)
            X_vd     = X_vd.to(device) if X_vd is not None else None
            features = [f.to(device) for f in features] if features else features

            raw = model(bmg, X_vd, features).cpu()

            if debug and first_batch:
                print(f"[DEBUG] First batch raw output (first 3 rows):\n"
                      f"{raw[:3].numpy()}")
                first_batch = False

            # Apply sigmoid only if the model returns logits
            if already_sigmoid:
                probs = raw.numpy()
            else:
                probs = torch.sigmoid(raw).numpy()

            # Clip to strict (0,1) — avoids log(0) in any downstream work
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)  # (n_valid, n_targets)

    prob_min = float(all_probs.min())
    prob_max = float(all_probs.max())
    print(f"[INFO] Probability range after processing: [{prob_min:.4f}, {prob_max:.4f}]")
    if prob_max - prob_min < 0.05:
        print(f"[WARN] Very narrow probability range ({prob_min:.4f}–{prob_max:.4f}). "
              f"Consider running with --debug to inspect raw model output.")

    # Build output rows
    rows = []
    for i, (orig_idx, smi) in enumerate(valid_smiles):
        row = {"original_index": orig_idx, "smiles": smi}
        for t, col in enumerate(target_cols):
            prob   = float(all_probs[i, t])
            thresh = thresholds[col]
            label  = "Active" if prob >= thresh else "Inactive"
            short  = col.replace("_class", "")
            row[f"{short}_prob"] = round(prob, 4)
            row[f"{short}_pred"] = label
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# TERMINAL PRINTING
# ═══════════════════════════════════════════════════════════
def print_results(df: pd.DataFrame, target_cols: list[str], thresholds: dict):
    targets_short = [c.replace("_class", "") for c in target_cols]

    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  AD MTDL PREDICTIONS — {len(df):,} compounds")
    print(f"  Thresholds: " +
          "  ".join(f"{c.replace('_class','')}≥{v}" for c, v in thresholds.items()))
    print(sep)

    header = (f"  {'#':>5}  {'SMILES':<45}  " +
              "  ".join(f"{t:<14}" for t in targets_short))
    print(header)
    print("  " + "-" * 88)

    for _, row in df.iterrows():
        smi_display = row["smiles"]
        if len(str(smi_display)) > 43:
            smi_display = str(smi_display)[:40] + "..."
        pred_str = "  ".join(
            f"{row[f'{t}_pred']:<6}({row[f'{t}_prob']:.3f})"
            for t in targets_short
        )
        print(f"  {int(row['original_index']):>5}  {str(smi_display):<45}  {pred_str}")

    print(f"\n{sep}")
    print("  SUMMARY")
    print(f"  {'Target':<15} {'Active':>8} {'Inactive':>10} {'Active%':>9}")
    print("  " + "-" * 45)
    for t in targets_short:
        n_active   = (df[f"{t}_pred"] == "Active").sum()
        n_inactive = (df[f"{t}_pred"] == "Inactive").sum()
        pct        = 100 * n_active / max(len(df), 1)
        print(f"  {t:<15} {n_active:>8,} {n_inactive:>10,} {pct:>8.1f}%")

    print(f"\n  MULTI-TARGET ACTIVE COUNTS")
    print(f"  {'Targets hit':<15} {'Compounds':>10}")
    print("  " + "-" * 27)
    df = df.copy()
    df["n_active"] = sum(
        (df[f"{t}_pred"] == "Active").astype(int) for t in targets_short
    )
    for n in sorted(df["n_active"].unique(), reverse=True):
        count = (df["n_active"] == n).sum()
        label = f"{n} target{'s' if n != 1 else ''}"
        print(f"  {label:<15} {count:>10,}")
    print(sep)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print(f"[INFO] AD_MTDL_predict  v{__version__}")

    args = parse_args()

    raw_smiles            = load_smiles(args)
    valid_smiles, invalid = validate_smiles(raw_smiles)

    if not valid_smiles:
        sys.exit("[ERROR] No valid SMILES to process.")

    model, target_cols, thresholds, device, already_sigmoid = load_model_and_meta(
        args.ckpt, args.meta, debug=args.debug
    )

    print(f"\n[INFO] Running inference on {len(valid_smiles):,} compounds...")
    results_df = run_inference(
        model, valid_smiles, target_cols, thresholds,
        device, args.batch_size, already_sigmoid, debug=args.debug
    )

    if invalid:
        invalid_rows = []
        for orig_idx, smi in invalid:
            row = {"original_index": orig_idx, "smiles": smi}
            for col in results_df.columns:
                if col not in ("original_index", "smiles"):
                    row[col] = None
            invalid_rows.append(row)
        invalid_df  = pd.DataFrame(invalid_rows)
        results_df  = (pd.concat([results_df, invalid_df])
                         .sort_values("original_index")
                         .reset_index(drop=True))

    if len(valid_smiles) > 100:
        print(f"\n[INFO] Large input ({len(valid_smiles):,} compounds) — "
              f"printing first 100 rows. Full results saved to CSV.")
        print_results(results_df.head(100), target_cols, thresholds)
    else:
        print_results(results_df, target_cols, thresholds)

    if not args.no_csv:
        results_df.to_csv(args.output, index=False)
        print(f"\n[INFO] Full results saved to: {args.output}")
        print(f"[INFO] Columns: {list(results_df.columns)}")


if __name__ == "__main__":
    main()
