#!/usr/bin/env python3
"""
AD_tanimoto_check.py  v1.2.0
Scaffold overlap analysis between the eval set and the training set.

Changelog vs v1.0.0:
  - FIX: Wide-form melt now uses pd.to_numeric(..., errors='coerce') before
    filtering _label == 1. Previously mixed-type columns (str '1', float 1.0)
    caused the filter to return 0 rows, crashing on empty train FPs.
  - FIX: IndexError on empty train_fps now raises a clear error message
    instead of an unhelpful traceback.
  - FIX: Melt now correctly handles all four class columns in one pass
    regardless of which target is checked first.
  - IMPROVEMENT: Post-melt row count logged per target for quick sanity check.

Usage:
  python AD_tanimoto_check.py \
      --eval   AD_eval_for_prediction.csv \
      --train  AD_MTDL_combined_chembl36.csv \
      --eval_smiles_col  canonical_smiles \
      --eval_target_col  target \
      --train_smiles_col smiles

  # With custom output directory:
  python AD_tanimoto_check.py --eval ... --train ... --output_dir results/
"""

__version__ = "1.2.0"

import argparse
import os
import sys
import warnings
from io import StringIO

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
TARGETS = ["AChE", "BuChE", "BACE1", "MAO_B"]

ZONE_BINS   = [0.0,  0.4,   0.6,    0.85,  1.01]
ZONE_LABELS = [
    "Novel      (< 0.40)",
    "Analogous  (0.40–0.60)",
    "Similar    (0.60–0.85)",
    "Memorised  (≥ 0.85)",
]


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description=f"AD Tanimoto overlap check  v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--eval",   required=True, help="Eval CSV path")
    p.add_argument("--train",  required=True, help="Training CSV path")
    p.add_argument("--eval_smiles_col",  default="canonical_smiles",
                   help="SMILES column in eval CSV  (default: canonical_smiles)")
    p.add_argument("--eval_target_col",  default="target",
                   help="Target column in eval CSV  (default: target)")
    p.add_argument("--train_smiles_col", default="smiles",
                   help="SMILES column in train CSV (default: smiles)")
    p.add_argument("--train_target_col", default="auto",
                   help="Target column in train CSV, or 'auto' to detect "
                        "long-form / wide-form layout (default: auto)")
    p.add_argument("--radius",     type=int, default=2,
                   help="Morgan fingerprint radius (default: 2)")
    p.add_argument("--nbits",      type=int, default=2048,
                   help="Fingerprint bit length    (default: 2048)")
    p.add_argument("--top_k",      type=int, default=3,
                   help="Nearest train neighbours to report (default: 3)")
    p.add_argument("--output_dir", default=".",
                   help="Output directory (default: current dir)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# FINGERPRINT HELPERS
# ═══════════════════════════════════════════════════════════
def make_fp_generator(radius: int, nbits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def smiles_to_fps(smiles_list, gen):
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
            valid_idx.append(i)
    return fps, valid_idx


# ═══════════════════════════════════════════════════════════
# LOAD + NORMALISE TRAINING SET
# ═══════════════════════════════════════════════════════════
def load_train(path: str, smiles_col: str, target_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Training file not found: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Train file loaded: {df.shape[0]:,} rows, "
          f"columns: {list(df.columns)}")

    if smiles_col not in df.columns:
        sys.exit(
            f"[ERROR] SMILES column '{smiles_col}' not found in train file.\n"
            f"        Available: {list(df.columns)}"
        )

    # ── Detect layout ────────────────────────────────────────────────────
    # Priority 1: explicit --train_target_col that exists as a column
    if target_col != "auto" and target_col in df.columns:
        print(f"[INFO] Using explicit target column: '{target_col}'")
        out = df[[smiles_col, target_col]].copy()
        out.columns = ["smiles", "target"]
        out["target"] = out["target"].str.replace("_class", "", regex=False).str.strip()
        out = out[out["target"].isin(TARGETS)]
        print(f"[INFO] Long-form train: {len(out):,} rows after target filter")
        return out

    # Priority 2: long-form column search
    for candidate in ["target", "Target", "task", "Task"]:
        if candidate in df.columns:
            print(f"[INFO] Auto-detected long-form target column: '{candidate}'")
            out = df[[smiles_col, candidate]].copy()
            out.columns = ["smiles", "target"]
            out["target"] = out["target"].str.replace("_class", "", regex=False).str.strip()
            out = out[out["target"].isin(TARGETS)]
            print(f"[INFO] Long-form train: {len(out):,} rows after target filter")
            return out

    # Priority 3: wide-form  (AChE_class / AChE columns)
    for suffix in ["_class", ""]:
        value_vars = [f"{t}{suffix}" for t in TARGETS if f"{t}{suffix}" in df.columns]
        if len(value_vars) >= 1:
            print(f"[INFO] Wide-form train detected (columns: {value_vars})")

            melted = df.melt(
                id_vars=[smiles_col],
                value_vars=value_vars,
                var_name="_raw_target",
                value_name="_label",
            )

            # ── v1.2.0 FIX: handle both numeric (0/1) and string label formats ──
            # Training CSVs may use:
            #   • integers / floats  : 1 / 0 / 1.0 / 0.0
            #   • strings            : 'Active' / 'Inactive' / 'Moderate'
            #   • NaN                : missing (treated as inactive / excluded)
            #
            # 'Moderate' was set to NaN during training (not used as a positive),
            # so only 'Active' (or numeric 1) counts as a positive here.
            label_str = melted["_label"].astype(str).str.strip().str.lower()
            label_num = pd.to_numeric(melted["_label"], errors="coerce")

            is_active = (label_str == "active") | (label_num == 1)
            melted = melted[is_active].copy()

            melted["target"] = (
                melted["_raw_target"]
                .str.replace(suffix, "", regex=False)
                .str.strip()
            )
            out = melted[[smiles_col, "target"]].rename(
                columns={smiles_col: "smiles"}
            )

            print(f"[INFO] Wide-form melt → {len(out):,} active rows")
            for t in TARGETS:
                n = (out["target"] == t).sum()
                print(f"         {t}: {n:,}")

            if len(out) == 0:
                sys.exit(
                    "[ERROR] Wide-form melt produced 0 active rows.\n"
                    "        Expected class columns to contain 'Active' (string) or 1 (numeric).\n"
                    f"        Sample values from your file: {df[value_vars].head(3).to_dict()}"
                )
            return out

    sys.exit(
        "[ERROR] Cannot find a target column in the training file.\n"
        "        Pass --train_target_col explicitly, or ensure your file has\n"
        "        either a 'target' column or columns named AChE_class, etc."
    )


# ═══════════════════════════════════════════════════════════
# CORE SIMILARITY SEARCH
# ═══════════════════════════════════════════════════════════
def compute_similarity(
    eval_df:          pd.DataFrame,
    train_df:         pd.DataFrame,
    eval_smiles_col:  str,
    eval_target_col:  str,
    gen,
    top_k:            int,
) -> pd.DataFrame:

    print("[INFO] Computing fingerprints for eval set ...")
    eval_fps, eval_valid = smiles_to_fps(eval_df[eval_smiles_col].tolist(), gen)
    print(f"[INFO] Eval  FPs: {len(eval_fps):,} valid / {len(eval_df):,} total")

    print("[INFO] Computing fingerprints for train set ...")
    train_fps, train_valid = smiles_to_fps(train_df["smiles"].tolist(), gen)
    train_df_valid = train_df.iloc[train_valid].reset_index(drop=True)
    print(f"[INFO] Train FPs: {len(train_fps):,} valid / {len(train_df):,} total")

    if len(train_fps) == 0:
        sys.exit(
            "[ERROR] Zero valid fingerprints from training set.\n"
            "        Check that --train_smiles_col points to a valid SMILES column."
        )

    # Per-target FP index
    target_fp_map: dict[str, tuple[list, list]] = {}
    for t in TARGETS:
        mask = train_df_valid["target"] == t
        idxs = mask[mask].index.tolist()
        target_fp_map[t] = ([train_fps[i] for i in idxs], idxs)
        print(f"[INFO]   {t} train FPs: {len(idxs):,}")

    rows = []
    n_eval = len(eval_fps)
    print(f"[INFO] Running similarity search for {n_eval:,} eval compounds ...")

    for rank, (ei, efp) in enumerate(zip(eval_valid, eval_fps)):
        if rank % 50 == 0:
            print(f"        {rank}/{n_eval} ...", flush=True)

        erow = eval_df.iloc[ei]
        esmi = erow[eval_smiles_col]
        etgt = erow[eval_target_col]

        # ── Overall nearest neighbours (all train) ──
        all_sims = np.array(BulkTanimotoSimilarity(efp, train_fps))
        top_idx  = np.argsort(all_sims)[::-1][:top_k]
        max_sim_overall = float(all_sims[top_idx[0]])
        nn_smiles  = train_df_valid.iloc[top_idx[0]]["smiles"]
        nn_target  = train_df_valid.iloc[top_idx[0]]["target"]

        # ── Same-target nearest neighbour ──
        same_fps, same_idxs = target_fp_map.get(etgt, ([], []))
        if same_fps:
            same_sims = np.array(BulkTanimotoSimilarity(efp, same_fps))
            best_i    = int(np.argmax(same_sims))
            max_sim_same = float(same_sims[best_i])
            nn_same_smi  = train_df_valid.iloc[same_idxs[best_i]]["smiles"]
        else:
            max_sim_same = np.nan
            nn_same_smi  = ""

        # ── Zone (based on same-target similarity, fallback to overall) ──
        sim_for_zone = max_sim_same if not np.isnan(max_sim_same) else max_sim_overall
        zone = ZONE_LABELS[0]
        for z, (lo, hi) in enumerate(zip(ZONE_BINS[:-1], ZONE_BINS[1:])):
            if lo <= sim_for_zone < hi:
                zone = ZONE_LABELS[z]
                break

        topk_str = "; ".join(
            f"{train_df_valid.iloc[i]['target']}:{all_sims[i]:.3f}"
            for i in top_idx
        )

        rows.append({
            "eval_index":              ei,
            "smiles":                  esmi,
            "eval_target":             etgt,
            "max_sim_overall":         round(max_sim_overall, 4),
            "max_sim_same_target":     round(max_sim_same, 4)
                                           if not np.isnan(max_sim_same) else np.nan,
            "nn_overall_smiles":       nn_smiles,
            "nn_overall_target":       nn_target,
            "nn_same_target_smiles":   nn_same_smi,
            f"top{top_k}_neighbours":  topk_str,
            "zone":                    zone,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════
def write_summary(results: pd.DataFrame, out_path: str):
    buf = StringIO()
    def p(s=""): buf.write(s + "\n")

    sep = "═" * 70
    p(sep)
    p(f"  AD TANIMOTO OVERLAP REPORT  (v{__version__})")
    p(f"  Total eval compounds analysed: {len(results):,}")
    p(sep)

    sim_col = results["max_sim_same_target"].fillna(results["max_sim_overall"])

    p("\n── OVERALL SIMILARITY STATISTICS ──")
    p(f"  {'Metric':<40} {'Value':>10}")
    p("  " + "-" * 52)
    for label, val in [
        ("Mean  max Tanimoto (same-target)",   sim_col.mean()),
        ("Median max Tanimoto (same-target)",  sim_col.median()),
        ("Std   max Tanimoto (same-target)",   sim_col.std()),
        ("% Memorised  (≥ 0.85)",              100*(sim_col >= 0.85).mean()),
        ("% Similar    (0.60–0.85)",           100*((sim_col >= 0.60) & (sim_col < 0.85)).mean()),
        ("% Analogous  (0.40–0.60)",           100*((sim_col >= 0.40) & (sim_col < 0.60)).mean()),
        ("% Novel      (< 0.40)",              100*(sim_col < 0.40).mean()),
    ]:
        suffix = "%" if label.startswith("%") else ""
        p(f"  {label:<40} {val:>9.2f}{suffix}")

    p("\n── PER-TARGET SIMILARITY STATISTICS ──")
    p(f"  {'Target':<10} {'N':>5} {'Mean':>7} {'Median':>8} "
      f"{'Mem≥.85':>8} {'Sim≥.60':>8} {'Novel<.40':>10}")
    p("  " + "-" * 62)
    for t in TARGETS:
        sub = results[results["eval_target"] == t]["max_sim_same_target"].dropna()
        if len(sub) == 0:
            continue
        p(f"  {t:<10} {len(sub):>5} {sub.mean():>7.3f} {sub.median():>8.3f} "
          f"{100*(sub >= 0.85).mean():>7.1f}% "
          f"{100*(sub >= 0.60).mean():>7.1f}% "
          f"{100*(sub <  0.40).mean():>9.1f}%")

    p("\n── ZONE BREAKDOWN (all targets) ──")
    zone_counts = results["zone"].value_counts()
    for zone in ZONE_LABELS[::-1]:
        n   = zone_counts.get(zone, 0)
        pct = 100 * n / max(len(results), 1)
        bar = "█" * int(pct / 2)
        p(f"  {zone:<28} {n:>4}  ({pct:5.1f}%)  {bar}")

    p("\n── ZONE BREAKDOWN PER TARGET ──")
    for t in TARGETS:
        sub = results[results["eval_target"] == t]
        if len(sub) == 0:
            continue
        p(f"\n  {t}  (n={len(sub)})")
        zc = sub["zone"].value_counts()
        for zone in ZONE_LABELS[::-1]:
            n   = zc.get(zone, 0)
            pct = 100 * n / max(len(sub), 1)
            p(f"    {zone:<28} {n:>4}  ({pct:5.1f}%)")

    p(f"\n{sep}")
    p("  INTERPRETATION GUIDE")
    p("  Memorised  (≥ 0.85) : near-identical to a training compound.")
    p("                         Model almost certainly saw this scaffold.")
    p("                         Good recall expected; not a true gen. test.")
    p("  Similar    (0.60–0.85): same chemotype with variation.")
    p("                         Predictions generally reliable.")
    p("  Analogous  (0.40–0.60): related scaffold, some novelty.")
    p("                         Moderate confidence; model interpolating.")
    p("  Novel      (< 0.40) : structurally distant from all training data.")
    p("                         Extrapolation — treat predictions cautiously.")
    p(sep)

    txt = buf.getvalue()
    print(txt)
    with open(out_path, "w") as f:
        f.write(txt)
    print(f"[INFO] Summary written to : {out_path}")


# ═══════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════
def make_plot(results: pd.DataFrame, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        vline_colors = ["#d62728", "#ff7f0e", "#2ca02c"]

        for ax, t in zip(axes, TARGETS):
            sub = results[results["eval_target"] == t]["max_sim_same_target"].dropna()
            if len(sub) == 0:
                ax.set_title(t)
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            ax.hist(sub, bins=40, range=(0, 1),
                    color="#4878d0", edgecolor="white", linewidth=0.4, alpha=0.85)
            for vl, col in zip([0.4, 0.6, 0.85], vline_colors):
                ax.axvline(vl, color=col, linestyle="--", linewidth=1.2, alpha=0.85)

            ax.set_title(f"{t}  (n={len(sub)})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Max Tanimoto to same-target train compound", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_xlim(0, 1)
            ax.text(0.02, 0.96,
                    f"mean={sub.mean():.3f}   median={sub.median():.3f}",
                    transform=ax.transAxes, fontsize=8, va="top", color="#333")

        zone_colors  = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        zone_short   = ["< 0.40  Novel", "0.40–0.60  Analogous",
                        "0.60–0.85  Similar", "≥ 0.85  Memorised"]
        patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(zone_colors, zone_short)]
        fig.legend(handles=patches, loc="lower center", ncol=4,
                   fontsize=9, bbox_to_anchor=(0.5, 0.01))

        fig.suptitle(
            "Eval Set — Tanimoto Similarity to Training Set\n"
            "(per-target, same-target nearest neighbour, Morgan r=2 2048-bit)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Plot saved to      : {out_path}")

    except ImportError:
        print("[WARN] matplotlib not available — skipping plot.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print(f"[INFO] AD_tanimoto_check.py  v{__version__}")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load eval
    if not os.path.exists(args.eval):
        sys.exit(f"[ERROR] Eval file not found: {args.eval}")
    eval_df = pd.read_csv(args.eval)
    print(f"[INFO] Eval file loaded: {eval_df.shape[0]:,} rows")
    for col, label in [(args.eval_smiles_col, "SMILES"),
                       (args.eval_target_col,  "target")]:
        if col not in eval_df.columns:
            sys.exit(f"[ERROR] Eval {label} column '{col}' not found.\n"
                     f"        Available: {list(eval_df.columns)}")

    # Load train
    train_df = load_train(args.train, args.train_smiles_col, args.train_target_col)

    # FP generator
    gen = make_fp_generator(args.radius, args.nbits)
    print(f"[INFO] Morgan FP: radius={args.radius}, nBits={args.nbits}")

    # Compute
    results = compute_similarity(
        eval_df, train_df,
        args.eval_smiles_col, args.eval_target_col,
        gen, args.top_k,
    )

    # Outputs
    csv_out = os.path.join(args.output_dir, "tanimoto_results.csv")
    results.to_csv(csv_out, index=False)
    print(f"[INFO] Per-compound CSV   : {csv_out}")

    txt_out = os.path.join(args.output_dir, "tanimoto_summary.txt")
    write_summary(results, txt_out)

    png_out = os.path.join(args.output_dir, "tanimoto_distribution.png")
    make_plot(results, png_out)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
