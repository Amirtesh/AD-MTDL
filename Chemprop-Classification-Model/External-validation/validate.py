#!/usr/bin/env python3
"""
validate.py — Comprehensive external PubChem validation for the AD-MTDL Chemprop classifier.

Updated to use:
- Unified AD and inference pathway via predict.py (ADMTDLPredictor, ad_features, knn_mean_similarity)
- Correct threshold loading from thresholds.json (probability threshold per target class)
- Dual nearest-neighbour similarity tracking (global training set and same-target training set)
- Stratification performed strictly on the primary evaluation set (nearest_train_tanimoto < max_similarity)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from chemprop.nn import BCELoss as ChempropBCELoss

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from predict import ADMTDLPredictor, knn_mean_similarity, ad_features

RDLogger.DisableLog("rdApp.*")

TARGETS = ["AChE", "BuChE", "BACE1", "MAO_B"]

EXTERNAL_VALIDATION_DATA = {
    "AChE": "AChE_pubchem_external_CLEAN.csv",
    "BuChE": "BuChE_pubchem_external_CLEAN.csv",
    "BACE1": "BACE1_pubchem_external_CLEAN.csv",
    "MAO_B": "MAO_B_pubchem_external_CLEAN.csv",
}


# ----------------------------- Checkpoint Compatibility -----------------------------

class WeightedBCELoss(ChempropBCELoss):
    """Loss stub for checkpoint unpickling."""
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        if pos_weight is None:
            pos_weight = torch.ones(len(TARGETS), dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)

    def _calc_unreduced_loss(self, preds, targets, *args, **kwargs):
        weight = self.pos_weight.to(preds.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            preds, targets, reduction="none", pos_weight=weight
        )


WeightedBCELoss.__module__ = "__main__"
setattr(sys.modules["__main__"], "WeightedBCELoss", WeightedBCELoss)


# ----------------------------- Thresholds & SMILES -----------------------------

def load_thresholds(path):
    with open(path) as f:
        j = json.load(f)
    return {t: float(j["thresholds_probability"][f"{t}_class"]) for t in TARGETS}


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


# ----------------------------- Metrics & Statistics -----------------------------

def logit_threshold(probability):
    prob = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return math.log(prob / (1.0 - prob))


def stable_sigmoid(values):
    return 0.5 * (1.0 + np.tanh(np.asarray(values, dtype=np.float64) / 2.0))


CI_KEYS = ["AUROC", "AUPRC", "MCC", "Sensitivity", "Specificity", "BalancedAccuracy"]
CI_COLUMNS = [f"{k}_{s}" for k in CI_KEYS for s in ("lo", "hi")]
CLUSTER_CI_COLUMNS = [f"{k}_{s}" for k in CI_KEYS for s in ("clo", "chi")]


def _mcc(tp, tn, fp, fn):
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def _point_metrics(y, s, thr_logit):
    p = s >= thr_logit
    tp = int((p & (y == 1)).sum())
    tn = int((~p & (y == 0)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPRC": float(average_precision_score(y, s)),
        "MCC": _mcc(tp, tn, fp, fn),
        "Sensitivity": sens,
        "Specificity": spec,
        "BalancedAccuracy": (sens + spec) / 2.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def scaffold_groups(smiles):
    """Bemis-Murcko scaffold of the parent fragment, used as the cluster id for the cluster bootstrap."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from predict import parent_mol

    out = []
    for s in smiles:
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=parent_mol(Chem.MolFromSmiles(s)), includeChirality=False))
        except Exception:
            out.append("")
    return np.asarray(out, dtype=object)


def bootstrap_metrics(labels, scores, thr_logit, n_boot, rng, groups=None):
    """95% percentile CIs. groups=None: compounds resampled within class (class counts held fixed).
    groups given: whole scaffold clusters resampled (accounts for analogue series)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if groups is None:
        pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        draw = lambda: np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    else:
        gid = pd.factorize(np.asarray(groups))[0]
        members = [np.flatnonzero(gid == k) for k in range(gid.max() + 1)]
        draw = lambda: np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])

    vals = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        idx = draw()
        y = labels[idx]
        if y.min() == y.max():
            continue
        m = _point_metrics(y, scores[idx], thr_logit)
        for k in CI_KEYS:
            vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def calculate_metrics(labels, logits, threshold_prob, n_boot=0, rng=None, groups=None):
    labels = np.asarray(labels).astype(int)
    logits = np.asarray(logits, dtype=np.float64)
    thresh_logit = logit_threshold(threshold_prob)
    m = _point_metrics(labels, logits, thresh_logit)
    tp, tn, fp, fn = m["TP"], m["TN"], m["FP"], m["FN"]
    metrics = {
        "N": int(len(labels)),
        "N_potent": int((labels == 1).sum()),
        "N_inactive": int((labels == 0).sum()),
        "Threshold": float(threshold_prob),
        "Threshold_logit": float(thresh_logit),
        **{k: m[k] for k in CI_KEYS},
        "Accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "F1_macro": float((2 * tp / max(2 * tp + fp + fn, 1) + 2 * tn / max(2 * tn + fn + fp, 1)) / 2.0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    metrics.update({c: np.nan for c in CI_COLUMNS + CLUSTER_CI_COLUMNS})
    if n_boot > 0 and rng is not None:
        for k, (lo, hi) in bootstrap_metrics(labels, logits, thresh_logit, n_boot, rng).items():
            metrics[f"{k}_lo"], metrics[f"{k}_hi"] = lo, hi
        if groups is not None:
            for k, (lo, hi) in bootstrap_metrics(labels, logits, thresh_logit, n_boot, rng, groups).items():
                metrics[f"{k}_clo"], metrics[f"{k}_chi"] = lo, hi
    return metrics


STRATIFIED_COLUMNS = ["AUPRC_minus_prevalence", "SimilarityAsScore_AUROC"] + CI_KEYS + CI_COLUMNS


def stratum_metrics(labels, logits, similarity, threshold_prob, min_per_class, n_boot, rng):
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    n = n_pos + n_neg
    row = {
        "N": n,
        "N_potent": n_pos,
        "N_inactive": n_neg,
        "prevalence": (n_pos / n) if n else np.nan,
        "status": "ok",
    }
    row.update({col: np.nan for col in STRATIFIED_COLUMNS})
    if min(n_pos, n_neg) < min_per_class:
        row["status"] = f"insufficient (<{min_per_class} per class)"
        return row
    metrics = calculate_metrics(labels, logits, threshold_prob, n_boot=n_boot, rng=rng)
    for k in CI_KEYS + CI_COLUMNS:
        row[k] = metrics[k]
    row["AUPRC_minus_prevalence"] = metrics["AUPRC"] - row["prevalence"]
    row["SimilarityAsScore_AUROC"] = float(roc_auc_score(labels, similarity))
    return row


def parse_edges(text):
    edges = [float(x) for x in text.split(",")]
    if len(edges) < 3 or sorted(edges) != edges:
        raise ValueError("--edges must be strictly increasing with at least 3 values.")
    return edges


def make_bin_labels(edges):
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= 0:
            labels.append(f"<{hi:.2f}")
        elif hi > 1:
            labels.append(f">={lo:.2f}")
        else:
            labels.append(f"{lo:.2f}-{hi:.2f}")
    return labels


def iter_strata(frame, edges, bin_labels, max_sim):
    sim = frame["nearest_train_tanimoto"].to_numpy()
    yield "primary", f"Primary set (<{max_sim:.2f})", np.ones(len(frame), dtype=bool)
    for label in bin_labels:
        yield "similarity_bin", label, (frame["sim_bin"] == label).to_numpy()
    for tier in ("Low", "Medium", "High"):
        yield "ad_tier", tier, (frame["AD_confidence"] == tier).to_numpy()
    for cutoff in edges[1:-1]:
        yield "similarity_below", f"<{cutoff:.2f}", sim < cutoff


# ----------------------------- Plotting Functions -----------------------------

def roc_legend_label(metrics):
    label = f"AUROC={metrics['AUROC']:.3f}"
    low = metrics.get("AUROC_lo", np.nan)
    high = metrics.get("AUROC_hi", np.nan)
    if np.isfinite(low) and np.isfinite(high):
        label += f" ({low:.3f}-{high:.3f})"
    return label


def save_target_roc_pr_plot(target, labels, logits, metrics, output_dir):
    fpr, tpr, _ = roc_curve(labels, logits)
    precision, recall, _ = precision_recall_curve(labels, logits)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=roc_legend_label(metrics))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
    axes[0].set_title(f"{target} ROC Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, color="#d62728", lw=2, label=f"AUPRC={metrics['AUPRC']:.3f}")
    axes[1].axhline(labels.mean(), color="gray", linestyle="--", lw=1.2, label=f"Prevalence={labels.mean():.2f}")
    axes[1].set_title(f"{target} Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / f"{target}_roc_pr.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "fpr": fpr, "tpr": tpr, "recall": recall, "precision": precision,
        "auroc": metrics["AUROC"], "auprc": metrics["AUPRC"]
    }


def save_combined_roc_pr_plot(curve_data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for target, d in curve_data.items():
        axes[0].plot(d["fpr"], d["tpr"], lw=2, label=f"{target} ({d['auroc']:.3f})")
        axes[1].plot(d["recall"], d["precision"], lw=2, label=f"{target} ({d['auprc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
    axes[0].set_title("Combined ROC Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Combined Precision-Recall Curves", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "all_targets_roc_pr.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metric_bar_plots(metrics_df, output_dir):
    metric_cols = ["AUROC", "AUPRC", "MCC", "Sensitivity", "Specificity", "F1_macro"]
    targets = metrics_df["Target"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / len(targets)
    x = np.arange(len(metric_cols))

    for i, target in enumerate(targets):
        row = metrics_df[metrics_df["Target"] == target].iloc[0]
        vals = [float(row[m]) for m in metric_cols]
        offset = (i - (len(targets) - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=target)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics Across Targets (Primary Set)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = output_dir / "external_metric_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_count_plot(metrics_df, output_dir):
    counts = ["TN", "FP", "FN", "TP"]
    targets = metrics_df["Target"].tolist()
    x = np.arange(len(targets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, c in enumerate(counts):
        vals = [metrics_df.loc[metrics_df["Target"] == t, c].iloc[0] for t in targets]
        offset = (i - (len(counts) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=c)

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Count")
    ax.set_title("Confusion Matrix Breakdown (Primary Set)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = output_dir / "external_confusion_counts.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ad_distribution_plot(frames, output_dir):
    comb = pd.concat(frames.values(), ignore_index=True)
    targets = sorted(comb["Target"].unique().tolist())
    bins = np.linspace(0, 1, 41)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, t in zip(axes.ravel(), targets):
        sub = comb[comb["Target"] == t]
        ax.hist(sub[sub["label_numeric"] == 1]["nearest_train_tanimoto"], bins=bins, alpha=0.6, label="Potent", color="#1f77b4")
        ax.hist(sub[sub["label_numeric"] == 0]["nearest_train_tanimoto"], bins=bins, alpha=0.6, label="Inactive", color="#d62728")
        ax.set_title(t, fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

    for ax in axes[1]:
        ax.set_xlabel("Nearest-train Tanimoto")

    fig.suptitle("External AD Coverage by Target (Unfiltered)", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "external_ad_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_auroc_by_bin(strat_df, bin_labels, targets, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, target in zip(axes.ravel(), targets):
        sub = strat_df[(strat_df["Target"] == target) & (strat_df["stratum_type"] == "similarity_bin")].set_index("stratum")
        pos, vals, err_lo, err_hi, labels = [], [], [], [], []
        for p, b in enumerate(bin_labels):
            if b not in sub.index:
                continue
            rec = sub.loc[b]
            labels.append(f"{b}\n(n={int(rec['N'])})")
            if rec["status"] == "ok":
                pos.append(p)
                vals.append(rec["AUROC"])
                err_lo.append(rec["AUROC"] - rec["AUROC_lo"])
                err_hi.append(rec["AUROC_hi"] - rec["AUROC"])

        if pos:
            ax.errorbar(pos, vals, yerr=[err_lo, err_hi], fmt="o-", capsize=4, color="#1f77b4")
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(target, fontweight="bold")
        ax.set_ylim(0.2, 1.05)
        ax.grid(alpha=0.25)
        ax.set_ylabel("AUROC")

    fig.suptitle("External AUROC by Nearest-Neighbour Similarity Bin (Primary Set)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_auroc_by_ad_tier(strat_df, targets, path):
    tiers = ["Low", "Medium", "High"]
    colors = {"Low": "#d62728", "Medium": "#ff7f0e", "High": "#2ca02c"}
    sub = strat_df[strat_df["stratum_type"] == "ad_tier"]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    pos = np.arange(len(targets))

    for idx, tier in enumerate(tiers):
        offset = (idx - 1) * width
        vals, err_lo, err_hi = [], [], []
        for t in targets:
            rec = sub[(sub["Target"] == t) & (sub["stratum"] == tier)]
            if not rec.empty and rec.iloc[0]["status"] == "ok":
                r = rec.iloc[0]
                vals.append(r["AUROC"])
                err_lo.append(r["AUROC"] - r["AUROC_lo"])
                err_hi.append(r["AUROC_hi"] - r["AUROC"])
            else:
                vals.append(0)
                err_lo.append(0)
                err_hi.append(0)

        ax.bar(pos + offset, vals, width=width, color=colors[tier], label=tier,
               yerr=[err_lo, err_hi], capsize=3)

    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(targets)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC")
    ax.set_title("External AUROC by Applicability Domain Tier (Primary Set)", fontweight="bold")
    ax.legend(title="AD Tier")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Main Pipeline -----------------------------

def score_target(target, pubchem_dir, predictor, device, _unused, batch_size, scored_dir, reuse_scored):
    scored_path = scored_dir / f"{target}_scored.csv"
    if reuse_scored:
        return (pd.read_csv(scored_path), None) if scored_path.exists() else (None, f"cached scores not found: {scored_path}")

    filename = EXTERNAL_VALIDATION_DATA.get(target, f"{target}_pubchem_external_CLEAN.csv")
    input_path = Path(pubchem_dir) / filename
    if not input_path.exists():
        return None, f"file not found: {input_path}"

    df = pd.read_csv(input_path)
    if "label" not in df.columns or "canonical_smiles" not in df.columns:
        return None, "missing canonical_smiles or label column"

    df = df[df["label"].isin(["Potent", "Inactive"])].copy()
    df["canonical_smiles"] = df["canonical_smiles"].map(canonicalize)
    df = df.dropna(subset=["canonical_smiles"]).copy()

    if df.empty or df["label"].nunique() < 2:
        return None, "insufficient binary labels"

    df["label_numeric"] = (df["label"] == "Potent").astype(int)
    smiles = df["canonical_smiles"].tolist()

    ti, ad = TARGETS.index(target), predictor.ad
    pred = predictor.predict(smiles, batch_size=batch_size)
    fps, _ = ad_features(smiles)

    df["raw_logit"] = pred[f"{target}_logit"].to_numpy()
    df["nearest_train_tanimoto"] = knn_mean_similarity(fps, ad.fp, 1, "tanimoto")
    df["nearest_train_tanimoto_target"] = knn_mean_similarity(fps, ad.fp[ad.mask[:, ti]], 1, "tanimoto")
    df["AD_confidence"] = pred[f"{target}_ad_level"].str.capitalize().to_numpy()
    df["AD_struct_sim5"] = pred[f"{target}_ad_struct_sim"].to_numpy()
    df["AD_embed_sim5"] = pred[f"{target}_ad_embed_sim"].to_numpy()
    df["Target"] = target

    scored_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(scored_path, index=False)
    return df, None


def main():
    parser = argparse.ArgumentParser(description="AD-MTDL Stratified External PubChem Validation")
    parser.add_argument("--output_dir", default="validation_results", help="Directory to save outputs")
    parser.add_argument("--checkpoint", default="AD_MTDL_best.ckpt", help="Model checkpoint path")
    parser.add_argument("--pubchem_dir", default=".", help="Directory containing clean PubChem files")
    parser.add_argument("--ad_reference", default="AD_reference.npz", help="Path to AD reference .npz")
    parser.add_argument("--thresholds_json", default="thresholds.json", help="Path to thresholds.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_similarity", type=float, default=0.85, help="Primary evaluation similarity cutoff")
    parser.add_argument("--edges", default="0,0.5,0.7,0.85", help="Similarity bin edges")
    parser.add_argument("--min_per_class", type=int, default=15, help="Min compounds required per class in a stratum")
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap iterations for AUROC CI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_scored", action="store_true", help="Reuse cached scored CSVs")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    scored_dir = out_dir / "scored"
    pred_dir = out_dir / "predictions"
    conf_dir = out_dir / "confusion_matrices"

    for d in (out_dir, plot_dir, scored_dir, pred_dir, conf_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    edges = parse_edges(args.edges)
    bin_labels = make_bin_labels(edges)

    thresholds = load_thresholds(args.thresholds_json)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    predictor = None if args.reuse_scored else ADMTDLPredictor(
        args.checkpoint, args.thresholds_json, args.ad_reference, device
    )

    frames = {}
    for target in TARGETS:
        frame, err = score_target(
            target, args.pubchem_dir, predictor, device, None, args.batch_size, scored_dir, args.reuse_scored
        )
        if frame is None:
            print(f"[WARN] Skipping {target}: {err}")
            continue
        frames[target] = frame
        print(f"[{target}] Scored {len(frame):,} compounds")

    if not frames:
        print("[ERROR] No target datasets available.", file=sys.stderr)
        sys.exit(1)

    primary_rows, unfiltered_rows, duplicate_rows, curve_data = [], [], [], {}

    for target, df in frames.items():
        thresh = thresholds[target]
        sim_all = df["nearest_train_tanimoto"].to_numpy()
        labels_all = df["label_numeric"].to_numpy().astype(int)
        logits_all = df["raw_logit"].to_numpy().astype(np.float64)

        duplicate_rows.append({
            "Target": target,
            "N_unfiltered": len(df),
            "N_identical_to_train": int((sim_all >= 0.999).sum()),
            "N_identical_same_target": int((df["nearest_train_tanimoto_target"] >= 0.999).sum()),
            f"N_similarity_ge_{args.max_similarity:.2f}": int((sim_all >= args.max_similarity).sum()),
            "fraction_excluded": float((sim_all >= args.max_similarity).mean()),
            "N_primary": int((sim_all < args.max_similarity).sum()),
        })

        unfiltered_rows.append({"Target": target, **calculate_metrics(labels_all, logits_all, thresh)})

        primary_mask = sim_all < args.max_similarity
        prim_df = df[primary_mask].copy()
        p_labels = prim_df["label_numeric"].to_numpy().astype(int)
        p_logits = prim_df["raw_logit"].to_numpy().astype(np.float64)

        if len(np.unique(p_labels)) < 2:
            print(f"[WARN] {target}: Primary subset contains only 1 class. Skipping primary metrics.")
            continue

        # primary-set metrics call: add the scaffold groups
        metrics = calculate_metrics(
            p_labels,
            p_logits,
            thresh,
            n_boot=args.n_boot,
            rng=rng,
            groups=scaffold_groups(prim_df["canonical_smiles"].tolist()),
        )
        note = f"<{args.min_per_class} compounds in minority class" if min(metrics["N_potent"], metrics["N_inactive"]) < args.min_per_class else ""
        primary_rows.append({"Target": target, **metrics, "Note": note})

        prim_df["predicted_prob"] = stable_sigmoid(p_logits)
        prim_df["predicted_label"] = np.where(p_logits >= metrics["Threshold_logit"], "Potent", "Inactive")
        prim_df.to_csv(pred_dir / f"{target}_primary_predictions.csv", index=False)

        pd.DataFrame([
            {"actual": "Inactive", "pred_inactive": metrics["TN"], "pred_potent": metrics["FP"]},
            {"actual": "Potent", "pred_inactive": metrics["FN"], "pred_potent": metrics["TP"]},
        ]).to_csv(conf_dir / f"{target}_confusion_matrix.csv", index=False)

        curve_data[target] = save_target_roc_pr_plot(target, p_labels, p_logits, metrics, plot_dir)

    prim_metrics_df = pd.DataFrame(primary_rows)
    if not prim_metrics_df.empty:
        prim_metrics_df.to_csv(out_dir / "pubchem_external_metrics.csv", index=False)
        save_combined_roc_pr_plot(curve_data, plot_dir)
        save_metric_bar_plots(prim_metrics_df, plot_dir)
        save_confusion_count_plot(prim_metrics_df, plot_dir)

    pd.DataFrame(unfiltered_rows).to_csv(out_dir / "pubchem_external_metrics_unfiltered.csv", index=False)
    dup_df = pd.DataFrame(duplicate_rows)
    dup_df.to_csv(out_dir / "pubchem_external_duplicate_summary.csv", index=False)
    save_ad_distribution_plot(frames, plot_dir)

    # ----------------------------- Stratified Evaluation -----------------------------
    strat_rows = []
    for target, df in frames.items():
        df = df[df["nearest_train_tanimoto"] < args.max_similarity].copy()
        df["sim_bin"] = pd.cut(df["nearest_train_tanimoto"], bins=edges, labels=bin_labels, right=False).astype(object)
        l_all = df["label_numeric"].to_numpy().astype(int)
        log_all = df["raw_logit"].to_numpy().astype(np.float64)
        s_all = df["nearest_train_tanimoto"].to_numpy()

        for s_type, s_name, mask in iter_strata(df, edges, bin_labels, args.max_similarity):
            row = {
                "Target": target,
                "stratum_type": s_type,
                "stratum": s_name,
                "threshold_prob": thresholds[target],
                "threshold_logit": logit_threshold(thresholds[target]),
            }
            row.update(stratum_metrics(l_all[mask], log_all[mask], s_all[mask], thresholds[target], args.min_per_class, args.n_boot, rng))
            strat_rows.append(row)

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(out_dir / "stratified_metrics.csv", index=False)

    valid_targets = [t for t in TARGETS if t in frames]
    plot_auroc_by_bin(strat_df, bin_labels, valid_targets, plot_dir / "auroc_by_similarity_bin.png")
    plot_auroc_by_ad_tier(strat_df, valid_targets, plot_dir / "auroc_by_ad_tier.png")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 60 + "\n  VALIDATION COMPLETE\n" + "=" * 60)
    print("\n--- Leakage / Duplicate Summary ---")
    print(dup_df.to_string(index=False))

    # final report: replace the primary-metrics print with this
    if not prim_metrics_df.empty:
        print(f"\n--- Primary set (similarity < {args.max_similarity:.2f}); 95% CI: compound [lo, hi] | scaffold-cluster [clo, chi] ---")
        for _, r in prim_metrics_df.iterrows():
            print(f"\n{r['Target']}  N={int(r['N'])}  potent={int(r['N_potent'])}  inactive={int(r['N_inactive'])}")
            for k in CI_KEYS:
                print(f"  {k:<17}{r[k]:.3f}  [{r[k+'_lo']:.3f}, {r[k+'_hi']:.3f}] | [{r[k+'_clo']:.3f}, {r[k+'_chi']:.3f}]")

    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
