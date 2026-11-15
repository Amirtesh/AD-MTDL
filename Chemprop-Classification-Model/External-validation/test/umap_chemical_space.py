#!/usr/bin/env python3
"""
UMAP chemical space comparison: PubChem external actives vs. generated top compounds for AChE.

Usage:
    python umap_chemical_space.py
    python umap_chemical_space.py --target ACHE

Requires: rdkit, umap-learn, scikit-learn, matplotlib, pandas, numpy
    pip install rdkit umap-learn scikit-learn matplotlib pandas numpy
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import umap

RDLogger.DisableLog("rdApp.*")

# --- Configuration per target ---------------------------------------------
# class_col      : column in the external csv holding the activity label
# external_file  : external dataset file (PubChem)
# top_file       : file containing the generated top compounds for this target
# label          : nice name for plot titles/legends
TARGET_CONFIG = {
    "ACHE": {
        "class_col": "label",
        "external_file": "AChE_pubchem_external.csv",
        "top_file": "top_compounds_ACHE.csv",
        "label": "AChE",
    }
}

EXTERNAL_FILE = "AChE_pubchem_external.csv"
FP_RADIUS = 2
FP_NBITS = 2048


def find_smiles_col(df):
    for c in df.columns:
        if c.strip().lower() in ("smiles", "smile", "canonical_smiles"):
            return c
    raise ValueError(f"Could not find a SMILES column among: {list(df.columns)}")


def smiles_to_fp(smiles_list):
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
        fps.append(np.array(fp, dtype=np.uint8))
        valid_idx.append(i)
    return np.array(fps), valid_idx


def main():
    parser = argparse.ArgumentParser(description="UMAP chemical space plot for AChE")
    parser.add_argument(
        "--target",
        default="ACHE",
        choices=["ACHE", "ache", "AChE"],
        help="Target to plot (default: ACHE)",
    )
    parser.add_argument(
        "--external-csv",
        "--combined-csv",
        dest="external_csv",
        default=None,
        help=f"Path to external PubChem csv (default: {EXTERNAL_FILE})",
    )
    parser.add_argument(
        "--top-csv",
        default=None,
        help="Path to top_compounds csv (default: inferred from --target)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: umap_<target>.png)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for UMAP (default: 42)"
    )
    args = parser.parse_args()

    target_key = args.target.upper()
    cfg = TARGET_CONFIG[target_key]
    external_csv = args.external_csv or cfg["external_file"]
    top_csv = args.top_csv or cfg["top_file"]
    out_path = args.output or f"umap_{target_key}.png"

    # --- Load data ----------------------------------------------------------
    print(f"[{target_key}] Loading PubChem external data from {external_csv} ...")
    external_df = pd.read_csv(external_csv)
    smi_col = find_smiles_col(external_df)

    if cfg["class_col"] not in external_df.columns:
        sys.exit(
            f"ERROR: column '{cfg['class_col']}' not found in {external_csv}. "
            f"Available columns: {list(external_df.columns)}"
        )

    # Filter active compounds (only 'potent' labelled)
    val_series = external_df[cfg["class_col"]].astype(str).str.strip().str.lower()
    active_mask = val_series == "potent"
    actives = external_df.loc[active_mask, smi_col].dropna().tolist()
    print(f"[{target_key}] Found {len(actives)} potent external PubChem compounds.")

    print(f"[{target_key}] Loading generated top compounds from {top_csv} ...")
    top_df = pd.read_csv(top_csv)
    top_smi_col = find_smiles_col(top_df)
    generated = top_df[top_smi_col].dropna().tolist()
    print(f"[{target_key}] Found {len(generated)} generated compounds.")

    # --- Fingerprints ---------------------------------------------------------
    print("Computing Morgan fingerprints...")
    actives_fp, _ = smiles_to_fp(actives)
    gen_fp, _ = smiles_to_fp(generated)

    n_actives = actives_fp.shape[0]
    n_gen = gen_fp.shape[0]
    print(f"Valid fingerprints -> actives: {n_actives}, generated: {n_gen}")

    if n_actives == 0 or n_gen == 0:
        sys.exit("ERROR: one of the fingerprint sets is empty, check input SMILES.")

    all_fp = np.vstack([actives_fp, gen_fp])

    # --- UMAP -------------------------------------------------------------
    print("Fitting UMAP (this may take a moment)...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="jaccard",  # appropriate for binary fingerprints
        random_state=args.seed,
    )
    embedding = reducer.fit_transform(all_fp)

    emb_actives = embedding[:n_actives]
    emb_gen = embedding[n_actives:]

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        emb_actives[:, 0],
        emb_actives[:, 1],
        s=12,
        alpha=0.5,
        color="#7f7f7f",
        label=f"PubChem potent ({cfg['label']}), n={n_actives}",
    )
    ax.scatter(
        emb_gen[:, 0],
        emb_gen[:, 1],
        s=14,
        alpha=0.7,
        color="#d62728",
        label=f"Generated top compounds, n={n_gen}",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Chemical space: {cfg['label']} actives vs. generated compounds")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
