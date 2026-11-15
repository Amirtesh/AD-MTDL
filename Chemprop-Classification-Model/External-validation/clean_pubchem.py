import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ── Paths ──
CHEMBL_COMBINED_PATH = "/home/amirtesh/Projects and Papers/AIML-Projects/Alzheimer-target-optimization/Chemprop-Classification-Model/External-Validation/AD_MTDL_combined_chembl36.csv"
TARGETS = ["AChE", "BuChE", "BACE1", "MAO_B"]
TARGET_LABEL_COLS = {t: f"{t}_class" for t in TARGETS}

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None

def get_fp(smi):
    mol = Chem.MolFromSmiles(str(smi))
    return morgan_gen.GetFingerprint(mol) if mol is not None else None

# ── Load full ChEMBL pool ──
df_chembl = pd.read_csv(CHEMBL_COMBINED_PATH)
df_chembl["canon_smiles"] = df_chembl["smiles"].apply(canonicalize)
df_chembl = df_chembl.dropna(subset=["canon_smiles"])

for target in TARGETS:
    label_col = TARGET_LABEL_COLS[target]
    ext_path = f"{target}_pubchem_external.csv"

    print(f"\n{'='*60}\n  {target}\n{'='*60}")

    df_ext = pd.read_csv(ext_path)
    n_before = len(df_ext)
    df_ext["canon_smiles"] = df_ext["canonical_smiles"].apply(canonicalize)
    df_ext = df_ext.dropna(subset=["canon_smiles"])

    # ChEMBL pool for THIS target only (compounds with a real label for that task)
    chembl_target_smiles = set(
        df_chembl.loc[~df_chembl[label_col].isna(), "canon_smiles"]
    )

    # ── Step 1: exact-duplicate removal (canonical SMILES match) ──
    exact_dup_mask = df_ext["canon_smiles"].isin(chembl_target_smiles)
    n_exact_dup = exact_dup_mask.sum()
    df_ext_no_exact = df_ext[~exact_dup_mask].copy()
    print(f"  Exact duplicates (identical compound already in ChEMBL): {n_exact_dup} / {n_before}")

    # ── Step 2: Tanimoto near-duplicate check on the remainder ──
    chembl_fps = [get_fp(s) for s in chembl_target_smiles]
    chembl_fps = [fp for fp in chembl_fps if fp is not None]

    max_sims = []
    for smi in df_ext_no_exact["canon_smiles"]:
        fp = get_fp(smi)
        if fp is None:
            max_sims.append(np.nan)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, chembl_fps)
        max_sims.append(max(sims) if sims else 0.0)

    df_ext_no_exact["max_sim_to_chembl"] = max_sims
    valid_sims = df_ext_no_exact["max_sim_to_chembl"].dropna()

    print(f"  Near-dup check on remaining {len(df_ext_no_exact)} compounds:")
    print(f"    mean={valid_sims.mean():.3f}  median={valid_sims.median():.3f}  max={valid_sims.max():.3f}")
    print(f"    fraction > 0.85 (near-duplicate): {(valid_sims > 0.85).mean():.1%}")
    print(f"    fraction > 0.95 (near-identical): {(valid_sims > 0.95).mean():.1%}")

    # ── Step 3: produce a CLEAN external set — drop exact dups AND near-dups > 0.85 ──
    df_clean = df_ext_no_exact[df_ext_no_exact["max_sim_to_chembl"] <= 0.85].copy()
    print(f"\n  Final clean external set: {len(df_clean)} / {n_before} "
          f"({100*len(df_clean)/n_before:.1f}% retained)")
    print(f"  Label distribution (clean set):")
    print(df_clean["label"].value_counts().to_string())

    df_clean.drop(columns=["canon_smiles", "max_sim_to_chembl"]).to_csv(
        f"{target}_pubchem_external_CLEAN.csv", index=False
    )
    print(f"  -> saved {target}_pubchem_external_CLEAN.csv")
