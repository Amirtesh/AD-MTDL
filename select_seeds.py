# select_seeds.py
# Selects seeds from the moderate zone (pChEMBL 5.0–7.0) for each campaign.
# Diversity filter: among candidates, pick the one with lowest maximum
# Tanimoto similarity to all other seeds (maximizes seed diversity).

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

CSV_PATH = "AD_MTDL_combined_chembl36.csv"
OUT_PATH = "campaign_seeds.csv"

TARGET_COLS = ["AChE_class", "BuChE_class", "BACE1_class", "MAO_B_class"]
TARGET_NAMES = ["AChE", "BuChE", "BACE1", "MAO_B"]

# ── Load and validate ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {len(df):,} compounds")
print(f"Columns: {list(df.columns)}")

valid_mask = df["smiles"].apply(
    lambda s: Chem.MolFromSmiles(str(s)) is not None
)
df = df[valid_mask].reset_index(drop=True)
print(f"Valid SMILES: {len(df):,}")

# ── Desalt — must happen before any selection ─────────────
def desalt(smi: str) -> str:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return smi
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return smi
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)

df["smiles"] = df["smiles"].apply(desalt)
print(f"Desalting applied.")

# ── Fingerprints for diversity selection ──────────────────
def get_fp(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

def max_tanimoto_to_set(fp, fp_set):
    """Max Tanimoto similarity of fp to any fingerprint in fp_set."""
    if not fp_set:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_set)
    return max(sims) if sims else 0.0

def pick_diverse_seed(candidates: pd.DataFrame,
                      existing_fps: list,
                      n_candidates: int = 50) -> pd.Series:
    """
    From candidates, pick the compound with lowest max Tanimoto
    to already-selected seeds. If no existing seeds, pick randomly
    from top n_candidates.
    """
    # Take up to n_candidates for efficiency
    pool = candidates.head(n_candidates)

    if not existing_fps:
        return pool.iloc[0]

    best_row  = None
    best_score = 1.0  # lower is more diverse

    for _, row in pool.iterrows():
        fp = get_fp(row["smiles"])
        if fp is None:
            continue
        sim = max_tanimoto_to_set(fp, existing_fps)
        if sim < best_score:
            best_score = sim
            best_row   = row

    return best_row if best_row is not None else pool.iloc[0]


# ══════════════════════════════════════════════════════════
# CAMPAIGN SEED SELECTION
# ══════════════════════════════════════════════════════════
selected_seeds = []
selected_fps   = []

print("\n" + "═" * 65)
print("SEED SELECTION")
print("═" * 65)

# ── Single-target campaigns ───────────────────────────────
# Criterion: Moderate for target X, not Active for any other target
# Rationale: oracle has real work to do (sub-threshold start),
#            but compound is in relevant chemical space.
#            Not Active for other targets ensures the RL learns
#            multi-target activity rather than just inheriting it.

for t_name, t_col in zip(TARGET_NAMES, TARGET_COLS):
    other_cols = [c for c in TARGET_COLS if c != t_col]

    mask = (df[t_col] == "Moderate")
    for oc in other_cols:
        mask = mask & (df[oc].isin(["Inactive", float("nan")]) |
                       df[oc].isna())

    candidates = df[mask].reset_index(drop=True)
    print(f"\n{t_name} single-target:")
    print(f"  Moderate for {t_name}, not Active for others: "
          f"{len(candidates):,} candidates")

    if candidates.empty:
        # Relax: allow any non-Active for other targets
        mask_relaxed = df[t_col] == "Moderate"
        candidates   = df[mask_relaxed].reset_index(drop=True)
        print(f"  Relaxed (any Moderate for {t_name}): "
              f"{len(candidates):,} candidates")

    if candidates.empty:
        print(f"  WARNING: No moderate candidates for {t_name}")
        continue

    seed_row = pick_diverse_seed(candidates, selected_fps)
    seed_fp  = get_fp(seed_row["smiles"])
    if seed_fp:
        selected_fps.append(seed_fp)

    # Show class profile of selected seed
    profile = {
        t: seed_row.get(c, "NaN")
        for t, c in zip(TARGET_NAMES, TARGET_COLS)
    }

    print(f"  Selected seed: {seed_row['smiles']}")
    print(f"  Activity profile: {profile}")

    selected_seeds.append({
        "campaign"  : f"single_{t_name}",
        "seed_smiles": seed_row["smiles"],
        "target_1"  : t_name,
        "target_2"  : "",
        "AChE_label": seed_row.get("AChE_class", ""),
        "BuChE_label": seed_row.get("BuChE_class", ""),
        "BACE1_label": seed_row.get("BACE1_class", ""),
        "MAO_B_label": seed_row.get("MAO_B_class", ""),
        "selection_criterion": f"Moderate for {t_name}, not Active for others",
    })

# ── Dual AChE+BACE1 campaign ──────────────────────────────
# Criterion: Moderate for AChE AND Moderate for BACE1
# Rationale: both targets start in the ambiguous activity zone,
#            forcing the oracle to independently push both
#            AChE and BACE1 activity above their respective
#            thresholds during optimization — the cleanest test
#            of genuine dual-target generative design.

print(f"\nDual AChE+BACE1 campaign:")

mask_dual = (
    (df["AChE_class"] == "Moderate") &
    (df["BACE1_class"] == "Moderate")
)
candidates_dual = df[mask_dual].reset_index(drop=True)
print(f"  Both Moderate (AChE+BACE1): "
      f"{len(candidates_dual):,} candidates")

if candidates_dual.empty:
    # Fallback: Moderate for either, not Active for both
    mask_dual = (
        (df["AChE_class"] == "Moderate") |
        (df["BACE1_class"] == "Moderate")
    ) & ~(
        (df["AChE_class"] == "Active") &
        (df["BACE1_class"] == "Active")
    )
    candidates_dual = df[mask_dual].reset_index(drop=True)
    print(f"  Fallback (Moderate for either, not Active for both): "
          f"{len(candidates_dual):,} candidates")

if not candidates_dual.empty:
    seed_dual    = pick_diverse_seed(candidates_dual, selected_fps)
    seed_dual_fp = get_fp(seed_dual["smiles"])
    if seed_dual_fp:
        selected_fps.append(seed_dual_fp)

    profile_dual = {
        t: seed_dual.get(c, "NaN")
        for t, c in zip(TARGET_NAMES, TARGET_COLS)
    }

    print(f"  Selected seed: {seed_dual['smiles']}")
    print(f"  Activity profile: {profile_dual}")

    selected_seeds.append({
        "campaign"   : "dual_AChE_BACE1",
        "seed_smiles": seed_dual["smiles"],
        "target_1"   : "AChE",
        "target_2"   : "BACE1",
        "AChE_label" : seed_dual.get("AChE_class", ""),
        "BuChE_label": seed_dual.get("BuChE_class", ""),
        "BACE1_label": seed_dual.get("BACE1_class", ""),
        "MAO_B_label": seed_dual.get("MAO_B_class", ""),
        "selection_criterion":
            "Moderate for AChE, Moderate for BACE1",
    })
else:
    print("  WARNING: No candidates found for dual campaign")

# ── Cross-seed diversity check ────────────────────────────
print(f"\n{'═' * 65}")
print("SEED DIVERSITY CHECK")
print("═" * 65)
print(f"{'Campaign':<25} {'SMILES (truncated)':<45}")
print("-" * 65)
for s in selected_seeds:
    smi = s["seed_smiles"]
    smi_display = smi[:43] + "..." if len(smi) > 43 else smi
    print(f"  {s['campaign']:<23} {smi_display}")

if len(selected_fps) >= 2:
    print("\nPairwise Tanimoto between seeds:")
    names = [s["campaign"] for s in selected_seeds]
    for i in range(len(selected_fps)):
        for j in range(i + 1, len(selected_fps)):
            sim = DataStructs.TanimotoSimilarity(
                selected_fps[i], selected_fps[j]
            )
            print(f"  {names[i]:<25} vs {names[j]:<25}: {sim:.3f}")

# ── Save ──────────────────────────────────────────────────
seeds_df = pd.DataFrame(selected_seeds)
seeds_df.to_csv(OUT_PATH, index=False)

print(f"\n{'═' * 65}")
print(f"Saved: {OUT_PATH}")
print(f"Total seeds selected: {len(selected_seeds)}")
print("═" * 65)

# Print final seeds in format ready to paste into run_campaigns.sh
print("\nCOPY INTO run_campaigns.sh:")
print("─" * 65)
for s in selected_seeds:
    camp = s["campaign"].upper()
    print(f"# {camp} — {s['selection_criterion']}")
    print(f"SEED_{camp.replace('SINGLE_', '').replace('DUAL_', '')} "
          f'= "{s["seed_smiles"]}"')
    print()

