import sqlite3
import pandas as pd

DB_PATH = "/home/amirtesh/chembl_36/chembl_36_sqlite/chembl_36.db"

TARGETS = {
    "AChE":  "CHEMBL220",
    "BuChE": "CHEMBL1914",
    "BACE1": "CHEMBL4822",
    "MAO_B": "CHEMBL2039",
}

# ───────────────────────────────────────────────
# Known compounds hardcoded with ChEMBL IDs
# SMILES will be fetched from ChEMBL itself
# ───────────────────────────────────────────────
KNOWN = {
    "AChE": [
        "CHEMBL502",      # DONEPEZIL
        "CHEMBL659",      # GALANTAMINE
        "CHEMBL636",      # RIVASTIGMINE
        "CHEMBL95",       # TACRINE
        "CHEMBL94",       # PHYSOSTIGMINE
        "CHEMBL278020",   # NEOSTIGMINE
        "CHEMBL1115",     # PYRIDOSTIGMINE
        "CHEMBL1104",     # EDROPHONIUM
        "CHEMBL167150",   # TRICHLORFON (METRIFONATE)
        "CHEMBL433041",   # EPTASTIGMINE
        "CHEMBL1652",     # AMBENONIUM
        "CHEMBL395280",   # HUPERZINE A

    ],

    "BuChE": [
        "CHEMBL636",      # RIVASTIGMINE
        "CHEMBL95",       # TACRINE
        "CHEMBL502",      # DONEPEZIL
        "CHEMBL521589",   # BAMBUTEROL
        "CHEMBL1206",     # ETHOPROPAZINE
        "CHEMBL659",      # GALANTAMINE
        "CHEMBL1190",     # DECAMETHONIUM
        "CHEMBL494887",   # TETRAISOPROPYL PYROPHOSPHORAMIDE
        "CHEMBL129837",   # Cymserine
        "CHEMBL494887",   # iso-OMPA
    ],

    "BACE1": [
        "CHEMBL3301601",  # VERUBECESTAT (MK-8931)
        "CHEMBL3916243",  # ATABECESTAT
        "CHEMBL3989948",  # LANABECESTAT
        "CHEMBL4204869",  # ELENBECESTAT
        "CHEMBL2396989",  # LY-2886721
        "CHEMBL2333941",  # LY-2811376
        "CHEMBL3670375",  # Umibecestat
        "CHEMBL3952064",  # Pf-06751979
        "CHEMBL3261067",  # NB-360
    ],

    "MAO_B": [
        "CHEMBL972",      # SELEGILINE (DEPRENYL)
        "CHEMBL887",      # RASAGILINE
        "CHEMBL396778",   # SAFINAMIDE
        "CHEMBL673",      # PARGYLINE
        "CHEMBL279390",   # LAZABEMIDE
        "CHEMBL3707394",  # SEMBRAGILINE
        "CHEMBL3989843",  # TRANYLCYPROMINE
        "CHEMBL1089",     # PHENELZINE
        "CHEMBL86304",    # MOCLOBEMIDE
    ],
}

# ───────────────────────────────────────────────
# Query 1: fetch SMILES + name for known IDs
# ───────────────────────────────────────────────
SMILES_QUERY = """
SELECT
    md.chembl_id,
    md.pref_name,
    cs.canonical_smiles,
    md.max_phase
FROM molecule_dictionary md
JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE md.chembl_id IN ({placeholders})
AND cs.canonical_smiles IS NOT NULL
"""

# ───────────────────────────────────────────────
# Query 2: clinical compounds (Phase 1-4) with
# activity against each target from ChEMBL assays
# ───────────────────────────────────────────────
CLINICAL_QUERY = """
SELECT DISTINCT
    md.chembl_id,
    md.pref_name,
    cs.canonical_smiles,
    md.max_phase,
    MAX(a.pchembl_value) AS best_pchembl
FROM activities a
JOIN assays              ass ON a.assay_id  = ass.assay_id
JOIN target_dictionary    td ON ass.tid     = td.tid
JOIN molecule_dictionary  md ON a.molregno  = md.molregno
JOIN compound_structures  cs ON md.molregno = cs.molregno
WHERE
    td.chembl_id            = ?
    AND md.max_phase        >= 1
    AND a.pchembl_value     IS NOT NULL
    AND a.potential_duplicate = 0
    AND a.data_validity_comment IS NULL
    AND cs.canonical_smiles IS NOT NULL
GROUP BY md.chembl_id, md.pref_name, cs.canonical_smiles, md.max_phase
ORDER BY md.max_phase DESC, best_pchembl DESC
"""

# ───────────────────────────────────────────────
# Also pull via drug_mechanism table — this catches
# approved drugs whose mechanism is annotated
# ───────────────────────────────────────────────
MECHANISM_QUERY = """
SELECT DISTINCT
    md.chembl_id,
    md.pref_name,
    cs.canonical_smiles,
    md.max_phase,
    dm.mechanism_of_action
FROM drug_mechanism dm
JOIN molecule_dictionary md ON dm.molregno = md.molregno
JOIN compound_structures cs ON md.molregno = cs.molregno
JOIN target_dictionary    td ON dm.tid     = td.tid
WHERE td.chembl_id = ?
AND cs.canonical_smiles IS NOT NULL
ORDER BY md.max_phase DESC
"""

# ───────────────────────────────────────────────
# Run queries
# ───────────────────────────────────────────────
con = sqlite3.connect(DB_PATH)
all_output = {}

for target_name, chembl_id in TARGETS.items():
    print(f"\n{'='*55}")
    print(f"  {target_name}  ({chembl_id})")
    print(f"{'='*55}")

    frames = []

    # --- Known compounds ---
    known_ids = KNOWN.get(target_name, [])
    if known_ids:
        placeholders = ",".join("?" * len(known_ids))
        df_known = pd.read_sql_query(
            SMILES_QUERY.format(placeholders=placeholders),
            con, params=known_ids
        )
        df_known["source"] = "known_inhibitor"
        frames.append(df_known)
        print(f"  Known inhibitors fetched : {len(df_known)}/{len(known_ids)}")
        missing = set(known_ids) - set(df_known["chembl_id"].tolist())
        if missing:
            print(f"  Not found in ChEMBL     : {missing}")

    # --- Clinical via activity table ---
    df_clinical = pd.read_sql_query(CLINICAL_QUERY, con, params=(chembl_id,))
    df_clinical["source"] = "clinical_activity"
    frames.append(df_clinical)
    print(f"  Clinical (Phase 1-4) via activities : {len(df_clinical)}")

    # --- Clinical via drug_mechanism table ---
    df_mech = pd.read_sql_query(MECHANISM_QUERY, con, params=(chembl_id,))
    df_mech["source"] = "drug_mechanism"
    if "best_pchembl" not in df_mech.columns:
        df_mech["best_pchembl"] = None
    frames.append(df_mech)
    print(f"  Annotated mechanisms               : {len(df_mech)}")

    # --- Merge, deduplicate on chembl_id ---
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset="chembl_id", keep="first")
    df_all["target"] = target_name

    # Phase label
    def phase_label(p):
        if pd.isna(p):  return "Unknown"
        p = int(p)
        return {4: "Approved", 3: "Phase 3", 2: "Phase 2",
                1: "Phase 1", 0: "Preclinical"}.get(p, f"Phase {p}")

    df_all["phase_label"] = df_all["max_phase"].apply(phase_label)

    print(f"\n  Phase breakdown:")
    for phase, grp in df_all.groupby("phase_label"):
        print(f"    {phase:<15}: {len(grp):>4} compounds")
    print(f"  TOTAL unique           : {len(df_all)}")

    all_output[target_name] = df_all

con.close()

# ───────────────────────────────────────────────
# Save per-target CSVs + combined SMILES file
# ───────────────────────────────────────────────
combined_rows = []

for target_name, df in all_output.items():
    fname = f"{target_name}_clinical_known.csv"
    df.to_csv(fname, index=False)
    print(f"\nSaved: {fname}  ({len(df)} compounds)")

    # For predictor: smiles only (one file per target, then combined)
    smiles_fname = f"{target_name}_eval_smiles.txt"
    df[["chembl_id", "canonical_smiles", "pref_name",
        "max_phase", "phase_label", "source"]].to_csv(
        smiles_fname.replace(".txt", ".csv"), index=False
    )

    combined_rows.append(df)

# Combined file across all targets — unique SMILES, annotated
df_combined = pd.concat(combined_rows, ignore_index=True)
df_combined = df_combined.sort_values(
    ["chembl_id", "target"]
).reset_index(drop=True)

df_combined.to_csv("AD_eval_all_targets.csv", index=False)

# Prediction-ready: unique SMILES, one per row
df_for_pred = (
    df_combined[["canonical_smiles", "chembl_id", "pref_name", "target", "phase_label"]]
    .drop_duplicates(subset="canonical_smiles")
    .reset_index(drop=True)
)
df_for_pred.to_csv("AD_eval_for_prediction.csv", index=False)

print(f"\n{'='*55}")
print(f"Saved: AD_eval_all_targets.csv    ({len(df_combined)} rows)")
print(f"Saved: AD_eval_for_prediction.csv ({len(df_for_pred)} unique SMILES)")
print("Run prediction with:")
print("  python predict.py --input AD_eval_for_prediction.csv --smiles_col canonical_smiles")

