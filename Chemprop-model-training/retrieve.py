import sqlite3
import pandas as pd
from rdkit import Chem

DB_PATH = "/home/amirtesh/chembl_36/chembl_36_sqlite/chembl_36.db"

TARGETS = {
    "AChE":  "CHEMBL220",
    "BuChE": "CHEMBL1914",
    "BACE1": "CHEMBL4822",
    "MAO-B": "CHEMBL2039",
}

QUERY = """
SELECT
    md.chembl_id        AS molecule_chembl_id,
    cs.canonical_smiles AS smiles,
    a.pchembl_value     AS pchembl_value
FROM activities a
JOIN assays              ass ON a.assay_id  = ass.assay_id
JOIN target_dictionary    td ON ass.tid     = td.tid
JOIN molecule_dictionary  md ON a.molregno  = md.molregno
JOIN compound_structures  cs ON md.molregno = cs.molregno
WHERE
    td.chembl_id              = ?
    AND ass.assay_type         IN ('B', 'F')
    AND ass.confidence_score   >= 8
    AND a.pchembl_value        IS NOT NULL
    AND a.standard_relation    = '='
    AND a.potential_duplicate  = 0
    AND a.data_validity_comment IS NULL
    AND cs.canonical_smiles    IS NOT NULL
"""

def bin_compound(pchembl: float) -> str:
    if pchembl >= 7.0:
        return "Active"
    elif pchembl >= 5.0:
        return "Moderate"
    else:
        return "Inactive"

def is_valid_smiles(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except Exception:
        return False

con = sqlite3.connect(DB_PATH)

target_dfs = {} 

for target_name, chembl_id in TARGETS.items():
    df = pd.read_sql_query(QUERY, con, params=(chembl_id,))

    df_dedup = (
        df.groupby(["molecule_chembl_id", "smiles"])["pchembl_value"]
        .median()
        .reset_index()
        .rename(columns={"pchembl_value": "pchembl_median"})
    )

    df_dedup = df_dedup[df_dedup["smiles"].apply(is_valid_smiles)].copy()

    df_dedup["class"] = df_dedup["pchembl_median"].apply(bin_compound)

    out_individual = df_dedup[["smiles", "class"]].reset_index(drop=True)
    fname = f"{target_name.replace('-', '_')}_chembl36.csv"
    out_individual.to_csv(fname, index=False)
    print(f"Wrote {fname}  ({len(out_individual):,} compounds)")

    target_dfs[target_name] = df_dedup[["molecule_chembl_id", "smiles", "class"]].copy()

con.close()

combined = None
for target_name, df in target_dfs.items():
    col = target_name.replace("-", "_") + "_class"
    df_renamed = df[["smiles", "class"]].rename(columns={"class": col})
    if combined is None:
        combined = df_renamed
    else:
        combined = pd.merge(combined, df_renamed, on="smiles", how="outer")

combined = combined.drop_duplicates(subset="smiles").reset_index(drop=True)

combined.to_csv("AD_MTDL_combined_chembl36.csv", index=False)

print(f"\nWrote AD_MTDL_combined_chembl36.csv  ({len(combined):,} unique compounds)")
print(f"\nCoverage breakdown:")
target_cols = [c for c in combined.columns if c.endswith("_class")]
for col in target_cols:
    n = combined[col].notna().sum()
    print(f"  {col:<20}: {n:>6,} measured  |  {len(combined)-n:>6,} NaN")

print(f"\nMulti-target compound counts:")
combined["n_targets"] = combined[target_cols].notna().sum(axis=1)
for n in sorted(combined["n_targets"].unique()):
    count = (combined["n_targets"] == n).sum()
    print(f"  Present in {n} target(s): {count:>6,}")

