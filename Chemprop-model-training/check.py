import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "/home/amirtesh/chembl_36/chembl_36_sqlite/chembl_36.db"

TARGETS = {
    "AChE":  "CHEMBL220",
    "BuChE": "CHEMBL1914",
    "BACE1": "CHEMBL4822",
    "MAO-B": "CHEMBL2039",
}

# pChEMBL bin edges — 3-class
# Active:   >= 7.0  (~IC50 <= 100 nM)
# Moderate: >= 5.0 and < 7.0  (~100 nM – 10 µM)
# Inactive: < 5.0

QUERY = """
SELECT
    md.chembl_id    AS molecule_chembl_id,
    a.pchembl_value AS pchembl_value
FROM activities a
JOIN assays              ass ON a.assay_id  = ass.assay_id
JOIN target_dictionary    td ON ass.tid     = td.tid
JOIN molecule_dictionary  md ON a.molregno  = md.molregno
WHERE
    td.chembl_id              = ?
    AND ass.assay_type         IN ('B', 'F')
    AND ass.confidence_score   >= 8
    AND a.pchembl_value        IS NOT NULL
    AND a.standard_relation    = '='
    AND a.potential_duplicate  = 0
    AND a.data_validity_comment IS NULL
"""

def bin_compound(pchembl: float) -> str:
    if pchembl >= 7.0:
        return "Active"
    elif pchembl >= 5.0:
        return "Moderate"
    else:
        return "Inactive"

def print_separator(char="-", width=55):
    print(char * width)

con = sqlite3.connect(DB_PATH)

print_separator("=")
print("ChEMBL 36 — AD Target Data Summary")
print("Filters: assay_type IN (B,F) | confidence >= 8")
print("         standard_relation='=' | no duplicates | pChEMBL not null")
print("Deduplication: median pChEMBL per unique compound")
print_separator("=")

for target_name, chembl_id in TARGETS.items():
    df = pd.read_sql_query(QUERY, con, params=(chembl_id,))

    raw_count = len(df)

    if raw_count == 0:
        print(f"\n{target_name} ({chembl_id}): NO DATA FOUND — check filters")
        continue

    # Deduplicate: median pChEMBL per compound
    df_dedup = (
        df.groupby("molecule_chembl_id")["pchembl_value"]
        .median()
        .reset_index()
        .rename(columns={"pchembl_value": "pchembl_median"})
    )

    df_dedup["class"] = df_dedup["pchembl_median"].apply(bin_compound)

    total   = len(df_dedup)
    counts  = df_dedup["class"].value_counts()
    pchembl = df_dedup["pchembl_median"]

    print(f"\n{target_name}  ({chembl_id})")
    print_separator()
    print(f"  Raw measurements (pre-dedup) : {raw_count:>6,}")
    print(f"  Unique compounds (post-dedup): {total:>6,}")
    print(f"  pChEMBL  min / median / max  : "
          f"{pchembl.min():.2f} / {pchembl.median():.2f} / {pchembl.max():.2f}")
    print()
    print(f"  {'Class':<12} {'Count':>6}   {'%':>6}")
    print(f"  {'-'*28}")
    for cls in ["Active", "Moderate", "Inactive"]:
        n   = counts.get(cls, 0)
        pct = 100 * n / total if total > 0 else 0.0
        print(f"  {cls:<12} {n:>6,}   {pct:>5.1f}%")
    print(f"  {'TOTAL':<12} {total:>6,}   100.0%")

con.close()
print_separator("=")

