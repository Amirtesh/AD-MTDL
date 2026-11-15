import sqlite3
import pandas as pd

DB_PATH = "/home/amirtesh/chembl_36/chembl_36_sqlite/chembl_36.db"

KNOWN_NAMES = {
    "AChE": [
        "donepezil", "galantamine", "rivastigmine", "tacrine",
        "physostigmine", "neostigmine", "pyridostigmine", "edrophonium",
        "huperzine", "metrifonate", "eptastigmine", "ambenonium",
    ],
    "BuChE": [
        "rivastigmine", "tacrine", "donepezil", "bambuterol",
        "ethopropazine", "galantamine", "cymserine", "decamethonium",
        "iso-ompa", "tetraisopropyl pyrophosphoramide",
    ],
    "BACE1": [
        "verubecestat", "atabecestat", "lanabecestat", "elenbecestat",
        "umibecestat", "ly2886721", "ly2811376", "cts-21166",
        "pf-06751979", "nb-360", "mk-8931",
    ],
    "MAO_B": [
        "selegiline", "rasagiline", "safinamide", "pargyline",
        "lazabemide", "sembragiline", "kds2010", "clorgyline",
        "tranylcypromine", "phenelzine", "moclobemide", "deprenyl",
    ],
}

EXACT_PREF_QUERY = """
SELECT DISTINCT
    md.chembl_id,
    md.pref_name,
    md.max_phase,
    cs.canonical_smiles
FROM molecule_dictionary md
JOIN compound_structures cs
    ON md.molregno = cs.molregno
WHERE LOWER(md.pref_name) = LOWER(?)
ORDER BY md.max_phase DESC
"""

EXACT_SYNONYM_QUERY = """
SELECT DISTINCT
    md.chembl_id,
    md.pref_name,
    md.max_phase,
    ms.synonyms,
    cs.canonical_smiles
FROM molecule_dictionary md
JOIN molecule_synonyms ms
    ON md.molregno = ms.molregno
JOIN compound_structures cs
    ON md.molregno = cs.molregno
WHERE LOWER(ms.synonyms) = LOWER(?)
ORDER BY md.max_phase DESC
"""

con = sqlite3.connect(DB_PATH)

resolved = {}
ambiguous = []

print("=" * 75)
print("  ChEMBL 36 — STRICT Name Resolution")
print("=" * 75)

for target, names in KNOWN_NAMES.items():

    print(f"\n{'─'*75}")
    print(f"  TARGET: {target}")
    print(f"{'─'*75}")

    print(
        f"  {'Search Term':<35}"
        f"{'ChEMBL ID':<16}"
        f"{'Pref Name':<30}"
        f"{'Phase':<8}"
        f"{'Match via'}"
    )

    print(
        f"  {'-'*33} "
        f"{'-'*14} "
        f"{'-'*28} "
        f"{'-'*6} "
        f"{'-'*10}"
    )

    target_resolved = []

    for name in names:

        pref = pd.read_sql_query(
            EXACT_PREF_QUERY,
            con,
            params=(name,)
        )

        syn = pd.read_sql_query(
            EXACT_SYNONYM_QUERY,
            con,
            params=(name,)
        )

        candidates = pd.concat([pref, syn], ignore_index=True)

        if candidates.empty:
            print(
                f"  {name:<35}"
                f"{'NOT FOUND':<16}"
                f"{'—':<30}"
                f"{'—':<8}"
                f"—"
            )
            continue

        candidates = candidates.drop_duplicates(
            subset=["chembl_id"]
        )

        if len(candidates) > 1:

            print(
                f"  {name:<35}"
                f"{'AMBIGUOUS':<16}"
                f"{str(len(candidates)) + ' matches':<30}"
                f"{'—':<8}"
                f"review"
            )

            ambiguous.append(
                {
                    "search_term": name,
                    "target": target,
                    "candidates": candidates[
                        ["chembl_id", "pref_name"]
                    ].to_dict("records"),
                }
            )

            continue

        row = candidates.iloc[0]

        phase = (
            int(row["max_phase"])
            if pd.notna(row["max_phase"])
            else "?"
        )

        match_via = (
            "pref_name"
            if len(pref) > 0
            else "synonym"
        )

        print(
            f"  {name:<35}"
            f"{row['chembl_id']:<16}"
            f"{str(row['pref_name']):<30}"
            f"{str(phase):<8}"
            f"{match_via}"
        )

        target_resolved.append(
            {
                "search_term": name,
                "chembl_id": row["chembl_id"],
                "pref_name": row["pref_name"],
                "max_phase": row["max_phase"],
                "canonical_smiles": row["canonical_smiles"],
                "match_via": match_via,
            }
        )

    resolved[target] = target_resolved

    print(
        f"\n  Resolved: "
        f"{len(target_resolved)}/{len(names)} unique compounds"
    )

con.close()

print(f"\n{'='*75}")
print("  AMBIGUOUS TERMS REQUIRING MANUAL REVIEW")
print("=" * 75)

if not ambiguous:
    print("\n  None")
else:
    for item in ambiguous:

        print(
            f"\n  {item['target']} :: "
            f"{item['search_term']}"
        )

        for c in item["candidates"]:
            print(
                f"      {c['chembl_id']:<15}"
                f"{c['pref_name']}"
            )

print(f"\n{'='*75}")
print("  RESOLVED DICTIONARY (copy-paste ready)")
print("=" * 75)

for target, entries in resolved.items():

    print(f"\n  {target}:")

    for e in entries:

        print(
            f"    {e['chembl_id']:<16}"
            f"# {e['pref_name']} "
            f"(Phase {e['max_phase']})"
        )
