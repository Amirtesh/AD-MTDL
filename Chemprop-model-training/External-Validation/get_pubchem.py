import pandas as pd
import numpy as np
import requests
import time
import io
from rdkit import Chem

# ── PubChem PUG REST base URL ──
PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# ── Targets: SAME UniProt accessions as get_bindingdb.py. ──
# FIX 1 (the core fix): querying by UniProt accession instead of gene symbol
# restricts assay retrieval to the specific human protein, by construction,
# rather than pulling every organism's ortholog under one gene-symbol query
# and trying to filter organisms out afterward. AChE and BuChE in particular
# have a long history of non-human (rat, mouse, bovine, Torpedo californica)
# assay use in the literature — a gene-symbol query silently pools all of
# these together as if they were one target.
TARGET_UNIPROT = {
    "AChE":  "P22303",
    "BuChE": "P06276",
    "BACE1": "P56817",
    "MAO_B": "P27338",
}

# FIX 2: restrict to standard confirmatory potency measures, matching the
# same set get_bindingdb.py draws from (Ki/Kd/IC50/EC50). Excludes AC50
# (typically single-concentration/HTS curve-fit, not a confirmatory binding
# or functional potency value) and other derived quantities (e.g. "Potency",
# "% Inhibition", "Activity") that PubChem concise data can carry under
# Activity Name for the same target.
ALLOWED_ACTIVITY_NAMES = {"IC50", "KI", "KD", "EC50"}

# FIX 3: PubChem concise data carries an assay-level Activity Outcome call
# (Active / Inactive / Inconclusive / Unspecified / Probe). Rows flagged
# Inconclusive/Unspecified can still carry a numeric Activity Value [uM]
# and would otherwise pass straight through a numeric-only filter.
ALLOWED_OUTCOMES = {"ACTIVE", "INACTIVE"}

# FIX 4 (defensive, belt-and-suspenders): if the concise response includes a
# target-identifying column, cross-check rows against our accession/gene —
# guards against multi-target assay panels where a single AID tests several
# targets/organisms and an accession-based AID lookup could still admit rows
# for a different target within that same AID.
TARGET_ACCESSION_COLS = ["Target Accession", "Target UniProt Accession"]
TARGET_ORGANISM_COLS = ["Target Source Organism", "Organism", "Target Organism"]
HUMAN_ORGANISM_STRINGS = {"homo sapiens", "human"}

# Rate-limit: PubChem allows max 5 requests/second
REQUEST_DELAY = 0.25  # seconds between requests


def rate_limit():
    """Sleep to respect PubChem's rate limit."""
    time.sleep(REQUEST_DELAY)


def to_pchembl(value_um):
    """Convert a potency value in µM to pChEMBL-equivalent (-log10 molar)."""
    try:
        v = float(value_um)
        return -np.log10(v * 1e-6) if v > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def bin_compound(pchembl):
    """Bin a compound by pChEMBL value — same thresholds as get_bindingdb.py."""
    if pd.isna(pchembl):
        return None
    if pchembl >= 7.0:
        return "Potent"
    elif pchembl >= 5.0:
        return "Moderate"
    return "Inactive"


def canonicalize(smi):
    """Canonical SMILES via RDKit, or None if unparseable."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(str(smi)) is not None
    except Exception:
        return False


def get_aids_for_accession(uniprot_accession):
    """Retrieve all Assay IDs (AIDs) for a given UniProt accession from
    PubChem — restricts retrieval to the specific human protein rather than
    every organism's ortholog under a shared gene symbol."""
    url = f"{PUG_BASE}/assay/target/accession/{uniprot_accession}/aids/TXT"
    rate_limit()
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"  WARNING: Failed to get AIDs for accession {uniprot_accession}: "
              f"HTTP {resp.status_code}")
        return []
    aids = [int(x.strip()) for x in resp.text.strip().split("\n") if x.strip().isdigit()]
    return aids


def get_concise_data(aids_batch):
    """Download concise activity data for a batch of AIDs.
    Returns a DataFrame with columns: AID, SID, CID, Activity Outcome,
    Activity Value [uM], Activity Name, Assay Name, Assay Type, and
    (when present) target-identifying columns, etc.
    """
    aids_str = ",".join(str(a) for a in aids_batch)
    url = f"{PUG_BASE}/assay/aid/{aids_str}/concise/JSON"
    rate_limit()
    try:
        resp = requests.get(url, timeout=60)
    except requests.exceptions.Timeout:
        print(f"    Timeout for AIDs batch starting with {aids_batch[0]}")
        return pd.DataFrame()

    if resp.status_code != 200:
        return pd.DataFrame()

    try:
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    table = data.get("Table", {})
    columns = table.get("Columns", {}).get("Column", [])
    rows = table.get("Row", [])

    if not columns or not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        cells = row.get("Cell", [])
        while len(cells) < len(columns):
            cells.append("")
        records.append(dict(zip(columns, cells)))

    return pd.DataFrame(records)


def get_smiles_for_cids(cids, max_retries=3):
    """Batch-fetch CanonicalSMILES for a list of CIDs via POST requests.
    Returns a dict {cid: smiles}, canonicalized via RDKit."""
    smiles_map = {}
    cid_list = list(set(cids))
    batch_size = 200

    for i in range(0, len(cid_list), batch_size):
        batch = cid_list[i:i + batch_size]
        cids_str = ",".join(str(c) for c in batch)
        url = f"{PUG_BASE}/compound/cid/property/CanonicalSMILES/CSV"

        resp = None
        for attempt in range(1, max_retries + 1):
            rate_limit()
            try:
                resp = requests.post(url, data={"cid": cids_str}, timeout=60)
            except requests.exceptions.Timeout:
                print(f"    Timeout fetching SMILES for CID batch at index {i} "
                      f"(attempt {attempt}/{max_retries})")
                resp = None

            if resp is not None and resp.status_code == 200:
                break
            elif resp is not None and resp.status_code in (500, 502, 503, 429):
                wait = 2 ** attempt
                print(f"    HTTP {resp.status_code} for CID batch at index {i} "
                      f"(attempt {attempt}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
                resp = None
            elif resp is not None:
                print(f"    Failed SMILES fetch for CID batch at index {i}: HTTP {resp.status_code}")
                break

        if resp is None or resp.status_code != 200:
            print(f"    Skipping CID batch at index {i} after {max_retries} attempts")
            continue

        try:
            df_smi = pd.read_csv(io.StringIO(resp.text))
            batch_count = 0
            smi_col = None
            for candidate in ("CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES"):
                if candidate in df_smi.columns:
                    smi_col = candidate
                    break
            if smi_col is None:
                print(f"    WARNING: No SMILES column found in batch at index {i}. "
                      f"Columns: {list(df_smi.columns)}")
                continue
            for _, row in df_smi.iterrows():
                cid_val = str(int(row["CID"])) if pd.notna(row.get("CID")) else None
                smi_val = row.get(smi_col, None)
                if cid_val and pd.notna(smi_val):
                    canon = canonicalize(str(smi_val))
                    if canon is not None:
                        smiles_map[cid_val] = canon
                        batch_count += 1
            if batch_count == 0 and len(df_smi) > 0:
                print(f"    WARNING: Batch at index {i} returned {len(df_smi)} rows "
                      f"but 0 valid SMILES. Columns: {list(df_smi.columns)}")
        except Exception as e:
            print(f"    Error parsing SMILES CSV for batch at index {i}: {e}")
            continue

        if (i // batch_size + 1) % 10 == 0:
            print(f"    Fetched SMILES for {min(i + batch_size, len(cid_list)):,}/{len(cid_list):,} CIDs "
                  f"(map size: {len(smiles_map):,})")

    return smiles_map


def apply_quality_filters(df_raw, target_name, uniprot_accession):
    """FIX 2, 3, 4 applied together. Returns filtered df with a printed
    breakdown of how many rows each filter removed, so the effect of each
    is auditable rather than silently applied."""
    n0 = len(df_raw)

    # FIX 2: Activity Name — keep only standard confirmatory potency types
    if "Activity Name" in df_raw.columns:
        name_norm = df_raw["Activity Name"].astype(str).str.strip().str.upper()
        df_raw = df_raw[name_norm.isin(ALLOWED_ACTIVITY_NAMES)].copy()
    else:
        print(f"  WARNING: No 'Activity Name' column present for {target_name} — "
              f"cannot filter by assay type. All rows retained at this step.")
    n1 = len(df_raw)
    print(f"  After Activity Name filter (IC50/Ki/Kd/EC50 only): {n1:,} / {n0:,}")

    # FIX 3: Activity Outcome — drop Inconclusive/Unspecified/Probe rows
    if "Activity Outcome" in df_raw.columns:
        outcome_norm = df_raw["Activity Outcome"].astype(str).str.strip().str.upper()
        df_raw = df_raw[outcome_norm.isin(ALLOWED_OUTCOMES)].copy()
    else:
        print(f"  WARNING: No 'Activity Outcome' column present for {target_name} — "
              f"cannot filter by outcome quality. All rows retained at this step.")
    n2 = len(df_raw)
    print(f"  After Activity Outcome filter (Active/Inactive only): {n2:,} / {n1:,}")

    # FIX 4: defensive target-accession cross-check, if the field exists
    acc_col = next((c for c in TARGET_ACCESSION_COLS if c in df_raw.columns), None)
    if acc_col is not None:
        df_raw = df_raw[
            df_raw[acc_col].astype(str).str.upper().str.contains(uniprot_accession, na=False)
        ].copy()
        print(f"  After target-accession cross-check ({acc_col}): {len(df_raw):,} / {n2:,}")
    else:
        print(f"  NOTE: No target-accession column present in concise response — "
              f"relying on accession-based AID retrieval alone for target specificity.")

    # FIX 4 (organism safety net), if the field exists
    org_col = next((c for c in TARGET_ORGANISM_COLS if c in df_raw.columns), None)
    if org_col is not None:
        n_before_org = len(df_raw)
        org_norm = df_raw[org_col].astype(str).str.strip().str.lower()
        df_raw = df_raw[org_norm.isin(HUMAN_ORGANISM_STRINGS)].copy()
        print(f"  After organism filter ({org_col} == human): {len(df_raw):,} / {n_before_org:,}")
    else:
        print(f"  NOTE: No organism column present in concise response — "
              f"relying on accession-based AID retrieval alone for species specificity.")

    return df_raw


# ── Main processing loop ──
for target_name, uniprot_accession in TARGET_UNIPROT.items():
    print(f"\n{'='*60}")
    print(f"  {target_name} (UniProt: {uniprot_accession}, human-only)")
    print(f"{'='*60}")

    print("Fetching assay IDs (accession-restricted)...")
    aids = get_aids_for_accession(uniprot_accession)
    print(f"  Found {len(aids):,} assays for {uniprot_accession}")

    if not aids:
        print(f"  No assays found for {target_name}, skipping.")
        continue

    print("Downloading concise activity data...")
    AID_BATCH_SIZE = 25
    all_data = []

    for i in range(0, len(aids), AID_BATCH_SIZE):
        batch = aids[i:i + AID_BATCH_SIZE]
        df_batch = get_concise_data(batch)
        if not df_batch.empty:
            all_data.append(df_batch)

        if ((i // AID_BATCH_SIZE) + 1) % 20 == 0:
            total_so_far = sum(len(d) for d in all_data)
            print(f"    Processed {min(i + AID_BATCH_SIZE, len(aids)):,}/{len(aids):,} AIDs, "
                  f"{total_so_far:,} activity rows so far")

    if not all_data:
        print(f"  No activity data found for {target_name}, skipping.")
        continue

    df_raw = pd.concat(all_data, ignore_index=True)
    print(f"  Total raw activity rows: {len(df_raw):,}")

    # Quantitative activity value
    activity_col = "Activity Value [uM]"
    if activity_col not in df_raw.columns:
        possible = [c for c in df_raw.columns if "activity value" in c.lower()]
        if possible:
            activity_col = possible[0]
        else:
            print(f"  No activity value column found for {target_name}, skipping.")
            continue

    df_raw[activity_col] = pd.to_numeric(df_raw[activity_col], errors="coerce")
    df_raw = df_raw.dropna(subset=[activity_col])
    df_raw = df_raw[df_raw[activity_col] > 0]
    print(f"  Rows with valid quantitative activity values: {len(df_raw):,}")

    if df_raw.empty:
        print(f"  No valid activity data for {target_name}, skipping.")
        continue

    # FIXES 2/3/4 applied here, with an auditable per-filter breakdown
    df_raw = apply_quality_filters(df_raw, target_name, uniprot_accession)
    if df_raw.empty:
        print(f"  No rows survived quality filtering for {target_name}, skipping.")
        continue

    # Clean CID column
    df_raw["CID"] = df_raw["CID"].astype(str).str.strip()
    df_raw = df_raw[df_raw["CID"].str.isdigit()]
    print(f"  Rows with valid CIDs: {len(df_raw):,}")

    # pChEMBL conversion
    df_raw["pchembl_equiv"] = df_raw[activity_col].apply(to_pchembl)
    df_raw = df_raw.dropna(subset=["pchembl_equiv"])
    print(f"  Rows after pChEMBL conversion: {len(df_raw):,}")

    # SMILES retrieval (already canonicalized inside get_smiles_for_cids)
    unique_cids = df_raw["CID"].unique().tolist()
    print(f"  Fetching SMILES for {len(unique_cids):,} unique CIDs...")
    smiles_map = get_smiles_for_cids(unique_cids)
    print(f"  Retrieved SMILES for {len(smiles_map):,} CIDs")

    df_raw["Ligand SMILES"] = df_raw["CID"].map(smiles_map)
    df_raw = df_raw.dropna(subset=["Ligand SMILES"])
    print(f"  Rows with SMILES: {len(df_raw):,}")

    if df_raw.empty:
        print(f"  No rows with SMILES for {target_name}, skipping.")
        continue

    print("Verifying SMILES (already canonicalized at retrieval)...")
    df_raw["canon_smiles"] = df_raw["Ligand SMILES"]
    df_raw = df_raw[df_raw["canon_smiles"].apply(is_valid_smiles)].copy()
    print(f"  {len(df_raw):,} rows with valid canonical SMILES")

    if df_raw.empty:
        print(f"  No valid SMILES for {target_name}, skipping.")
        continue

    # Deduplicate on canonical SMILES, median pChEMBL across all surviving
    # (already quality-filtered) measurements for that molecule
    df_dedup = (
        df_raw.groupby("canon_smiles")["pchembl_equiv"]
        .median()
        .reset_index()
        .rename(columns={"canon_smiles": "canonical_smiles", "pchembl_equiv": "pchembl_median"})
    )
    df_dedup["label"] = df_dedup["pchembl_median"].apply(bin_compound)

    out_path = f"{target_name}_pubchem_external.csv"
    df_dedup.to_csv(out_path, index=False)

    print(f"\n  {target_name}: {len(df_dedup):,} unique compounds (human-only, filtered)")
    print(df_dedup["label"].value_counts())
    print(f"  -> saved {out_path}")

print("\n===== PubChem external set (human-only, filtered) — all targets done =====")
