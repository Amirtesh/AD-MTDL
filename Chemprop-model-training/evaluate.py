import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    recall_score, f1_score
)

DB_PATH    = "/home/amirtesh/chembl_36/chembl_36_sqlite/chembl_36.db"
PRED_PATH  = "predictions.csv"
EVAL_PATH  = "AD_eval_for_prediction.csv"

TARGETS = {
    "AChE":  "CHEMBL220",
    "BuChE": "CHEMBL1914",
    "BACE1": "CHEMBL4822",
    "MAO_B": "CHEMBL2039",
}

THRESHOLDS = {
    "AChE":  0.59,
    "BuChE": 0.68,
    "BACE1": 0.60,
    "MAO_B": 0.51,
}

ACTIVE_PCHEMBL   = 7.0
INACTIVE_PCHEMBL = 5.0


EXCLUDE_CHEMBL_IDS = {
    "CHEMBL9113",  # Toluene — pharmacologically implausible AChE activity
    "CHEMBL8706",  # Clorgyline — MAO-A selective; MAO-B data are outlier artifacts
}


pred_df = pd.read_csv(PRED_PATH)
eval_df = pd.read_csv(EVAL_PATH)

merged = pred_df.merge(
    eval_df[["canonical_smiles", "chembl_id", "pref_name"]],
    left_on="smiles", right_on="canonical_smiles", how="left"
)

ACTIVITY_QUERY = """
SELECT
    md.chembl_id,
    td.chembl_id        AS target_chembl_id,
    a.pchembl_value
FROM activities a
JOIN assays              ass ON a.assay_id = ass.assay_id
JOIN target_dictionary    td ON ass.tid    = td.tid
JOIN molecule_dictionary  md ON a.molregno = md.molregno
WHERE
    md.chembl_id              = ?
    AND td.chembl_id          = ?
    AND ass.assay_type         IN ('B', 'F')
    AND ass.confidence_score   >= 8
    AND a.standard_relation    = '='
    AND a.pchembl_value        IS NOT NULL
    AND a.potential_duplicate  = 0
    AND a.data_validity_comment IS NULL
"""

con = sqlite3.connect(DB_PATH)
chembl_ids = merged["chembl_id"].dropna().unique().tolist()

gt_rows = []
for cid in chembl_ids:
    if cid in EXCLUDE_CHEMBL_IDS:
        continue
    for target_name, target_chembl in TARGETS.items():
        rows = pd.read_sql_query(
            ACTIVITY_QUERY, con, params=(cid, target_chembl)
        )
        if rows.empty:
            continue

        median_pchembl = float(rows["pchembl_value"].median())

        if median_pchembl >= ACTIVE_PCHEMBL:
            label = 1
        elif median_pchembl < INACTIVE_PCHEMBL:
            label = 0
        else:
            label = np.nan  # Moderate — exclude, same as training

        gt_rows.append({
            "chembl_id"     : cid,
            "target_name"   : target_name,
            "median_pchembl": round(median_pchembl, 3),
            "n_measurements": len(rows),
            "gt_label"      : label,
        })

con.close()

gt_df = pd.DataFrame(gt_rows)

print("Ground truth (median pChEMBL, training-consistent filters):")
print(f"  Total records      : {len(gt_df)}")
print(f"  Moderate (excluded): {gt_df['gt_label'].isna().sum()}")

gt_df = gt_df.dropna(subset=["gt_label"]).reset_index(drop=True)
print(f"  Usable records     : {len(gt_df)}")
print()

# Show what changed vs MAX-based evaluation
print("Compounds reclassified vs MAX-based evaluation:")
print(f"  {'ChEMBL ID':<16} {'Name':<20} {'Target':<8} "
      f"{'Median':>8} {'GT Label':>10}")
print(f"  {'-'*65}")
for _, r in gt_df.iterrows():
    name = merged[merged["chembl_id"] == r["chembl_id"]]["pref_name"].iloc[0] \
           if not merged[merged["chembl_id"] == r["chembl_id"]].empty else "?"
    label_str = "Active" if r["gt_label"] == 1 else "Inactive"
    print(f"  {r['chembl_id']:<16} {str(name):<20} {r['target_name']:<8} "
          f"{r['median_pchembl']:>8.3f} {label_str:>10}")


gt_wide = gt_df.pivot_table(
    index="chembl_id",
    columns="target_name",
    values="gt_label",
    aggfunc="first"
).reset_index()

eval_pred = (
    merged[["smiles", "chembl_id", "pref_name",
            "AChE_prob", "BuChE_prob", "BACE1_prob", "MAO_B_prob"]]
    .drop_duplicates(subset="chembl_id")
    .merge(gt_wide, on="chembl_id", how="inner")
)

print(f"\nCompounds with prediction + corrected ground truth: {len(eval_pred)}")

print("\n" + "=" * 70)
print("  EXTERNAL VALIDATION — CORRECTED (median pChEMBL ground truth)")
print("=" * 70)

summary_rows = []

for target in ["AChE", "BuChE", "BACE1", "MAO_B"]:
    prob_col = f"{target}_prob"

    if target not in eval_pred.columns:
        print(f"\n{target}: no ground truth after correction")
        continue

    sub = eval_pred[[prob_col, target, "chembl_id", "pref_name"]].dropna(
        subset=[target]
    ).copy()
    sub["gt_label"] = sub[target].astype(int)

    if len(sub) < 3:
        print(f"\n{target}: only {len(sub)} compounds — insufficient for metrics")
        continue

    n_active   = (sub["gt_label"] == 1).sum()
    n_inactive = (sub["gt_label"] == 0).sum()
    thresh     = THRESHOLDS[target]
    sub["pred_label"] = (sub[prob_col] >= thresh).astype(int)

    try:
        auroc = roc_auc_score(sub["gt_label"], sub[prob_col])
    except ValueError:
        auroc = float("nan")

    mcc  = matthews_corrcoef(sub["gt_label"], sub["pred_label"])
    sens = recall_score(sub["gt_label"], sub["pred_label"],
                        pos_label=1, zero_division=0)
    spec = recall_score(sub["gt_label"], sub["pred_label"],
                        pos_label=0, zero_division=0)
    f1m  = f1_score(sub["gt_label"], sub["pred_label"],
                    average="macro", zero_division=0)

    tn = fp = fn = tp = 0
    if len(np.unique(sub["pred_label"])) == 2:
        tn, fp, fn, tp = confusion_matrix(
            sub["gt_label"], sub["pred_label"]
        ).ravel()

    print(f"\n  {target}")
    print(f"  {'─'*50}")
    print(f"  N = {len(sub)}  (Active={n_active}, Inactive={n_inactive})")
    print(f"  AUROC       : {auroc:.4f}" if not np.isnan(auroc)
          else "  AUROC       : N/A")
    print(f"  MCC         : {mcc:.4f}")
    print(f"  Sensitivity : {sens:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  F1 macro    : {f1m:.4f}")
    print(f"  Confusion   : TP={tp} FP={fp} TN={tn} FN={fn}")

    fn_df = sub[(sub["gt_label"] == 1) & (sub["pred_label"] == 0)]
    fp_df = sub[(sub["gt_label"] == 0) & (sub["pred_label"] == 1)]

    if len(fn_df):
        print(f"\n  False Negatives:")
        for _, r in fn_df.iterrows():
            print(f"    {r['chembl_id']:<16} {str(r['pref_name']):<28} "
                  f"prob={r[prob_col]:.4f}")
    if len(fp_df):
        print(f"\n  False Positives:")
        for _, r in fp_df.iterrows():
            print(f"    {r['chembl_id']:<16} {str(r['pref_name']):<28} "
                  f"prob={r[prob_col]:.4f}")

    summary_rows.append({
        "Target": target, "N": len(sub),
        "Active": n_active, "Inactive": n_inactive,
        "AUROC": round(auroc, 4) if not np.isnan(auroc) else "N/A",
        "MCC": round(mcc, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "F1_macro": round(f1m, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    })

print(f"\n{'='*70}")
print("  SUMMARY")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows).set_index("Target")
print(summary_df[[
    "N", "Active", "Inactive",
    "AUROC", "MCC", "Sensitivity", "Specificity", "F1_macro"
]].to_string())

summary_df.reset_index().to_csv(
    "external_validation_corrected_summary.csv", index=False
)
eval_pred.to_csv("external_validation_corrected_results.csv", index=False)
print("\nSaved: external_validation_corrected_summary.csv")
print("Saved: external_validation_corrected_results.csv")
