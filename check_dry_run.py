#!/usr/bin/env python
"""check_dry_run.py - save in ~/RL (next to predict.py), run from ~/RL in the ad_mtdl env:
       cd ~/RL && conda activate ad_mtdl && python check_dry_run.py dry_run AChE
Needs in ~/RL: AD_reference.npz, AD_MTDL_combined_chembl36.csv, <run_dir>/results_1.csv, <run_dir>/top_compounds.csv"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # find predict.py even if this file is run from elsewhere
import numpy as np
import pandas as pd

run_dir = sys.argv[1] if len(sys.argv) > 1 else "dry_run"
target = sys.argv[2] if len(sys.argv) > 2 else "AChE"
TARGETS = ["AChE", "BuChE", "BACE1", "MAO_B"]; ti = TARGETS.index(target)

# ---- 1. reward transform and display columns ----
r = pd.read_csv(f"{run_dir}/results_1.csv")
cols = [c for c in r.columns if any(k in c.lower() for k in ("margin", "logit", "ad_level"))]
print("matching columns:", cols)
raw = [c for c in cols if "margin" in c.lower() and "raw" in c.lower()]
trf = [c for c in cols if "margin" in c.lower() and "raw" not in c.lower()]
if raw and trf:
    exp = 1 / (1 + 10 ** (-10 * 0.25 * r[raw[0]] / 6))
    print(f"transform check: max |{trf[0]} - formula| = {float((r[trf[0]] - exp).abs().max()):.2e}  (should be ~0)")
else:
    print("could not find raw + transformed margin columns; paste the list above")

# ---- 2. what are the generated compounds extrapolating from? ----
from predict import ad_features
z = np.load("AD_reference.npz", allow_pickle=False)
raw_df = pd.read_csv("AD_MTDL_combined_chembl36.csv")
cls = dict(zip(raw_df["smiles"].astype(str), raw_df[f"{target}_class"]))
mask = z["target_mask"][:, ti]
lab = np.array([{"Active": 1.0, "Inactive": 0.0}.get(cls.get(s), np.nan) for s in z["train_smiles"][mask]])
ref = np.unpackbits(z["train_fp_packed"], axis=1)[mask].astype(np.float32); rn = ref.sum(1)
print(f"\n{target}: {len(lab)} reference compounds, {int(np.isfinite(lab).sum())} with an active/inactive label")

top = pd.read_csv(f"{run_dir}/top_compounds.csv")
scol = next(c for c in top.columns if c.lower() == "smiles")
q, _ = ad_features(top[scol].astype(str).tolist()); q = q.astype(np.float32)
inter = q @ ref.T; sim = inter / np.maximum(q.sum(1)[:, None] + rn[None, :] - inter, 1)
nn = np.argsort(-sim, axis=1)[:, :5]
top["nn1_sim"] = sim[np.arange(len(q)), nn[:, 0]]
top["nn5_frac_active"] = np.nanmean(lab[nn], axis=1)
show = [scol] + ([f"{target}_margin"] if f"{target}_margin" in top.columns else []) + ["nn1_sim", "nn5_frac_active"]
print(top[show].round(2).assign(**{scol: top[scol].str.slice(0, 46)}).to_string(index=False))
print(f"\nmedian nn1_sim = {top['nn1_sim'].median():.2f} | median nn5_frac_active = {top['nn5_frac_active'].median():.2f} | "
      f"share of compounds whose 5 nearest labelled neighbours are mostly inactive = {(top['nn5_frac_active'] < 0.5).mean():.0%}")
