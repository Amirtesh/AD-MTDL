#!/usr/bin/env python
"""
build_ad.py - build the AD-MTDL applicability-domain reference (AD_reference.npz).

Reference = TRAINING split only (never val/test). Per target, the reference set is the training
compounds that carry a label for that target. Cutoffs are percentiles of scaffold-excluded kNN
similarity among the training reference (each compound compared only to compounds of OTHER
Bemis-Murcko scaffolds), i.e. the similarity a genuinely new scaffold would see.

Run:  python build_ad.py --eval-dir <corrected_eval_dir> --out AD_reference.npz
"""
import argparse, datetime, json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

import chemprop
from predict import (DESC_NAMES, FP_BITS, FP_RADIUS, KNN_K, LEVELS, TARGETS, ADReference, ad_features,
                     compute_ad, knn_mean_similarity, load_model, model_outputs, pick_device, sha256_file)

ap = argparse.ArgumentParser()
ap.add_argument("--eval-dir", default="/kaggle/working/AD_MTDL_v1ckpt_corrected_eval")
ap.add_argument("--ckpt", default=None, help="default: <eval-dir>/AD_MTDL_v1_best.ckpt")
ap.add_argument("--out", default="AD_reference.npz")
ap.add_argument("--k", type=int, default=KNN_K)
ap.add_argument("--high-pct", type=float, default=25.0, help="HIGH if score >= this percentile of the scaffold-excluded training distribution")
ap.add_argument("--med-pct", type=float, default=5.0, help="MEDIUM if score >= this percentile; else LOW")
ap.add_argument("--device", default=None)
args = ap.parse_args()
assert args.high_pct > args.med_pct

eval_dir = Path(args.eval_dir)
ckpt = Path(args.ckpt) if args.ckpt else eval_dir / "AD_MTDL_v1_best.ckpt"
device = pick_device(args.device)

df = pd.read_csv(eval_dir / "dataset_after_cleaning.csv")
split = json.load(open(eval_dir / "split_indices.json"))
thr = json.load(open(eval_dir / "thresholds.json"))["thresholds_logit"]
smiles = df["smiles"].astype(str).tolist()
Y = df[[f"{t}_class" for t in TARGETS]].to_numpy(float)
tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))
assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0, "train/val/test overlap"
print(f"compounds={len(df):,} train={len(tr):,} val={len(va):,} test={len(te):,} | device={device} | chemprop {chemprop.__version__}")

model = load_model(ckpt, device)
L, Z = model_outputs(model, smiles, device)          # logits (n,4), latent embeddings (n,300)
fps, desc = ad_features(smiles)
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False) for s in smiles]
gid = pd.factorize(pd.Series(scaf))[0]

# ---- calibrate cutoffs on the training reference (scaffold-excluded kNN) ----
struct_cuts = np.zeros((len(TARGETS), 2)); embed_cuts = np.zeros((len(TARGETS), 2)); n_ref = {}
for ti, t in enumerate(TARGETS):
    ref = tr[~np.isnan(Y[tr, ti])]
    n_ref[t] = int(len(ref))
    s = knn_mean_similarity(fps[ref], fps[ref], args.k, "tanimoto", gid[ref], gid[ref])
    e = knn_mean_similarity(Z[ref], Z[ref], args.k, "cosine", gid[ref], gid[ref])
    struct_cuts[ti] = [np.nanpercentile(s, args.high_pct), np.nanpercentile(s, args.med_pct)]
    embed_cuts[ti] = [np.nanpercentile(e, args.high_pct), np.nanpercentile(e, args.med_pct)]
    print(f"  {t:<6} n_ref={len(ref):>5} | struct cuts high>={struct_cuts[ti,0]:.3f} med>={struct_cuts[ti,1]:.3f}"
          f" | embed cuts high>={embed_cuts[ti,0]:.3f} med>={embed_cuts[ti,1]:.3f}")

desc_lo = np.percentile(desc[tr], 0.5, axis=0); desc_hi = np.percentile(desc[tr], 99.5, axis=0)
meta = {
    "created": datetime.datetime.now().isoformat(timespec="seconds"), "chemprop_version": chemprop.__version__,
    "checkpoint_sha256": sha256_file(ckpt), "targets": TARGETS, "k": args.k, "fp_radius": FP_RADIUS, "fp_bits": FP_BITS,
    "desc_names": DESC_NAMES, "high_pct": args.high_pct, "med_pct": args.med_pct, "n_reference_per_target": n_ref,
    "rule": "level = min(structural_tier, embedding_tier); capped at 'medium' if any physchem descriptor is outside the "
            "training 0.5-99.5 percentile envelope. Cutoffs = percentiles of scaffold-excluded kNN similarity on training compounds.",
    "levels": ["low", "medium", "high"],
}
out = Path(args.out)
np.savez_compressed(
    out, train_fp_packed=np.packbits(fps[tr], axis=1), train_embed=Z[tr].astype(np.float32),
    target_mask=~np.isnan(Y[tr]), train_index=tr, train_smiles=np.array([smiles[i] for i in tr], dtype=str),
    struct_cuts=struct_cuts, embed_cuts=embed_cuts, desc_lo=desc_lo, desc_hi=desc_hi,
    meta_json=np.array(json.dumps(meta)),
)
print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")

# ---- evidence that the AD is meaningful: performance must degrade high -> medium -> low ----
ref_ad = ADReference(out)   # also tests the loader round trip
rows = []
for split_name, idx in [("val", va), ("test", te)]:
    ad, viol = compute_ad(ref_ad, fps[idx], Z[idx], desc[idx])
    for ti, t in enumerate(TARGETS):
        m = ~np.isnan(Y[idx, ti]); y = Y[idx, ti][m].astype(int); s = L[idx, ti][m]; lvl = ad[t]["level"][m]
        pred = (s >= thr[f"{t}_class"]).astype(int)
        for band, sel in [("high", lvl == 2), ("medium", lvl == 1), ("low", lvl == 0), ("all", np.ones(len(lvl), bool))]:
            n = int(sel.sum()); ok = n >= 20 and len(set(y[sel])) == 2
            rows.append({"split": split_name, "target": t, "band": band, "n": n, "n_active": int(y[sel].sum()) if n else 0,
                         "AUROC": roc_auc_score(y[sel], s[sel]) if ok else np.nan,
                         "accuracy": accuracy_score(y[sel], pred[sel]) if n else np.nan,
                         "MCC": matthews_corrcoef(y[sel], pred[sel]) if ok else np.nan})
val_tab = pd.DataFrame(rows)
val_tab.to_csv(out.with_name("AD_validation_by_band.csv"), index=False)
pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.3f}".format)
print("\n===== PERFORMANCE BY AD BAND (val and test; AUROC/MCC shown only if n>=20 and both classes present) =====")
print(val_tab.to_string(index=False))
print("\nIf performance is not monotone high >= medium >= low, report that honestly. If you change --high-pct/--med-pct, "
      "choose them on VAL only; test is for reporting.")
