#!/usr/bin/env python
"""
model_extras.py - remaining model-level analyses for the FROZEN AD-MTDL classifier (no retraining, no baselines):
  1. Calibration: Brier / ECE before and after Platt scaling (fit on VALIDATION logits only), on test and external primary sets.
  2. Ordinal check on the held-out intermediate-potency pool: does the score order Active > Moderate > Inactive?
  3. Per-assay breakdown of external errors, if the scored CSVs carry an assay-id column.

python model_extras.py --eval-dir <corrected_eval_dir> --scored-dir validation_results/scored \
       --raw-csv <AD_MTDL_combined_chembl36.csv> --out extras_results
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

TARGETS = ["AChE", "BuChE", "BACE1", "MAO_B"]


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); rows = []
    for b in range(bins):
        sel = (p >= edges[b]) & ((p < edges[b + 1]) if b < bins - 1 else (p <= edges[b + 1]))
        if sel.any():
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(sel.sum()),
                         "mean_pred": float(p[sel].mean()), "frac_pos": float(y[sel].mean())})
    return rows


def ece(y, p, bins=10):
    return float(sum(r["n"] / len(y) * abs(r["mean_pred"] - r["frac_pos"]) for r in reliability(y, p, bins)))


def fit_platt(s, y):
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(s.reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    assert a > 0, "negative Platt slope: scores anti-correlated with labels on validation"
    return a, b


def auc(pos, neg, min_n=10):
    if len(pos) < min_n or len(neg) < min_n:
        return np.nan
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True); ap.add_argument("--scored-dir", default="validation_results/scored")
    ap.add_argument("--raw-csv", default=None, help="original class CSV incl. 'Moderate' labels; enables analysis 2")
    ap.add_argument("--thresholds", default=None); ap.add_argument("--ckpt", default="AD_MTDL_best.ckpt")
    ap.add_argument("--ad-reference", default="AD_reference.npz"); ap.add_argument("--max-sim", type=float, default=0.85)
    ap.add_argument("--out", default="extras_results")
    args = ap.parse_args()

    ev, out = Path(args.eval_dir), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    VL, VT = np.load(ev / "val_logits.npy"), np.load(ev / "val_targets.npy")
    TL, TT = np.load(ev / "test_logits.npy"), np.load(ev / "test_targets.npy")
    thr_path = Path(args.thresholds) if args.thresholds else ev / "thresholds.json"
    thr = json.load(open(thr_path))["thresholds_logit"]
    pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.3f}".format)

    # ---------------- 1. calibration ----------------
    cal_rows, rel_rows, platt, ext_frames = [], [], {}, {}
    for ti, t in enumerate(TARGETS):
        thr_t = float(thr[f"{t}_class"])
        mv, mt = ~np.isnan(VT[:, ti]), ~np.isnan(TT[:, ti])
        sv, yv = VL[mv, ti], VT[mv, ti].astype(int)
        st, yt = TL[mt, ti], TT[mt, ti].astype(int)
        sets = [("validation (Platt fit)", sv, yv), ("test", st, yt)]
        ep = Path(args.scored_dir) / f"{t}_scored.csv"
        if ep.exists():
            e = pd.read_csv(ep); ext_frames[t] = e
            e = e[e["nearest_train_tanimoto"] < args.max_sim]
            sets.append(("external (primary)", e["raw_logit"].to_numpy(float), e["label_numeric"].to_numpy(int)))
        a, b = fit_platt(sv, yv)
        platt[t] = {"a": a, "b": b, "fit_set": "validation", "val_prevalence": float(yv.mean()),
                    "frozen_threshold_logit": thr_t, "frozen_threshold_calibrated_prob": float(expit(a * thr_t + b))}
        for name, s, y in sets:
            p_raw, p_cal = expit(s), expit(a * s + b)
            cal_rows.append({"target": t, "set": name, "n": len(y), "prevalence": float(y.mean()), "platt_slope_a": a,
                             "Brier_raw": brier_score_loss(y, p_raw), "Brier_platt": brier_score_loss(y, p_cal),
                             "ECE_raw": ece(y, p_raw), "ECE_platt": ece(y, p_cal)})
            for variant, p in (("raw", p_raw), ("platt", p_cal)):
                rel_rows += [{"target": t, "set": name, "variant": variant, **r} for r in reliability(y, p)]
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(out / "calibration_summary.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "reliability_bins.csv", index=False)
    json.dump({"note": "Platt scaling fit on validation logits; AUROC and the ranking are unchanged (a>0). Calibrated "
                       "probabilities reflect validation prevalence and are not guaranteed on sets with different prevalence.",
               "targets": platt}, open(out / "platt.json", "w"), indent=2)
    print("\n===== 1. CALIBRATION (raw = single sigmoid of the logit) =====\n" + cal.to_string(index=False))

    # ---------------- 2. ordinal check on the intermediate pool ----------------
    if args.raw_csv:
        from predict import ADMTDLPredictor
        raw = pd.read_csv(args.raw_csv)
        ds = pd.read_csv(ev / "dataset_after_cleaning.csv"); split = json.load(open(ev / "split_indices.json"))
        train_smiles = set(ds["smiles"].iloc[split["train"]].astype(str))
        predictor = ADMTDLPredictor(args.ckpt, thr_path, args.ad_reference)
        rows = []
        for ti, t in enumerate(TARGETS):
            thr_t = float(thr[f"{t}_class"])
            mod = (raw.loc[(raw[f"{t}_class"] == "Moderate") & (~raw["smiles"].astype(str).isin(train_smiles)), "smiles"]
                   .astype(str).drop_duplicates().tolist())
            pr = predictor.predict(mod)
            sm = pr.loc[pr["valid"], f"{t}_logit"].to_numpy(float)
            s_all = np.concatenate([VL[:, ti], TL[:, ti]]); y_all = np.concatenate([VT[:, ti], TT[:, ti]]); ok = ~np.isnan(y_all)
            sa, si = s_all[ok & (y_all == 1)], s_all[ok & (y_all == 0)]
            med = {k: (float(np.median(v)) if len(v) else np.nan) for k, v in (("active", sa), ("moderate", sm), ("inactive", si))}
            rows.append({"target": t, "n_active_heldout": len(sa), "n_moderate_unseen": len(sm), "n_inactive_heldout": len(si),
                         "median_logit_active": med["active"], "median_logit_moderate": med["moderate"],
                         "median_logit_inactive": med["inactive"],
                         "AUROC_active_vs_moderate": auc(sa, sm), "AUROC_moderate_vs_inactive": auc(sm, si),
                         "frac_moderate_predicted_active": float((sm >= thr_t).mean()) if len(sm) else np.nan,
                         "ordered_A>M>I": bool(med["active"] > med["moderate"] > med["inactive"])})
        ordd = pd.DataFrame(rows); ordd.to_csv(out / "intermediate_pool_ordinal.csv", index=False)
        print("\n===== 2. INTERMEDIATE-POTENCY POOL (5 <= pChEMBL < 7), never seen in training =====\n" + ordd.to_string(index=False))
    else:
        print("\n(2 skipped: pass --raw-csv to run the intermediate-pool check)")

    # ---------------- 3. per-assay breakdown ----------------
    found, rows = False, []
    for t, e in ext_frames.items():
        e = e[e["nearest_train_tanimoto"] < args.max_sim]
        cols = [c for c in e.columns if any(k in c.lower() for k in ("aid", "assay"))]
        if not cols:
            continue
        found = True; col = cols[0]; thr_t = float(thr[f"{t}_class"])
        for key, g in e.groupby(col):
            y, s = g["label_numeric"].to_numpy(int), g["raw_logit"].to_numpy(float); pred = s >= thr_t
            rows.append({"target": t, "assay_col": col, "assay": key, "n": len(g), "n_potent": int(y.sum()),
                         "n_inactive": int((y == 0).sum()), "FP": int((pred & (y == 0)).sum()),
                         "FP_rate": float((pred & (y == 0)).sum() / max((y == 0).sum(), 1)),
                         "sens": float((pred & (y == 1)).sum() / max((y == 1).sum(), 1)),
                         "AUROC": auc(s[y == 1], s[y == 0])})
    if found:
        aid = pd.DataFrame(rows).sort_values(["target", "n"], ascending=[True, False])
        aid.to_csv(out / "per_assay_errors.csv", index=False)
        print("\n===== 3. PER-ASSAY ERRORS (top assays per target) =====\n" + aid.groupby("target").head(10).to_string(index=False))
    else:
        print("\n(3 skipped: no assay-id column in the scored CSVs; carry the PubChem AID through get_pubchem.py)")
    print(f"\noutputs in {out.resolve()}")


if __name__ == "__main__":
    main()
