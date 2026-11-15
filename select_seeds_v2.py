#!/usr/bin/env python
"""select_seeds_v2.py - programmatic, oracle-aware, diverse seed selection for the v2 campaigns (no manual choices).

A seed must (1) come from the moderate-potency pool (pChEMBL 5-7) of the campaign target(s), not Active for any other target,
(2) be scored by the frozen oracle BELOW its decision threshold (margin <= -0.5) and inside the HIGH applicability domain, so a
campaign has to cross the boundary; if too few qualify, the rule relaxes once (margin <= 0, AD high/medium) and says so.
Seeds are then picked by MaxMin Tanimoto diversity, starting from the candidate closest to the boundary.

Run in the ad_mtdl env from ~/RL:   python select_seeds_v2.py --n 3 --out campaign_seeds_v2.csv"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from predict import ADMTDLPredictor, TARGETS, parent_mol

RDLogger.DisableLog("rdApp.*")
CAMPAIGNS = {"AChE": ["AChE"], "BuChE": ["BuChE"], "BACE1": ["BACE1"], "MAO_B": ["MAO_B"], "AChE_BACE1": ["AChE", "BACE1"]}
RULES = [(-0.5, {"high"}), (0.0, {"high", "medium"})]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ap = argparse.ArgumentParser()
ap.add_argument("--raw-csv", default="AD_MTDL_combined_chembl36.csv")
ap.add_argument("--ckpt", default="AD_MTDL_best.ckpt"); ap.add_argument("--thresholds", default="thresholds.json")
ap.add_argument("--platt", default="platt.json"); ap.add_argument("--ad-reference", default="AD_reference.npz")
ap.add_argument("--campaigns", nargs="+", default=list(CAMPAIGNS), choices=list(CAMPAIGNS))
ap.add_argument("--n", type=int, default=3); ap.add_argument("--out", default="campaign_seeds_v2.csv")
ap.add_argument("--allow-pains", action="store_true", help="do not exclude PAINS matches")
args = ap.parse_args()

raw = pd.read_csv(args.raw_csv)
platt = json.load(open(args.platt))["targets"]
predictor = ADMTDLPredictor(args.ckpt, args.thresholds, args.ad_reference)

# candidate pools -> parent SMILES (salt-free, what REINVENT's Mol2Mol should see), size window of the MW score component
pools = {}
for name in args.campaigns:
    tg = CAMPAIGNS[name]
    m = np.logical_and.reduce([raw[f"{t}_class"] == "Moderate" for t in tg])
    for o in (o for o in TARGETS if o not in tg):
        m &= raw[f"{o}_class"] != "Active"
    pools[name] = raw.loc[m, "smiles"].astype(str)
import ast
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
_tree = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize.py")).read())
ALERTS = [Chem.MolFromSmarts(a) for a in next(ast.literal_eval(n.value) for n in _tree.body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CUSTOM_ALERTS")]
_pp = FilterCatalogParams(); _pp.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS); PAINS = FilterCatalog(_pp)
n_alert = n_pains = 0
def parent_smiles(s):
    global n_alert, n_pains
    mol = Chem.MolFromSmiles(s)
    if mol is None: return None
    p = parent_mol(mol)
    if not 200 <= Descriptors.MolWt(p) <= 550: return None
    if any(p.HasSubstructMatch(a) for a in ALERTS): n_alert += 1; return None     # would score 0 from step 1
    if not args.allow_pains and PAINS.HasMatch(p): n_pains += 1; return None
    return Chem.MolToSmiles(p)
cache = {s: parent_smiles(s) for s in set().union(*[set(p) for p in pools.values()])}
print(f"excluded as seeds: {n_alert} match the scoring function's structural alerts, {n_pains} match PAINS")
universe = sorted({v for v in cache.values() if v})
print(f"scoring {len(universe)} unique parent molecules with the frozen oracle ...")
scored = predictor.predict(universe).set_index("smiles")
for t in TARGETS:
    a = float(platt[t]["a"]); scored[f"{t}_margin"] = a * (scored[f"{t}_logit"] - predictor.thr[t])
    scored[f"{t}_ad"] = scored[f"{t}_ad_level"]

rows = []
for name in args.campaigns:
    tg = CAMPAIGNS[name]
    cand_all = pd.DataFrame({"smiles": sorted({cache[s] for s in pools[name] if cache[s]})}).set_index("smiles")
    cand_all = cand_all.join(scored[[f"{t}_{k}" for t in tg for k in ("margin", "ad")]])
    cand_all = cand_all[scored.loc[cand_all.index, "valid"].astype(bool)]
    chosen = None
    for r, (mmax, allowed) in enumerate(RULES):
        ok = np.logical_and.reduce([(cand_all[f"{t}_margin"] <= mmax) & cand_all[f"{t}_ad"].isin(allowed) for t in tg])
        cand = cand_all[ok]
        print(f"{name}: pool {len(cand_all)} | rule {r} (margin <= {mmax}, AD in {sorted(allowed)}): {len(cand)} candidates")
        if len(cand) >= args.n or r == len(RULES) - 1:
            chosen, rule = cand, r; break
    if len(chosen) == 0:
        print(f"  !! no seeds for {name}"); continue
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in chosen.index]
    closeness = chosen[[f"{t}_margin" for t in tg]].max(axis=1).to_numpy()          # nearest to the boundary from below
    picked = [int(np.argmax(closeness))]
    while len(picked) < min(args.n, len(chosen)):
        dmin = [min(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in picked) for i in range(len(chosen))]
        for j in picked: dmin[j] = -1
        picked.append(int(np.argmax(dmin)))
    for k, i in enumerate(picked, 1):
        row = {"campaign": name, "targets": " ".join(tg), "seed_idx": k, "smiles": chosen.index[i], "rule": rule}
        for t in tg: row[f"{t}_margin"] = round(float(chosen.iloc[i][f"{t}_margin"]), 3); row[f"{t}_ad"] = chosen.iloc[i][f"{t}_ad"]
        if len(picked) > 1:
            row["min_dist_to_other_seeds"] = round(min(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in picked if j != i), 3)
        rows.append(row)
out = pd.DataFrame(rows); out.to_csv(args.out, index=False)
print(out.to_string(index=False)); print(f"\nwrote {args.out}")
if (out["rule"] > 0).any():
    print("NOTE: some campaigns needed the relaxed rule; say so in Methods.")
