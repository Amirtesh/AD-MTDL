#!/usr/bin/env python
"""
predict.py - AD-MTDL multi-task D-MPNN inference (corrected) with an always-on applicability domain.

Scoring = RAW LOGITS from predictor.train_step (no sigmoid anywhere in decisions); prob = expit(logit) in float64.
Decision = logit >= thresholds.json["thresholds_logit"][<target>_class].
Per-target AD level (low/medium/high) is always computed; the AD reference file is mandatory.

CLI:  python predict.py --input in.csv --smiles-col smiles --output out.csv
API:  from predict import ADMTDLPredictor; df = ADMTDLPredictor().predict(list_of_smiles)
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.serialization
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.special import expit

import chemprop
from chemprop import data, featurizers, models
from chemprop import nn as cpnn
from chemprop.nn import BCELoss as ChempropBCELoss

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
TARGETS = ["AChE", "BuChE", "BACE1", "MAO_B"]
LEVELS = np.array(["low", "medium", "high"])
DESC_NAMES = ["MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "HeavyAtoms"]
FP_RADIUS, FP_BITS, KNN_K = 2, 2048, 5
FEATURIZER = featurizers.SimpleMoleculeMolGraphFeaturizer()


# ----------------------------- checkpoint loading -----------------------------
class WeightedBCELoss(ChempropBCELoss):
    """v1 training loss; defined only so the checkpoint can be unpickled (never used for inference)."""
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("pos_weight", pos_weight)

    def _calc_unreduced_loss(self, preds, targets, *args, **kwargs):
        weight = self.pos_weight.to(preds.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none", pos_weight=weight)

# the checkpoint references __main__.WeightedBCELoss (it was trained in a notebook)
WeightedBCELoss.__module__ = "__main__"
setattr(sys.modules["__main__"], "WeightedBCELoss", WeightedBCELoss)


def pick_device(arg=None):
    return torch.device(arg) if arg else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt, device):
    torch.serialization.add_safe_globals([WeightedBCELoss, cpnn.metrics.BinaryAUPRC, cpnn.metrics.BinaryAUROC])
    return models.MPNN.load_from_checkpoint(str(ckpt), map_location=device, strict=False).to(device).eval()


@torch.no_grad()
def model_outputs(model, smiles, device, batch_size=256):
    """Returns raw logits (n,4, float64) and latent embeddings (n,300, float32). Self-checks sigmoid(logit)==forward()."""
    dps = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
    loader = data.build_dataloader(data.MoleculeDataset(dps, FEATURIZER), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    L, Z = [], []
    for batch in loader:
        bmg, V_d, X_d, *_ = batch
        bmg.V = bmg.V.to(device); bmg.E = bmg.E.to(device)
        bmg.edge_index = bmg.edge_index.to(device); bmg.batch = bmg.batch.to(device)
        V_d = V_d.to(device) if V_d is not None else None
        X_d = X_d.to(device) if X_d is not None else None
        z = model.fingerprint(bmg, V_d, X_d)
        logits = model.predictor.train_step(z)            # raw logits (what the loss consumed)
        err = (torch.sigmoid(logits.double()) - model.predictor(z).double()).abs().max().item()
        if err > 1e-4:
            raise RuntimeError(f"train_step logits inconsistent with forward() probabilities (max err {err:.2e}); head semantics changed.")
        L.append(logits.double().cpu().numpy()); Z.append(z.float().cpu().numpy())
    return np.vstack(L), np.vstack(Z)


# ----------------------------- AD features -----------------------------
_LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def parent_mol(mol):
    try:
        return _LFC.choose(mol)       # structural AD ignores counter-ions/salts
    except Exception:
        return mol


def physchem(m):
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
            m.GetNumHeavyAtoms()]


def ad_features(smiles):
    """Morgan bits (uint8) and physchem descriptors of the parent fragment. All SMILES must be valid."""
    fps = np.zeros((len(smiles), FP_BITS), np.uint8); desc = np.zeros((len(smiles), len(DESC_NAMES)), np.float64)
    for i, s in enumerate(smiles):
        m = parent_mol(Chem.MolFromSmiles(s))
        fps[i] = np.asarray(_MORGAN.GetFingerprintAsNumPy(m), dtype=np.uint8)
        desc[i] = physchem(m)
    return fps, desc


def knn_mean_similarity(Q, R, k, metric, q_groups=None, r_groups=None, chunk=1024):
    """Mean similarity of each query to its k nearest reference rows. metric: 'tanimoto' (bits) or 'cosine' (embeddings).
    If groups are given, reference rows sharing a group id with the query are excluded (scaffold-excluded calibration)."""
    Rf = R.astype(np.float32)
    if metric == "tanimoto":
        rn = Rf.sum(1)
    else:
        Rf = Rf / np.maximum(np.linalg.norm(Rf, axis=1, keepdims=True), 1e-12)
    out = np.empty(len(Q), np.float32)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk].astype(np.float32)
        if metric == "tanimoto":
            inter = q @ Rf.T
            sim = inter / np.maximum(q.sum(1)[:, None] + rn[None, :] - inter, 1.0)
        else:
            sim = (q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)) @ Rf.T
        if q_groups is not None:
            sim = np.where(q_groups[s:s + chunk, None] == r_groups[None, :], -np.inf, sim)
        kk = min(k, sim.shape[1])
        top = -np.partition(-sim, kk - 1, axis=1)[:, :kk]
        valid = np.isfinite(top)
        cnt = valid.sum(1)
        out[s:s + chunk] = np.where(cnt > 0, np.where(valid, top, 0).sum(1) / np.maximum(cnt, 1), np.nan)
    return out


class ADReference:
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AD reference not found: {path}. Run build_ad.py first (AD is always computed).")
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta_json"]))
        self.fp = np.unpackbits(z["train_fp_packed"], axis=1)
        assert self.fp.shape[1] == self.meta["fp_bits"]
        self.emb = z["train_embed"].astype(np.float32)
        self.mask = z["target_mask"].astype(bool)
        self.struct_cuts, self.embed_cuts = z["struct_cuts"], z["embed_cuts"]
        self.desc_lo, self.desc_hi = z["desc_lo"], z["desc_hi"]
        self.k = int(self.meta["k"])


def tier(score, cuts):
    s = np.nan_to_num(score.astype(np.float64), nan=-np.inf)
    return np.where(s >= cuts[0], 2, np.where(s >= cuts[1], 1, 0))


def compute_ad(ad, fps, emb, desc):
    """Per-target AD. level 2=high, 1=medium, 0=low. Returns ({target: dict}, n_out_of_range)."""
    viol = ((desc < ad.desc_lo) | (desc > ad.desc_hi)).sum(1)
    res = {}
    for ti, t in enumerate(TARGETS):
        m = ad.mask[:, ti]
        s = knn_mean_similarity(fps, ad.fp[m], ad.k, "tanimoto")
        e = knn_mean_similarity(emb, ad.emb[m], ad.k, "cosine")
        lvl = np.minimum(tier(s, ad.struct_cuts[ti]), tier(e, ad.embed_cuts[ti]))
        lvl = np.where(viol > 0, np.minimum(lvl, 1), lvl)
        res[t] = {"struct_sim": s, "embed_sim": e, "level": lvl}
    return res, viol


# ----------------------------- predictor -----------------------------
class ADMTDLPredictor:
    def __init__(self, ckpt=HERE / "AD_MTDL_v1_best.ckpt", thresholds=HERE / "thresholds.json",
                 ad_reference=HERE / "AD_reference.npz", device=None):
        self.device = pick_device(device)
        thr = json.load(open(thresholds))["thresholds_logit"]
        self.thr = {t: float(thr[f"{t}_class"]) for t in TARGETS}
        self.ad = ADReference(ad_reference)                      # mandatory: AD is always computed
        if sha256_file(ckpt) != self.ad.meta["checkpoint_sha256"]:
            raise RuntimeError("AD reference was built for a different checkpoint; rebuild it with build_ad.py.")
        self.model = load_model(ckpt, self.device)

    def predict(self, smiles, batch_size=256):
        smiles = [str(s) for s in smiles]
        n = len(smiles)
        valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles])
        out = {"smiles": smiles, "valid": valid}
        for t in TARGETS:
            for c in ("logit", "prob", "pred", "ad_struct_sim", "ad_embed_sim"):
                out[f"{t}_{c}"] = np.full(n, np.nan)
            out[f"{t}_ad_level"] = np.full(n, "invalid", dtype=object)
        out["ad_physchem_n_out_of_range"] = np.full(n, np.nan)
        if valid.any():
            idx = np.where(valid)[0]
            vs = [smiles[i] for i in idx]
            L, Z = model_outputs(self.model, vs, self.device, batch_size)
            fps, desc = ad_features(vs)
            ad, viol = compute_ad(self.ad, fps, Z, desc)
            out["ad_physchem_n_out_of_range"][idx] = viol
            for ti, t in enumerate(TARGETS):
                out[f"{t}_logit"][idx] = L[:, ti]
                out[f"{t}_prob"][idx] = expit(L[:, ti])
                out[f"{t}_pred"][idx] = (L[:, ti] >= self.thr[t]).astype(int)
                out[f"{t}_ad_struct_sim"][idx] = ad[t]["struct_sim"]
                out[f"{t}_ad_embed_sim"][idx] = ad[t]["embed_sim"]
                out[f"{t}_ad_level"][idx] = LEVELS[ad[t]["level"]]
        df = pd.DataFrame(out)
        for t in TARGETS:
            df[f"{t}_pred"] = df[f"{t}_pred"].astype("Int64")
        return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--smiles-col", default="smiles"); ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt", default=str(HERE / "AD_MTDL_v1_best.ckpt")); ap.add_argument("--thresholds", default=str(HERE / "thresholds.json"))
    ap.add_argument("--ad-reference", default=str(HERE / "AD_reference.npz"))
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    pred = ADMTDLPredictor(a.ckpt, a.thresholds, a.ad_reference, a.device).predict(df[a.smiles_col], a.batch_size)
    out = pd.concat([df.reset_index(drop=True), pred.drop(columns=["smiles"])], axis=1)
    out.to_csv(a.output, index=False)
    print(f"wrote {a.output} | chemprop {chemprop.__version__} | invalid SMILES: {int((~pred['valid']).sum())}")
    for t in TARGETS:
        print(f"  {t:<6} AD levels:", pred[f"{t}_ad_level"].value_counts().to_dict())


if __name__ == "__main__":
    main()
