# scoring_server.py
# Runs in: ad_mtdl env (Python 3.11, ChemProp 2.2.3, Flask)
# Start: python scoring_server.py --ckpt AD_MTDL_best.ckpt --meta AD_MTDL_metadata.json
# Stays alive for entire REINVENT4 run

import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
from flask import Flask, jsonify, request
from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

from chemprop import data, featurizers, models
from chemprop import nn as cpnn
from chemprop.nn import BCELoss as ChempropBCELoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── WeightedBCELoss — required for checkpoint loading ─────
class WeightedBCELoss(ChempropBCELoss):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)

    def _calc_unreduced_loss(self, preds, targets, *args, **kwargs):
        return F.binary_cross_entropy_with_logits(
            preds, targets, reduction="none",
            pos_weight=self.pos_weight.to(preds.device),
        )


# ── Global state ──────────────────────────────────────────
_MODEL      = None
_FEATURIZER = None
_TARGETS    = None
_THRESHOLDS = None
_DEVICE     = torch.device("cpu")
_BATCH_SIZE = 128


def load_model(ckpt_path: str, meta_path: str, n_threads: int):
    global _MODEL, _FEATURIZER, _TARGETS, _THRESHOLDS

    torch.set_num_threads(n_threads)
    log.info(f"PyTorch CPU threads: {n_threads}")

    for label, path in [("checkpoint", ckpt_path), ("metadata", meta_path)]:
        if not os.path.exists(path):
            log.error(f"{label} not found: {path}")
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    _TARGETS    = meta["targets"]
    _THRESHOLDS = meta["optimal_thresholds"]

    torch.serialization.add_safe_globals([
        WeightedBCELoss,
        cpnn.metrics.BinaryAUPRC,
        cpnn.metrics.BinaryAUROC,
    ])

    _MODEL = models.MPNN.load_from_checkpoint(
        ckpt_path, map_location=_DEVICE, strict=False
    )
    _MODEL.eval()

    _FEATURIZER = featurizers.SimpleMoleculeMolGraphFeaturizer()

    log.info(f"Model loaded: {ckpt_path}")
    log.info(f"Targets    : {_TARGETS}")
    log.info(f"Thresholds : {_THRESHOLDS}")


def _run_inference(smiles_list: list[str]) -> dict:
    """
    Returns dict: target_col -> [prob, ...] for each input SMILES.
    Invalid SMILES get prob=0.5 (model uncertainty default).
    """
    valid_idx, valid_smi = [], []
    for i, smi in enumerate(smiles_list):
        if Chem.MolFromSmiles(str(smi)) is not None:
            valid_idx.append(i)
            valid_smi.append(smi)

    n = len(smiles_list)
    result = {t: [0.5] * n for t in _TARGETS}
    result["valid_mask"] = [False] * n

    if not valid_smi:
        return result

    dummy_y    = np.zeros((len(valid_smi), len(_TARGETS)), dtype=np.float32)
    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(valid_smi, dummy_y)
    ]
    dataset = data.MoleculeDataset(datapoints, _FEATURIZER)
    loader  = data.build_dataloader(
        dataset,
        shuffle=False,
        batch_size=min(_BATCH_SIZE, len(dataset)),
        num_workers=0,
        drop_last=False,
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            bmg, X_vd, features, targets, weights, lt_mask, gt_mask = batch
            bmg.V          = bmg.V.to(_DEVICE)
            bmg.E          = bmg.E.to(_DEVICE)
            bmg.edge_index = bmg.edge_index.to(_DEVICE)
            bmg.batch      = bmg.batch.to(_DEVICE)
            logits = _MODEL(bmg, X_vd, features)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    for t_idx, target in enumerate(_TARGETS):
        for rank, orig_idx in enumerate(valid_idx):
            result[target][orig_idx] = float(all_probs[rank, t_idx])
            result["valid_mask"][orig_idx] = True

    return result


# ── Flask app ─────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"    : "ok",
        "targets"   : _TARGETS,
        "thresholds": _THRESHOLDS,
    })


@app.route("/score", methods=["POST"])
def score():
    """
    POST {"smiles": ["CCO", ...]}
    Returns per-target probabilities for all input SMILES.
    """
    payload = request.get_json(force=True)
    smiles  = payload.get("smiles", [])
    if not smiles:
        return jsonify({"error": "no smiles provided"}), 400

    preds = _run_inference(smiles)
    return jsonify(preds)



# ── Entry point ───────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="AD MTDL scoring server")
    p.add_argument("--ckpt",    default="AD_MTDL_best.ckpt")
    p.add_argument("--meta",    default="AD_MTDL_metadata.json")
    p.add_argument("--port",    type=int, default=8765)
    p.add_argument("--threads", type=int, default=64,
                   help="PyTorch CPU inference threads")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.ckpt, args.meta, args.threads)
    log.info(f"Listening on port {args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True)

