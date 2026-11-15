#!/usr/bin/env python
"""
scoring_server.py (v2) - HTTP bridge between the ChemProp v2 AD-MTDL classifier and REINVENT4.

Runs in the ad_mtdl env (ChemProp 2.2.x). Uses the SAME inference + applicability-domain code as validation
(predict.py), so the bridge cannot drift from what was evaluated:
  - scores are raw logits (predictor.train_step); there is NO sigmoid anywhere except in the calibrated probability field
  - no silent fallback scores: invalid molecules are flagged, failed requests return HTTP 500
  - every request is counted (/stats); optional per-molecule log for post-hoc analysis

Start:  python scoring_server.py --port 8765 --threads 8 [--log-molecules run1_scored.csv]
"""
import argparse, atexit, json, logging, threading, time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from scipy.special import expit

from predict import ADMTDLPredictor, TARGETS, sha256_file

HERE = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("scoring_server")
LEVEL_CODE = {"low": 0, "medium": 1, "high": 2}

app = Flask(__name__)
_P = None            # ADMTDLPredictor
_PLATT = {}          # target -> (a, b)
_PROV = {}
_LOCK = threading.Lock()
_STATS = {"requests": 0, "molecules": 0, "invalid_molecules": 0, "failed_requests": 0, "seconds": 0.0,
          "ad_counts": {t: {"low": 0, "medium": 0, "high": 0} for t in TARGETS}}
_ARGS = None
_LOGGED_HEADER = False


def _clean(arr):
    """JSON-safe list: NaN/inf -> None."""
    return [float(x) if np.isfinite(x) else None for x in np.asarray(arr, dtype=float)]


def load(args):
    global _P, _PLATT, _PROV
    torch.set_num_threads(args.threads)
    _P = ADMTDLPredictor(args.ckpt, args.thresholds, args.ad_reference)
    platt = json.load(open(args.platt))["targets"]
    _PLATT = {t: (float(platt[t]["a"]), float(platt[t]["b"])) for t in TARGETS}
    _PROV = {"checkpoint": str(args.ckpt), "checkpoint_sha256": sha256_file(args.ckpt),
             "ad_reference_sha256": sha256_file(args.ad_reference), "thresholds_sha256": sha256_file(args.thresholds),
             "platt_sha256": sha256_file(args.platt), "server_sha256": sha256_file(__file__),
             "predict_sha256": sha256_file(HERE / "predict.py"), "torch_threads": args.threads}
    log.info("loaded | logit thresholds %s | platt slopes %s",
             {t: round(_P.thr[t], 3) for t in TARGETS}, {t: round(_PLATT[t][0], 3) for t in TARGETS})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "targets": TARGETS, "logit_thresholds": _P.thr,
                    "platt": {t: {"a": a, "b": b} for t, (a, b) in _PLATT.items()},
                    "quantities": ["margin", "logit", "prob_platt", "ad_level", "ad_struct_sim", "ad_embed_sim"],
                    "provenance": _PROV})


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(_STATS)


@app.route("/score", methods=["POST"])
def score():
    """POST {"smiles": [...]} -> {"valid": [...], "targets": {T: {margin, logit, prob_platt, ad_level, ad_*_sim}}}.
    margin = platt_slope * (logit - frozen_logit_threshold): 0 at the decision boundary, calibrated log-odds relative to it.
    ad_level: 0 low / 1 medium / 2 high. Invalid molecules are valid=False with null scores (never a fake number)."""
    global _LOGGED_HEADER
    smiles = request.get_json(force=True).get("smiles", [])
    if not smiles:
        return jsonify({"error": "no smiles provided"}), 400
    t0 = time.time()
    try:
        with _LOCK:
            df = _P.predict(smiles, batch_size=_ARGS.batch)
    except Exception as exc:
        _STATS["failed_requests"] += 1
        log.exception("scoring failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    valid = df["valid"].to_numpy(bool)
    out = {"valid": valid.tolist(), "targets": {}}
    for t in TARGETS:
        z = df[f"{t}_logit"].to_numpy(float); a, b = _PLATT[t]
        lvl = df[f"{t}_ad_level"].map(LEVEL_CODE).astype(float).to_numpy()
        out["targets"][t] = {
            "logit": _clean(z), "margin": _clean(a * (z - _P.thr[t])), "prob_platt": _clean(expit(a * z + b)),
            "ad_level": _clean(lvl), "ad_struct_sim": _clean(df[f"{t}_ad_struct_sim"]),
            "ad_embed_sim": _clean(df[f"{t}_ad_embed_sim"])}

    _STATS["requests"] += 1; _STATS["molecules"] += len(smiles); _STATS["invalid_molecules"] += int((~valid).sum())
    _STATS["seconds"] += time.time() - t0
    for t in TARGETS:                      # applicability-domain mix per target, over valid molecules only
        lv = np.array(out["targets"][t]["ad_level"], dtype=float)
        for k, name in enumerate(("low", "medium", "high")):
            _STATS["ad_counts"][t][name] += int(np.sum(lv == k))
    if _ARGS.log_molecules:
        rec = df[["smiles", "valid"] + [f"{t}_{c}" for t in TARGETS for c in ("logit", "ad_level")]].copy()
        rec.insert(0, "request", _STATS["requests"])
        with _LOCK:
            rec.to_csv(_ARGS.log_molecules, mode="a", header=not _LOGGED_HEADER, index=False)
            _LOGGED_HEADER = True
    if _STATS["requests"] % 100 == 0:
        log.info("requests=%d molecules=%d invalid=%d failed=%d avg=%.2fs", _STATS["requests"], _STATS["molecules"],
                 _STATS["invalid_molecules"], _STATS["failed_requests"], _STATS["seconds"] / _STATS["requests"])
    return jsonify(out)


def parse_args():
    p = argparse.ArgumentParser(description="AD-MTDL scoring server (v2)")
    p.add_argument("--ckpt", default=str(HERE / "AD_MTDL_v1_best.ckpt"))
    p.add_argument("--thresholds", default=str(HERE / "thresholds.json"))
    p.add_argument("--ad-reference", default=str(HERE / "AD_reference.npz"))
    p.add_argument("--platt", default=str(HERE / "platt.json"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=8, help="torch CPU threads for THIS server")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--log-molecules", default=None, help="append every scored molecule (SMILES, logits, AD level) to this CSV")
    return p.parse_args()


if __name__ == "__main__":
    _ARGS = parse_args()
    load(_ARGS)
    atexit.register(lambda: log.info("final stats: %s", json.dumps(_STATS)))
    log.info("listening on 127.0.0.1:%d", _ARGS.port)
    app.run(host="127.0.0.1", port=_ARGS.port, threaded=True)
