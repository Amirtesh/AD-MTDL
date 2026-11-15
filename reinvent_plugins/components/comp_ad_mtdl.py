# reinvent_plugins/components/comp_ad_mtdl.py  (v2)
#
# REINVENT4 contrib plugin: HTTP bridge to the ChemProp v2 AD-MTDL scoring server.
#
# What changed vs v1
#   * No silent fallback scores. Invalid molecules and failed requests return NaN (REINVENT's convention for
#     "could not be scored"), and the two causes are counted separately.
#   * Fail fast: the server must be reachable at init; MAX_CONSECUTIVE_FAILED_CALLS failed batches abort the run
#     instead of letting it continue on garbage.
#   * Each endpoint chooses WHICH server quantity it returns (param "quantity"), so diagnostics (raw logit,
#     applicability-domain level) can ride along as zero-weight endpoints and appear in REINVENT's output CSV.
#   * Failure statistics are written to ADMTDL_bridge_stats.json (path via env ADMTDL_BRIDGE_STATS).
#
# TOML (one endpoint per quantity; weight = 0 only displays a column and does not affect the aggregate):
#
#   [[stage.scoring.component]]
#   [stage.scoring.component.ADMTDLOracle]
#   [[stage.scoring.component.ADMTDLOracle.endpoint]]
#   name = "AChE margin"
#   weight = 1.0
#   params.target = "AChE_class"
#   params.quantity = "margin"
#   params.server_url = "http://127.0.0.1:8765"
#   [[stage.scoring.component.ADMTDLOracle.endpoint]]
#   name = "AChE AD level"
#   weight = 0.0
#   params.target = "AChE_class"
#   params.quantity = "ad_level"
#   params.server_url = "http://127.0.0.1:8765"

__all__ = ["ADMTDLOracle"]

import atexit
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import requests
from rdkit import Chem

from .component_results import ComponentResults
from .add_tag import add_tag

logger = logging.getLogger("reinvent")

VALID_TARGETS = {"AChE_class", "BuChE_class", "BACE1_class", "MAO_B_class"}
VALID_QUANTITIES = {"margin", "logit", "prob_platt", "ad_level", "ad_struct_sim", "ad_embed_sim"}
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILED_CALLS = 5
INIT_RETRIES, INIT_WAIT = 10, 3.0
STATS_PATH = os.environ.get("ADMTDL_BRIDGE_STATS", "ADMTDL_bridge_stats.json")

_STATS = {"calls": 0, "molecules": 0, "invalid_smiles": 0, "failed_calls": 0, "failed_molecules": 0,
          "retries": 0, "server_invalid": 0}


def _write_stats():
    try:
        with open(STATS_PATH, "w") as f:
            json.dump(_STATS, f, indent=2)
    except OSError:
        pass


atexit.register(_write_stats)


@add_tag("__parameters")
@dataclass
class Parameters:
    """REINVENT collects params from ALL endpoints into lists (one entry per endpoint)."""
    target: List[str]
    server_url: List[str]
    quantity: Optional[List[str]] = None


@add_tag("__component")
class ADMTDLOracle:
    """One HTTP call per batch regardless of how many endpoints are configured."""

    def __init__(self, params: Parameters):
        self.targets = params.target
        self.quantities = params.quantity or ["margin"] * len(self.targets)
        self.server_url = params.server_url[0].rstrip("/")
        if len(self.quantities) != len(self.targets):
            raise ValueError("ADMTDLOracle: 'quantity' must be given on every endpoint or on none")
        bad_t = [t for t in self.targets if t not in VALID_TARGETS]
        bad_q = [q for q in self.quantities if q not in VALID_QUANTITIES]
        if bad_t or bad_q:
            raise ValueError(f"ADMTDLOracle: unknown targets {bad_t} / quantities {bad_q}. "
                             f"Valid: {sorted(VALID_TARGETS)} / {sorted(VALID_QUANTITIES)}")
        self._consecutive_failures = 0
        self.number_of_endpoints = len(self.targets)   # REINVENT checks one score array per endpoint

        info, last = None, None
        for _ in range(INIT_RETRIES):
            try:
                r = requests.get(f"{self.server_url}/health", timeout=10)
                r.raise_for_status()
                info = r.json()
                break
            except Exception as exc:
                last = exc
                time.sleep(INIT_WAIT)
        if info is None:
            raise RuntimeError(f"ADMTDLOracle: scoring server not reachable at {self.server_url} "
                               f"after {INIT_RETRIES} attempts ({last}). Start scoring_server.py first.")
        logger.info(f"ADMTDLOracle: connected to {self.server_url}")
        logger.info(f"ADMTDLOracle: endpoints = {list(zip(self.targets, self.quantities))}")
        logger.info(f"ADMTDLOracle: server provenance = {json.dumps(info.get('provenance', {}))}")

    def _post(self, smiles):
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(f"{self.server_url}/score", json={"smiles": smiles}, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last = exc
                _STATS["retries"] += 1
                logger.warning(f"ADMTDLOracle: request failed (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")
                time.sleep(2 ** attempt)
        raise last

    def __call__(self, smilies: List[str]) -> ComponentResults:
        n = len(smilies)
        arrays = [np.full(n, np.nan, dtype=np.float32) for _ in self.targets]
        _STATS["calls"] += 1
        _STATS["molecules"] += n

        valid_idx = [i for i, s in enumerate(smilies) if Chem.MolFromSmiles(str(s)) is not None]
        _STATS["invalid_smiles"] += n - len(valid_idx)
        if not valid_idx:
            return ComponentResults(arrays)

        try:
            preds = self._post([smilies[i] for i in valid_idx])
            self._consecutive_failures = 0
        except Exception as exc:
            _STATS["failed_calls"] += 1
            _STATS["failed_molecules"] += len(valid_idx)
            self._consecutive_failures += 1
            logger.error(f"ADMTDLOracle: batch could not be scored ({exc}); returning NaN for {len(valid_idx)} molecules.")
            _write_stats()
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILED_CALLS:
                raise RuntimeError(f"ADMTDLOracle: {self._consecutive_failures} consecutive failed batches; aborting run.")
            return ComponentResults(arrays)

        server_valid = preds.get("valid", [True] * len(valid_idx))
        _STATS["server_invalid"] += int(sum(1 for v in server_valid if not v))
        for k, (target, quantity) in enumerate(zip(self.targets, self.quantities)):
            vals = preds["targets"][target[:-len("_class")]][quantity]
            for rank, orig in enumerate(valid_idx):
                v = vals[rank]
                if v is not None and server_valid[rank]:
                    arrays[k][orig] = float(v)

        if _STATS["calls"] % 100 == 0:
            _write_stats()
            logger.info(f"ADMTDLOracle stats: {_STATS}")
        return ComponentResults(arrays)
