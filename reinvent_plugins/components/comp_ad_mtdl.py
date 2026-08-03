# reinvent_plugins/components/comp_ad_mtdl.py
#
# REINVENT4 contrib plugin — AD Multi-Target Drug Design Oracle
# Bridges ChemProp v2.x multi-task classifier to REINVENT4 4.x via HTTP.
# This resolves the ChemProp v1.x / v2.x incompatibility in REINVENT4.
#
# INSTALL:
#   Place this file at:
#     ~/AD-RL/reinvent_plugins/components/comp_ad_mtdl.py
#   No __init__.py needed (native namespace package).
#   REINVENT4 discovers it via:
#     export PYTHONPATH=~/AD-RL
#
# REQUIRES:
#   scoring_server.py must be running in ad_mtdl env before REINVENT4 starts.
#
# TOML usage (one endpoint per target):
#
#   [[stage.scoring.component]]
#   [stage.scoring.component.ADMTDLOracle]
#
#   [[stage.scoring.component.ADMTDLOracle.endpoint]]
#   name = "AChE_activity"
#   weight = 1.0
#   [stage.scoring.component.ADMTDLOracle.endpoint.params]
#   target = "AChE_class"
#   server_url = "http://127.0.0.1:8765"
#
#   [[stage.scoring.component.ADMTDLOracle.endpoint]]
#   name = "BACE1_activity"
#   weight = 1.0
#   [stage.scoring.component.ADMTDLOracle.endpoint.params]
#   target = "BACE1_class"
#   server_url = "http://127.0.0.1:8765"

__all__ = ["ADMTDLOracle"]

from dataclasses import dataclass
from typing import List
import logging

import numpy as np
import requests
from rdkit import Chem

from .component_results import ComponentResults
from .add_tag import add_tag

logger = logging.getLogger("reinvent")

VALID_TARGETS  = {"AChE_class", "BuChE_class", "BACE1_class", "MAO_B_class"}
REQUEST_TIMEOUT = 180  # seconds — generous for CPU inference on large batches


@add_tag("__parameters")
@dataclass
class Parameters:
    """
    REINVENT4 collects params from ALL endpoints into lists.
    With 2 endpoints, params.target = ["AChE_class", "BACE1_class"].
    """
    target:     List[str]
    server_url: List[str]


@add_tag("__component")
class ADMTDLOracle:
    """
    Multi-target AD activity oracle backed by ChemProp v2.x.

    One HTTP call per batch regardless of number of target endpoints.
    Invalid SMILES receive default score 0.5 (model uncertainty).
    """

    def __init__(self, params: Parameters):
        self.targets    = params.target
        # All endpoints share the same server; take first URL
        self.server_url = params.server_url[0].rstrip("/")

        # Validate target names at startup — fail fast
        invalid = [t for t in self.targets if t not in VALID_TARGETS]
        if invalid:
            raise ValueError(
                f"ADMTDLOracle: unknown targets {invalid}. "
                f"Valid: {sorted(VALID_TARGETS)}"
            )

        # Confirm server is reachable before RL starts
        try:
            r = requests.get(
                f"{self.server_url}/health", timeout=10
            )
            r.raise_for_status()
            info = r.json()
            logger.info(
                f"ADMTDLOracle: connected to scoring server "
                f"at {self.server_url}"
            )
            logger.info(
                f"ADMTDLOracle: server targets = {info.get('targets')}"
            )
            logger.info(
                f"ADMTDLOracle: active targets  = {self.targets}"
            )
        except Exception as exc:
            # Warn but don't crash init — server might start between
            # init and first __call__
            logger.warning(
                f"ADMTDLOracle: cannot reach server at "
                f"{self.server_url} during init: {exc}. "
                f"Ensure scoring_server.py is running."
            )

    def __call__(self, smilies: List[str]) -> ComponentResults:
        """
        Score a batch of SMILES. Called once per RL step by REINVENT4.

        Makes ONE HTTP request for all targets simultaneously.
        Returns ComponentResults with one score array per endpoint.
        """
        n = len(smilies)

        # Separate valid / invalid SMILES up front
        valid_mask    = [Chem.MolFromSmiles(str(s)) is not None for s in smilies]
        valid_smiles  = [s for s, v in zip(smilies, valid_mask) if v]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]

        # Default: 0.5 for invalid SMILES (mid-range, model uncertainty)
        scores_per_target = {
            t: np.full(n, 0.5, dtype=np.float32)
            for t in self.targets
        }

        if not valid_smiles:
            logger.warning("ADMTDLOracle: no valid SMILES in batch")
            return ComponentResults(
                [scores_per_target[t] for t in self.targets]
            )

        # Single HTTP call — server scores all targets at once
        try:
            resp = requests.post(
                f"{self.server_url}/score",
                json={"smiles": valid_smiles},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            preds = resp.json()

            # Map scores back to original batch positions
            for target in self.targets:
                if target not in preds:
                    logger.warning(
                        f"ADMTDLOracle: target '{target}' "
                        f"missing from server response"
                    )
                    continue
                target_preds = preds[target]
                for rank, orig_idx in enumerate(valid_indices):
                    scores_per_target[target][orig_idx] = float(
                        target_preds[rank]
                    )

        except requests.exceptions.ConnectionError:
            logger.error(
                f"ADMTDLOracle: lost connection to scoring server at "
                f"{self.server_url}. Returning default scores (0.5)."
            )
        except requests.exceptions.Timeout:
            logger.error(
                f"ADMTDLOracle: request timed out ({REQUEST_TIMEOUT}s). "
                f"Consider reducing batch_size in TOML."
            )
        except Exception as exc:
            logger.error(
                f"ADMTDLOracle: unexpected error: {exc}. "
                f"Returning default scores."
            )

        return ComponentResults(
            [scores_per_target[t] for t in self.targets]
        )

