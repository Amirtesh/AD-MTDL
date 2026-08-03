#!/usr/bin/env python3
# optimize.py — AD-MTDL Molecular Optimizer
# Changes from previous version:
#   1. Oracle endpoints now apply threshold-centered sigmoid transform
#   2. results_1.csv is primary output path (REINVENT4 4.7.15 convention)

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OPTIMIZE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR      = Path(__file__).parent.resolve()
VALID_TARGETS   = ["AChE", "BuChE", "BACE1", "MAO_B"]
TARGET_COL_MAP  = {t: f"{t}_class" for t in VALID_TARGETS}

DEFAULT_CKPT           = str(SCRIPT_DIR / "AD_MTDL_best.ckpt")
DEFAULT_META           = str(SCRIPT_DIR / "AD_MTDL_metadata.json")
DEFAULT_PRIOR_SCAFFOLD = str(SCRIPT_DIR / "priors" / "mol2mol_medium_similarity.prior")
DEFAULT_PRIOR_DENOVO   = str(SCRIPT_DIR / "priors" / "reinvent.prior")
DEFAULT_SERVER_PORT    = 8765
DEFAULT_STEPS          = 10000
DEFAULT_BATCH_SIZE     = 64
DEFAULT_TOP_K          = 50
DEFAULT_AD_MTDL_ENV    = "ad_mtdl"

CUSTOM_ALERTS = [
    "[*;r{8-17}]",    "[#8][#8]",         "[#6;+]",
    "[#16][#16]",     "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]", "C#C",              "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
]

SIGMOID_SPREAD = 0.20  # half-width around threshold for sigmoid transform


# ══════════════════════════════════════════════════════════
# ARG PARSING
# ══════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="AD-MTDL Scaffold-hop / de novo optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    seed = p.add_mutually_exclusive_group()
    seed.add_argument("--smiles",      type=str)
    seed.add_argument("--smiles_file", type=str)
    p.add_argument("--smiles_col",   type=str,   default="smiles")
    p.add_argument(
        "--targets", nargs="+", choices=VALID_TARGETS, required=True,
        metavar="TARGET", help=f"One or more of: {VALID_TARGETS}",
    )
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument(
        "--mode", choices=["scaffold_hop", "de_novo"],
        default="scaffold_hop",
    )
    p.add_argument(
        "--tanimoto_weight", type=float, default=0.0,
        help="Tanimoto similarity to seed as scoring component "
             "(scaffold_hop only, 0.0 = disabled)",
    )
    p.add_argument("--ckpt",           default=DEFAULT_CKPT)
    p.add_argument("--meta",           default=DEFAULT_META)
    p.add_argument("--prior_scaffold", default=DEFAULT_PRIOR_SCAFFOLD)
    p.add_argument("--prior_denovo",   default=DEFAULT_PRIOR_DENOVO)
    p.add_argument("--steps",          type=int,   default=DEFAULT_STEPS)
    p.add_argument("--batch_size",     type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--min_score",      type=float, default=0.4)
    p.add_argument("--top_k",          type=int,   default=DEFAULT_TOP_K)
    p.add_argument("--output",         type=str,   default=None)
    p.add_argument("--keep_temp",      action="store_true")
    p.add_argument("--ad_mtdl_env",    default=DEFAULT_AD_MTDL_ENV)
    p.add_argument("--server_port",    type=int,   default=DEFAULT_SERVER_PORT)
    p.add_argument("--n_cpus",         type=int,   default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════
def validate(args):
    errors = []
    if args.mode == "scaffold_hop" and not args.smiles and not args.smiles_file:
        errors.append("scaffold_hop requires --smiles or --smiles_file")
    if args.mode == "de_novo" and args.tanimoto_weight > 0:
        log.warning("--tanimoto_weight ignored in de_novo mode")
        args.tanimoto_weight = 0.0
    if args.smiles and Chem.MolFromSmiles(args.smiles) is None:
        errors.append(f"Invalid seed SMILES: {args.smiles}")
    if args.smiles_file and not os.path.exists(args.smiles_file):
        errors.append(f"--smiles_file not found: {args.smiles_file}")
    if args.weights:
        if len(args.weights) != len(args.targets):
            errors.append("--weights count must match --targets count")
        elif abs(sum(args.weights) - 1.0) > 1e-3:
            errors.append("--weights must sum to 1.0")
    for label, path in [("--ckpt", args.ckpt), ("--meta", args.meta)]:
        if not os.path.exists(path):
            errors.append(f"{label} not found: {path}")
    prior = args.prior_scaffold if args.mode == "scaffold_hop" \
            else args.prior_denovo
    if not os.path.exists(prior):
        errors.append(f"Prior not found: {prior}")
    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)


# ══════════════════════════════════════════════════════════
# LOAD THRESHOLDS FROM METADATA
# ══════════════════════════════════════════════════════════
def load_thresholds(meta_path: str) -> dict:
    """
    Returns e.g. {'AChE_class': 0.59, 'BuChE_class': 0.68, ...}
    """
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["optimal_thresholds"]


# ══════════════════════════════════════════════════════════
# SEED FILE
# ══════════════════════════════════════════════════════════
def build_seed_file(args, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "seeds.smi")
    if args.smiles:
        smiles = [args.smiles]
    else:
        ext = os.path.splitext(args.smiles_file)[1].lower()
        if ext == ".txt":
            with open(args.smiles_file) as f:
                smiles = [l.strip() for l in f if l.strip()]
        else:
            df  = pd.read_csv(args.smiles_file)
            col = next(
                (c for c in df.columns
                 if c.lower() == args.smiles_col.lower()), None
            )
            if col is None:
                log.error(
                    f"Column '{args.smiles_col}' not found. "
                    f"Available: {list(df.columns)}"
                )
                sys.exit(1)
            smiles = df[col].astype(str).tolist()

    valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    if not valid:
        log.error("No valid seed SMILES.")
        sys.exit(1)

    log.info(f"Seed SMILES: {len(valid)} valid")
    with open(path, "w") as f:
        f.write("\n".join(valid) + "\n")
    return path


# ══════════════════════════════════════════════════════════
# TOML GENERATION
# ══════════════════════════════════════════════════════════
def _toml_smarts_list(smarts: list) -> str:
    lines = ["["]
    for s in smarts:
        lines.append(f'    "{s}",')
    lines.append("]")
    return "\n".join(lines)


def _oracle_endpoints(targets: list,
                      target_cols: list,
                      server_url: str,
                      thresholds: dict) -> str:
    """
    Build one endpoint block per target.

    Each endpoint applies a sigmoid transform centered at the
    training threshold for that target, with half-width SIGMOID_SPREAD.

    Sigmoid midpoint = threshold = (high + low) / 2
    → high = threshold + SIGMOID_SPREAD
    → low  = threshold - SIGMOID_SPREAD

    Effect: compounds at exactly the activity threshold score 0.5.
    Compounds well above threshold score ~0.95.
    Compounds well below threshold score ~0.05.
    RL agent receives meaningful gradient across the full probability range,
    with strongest learning signal at the activity decision boundary.
    """
    blocks = []
    for t, tc in zip(targets, target_cols):
        thresh = thresholds.get(tc, 0.5)
        high   = round(thresh + SIGMOID_SPREAD, 4)
        low    = round(thresh - SIGMOID_SPREAD, 4)

        blocks.append(textwrap.dedent(f"""\
            [[stage.scoring.component.ADMTDLOracle.endpoint]]
            name   = "{t}_activity"
            weight = 1.0
            [stage.scoring.component.ADMTDLOracle.endpoint.params]
            target     = "{tc}"
            server_url = "{server_url}"
            [stage.scoring.component.ADMTDLOracle.endpoint.transform]
            type = "sigmoid"
            high = {high}
            low  = {low}
            k    = 0.5
        """))
    return "\n".join(blocks)


def build_toml(args,
               seed_file: str,
               output_dir: str,
               tmp_dir: str,
               target_cols: list,
               server_url: str,
               thresholds: dict) -> str:

    prior  = args.prior_scaffold if args.mode == "scaffold_hop" \
             else args.prior_denovo
    tb_dir = os.path.join(output_dir, "tb_logs")
    os.makedirs(tb_dir, exist_ok=True)
    chkpt  = os.path.join(output_dir, "campaign.chkpt")
    csv_px = os.path.join(output_dir, "results")

    if args.mode == "scaffold_hop":
        params_block = textwrap.dedent(f"""\
            prior_file         = "{prior}"
            agent_file         = "{prior}"
            smiles_file        = "{seed_file}"
            sample_strategy    = "multinomial"
            distance_threshold = 100
            batch_size         = {args.batch_size}
            unique_sequences   = true
            randomize_smiles   = false
            summary_csv_prefix = "{csv_px}"
            use_checkpoint     = false
            purge_memories     = false
        """)
    else:
        params_block = textwrap.dedent(f"""\
            prior_file         = "{prior}"
            agent_file         = "{prior}"
            batch_size         = {args.batch_size}
            unique_sequences   = true
            randomize_smiles   = true
            summary_csv_prefix = "{csv_px}"
            use_checkpoint     = false
            purge_memories     = false
        """)

    oracle_block = _oracle_endpoints(
        args.targets, target_cols, server_url, thresholds
    )

    if args.tanimoto_weight > 0.0 and args.smiles:
        tanimoto_block = textwrap.dedent(f"""\
            [[stage.scoring.component]]
            [stage.scoring.component.TanimotoDistance]
            [[stage.scoring.component.TanimotoDistance.endpoint]]
            name   = "Tanimoto_seed"
            weight = 1.0
            params.smiles     = ["{args.smiles}"]
            params.radius     = 2
            params.use_counts  = true
            params.use_features = false
            [stage.scoring.component.TanimotoDistance.endpoint.transform]
            type = "sigmoid"
            high = {round(args.tanimoto_weight + 0.20, 2)}
            low  = {round(max(0.05, args.tanimoto_weight - 0.20), 2)}
            k    = 0.5
        """)
    else:
        tanimoto_block = ""

    alerts_str = _toml_smarts_list(CUSTOM_ALERTS)

    toml = textwrap.dedent(f"""\
        run_type        = "staged_learning"
        device          = "cpu"
        tb_logdir       = "{tb_dir}"
        json_out_config = "{os.path.join(output_dir, 'campaign.json')}"

        [parameters]
        {params_block}

        [learning_strategy]
        type  = "dap"
        sigma = 128
        rate  = 0.0001

        [diversity_filter]
        type               = "IdenticalMurckoScaffold"
        bucket_size        = 25
        minscore           = {args.min_score:.2f}
        minsimilarity      = 0.4
        penalty_multiplier = 0.5

        [[stage]]
        chkpt_file  = "{chkpt}"
        termination = "simple"
        max_score   = 0.99
        min_steps   = 50
        max_steps   = {args.steps}

        [stage.scoring]
        type = "geometric_mean"

        # ── Custom alerts ─────────────────────────────────
        [[stage.scoring.component]]
        [stage.scoring.component.custom_alerts]
        [[stage.scoring.component.custom_alerts.endpoint]]
        name          = "Unwanted_SMARTS"
        weight        = 1.0
        params.smarts = {alerts_str}

        # ── AD Oracle (ChemProp v2 via HTTP) ──────────────
        [[stage.scoring.component]]
        [stage.scoring.component.ADMTDLOracle]
        {oracle_block}

        # ── QED ───────────────────────────────────────────
        [[stage.scoring.component]]
        [stage.scoring.component.QED]
        [[stage.scoring.component.QED.endpoint]]
        name   = "QED"
        weight = 1.0

        # ── Synthetic accessibility ───────────────────────
        [[stage.scoring.component]]
        [stage.scoring.component.SAScore]
        [[stage.scoring.component.SAScore.endpoint]]
        name   = "SA_score"
        weight = 1.0
        [stage.scoring.component.SAScore.endpoint.transform]
        type = "reverse_sigmoid"
        high = 4.5
        low  = 2.0
        k    = 0.5

        # ── Molecular weight ──────────────────────────────
        [[stage.scoring.component]]
        [stage.scoring.component.MolecularWeight]
        [[stage.scoring.component.MolecularWeight.endpoint]]
        name   = "MW"
        weight = 1.0
        [stage.scoring.component.MolecularWeight.endpoint.transform]
        type     = "double_sigmoid"
        high     = 550.0
        low      = 200.0
        coef_div = 500.0
        coef_si  = 20.0
        coef_se  = 20.0

        {tanimoto_block}
    """)

    toml_path = os.path.join(tmp_dir, "campaign.toml")
    with open(toml_path, "w") as fh:
        fh.write(toml)
    return toml_path


# ══════════════════════════════════════════════════════════
# SCORING SERVER
# ══════════════════════════════════════════════════════════
def start_server(args, n_cpus: int, log_path: str) -> subprocess.Popen:
    server_script  = str(SCRIPT_DIR / "scoring_server.py")
    server_threads = max(1, int(n_cpus * 0.75))
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", args.ad_mtdl_env,
        "python", server_script,
        "--ckpt",    args.ckpt,
        "--meta",    args.meta,
        "--port",    str(args.server_port),
        "--threads", str(server_threads),
    ]
    log.info(
        f"Starting scoring server "
        f"(env={args.ad_mtdl_env}, threads={server_threads})..."
    )
    log_fh = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log_fh, stderr=log_fh, preexec_fn=os.setsid
    )


def wait_for_server(server_url: str, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# RESULTS PARSING
# ══════════════════════════════════════════════════════════
def parse_results(output_dir: str,
                  targets: list,
                  target_cols: list,
                  server_url: str,
                  min_score: float,
                  top_k: int) -> pd.DataFrame:

    # REINVENT4 4.7.15 uses 1-indexed output: results_1.csv
    csv_path = os.path.join(output_dir, "results_1.csv")

    if not os.path.exists(csv_path):
        candidates = sorted(Path(output_dir).glob("results*.csv"))
        if not candidates:
            candidates = sorted(Path(output_dir).rglob("*.csv"))
        if not candidates:
            log.warning("No results CSV found. Check REINVENT4 log.")
            return pd.DataFrame()
        csv_path = str(candidates[-1])

    log.info(f"Parsing results from: {csv_path}")
    df = pd.read_csv(csv_path)

    smi_col = next(
        (c for c in df.columns if c.upper() == "SMILES"), None
    )
    if smi_col is None:
        log.warning(f"No SMILES column. Columns: {list(df.columns)}")
        return df

    score_col = next(
        (c for c in df.columns
         if "total" in c.lower() or "score" in c.lower()), None
    )

    seen, rows = set(), []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smi_col]))
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        rows.append({
            "smiles"     : canon,
            "total_score": float(row[score_col]) if score_col else 0.0,
        })

    if not rows:
        return pd.DataFrame()

    results = (
        pd.DataFrame(rows)
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )
    results = results[results["total_score"] >= min_score]
    if top_k > 0:
        results = results.head(top_k)
    if results.empty:
        return results

    log.info(f"Re-scoring {len(results)} compounds for per-target breakdown...")
    try:
        resp = requests.post(
            f"{server_url}/score",
            json={"smiles": results["smiles"].tolist()},
            timeout=300,
        )
        resp.raise_for_status()
        preds = resp.json()
        for t, tc in zip(targets, target_cols):
            if tc in preds:
                results[f"{t}_prob"] = [
                    round(float(p), 4) for p in preds[tc]
                ]
    except Exception as e:
        log.warning(f"Re-scoring failed: {e}")

    return results.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════
def print_summary(results: pd.DataFrame, args, elapsed: float, thresholds: dict):
    sep = "═" * 82
    print(f"\n{sep}")
    print(f"  AD-MTDL OPTIMIZATION RESULTS")
    print(f"  Mode    : {args.mode}")
    print(f"  Targets : {', '.join(args.targets)}")
    if args.smiles:
        print(f"  Seed    : {args.smiles}")
    print(f"  Steps   : {args.steps}   Time: {elapsed/60:.1f} min")

    # Show thresholds used for this run
    thresh_str = "  ".join(
        f"{t}≥{thresholds.get(TARGET_COL_MAP[t], '?')}"
        for t in args.targets
    )
    print(f"  Thresholds (sigmoid midpoints): {thresh_str}")
    print(sep)

    if results.empty:
        print(f"  No molecules above score threshold ({args.min_score}).")
        print(sep)
        return

    header = (
        f"  {'#':>4}  {'SMILES':<45}  {'Score':>7}  "
        + "  ".join(f"{t:>9}" for t in args.targets)
    )
    print(header)
    print("  " + "─" * 80)

    for i, row in results.iterrows():
        smi = row["smiles"]
        if len(smi) > 43:
            smi = smi[:40] + "..."
        prob_str = "  ".join(
            f"{row.get(f'{t}_prob', float('nan')):>9.4f}"
            for t in args.targets
        )
        print(f"  {i+1:>4}  {smi:<45}  "
              f"{row['total_score']:>7.4f}  {prob_str}")

    print(f"\n  Total reported : {len(results)}")
    print(
        f"  Score range    : "
        f"{results['total_score'].min():.4f} – "
        f"{results['total_score'].max():.4f}"
    )
    for t in args.targets:
        col    = f"{t}_prob"
        thresh = thresholds.get(TARGET_COL_MAP[t], 0.5)
        if col in results.columns:
            n_active = (results[col] >= thresh).sum()
            print(
                f"  {t} predicted active "
                f"(prob ≥ {thresh}): {n_active}/{len(results)}"
            )
    print(sep)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    args = parse_args()
    validate(args)

    # Load thresholds from metadata — used in TOML and summary
    thresholds = load_thresholds(args.meta)
    log.info(f"Loaded thresholds: {thresholds}")

    total_cpus    = args.n_cpus or os.cpu_count() or 8
    reinvent_cpus = max(1, total_cpus // 4)

    os.environ["OMP_NUM_THREADS"]      = str(reinvent_cpus)
    os.environ["MKL_NUM_THREADS"]      = str(reinvent_cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(reinvent_cpus)

    # Plugin discovery
    plugin_dir = str(SCRIPT_DIR)
    current_pp = os.environ.get("PYTHONPATH", "")
    if plugin_dir not in current_pp:
        os.environ["PYTHONPATH"] = (
            f"{plugin_dir}:{current_pp}" if current_pp else plugin_dir
        )

    target_cols = [TARGET_COL_MAP[t] for t in args.targets]
    server_url  = f"http://127.0.0.1:{args.server_port}"

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Output directory : {output_dir}")
    log.info(f"Mode             : {args.mode}")
    log.info(f"Targets          : {args.targets}")
    log.info(f"Steps            : {args.steps}")
    log.info(f"Total CPUs       : {total_cpus}")
    log.info(f"REINVENT4 CPUs   : {reinvent_cpus}")

    tmp_dir     = tempfile.mkdtemp(prefix="ad_mtdl_")
    server_proc = None

    try:
        seed_file = None
        if args.mode == "scaffold_hop":
            seed_file = build_seed_file(args, tmp_dir)

        toml_path = build_toml(
            args, seed_file or "", output_dir, tmp_dir,
            target_cols, server_url, thresholds,
        )
        log.info(f"TOML written: {toml_path}")

        server_log  = os.path.join(output_dir, "scoring_server.log")
        server_proc = start_server(args, total_cpus, server_log)

        log.info("Waiting for scoring server...")
        if not wait_for_server(server_url, timeout=90):
            log.error(
                f"Server failed to start within 90s. Check: {server_log}"
            )
            sys.exit(1)
        log.info("Scoring server ready.")

        if args.smiles:
            try:
                r = requests.post(
                    f"{server_url}/score",
                    json={"smiles": [args.smiles]},
                    timeout=30,
                )
                preds = r.json()
                scores = {
                    t: round(float(preds[tc][0]), 4)
                    for t, tc in zip(args.targets, target_cols)
                    if tc in preds
                }
                log.info(f"Seed SMILES scores: {scores}")
            except Exception as e:
                log.warning(f"Seed sanity check failed: {e}")

        reinvent_log = os.path.join(output_dir, "reinvent.log")
        log.info(f"Starting REINVENT4 ({args.steps} steps)...")

        start_time = time.time()
        proc = subprocess.run(
            ["reinvent", "-l", reinvent_log, toml_path],
            env=os.environ,
        )
        elapsed = time.time() - start_time

        log.info(
            f"REINVENT4 finished in {elapsed/60:.1f} min "
            f"(exit code: {proc.returncode})"
        )
        if proc.returncode != 0:
            log.error(
                f"REINVENT4 exit code {proc.returncode}. "
                f"Check: {reinvent_log}"
            )

        log.info("Parsing results...")
        results = parse_results(
            output_dir  = output_dir,
            targets     = args.targets,
            target_cols = target_cols,
            server_url  = server_url,
            min_score   = args.min_score,
            top_k       = args.top_k,
        )

        if not results.empty:
            out_csv = os.path.join(output_dir, "top_compounds.csv")
            results.to_csv(out_csv, index=False)
            log.info(f"Saved {len(results)} compounds to {out_csv}")

        with open(os.path.join(output_dir, "run_config.json"), "w") as f:
            json.dump({
                "timestamp"      : timestamp,
                "mode"           : args.mode,
                "seed_smiles"    : args.smiles,
                "seed_file"      : args.smiles_file,
                "targets"        : args.targets,
                "target_cols"    : target_cols,
                "thresholds_used": {
                    t: thresholds.get(tc)
                    for t, tc in zip(args.targets, target_cols)
                },
                "weights"        : args.weights,
                "tanimoto_weight": args.tanimoto_weight,
                "steps"          : args.steps,
                "batch_size"     : args.batch_size,
                "min_score"      : args.min_score,
                "top_k"          : args.top_k,
                "elapsed_min"    : round(elapsed / 60, 2),
                "n_results"      : len(results),
            }, f, indent=2)

        print_summary(results, args, elapsed, thresholds)
        print(f"\n  Output directory : {output_dir}")
        print(f"  Top compounds    : "
              f"{os.path.join(output_dir, 'top_compounds.csv')}")
        print(f"  REINVENT4 log    : {reinvent_log}")
        print(f"  Server log       : {server_log}\n")

    except KeyboardInterrupt:
        log.info("Interrupted.")

    finally:
        if server_proc is not None:
            log.info("Stopping scoring server...")
            stop_server(server_proc)
        if not args.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            log.info(f"Temp files at: {tmp_dir}")


if __name__ == "__main__":
    main()

