#!/usr/bin/env python
"""
ad_mtdl_final.py - single script: no-RL control (chunked parallel sampling + scoring + merge) and campaign analysis.

  smoke test : python ad_mtdl_final.py --stage control --workers 2 --chunks 2 --limit 1 --n 200 --ctrl ctrl_test
  full run   : nohup python ad_mtdl_final.py > final.log 2>&1 &     (90 single-thread jobs, then analysis)
  analysis   : python ad_mtdl_final.py --stage analyze
"""
import os, sys, re, json, math, argparse, itertools, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=["all", "control", "analyze", "worker"])
ap.add_argument("--workers", type=int, default=90)
ap.add_argument("--chunks", type=int, default=6, help="independent sampling chunks per seed")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--first-core", type=int, default=0)
ap.add_argument("--n", type=int, default=20000, help="total samples per seed (split over chunks)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--rng", type=int, default=1)
ap.add_argument("--worker", default="0/1")
ap.add_argument("--root", default="campaigns_v2")
ap.add_argument("--ctrl", default="control_v2")
ap.add_argument("--out", default="analysis_v2")
ap.add_argument("--seeds", default="campaign_seeds_v2.csv")
ap.add_argument("--prior", default="priors/mol2mol_medium_similarity.prior")
ap.add_argument("--reinvent-env", default="reinvent4")
a = ap.parse_args()

NT = str(a.threads if a.stage in ("worker", "control") else 4)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = NT

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

# =============================== CONTROL ===============================
TOML = """run_type = "sampling"
device = "cpu"
json_out_config = "_sampling.json"

[parameters]
model_file = "{prior}"
smiles_file = "seed.smi"
sample_strategy = "multinomial"
temperature = 1.0
num_smiles = {k}
unique_molecules = true
randomize_smiles = false
output_file = "sampling.csv"
"""


def read_seeds():
    import pandas as pd
    sd = pd.read_csv(a.seeds)
    cc = next(c for c in sd.columns if norm(c) in ("campaign", "campaignname"))
    sc = next(c for c in sd.columns if "smiles" in c.lower())
    kc = next((c for c in sd.columns if norm(c) in ("seed", "seedid", "seedidx", "seedindex", "seedk", "k")), None)
    if kc is None:
        sd["_k"] = sd.groupby(cc).cumcount() + 1
        kc = "_k"
    if a.limit:
        sd = sd.head(a.limit)
    return sd, cc, sc, kc


def sample(wd, smi, k, rng):
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "seed.smi").write_text(smi + "\n")
    (wd / "sampling.toml").write_text(TOML.format(prior=str(Path(a.prior).resolve()), k=k))
    if (wd / "sampling.csv").exists():
        (wd / "sampling.csv").unlink()
    subprocess.run(["conda", "run", "--no-capture-output", "-n", a.reinvent_env, "reinvent",
                    "-l", "log.txt", "-s", str(rng), "sampling.toml"], cwd=wd, check=True)


def unique_smiles(wd):
    import pandas as pd
    smp = pd.read_csv(wd / "sampling.csv")
    col = next(c for c in smp.columns if c.lower() == "smiles")
    return smp[col].dropna().astype(str).drop_duplicates().tolist()


def worker():
    import torch
    torch.set_num_threads(a.threads)
    import pandas as pd
    sys.path.insert(0, str(Path.cwd()))
    from predict import ADMTDLPredictor, TARGETS
    sd, cc, sc, kc = read_seeds()
    units = [(r[cc], int(r[kc]), r[sc], c) for _, r in sd.iterrows() for c in range(a.chunks)]
    i, n = map(int, a.worker.split("/"))
    keep = ["smiles", "valid"] + [f"{t}_{c}" for t in TARGETS for c in ("logit", "ad_level", "ad_struct_sim")]
    per = math.ceil(a.n / a.chunks)
    pred = None
    for camp, k, smi, c in units[i::n]:
        wd = Path(a.ctrl) / f"{camp}_s{k}" / f"chunk_{c}"
        if (wd / "scored.csv").exists():
            print("skip", camp, k, c, flush=True)
            continue
        if not (wd / "sampling.csv").exists():
            sample(wd, smi, per, a.rng * 1000 + c)
        smis = unique_smiles(wd)
        print(camp, k, c, "requested", per, "unique", len(smis), flush=True)
        if pred is None:
            pred = ADMTDLPredictor(device="cpu")
        res = pd.concat([pred.predict(smis[j:j + 2000]) for j in range(0, len(smis), 2000)], ignore_index=True)
        res[keep].to_csv(wd / "scored.csv", index=False)
        print("scored", camp, k, c, flush=True)


def merge():
    import pandas as pd
    for sdir in sorted(Path(a.ctrl).glob("*_s*")):
        out = sdir / "scored_control.csv"
        if out.exists():
            continue
        parts = [sdir / f"chunk_{c}" / "scored.csv" for c in range(a.chunks)]
        if not all(p.exists() for p in parts):
            print("WARNING incomplete chunks for", sdir.name)
            continue
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).drop_duplicates("smiles")
        df.to_csv(out, index=False)
        print("merged", sdir.name, "unique", len(df))


def control():
    procs = []
    for i in range(a.workers):
        s = a.first_core + i * a.threads
        cmd = ["taskset", "-c", f"{s}-{s + a.threads - 1}", sys.executable, os.path.abspath(__file__),
               "--stage", "worker", "--worker", f"{i}/{a.workers}", "--threads", str(a.threads),
               "--chunks", str(a.chunks), "--n", str(a.n), "--limit", str(a.limit), "--rng", str(a.rng),
               "--ctrl", a.ctrl, "--seeds", a.seeds, "--prior", a.prior, "--reinvent-env", a.reinvent_env]
        procs.append(subprocess.Popen(cmd, stdout=open(f"ctrl_worker_{i}.log", "w"), stderr=subprocess.STDOUT))
    bad = [i for i, p in enumerate(procs) if p.wait() != 0]
    if bad:
        sys.exit(f"control workers failed: {bad} (see ctrl_worker_<i>.log); rerun is safe, finished chunks are skipped")
    merge()
    print("control done")


# =============================== ANALYSIS ===============================
def _walk_fail(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, dict):
                yield from _walk_fail(v)
            elif "fail" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                yield k, v


def analyze():
    import numpy as np, pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    if Path(a.ctrl).exists():
        merge()
    ROOT, CTRL, OUT = Path(a.root), Path(a.ctrl), Path(a.out)
    OUT.mkdir(exist_ok=True)
    THR = {"ache": -0.50, "buche": 3.20, "bace1": -0.40, "maob": -2.55}
    if Path("thresholds.json").exists():
        tj = json.load(open("thresholds.json"))["thresholds_logit"]
        THR = {norm(k.replace("_class", "")): float(v) for k, v in tj.items()}
    RANK = {"low": 0, "medium": 1, "high": 2}
    STEPROWS = 64
    GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def col(df, t, suf):
        k = norm(t) + norm(suf)
        return next((c for c in df.columns if norm(c) == k), None)

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return None if m is None else GEN.GetFingerprint(m)

    def scaf(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            return None

    def prep(df):
        n0 = len(df)
        for c in df.columns:
            if c == "request" or c.endswith("_logit") or c.endswith("_ad_struct_sim"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "request" in df:
            df = df[df["request"].notna()]
            if len(df) < n0:
                print("   dropped", n0 - len(df), "non-numeric rows (repeated headers?)", flush=True)
        if "valid" in df:
            df = df[df["valid"].astype(str).str.lower().isin(["true", "1", "1.0"])]
        sc = next((c for c in df.columns if norm(c) in ("smiles", "canonicalsmiles")), None)
        return df.drop_duplicates(sc) if sc else df

    def ad_rank(df, targets, suf):
        r = []
        for t in targets:
            c = col(df, t, suf)
            if c is None:
                raise KeyError(f"AD column for {t} ({suf}) missing")
            r.append(df[c].astype(str).str.lower().map(RANK))
        return pd.concat(r, axis=1).min(axis=1)

    def passes(df, targets):
        ok = pd.Series(True, index=df.index)
        for t in targets:
            ok &= df[col(df, t, "logit")] >= THR[norm(t)]
        return ok

    def shares(rk):
        rk = rk.dropna()
        n = max(len(rk), 1)
        return {"low": (rk == 0).sum() / n, "medium": (rk == 1).sum() / n, "high": (rk == 2).sum() / n}

    def flat_nums(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from flat_nums(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from flat_nums(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield float(o)

    def reward_curve(rdir):
        f = rdir / "results_1.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        sc = next((c for c in df.columns if c.lower() == "score"), None)
        if sc is None:
            return None
        st = df["step"] if "step" in df.columns else pd.Series(np.arange(len(df)) // STEPROWS)
        return df.groupby(st.values)[sc].mean()

    def plateau(curve):
        if curve is None or len(curve) < 100:
            return np.nan
        b = curve.groupby((curve.index // 100) * 100).mean()
        hit = b[b >= 0.95 * b.iloc[-1]]
        return int(hit.index[0]) if len(hit) else np.nan

    ctrl_cache = {}

    def ctrl_stats(campaign, k, targets):
        f = CTRL / f"{campaign}_s{k}" / "scored_control.csv"
        if not f.exists():
            return None
        if f not in ctrl_cache:
            ctrl_cache[f] = prep(pd.read_csv(f))
        d = ctrl_cache[f]
        return {"pass": passes(d, targets).mean(), "n": len(d),
                **{f"ad_{x}": v for x, v in shares(ad_rank(d, targets, "ad_level")).items()}}

    rows, store, curves, adcurves = [], {}, {}, {}
    for cdir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for rdir in sorted(cdir.glob("s*_r*")):
            m = re.fullmatch(r"s(\d+)_r(\d+)", rdir.name)
            if not m or not (rdir / "top_compounds.csv").exists() or not (rdir / "run_config.json").exists():
                continue
            k, r = int(m[1]), int(m[2])
            cfg = json.loads((rdir / "run_config.json").read_text())
            T = cfg["targets"]
            T = T if isinstance(T, list) else list(T)
            seed = cfg.get("seed_smiles")
            seed = seed[0] if isinstance(seed, list) else seed
            top = pd.read_csv(rdir / "top_compounds.csv")
            sm = prep(pd.read_csv(rdir / "scored_molecules.csv", low_memory=False)) if (rdir / "scored_molecules.csv").exists() else None
            tsm = next(c for c in top.columns if "smiles" in c.lower())
            mols = [Chem.MolFromSmiles(s) for s in top[tsm]]
            mols = [x for x in mols if x is not None]
            fps = [GEN.GetFingerprint(x) for x in mols]
            scs = {scaf(s) for s in top[tsm]} - {None}
            sfp, smol = fp(seed), Chem.MolFromSmiles(seed)
            nums = list(flat_nums(cfg.get("seed_scores")))
            row = dict(campaign=cdir.name, seed=k, rng=r, targets="+".join(T), seed_smiles=seed,
                       seed_score_min_raw=min(nums) if nums else np.nan, elapsed_min=cfg.get("elapsed_min"))
            crv = reward_curve(rdir)
            row["plateau_step"] = plateau(crv)
            if sm is not None:
                for x, v in shares(ad_rank(sm, T, "ad_level")).items():
                    row[f"all_ad_{x}"] = v
                row["all_pass_rate"] = passes(sm, T).mean()
                row["n_scored_unique"] = len(sm)
            try:
                for x, v in shares(ad_rank(top, T, "ad")).items():
                    row[f"top_ad_{x}"] = v
            except KeyError:
                pass
            row["top_n"] = len(top)
            row["top_scaffolds"] = len(scs)
            row["top_mean_tanimoto_seed"] = float(np.mean(DataStructs.BulkTanimotoSimilarity(sfp, fps))) if fps and sfp else np.nan
            row["top_mean_MW"] = float(np.mean([Descriptors.MolWt(x) for x in mols])) if mols else np.nan
            row["seed_MW"] = Descriptors.MolWt(smol) if smol else np.nan
            bs = rdir / "bridge_stats.json"
            if bs.exists():
                row["failed_bridge_calls"] = sum(v for kk, v in _walk_fail(json.loads(bs.read_text())))
            c = ctrl_stats(cdir.name, k, T)
            if c:
                row.update({f"ctrl_{x}": v for x, v in c.items()})
                row["d_ad_low_vs_ctrl"] = row.get("all_ad_low", np.nan) - c["ad_low"]
                row["d_pass_vs_ctrl"] = row.get("all_pass_rate", np.nan) - c["pass"]
            else:
                print("WARNING no control for", cdir.name, f"s{k}")
            rows.append(row)
            store[(cdir.name, k, r)] = (fps, scs)
            curves[(cdir.name, k, r)] = crv
            if sm is not None and "request" in sm.columns:
                low = (ad_rank(sm, T, "ad_level") == 0).astype(float)
                adcurves[(cdir.name, k, r)] = low.groupby((sm["request"] // 100) * 100).mean()
            print("done", cdir.name, rdir.name, flush=True)

    if not rows:
        sys.exit("no runs found under " + str(ROOT))
    rt = pd.DataFrame(rows)
    rt.to_csv(OUT / "run_table.csv", index=False)
    cc = ["campaign", "seed", "rng", "all_pass_rate", "ctrl_pass", "d_pass_vs_ctrl", "all_ad_low", "all_ad_medium",
          "all_ad_high", "ctrl_ad_low", "ctrl_ad_medium", "ctrl_ad_high", "d_ad_low_vs_ctrl"]
    rt[[c for c in cc if c in rt]].to_csv(OUT / "rl_vs_control.csv", index=False)

    rep = []
    for camp in rt["campaign"].unique():
        keys = [k for k in store if k[0] == camp]
        for x, y in itertools.combinations(keys, 2):
            fa, sa = store[x]
            fb, sb = store[y]
            if not fa or not fb:
                continue
            nn = (np.mean([max(DataStructs.BulkTanimotoSimilarity(f, fb)) for f in fa]) +
                  np.mean([max(DataStructs.BulkTanimotoSimilarity(f, fa)) for f in fb])) / 2
            rep.append(dict(campaign=camp, run_a=f"s{x[1]}_r{x[2]}", run_b=f"s{y[1]}_r{y[2]}",
                            pair_type="same_seed_diff_rng" if x[1] == y[1] else "diff_seed",
                            nn_tanimoto=nn, scaffold_jaccard=len(sa & sb) / max(len(sa | sb), 1)))
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "reproducibility.csv", index=False)

    camps = sorted(rt["campaign"].unique())

    def panels(data, fname, ylabel, ctrl_col=None, smooth=False):
        fig, ax = plt.subplots(1, len(camps), figsize=(4 * len(camps), 3.4), squeeze=False)
        for axx, camp in zip(ax[0], camps):
            for key, s in data.items():
                if key[0] == camp and s is not None:
                    if smooth:
                        s = s.groupby((s.index // 20) * 20).mean()
                    axx.plot(s.index, s.values, lw=1, label=f"s{key[1]}_r{key[2]}")
            if ctrl_col and ctrl_col in rt:
                v = rt.loc[rt.campaign == camp, ctrl_col].dropna()
                if len(v):
                    axx.axhline(v.mean(), ls="--", c="k", lw=1, label="no-RL control")
            axx.set_xlabel("step")
            axx.set_ylabel(ylabel)
            axx.set_title(camp, fontsize=9)
        ax[0][0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=300)
        plt.close(fig)

    panels(curves, "reward_curves.png", "mean REINVENT score", smooth=True)
    panels(adcurves, "ad_low_share_vs_step.png", "share low-AD", ctrl_col="ctrl_ad_low")

    num = [c for c in rt.select_dtypes(include="number").columns if c not in ("seed", "rng")]
    sm_ = rt.groupby("campaign")[num].agg(["mean", "std"])
    sm_.columns = ["_".join(c) for c in sm_.columns]
    sm_.round(4).to_csv(OUT / "summary_by_campaign.csv")

    show = [c for c in ("all_pass_rate", "all_ad_low", "ctrl_ad_low", "d_ad_low_vs_ctrl", "d_pass_vs_ctrl",
                        "plateau_step", "top_scaffolds", "top_mean_tanimoto_seed", "failed_bridge_calls") if c in rt]
    print("\n=== per-campaign means ===")
    print(rt.groupby("campaign")[show].mean().round(3).to_string())
    if len(rep):
        print("\n=== reproducibility (NN Tanimoto / scaffold Jaccard) ===")
        print(rep.groupby("pair_type")[["nn_tanimoto", "scaffold_jaccard"]].agg(["mean", "std", "count"]).round(3).to_string())
    if "d_ad_low_vs_ctrl" in rt:
        d = rt["d_ad_low_vs_ctrl"].dropna()
        print(f"\nRL minus control low-AD share: mean {d.mean():+.3f}, range [{d.min():+.3f}, {d.max():+.3f}], n={len(d)}")
        print("-> ~0: drift comes from the Mol2Mol prior. Clearly >0: RL exploits the oracle -> AD-aware reward arm is required.")
    print("\nwrote", OUT)


if __name__ == "__main__":
    if a.stage == "worker":
        worker()
    elif a.stage == "control":
        control()
    elif a.stage == "analyze":
        analyze()
    else:
        control()
        analyze()
