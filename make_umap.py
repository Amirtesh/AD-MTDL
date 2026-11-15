#!/usr/bin/env python
"""
make_umap.py - UMAP figures, local. One embedding (fit on training chemistry), then:
  * every panel as its own PDF + TIFF  (Dovepress: one file per figure, no titles)
  * one combined multi-panel figure with panel letters A-K

  A      : training space + the 15 generation seeds
  B-F    : RL-generated molecules per campaign, coloured by applicability-domain (AD) level
  G-K    : no-RL control molecules per campaign, same colouring

  pip install umap-learn rdkit pandas matplotlib numpy
  python make_umap.py                 # fit + plot (coords cached to umap_coords.csv)
  python make_umap.py --replot        # restyle only, reuses cached coordinates
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="campaigns_v2")
ap.add_argument("--ctrl", default="control_v2")
ap.add_argument("--seeds", default="campaign_seeds_v2.csv")
ap.add_argument("--train", default="dataset_after_cleaning.csv")
ap.add_argument("--out", default="figures_umap")
ap.add_argument("--nfit", type=int, default=20000, help="training molecules used to fit UMAP")
ap.add_argument("--nrl", type=int, default=6000, help="RL molecules plotted per campaign (pooled over runs)")
ap.add_argument("--nctl", type=int, default=6000, help="control molecules plotted per campaign")
ap.add_argument("--neighbors", type=int, default=30)
ap.add_argument("--min-dist", type=float, default=0.3)
ap.add_argument("--replot", action="store_true")
ap.add_argument("--rng", type=int, default=0)
a = ap.parse_args()

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
OUT = Path(a.out); OUT.mkdir(exist_ok=True)
CACHE = OUT / "umap_coords.csv"
RANK = {"low": 0, "medium": 1, "high": 2}
INV = {0: "low", 1: "medium", 2: "high"}
ADCOL = {"low": "#D55E00", "medium": "#E69F00", "high": "#009E73"}


def smiles_col(df):
    return next(c for c in df.columns if norm(c) in ("smiles", "canonicalsmiles"))


def worst_ad(df, T):
    r = []
    for t in T:
        c = next(c for c in df.columns if norm(c) == norm(t) + "adlevel")
        r.append(df[c].astype(str).str.lower().map(RANK))
    return pd.concat(r, axis=1).min(axis=1).map(INV)


def collect():
    rng = np.random.default_rng(a.rng)
    tr = pd.read_csv(a.train)
    tr = tr[[smiles_col(tr)]].rename(columns=lambda c: "smiles").dropna().drop_duplicates()
    tr = tr.sample(min(a.nfit, len(tr)), random_state=a.rng)
    tr.insert(0, "set", "train"); tr["campaign"] = ""; tr["ad"] = ""

    sd = pd.read_csv(a.seeds)
    cc = next(c for c in sd.columns if norm(c) in ("campaign", "campaignname"))
    sc = next(c for c in sd.columns if "smiles" in c.lower())
    seeds = pd.DataFrame({"set": "seed", "campaign": sd[cc], "smiles": sd[sc], "ad": ""})

    rl, ct = [], []
    for cdir in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        runs = [r for r in sorted(cdir.glob("s*_r*")) if (r / "scored_molecules.csv").exists()]
        if not runs:
            continue
        T = json.load(open(runs[0] / "run_config.json"))["targets"]
        T = T if isinstance(T, list) else list(T)
        for r in runs:
            d = pd.read_csv(r / "scored_molecules.csv", low_memory=False)
            if "valid" in d:
                d = d[d["valid"].astype(str).str.lower().isin(["true", "1", "1.0"])]
            d = d.drop_duplicates(smiles_col(d))
            d = d.assign(ad=worst_ad(d, T)).dropna(subset=["ad"])
            d = d.sample(min(len(d), a.nrl // len(runs)), random_state=a.rng)
            rl.append(pd.DataFrame({"set": "rl", "campaign": cdir.name, "smiles": d[smiles_col(d)].values, "ad": d["ad"].values}))
        for f in sorted(Path(a.ctrl).glob(f"{cdir.name}_s*/scored_control.csv")):
            d = pd.read_csv(f, low_memory=False)
            d = d[d["valid"].astype(str).str.lower().isin(["true", "1", "1.0"])].drop_duplicates("smiles")
            nseed = len(list(Path(a.ctrl).glob(f"{cdir.name}_s*/scored_control.csv")))
            d = d.assign(ad=worst_ad(d, T)).dropna(subset=["ad"])
            d = d.sample(min(len(d), a.nctl // nseed), random_state=a.rng)
            ct.append(pd.DataFrame({"set": "ctrl", "campaign": cdir.name, "smiles": d["smiles"].values, "ad": d["ad"].values}))
    return tr, seeds, pd.concat(rl, ignore_index=True), pd.concat(ct, ignore_index=True)


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        ok.append(m is not None)
        if m is not None:
            X.append(gen.GetFingerprintAsNumPy(m).astype(bool))
    return np.vstack(X), np.array(ok)


def embed():
    import umap
    tr, seeds, rl, ct = collect()
    Xt, okt = fingerprints(tr["smiles"]); tr = tr[okt].reset_index(drop=True)
    print("fitting UMAP on", len(tr), "training molecules (Jaccard on Morgan r2/2048) ...", flush=True)
    reducer = umap.UMAP(n_neighbors=a.neighbors, min_dist=a.min_dist, metric="jaccard", random_state=a.rng)
    tr[["x", "y"]] = reducer.fit_transform(Xt)
    parts = [tr]
    for df in (seeds, rl, ct):
        X, ok = fingerprints(df["smiles"]); df = df[ok].reset_index(drop=True)
        print("transforming", df["set"].iloc[0], len(df), flush=True)
        df[["x", "y"]] = reducer.transform(X)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out


# ------------------------------- plotting -------------------------------
def draw(ax, c, key, camp=None, letter=None, legend=True):
    tr = c[c["set"] == "train"]
    ax.scatter(tr.x, tr.y, s=1.2, c="#BDBDBD", alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    handles = [Line2D([], [], marker="o", ls="", c="#BDBDBD", ms=4, label="Training set")]
    if key == "seeds":
        sd = c[c["set"] == "seed"]
        cm = plt.get_cmap("tab10")
        for i, (name, g) in enumerate(sd.groupby("campaign")):
            ax.scatter(g.x, g.y, s=70, marker="*", color=cm(i), edgecolor="k", linewidths=0.5, zorder=3)
            handles.append(Line2D([], [], marker="*", ls="", color=cm(i), mec="k", ms=9, label=f"{name} seeds"))
    else:
        d = c[(c["set"] == key) & (c["campaign"] == camp)]
        for lvl in ("high", "medium", "low"):
            g = d[d["ad"] == lvl]
            ax.scatter(g.x, g.y, s=2.5, color=ADCOL[lvl], alpha=0.6, linewidths=0, rasterized=True, zorder=2)
            handles.append(Line2D([], [], marker="o", ls="", color=ADCOL[lvl], ms=4, label=f"{lvl.capitalize()} AD"))
        s = c[(c["set"] == "seed") & (c["campaign"] == camp)]
        ax.scatter(s.x, s.y, s=80, marker="*", color="k", edgecolor="white", linewidths=0.6, zorder=4)
        handles.append(Line2D([], [], marker="*", ls="", color="k", mec="white", ms=9, label="Seeds"))
        ax.text(0.03, 0.96, camp.replace("_", "-"), transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    if letter:
        ax.text(-0.04, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    if legend:
        ax.legend(handles=handles, fontsize=6, loc="lower right", frameon=True, framealpha=0.8, markerscale=0.9)
    return handles


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


coords = pd.read_csv(CACHE).fillna({"ad": "", "campaign": ""}) if (a.replot and CACHE.exists()) else embed()
coords["ad"] = coords["ad"].astype(str); coords["campaign"] = coords["campaign"].astype(str)
pad = 0.04
XL = (coords.x.min() - pad * np.ptp(coords.x), coords.x.max() + pad * np.ptp(coords.x))
YL = (coords.y.min() - pad * np.ptp(coords.y), coords.y.max() + pad * np.ptp(coords.y))
camps = sorted(c for c in coords.loc[coords["set"] == "rl", "campaign"].unique())

panels = [("A_seeds", "seeds", None)]
panels += [(f"{chr(66 + i)}_RL_{c}", "rl", c) for i, c in enumerate(camps)]
panels += [(f"{chr(66 + len(camps) + i)}_control_{c}", "ctrl", c) for i, c in enumerate(camps)]

for name, key, camp in panels:                                   # individual files
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    draw(ax, coords, key, camp, letter=None)
    save(fig, f"umap_{name}")

n = len(camps)                                                   # combined figure
fig = plt.figure(figsize=(3.4 * n, 3.3 * 3))
gs = GridSpec(3, n, figure=fig, hspace=0.22, wspace=0.12)
ax = fig.add_subplot(gs[0, 0]); h = draw(ax, coords, "seeds", letter="A", legend=False)
lg = fig.add_subplot(gs[0, 1]); lg.axis("off")
lg.legend(handles=h + [Line2D([], [], marker="o", ls="", color=ADCOL[l], ms=5, label=f"{l.capitalize()} AD") for l in ("high", "medium", "low")]
          + [Line2D([], [], marker="*", ls="", color="k", mec="white", ms=10, label="Seed (RL/control panels)")],
          loc="center left", fontsize=8, frameon=False)
for i, cp in enumerate(camps):
    draw(fig.add_subplot(gs[1, i]), coords, "rl", cp, letter=chr(66 + i), legend=False)
    draw(fig.add_subplot(gs[2, i]), coords, "ctrl", cp, letter=chr(66 + n + i), legend=False)
save(fig, "umap_combined")
print("wrote", len(panels), "individual panels + combined to", OUT)
