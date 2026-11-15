#!/usr/bin/env python
"""launch_parallel.py - run many optimize.py campaigns concurrently on ONE server: one pinned CPU slot, one port, one output
directory per campaign. Each job is an independent process (its own scoring server + REINVENT), so nothing is shared.

  python launch_parallel.py --seeds campaign_seeds_v2.csv --rng 1 2 --steps 2000 --cores 0-95 --slot 12 --shard 0,1/3 --dry-run
Two machines of 96 and 48 cores (8 and 4 slots): run `--shard 0,1/3` on the big one and `--shard 2/3` on the small one.
Finished jobs (top_compounds.csv present) are skipped, so the launcher can simply be restarted."""
import argparse, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="campaign_seeds_v2.csv"); ap.add_argument("--only", nargs="+", default=None)
ap.add_argument("--rng", nargs="+", type=int, default=[1, 2]); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--cores", default="0-95", help='e.g. "0-95" or "0-47,96-143"'); ap.add_argument("--slot", type=int, default=12)
ap.add_argument("--reinvent-threads", type=int, default=10); ap.add_argument("--server-threads", type=int, default=2)
ap.add_argument("--shard", default=None, help='"0,1/3" = keep jobs whose index mod 3 is 0 or 1')
ap.add_argument("--base-port", type=int, default=8900); ap.add_argument("--out-root", default="campaigns_v2")
ap.add_argument("--reinvent-env", default="reinvent4"); ap.add_argument("--ad-env", default="ad_mtdl")
ap.add_argument("--ckpt", default="AD_MTDL_best.ckpt"); ap.add_argument("--stagger", type=float, default=20.0)
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--fake", action="store_true", help="scheduler self-test only")
args = ap.parse_args()


def parse_cores(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b or a) + 1))
    return out


seeds = pd.read_csv(args.seeds)
if args.only:
    seeds = seeds[seeds["campaign"].isin(args.only)]
jobs = []
for rng in args.rng:                                     # interleave campaigns so partial results stay balanced
    for k in sorted(seeds["seed_idx"].unique()):
        for _, r in seeds[seeds["seed_idx"] == k].iterrows():
            jobs.append({"campaign": r["campaign"], "targets": str(r["targets"]).split(), "smiles": r["smiles"], "seed_idx": int(k), "rng": rng})
for i, j in enumerate(jobs):                             # global index -> unique port and stable shard assignment
    j["idx"], j["port"] = i, args.base_port + i
    j["out"] = Path(args.out_root) / j["campaign"] / f"s{j['seed_idx']}_r{j['rng']}"
if args.shard:
    keep, _, mod = args.shard.partition("/"); keep, mod = {int(x) for x in keep.split(",")}, int(mod)
    jobs = [j for j in jobs if j["idx"] % mod in keep]
pending = [j for j in jobs if not (j["out"] / "top_compounds.csv").exists()]

cores = parse_cores(args.cores)
slots = [cores[i:i + args.slot] for i in range(0, len(cores) - args.slot + 1, args.slot)]
cs = lambda c: f"{c[0]}-{c[-1]}"


def build_cmd(j, slot):
    if args.fake:
        return ["bash", "-c", f"mkdir -p {j['out']} && sleep 3 && touch {j['out']}/top_compounds.csv"]
    pin = ["taskset", "-c", cs(slot)] if shutil.which("taskset") else []
    return pin + ["conda", "run", "--no-capture-output", "-n", args.reinvent_env, "python", "optimize.py",
                  "--smiles", j["smiles"], "--targets", *j["targets"], "--mode", "scaffold_hop", "--steps", str(args.steps),
                  "--batch_size", "64", "--top_k", "1000", "--min_score", "0.40", "--n_cpus", str(args.slot),
                  "--reinvent_threads", str(args.reinvent_threads), "--server_threads", str(args.server_threads),
                  "--server_port", str(j["port"]), "--rng_seed", str(j["rng"]), "--ad_mtdl_env", args.ad_env,
                  "--ckpt", args.ckpt, "--output", str(j["out"])]


print(f"{len(jobs)} jobs in this shard, {len(pending)} pending | {len(slots)} slots of {args.slot} cores")
if args.dry_run:
    for n, j in enumerate(pending):
        print(f"[slot {n % len(slots)}] port {j['port']}: " + " ".join(build_cmd(j, slots[n % len(slots)])))
    sys.exit(0)

Path(args.out_root).mkdir(parents=True, exist_ok=True)
status = open(Path(args.out_root) / "launcher_status.tsv", "a")
running, free = {}, list(range(len(slots)))
def kill_all(*_):
    for p, *_ in running.values():
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    sys.exit(1)
signal.signal(signal.SIGINT, kill_all); signal.signal(signal.SIGTERM, kill_all)

while pending or running:
    while pending and free:
        s, j = free.pop(0), pending.pop(0)
        j["out"].mkdir(parents=True, exist_ok=True)
        fh = open(j["out"] / "launcher_stdout.log", "w")
        p = subprocess.Popen(build_cmd(j, slots[s]), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        running[s] = (p, j, time.time(), fh)
        print(time.strftime("%H:%M:%S"), f"started {j['campaign']}/s{j['seed_idx']}_r{j['rng']} slot {s} cores {cs(slots[s])} port {j['port']}", flush=True)
        time.sleep(args.stagger)
    time.sleep(10)
    for s, (p, j, t0, fh) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        fh.close(); del running[s]; free.append(s)
        ok = rc == 0 and (j["out"] / "top_compounds.csv").exists()
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{j['campaign']}\ts{j['seed_idx']}_r{j['rng']}\t{'OK' if ok else 'FAILED rc=' + str(rc)}\t{(time.time() - t0) / 60:.1f} min"
        print(msg, flush=True); status.write(msg + "\n"); status.flush()
print("all jobs in this shard finished; see", Path(args.out_root) / "launcher_status.tsv")
