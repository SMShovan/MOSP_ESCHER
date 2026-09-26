#!/usr/bin/env python3
"""Summarize hsospBench --hg runs (see results/README.md).

Input per build, either
  - a directory with one <kind>_<batch>_r<run>.csv (+ .log) per run, each
    run = load + consecutive batches, as hsospBench writes them; or
  - a merged CSV (all rows with a leading run column, as --merge writes
    them, e.g. results/data/dblp_final.csv) and, next to it, the
    end-to-end numbers of every run in <name>_e2e.csv (kind, batch, run,
    load_ms, batches, dynamic_ms, static_ms, wall_ms).
Per (kind, batch size) and build it reports medians over the runs of the
per-run mean over the batches: dynamic time (the paper's metric: ESCHER
maintenance + unification + CSR apply + update) and its stages, the
static recompute, fallbacks and iterations, and the end-to-end numbers of
the [e2e] log line (load time, sum of the dynamic times, process wall
time).

  summarize_hg.py --build NAME PATH [--build NAME PATH ...] [--merge OUTDIR]

--merge writes, per build, NAME.csv (all rows, with a leading run column)
and NAME_e2e.csv, the two files a merged-CSV build reads.
"""
import argparse
import csv
import os
import re
import statistics as st
from collections import defaultdict

FILE_RE = re.compile(r"(hyperedge|vertex)_(\d+)_r(\d+)\.csv$")
E2E_RE = re.compile(r"\[e2e\].*load (\d+) ms, (\d+) batches: dynamic (\d+) ms, "
                    r"static (\d+) ms, wall (\d+) ms")


E2E_FIELDS = ["kind", "batch", "run", "load_ms", "batches", "dynamic_ms",
              "static_ms", "wall_ms"]


def e2e_path(csv_path):
    return csv_path[:-len(".csv")] + "_e2e.csv"


def load_merged(path):
    """A merged CSV plus its _e2e.csv (see the module docstring)."""
    runs = defaultdict(dict)   # (kind, size) -> run -> dict
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["batch_kind"], int(r["batch_size"]))
            run = int(r["run"])
            d = runs[key].setdefault(run, {"rows": [], "e2e": None,
                                           "file": path})
            d["rows"].append({k: v for k, v in r.items() if k != "run"})
    if os.path.exists(e2e_path(path)):
        with open(e2e_path(path)) as fh:
            for r in csv.DictReader(fh):
                key = (r["kind"], int(r["batch"]))
                d = runs.get(key, {}).get(int(r["run"]))
                if d is not None:
                    d["e2e"] = [int(r[k]) for k in E2E_FIELDS[3:]]
    return runs


def load(dirname):
    if os.path.isfile(dirname):
        return load_merged(dirname)
    runs = defaultdict(dict)   # (kind, size) -> run -> dict
    for f in sorted(os.listdir(dirname)):
        m = FILE_RE.match(f)
        if not m:
            continue
        kind, size, run = m.group(1), int(m.group(2)), int(m.group(3))
        with open(os.path.join(dirname, f)) as fh:
            rows = list(csv.DictReader(fh))
        e2e = None
        logf = os.path.join(dirname, f[:-4] + ".log")
        if os.path.exists(logf):
            with open(logf) as fh:
                for line in fh:
                    mm = E2E_RE.search(line)
                    if mm:
                        e2e = [int(x) for x in mm.groups()]
        runs[(kind, size)][run] = {"rows": rows, "e2e": e2e, "file": f}
    return runs


def fnum(r, k, default=0.0):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else default


def summarize(runs):
    out = {}
    for key, per in runs.items():
        vals = defaultdict(list)
        fallbacks = batches = 0
        iters = []
        verified = 0
        for run, d in per.items():
            rows = d["rows"]
            if not rows:
                continue
            for k in ("t_escher_ms", "t_delta_ms", "t_csr_apply_ms",
                      "t_sosp_update_ms", "t_dynamic_total_ms",
                      "t_static_ms"):
                vals[k].append(st.mean(fnum(r, k) for r in rows))
            fallbacks += sum(int(fnum(r, "fallback")) for r in rows)
            batches += len(rows)
            iters += [int(fnum(r, "iters")) for r in rows]
            verified += sum(int(fnum(r, "verified")) for r in rows)
            if d["e2e"]:
                load_ms, nb, dyn, stat, wall = d["e2e"]
                vals["load_ms"].append(load_ms)
                vals["e2e_ms"].append(load_ms + dyn)
                vals["wall_ms"].append(wall)
        med = {k: st.median(v) for k, v in vals.items() if v}
        med["runs"] = len(per)
        med["batches"] = batches
        med["fallbacks"] = fallbacks
        med["iters"] = (min(iters), max(iters)) if iters else (0, 0)
        med["verified"] = verified
        out[key] = med
    return out


def fmt(x):
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs=2, action="append", required=True,
                    metavar=("NAME", "DIR"))
    ap.add_argument("--merge", metavar="OUTDIR")
    a = ap.parse_args()
    data = {name: load(d) for name, d in a.build}
    summ = {name: summarize(r) for name, r in data.items()}
    names = [n for n, _ in a.build]
    keys = sorted({k for s in summ.values() for k in s},
                  key=lambda k: (k[0], k[1]))
    print("| batch | build | dynamic ms | ESCHER | unification | CSR | "
          "update | static ms | static/dynamic | fallbacks | iters | "
          "load s | load+dynamic s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        for n in names:
            s = summ[n].get(k)
            if not s:
                continue
            ratio = s["t_static_ms"] / s["t_dynamic_total_ms"]
            print(f"| {k[0]} {k[1] // 1000}K | {n} | "
                  f"{fmt(s['t_dynamic_total_ms'])} | {fmt(s['t_escher_ms'])} | "
                  f"{fmt(s['t_delta_ms'])} | {fmt(s['t_csr_apply_ms'])} | "
                  f"{fmt(s['t_sosp_update_ms'])} | {fmt(s['t_static_ms'])} | "
                  f"{ratio:.2f}x | {s['fallbacks']}/{s['batches']} | "
                  f"{s['iters'][0]}-{s['iters'][1]} | "
                  f"{s.get('load_ms', 0) / 1000:.1f} | "
                  f"{s.get('e2e_ms', 0) / 1000:.2f} |")
    if a.merge:
        os.makedirs(a.merge, exist_ok=True)
        for n in names:
            path = os.path.join(a.merge, f"{n}.csv")
            header = None
            with open(path, "w", newline="") as fh, \
                    open(e2e_path(path), "w", newline="") as eh:
                w = None
                ew = csv.writer(eh)
                ew.writerow(E2E_FIELDS)
                for k in keys:
                    for run, d in sorted(data[n].get(k, {}).items()):
                        for r in d["rows"]:
                            if w is None:
                                header = ["run"] + list(r.keys())
                                w = csv.DictWriter(fh, fieldnames=header)
                                w.writeheader()
                            w.writerow({"run": run, **r})
                        if d["e2e"]:
                            ew.writerow([k[0], k[1], run] + d["e2e"])


if __name__ == "__main__":
    main()
