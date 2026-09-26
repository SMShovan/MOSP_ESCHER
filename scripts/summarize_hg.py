#!/usr/bin/env python3
"""Summarize hsospBench --hg runs (see results/README.md).

Input: directories with one <kind>_<batch>_r<run>.csv (+ .log) per run,
each run = load + consecutive batches. Per (kind, batch size) and build it
reports medians over the runs of the per-run mean over the batches:
dynamic time (the paper's metric: ESCHER maintenance + unification + CSR
apply + update) and its stages, the static recompute, fallbacks and
iterations, and the end-to-end numbers of the [e2e] log line (load time,
sum of the dynamic times, process wall time).

  summarize_hg.py --build NAME DIR [--build NAME DIR ...] [--merge OUTDIR]

--merge writes one CSV per build (all rows, with a leading run column).
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


def load(dirname):
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
            with open(path, "w", newline="") as fh:
                w = None
                for k in keys:
                    for run, d in sorted(data[n].get(k, {}).items()):
                        for r in d["rows"]:
                            if w is None:
                                header = ["run"] + list(r.keys())
                                w = csv.DictWriter(fh, fieldnames=header)
                                w.writeheader()
                            w.writerow({"run": run, **r})


if __name__ == "__main__":
    main()
