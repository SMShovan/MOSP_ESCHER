#!/usr/bin/env python3
"""Generate every H-SOSP paper figure from the benchmark CSV.

Usage:
    python3 scripts/plot_results.py results/results.csv results/figures/

Follows the plotting conventions of "Our plotting style.ipynb": seaborn
barplots with the rocket palette and dodged hue groups, y-grid behind the
bars, framed legends, lineplots with 'o' markers for the del-vary sweeps,
stacked percentage bars for the phase breakdown, bar_label annotations on
speedup plots, figsize (6, 4), and PDF output with bbox_inches='tight'.

Every figure is derived from the CSV alone; figures whose experiment rows
are missing are skipped with a note, so partial runs still plot.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
})

FIGSIZE = (6, 4)
PALETTE = "rocket"

DATASET_ORDER = ["HG-S", "HG-M", "HG-L", "HG-XL", "HG-C", "SM-A", "SM-B"]


def human_batch(n):
    n = int(n)
    return f"{n // 1000}K" if n >= 1000 else str(n)


def dataset_order(df):
    present = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    extras = sorted(set(df["dataset"]) - set(present))
    return present + extras


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def agg_mean(df, keys, cols):
    return df.groupby(keys, as_index=False)[cols].mean(numeric_only=True)


# ---------------------------------------------------------------------------
# E1: time vs changed-batch size (one figure per batch kind)
# ---------------------------------------------------------------------------
def fig_time_vs_deltaE(df, outdir):
    e1 = df[df["experiment"] == "E1"]
    if e1.empty:
        print("  [skip] E1 rows missing (time_vs_DeltaE)")
        return
    for kind, tag in [("hyperedge", "he"), ("vertex", "vtx")]:
        sub = e1[e1["batch_kind"] == kind].copy()
        if sub.empty:
            continue
        sub["Changed Edges"] = sub["batch_size"].map(human_batch)
        order = dataset_order(sub)
        hue_order = [human_batch(b) for b in sorted(sub["batch_size"].unique())]
        data = agg_mean(sub, ["dataset", "Changed Edges"],
                        ["t_dynamic_total_ms"])
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.barplot(data=data, x="dataset", y="t_dynamic_total_ms",
                    hue="Changed Edges", order=order, hue_order=hue_order,
                    palette=PALETTE, dodge=True, ax=ax)
        style_axis(ax)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Time (ms)")
        ax.legend(title="Changed Hyperedges" if kind == "hyperedge"
                  else "Changed Incident Vertices",
                  loc="upper left", frameon=True)
        save(fig, outdir, f"time_vs_DeltaE_{tag}.pdf")


# ---------------------------------------------------------------------------
# Per-dataset dynamic vs static across batch sizes
# ---------------------------------------------------------------------------
def fig_base_vs_deltaE(df, outdir):
    e1 = df[(df["experiment"] == "E1") & (df["batch_kind"] == "hyperedge")]
    if e1.empty:
        print("  [skip] E1 rows missing (base_vs_DeltaE)")
        return
    for ds in dataset_order(e1):
        sub = e1[e1["dataset"] == ds].copy()
        rows = []
        for b in sorted(sub["batch_size"].unique()):
            bb = sub[sub["batch_size"] == b]
            rows.append({"Changed Edges": human_batch(b), "Mode": "H-SOSP update",
                         "Time (ms)": bb["t_dynamic_total_ms"].mean()})
            rows.append({"Changed Edges": human_batch(b), "Mode": "Static recompute",
                         "Time (ms)": bb["t_static_ms"].mean()})
        data = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.barplot(data=data, x="Changed Edges", y="Time (ms)", hue="Mode",
                    palette=PALETTE, dodge=True, ax=ax)
        style_axis(ax)
        ax.set_xlabel("Changed Hyperedges")
        ax.legend(loc="upper left", frameon=True)
        save(fig, outdir, f"{ds}_base_vs_DeltaE.pdf")


# ---------------------------------------------------------------------------
# E2: deletion-percentage sweep (per dataset, line plot)
# ---------------------------------------------------------------------------
def fig_del_vary(df, outdir):
    e2 = df[df["experiment"] == "E2"]
    if e2.empty:
        print("  [skip] E2 rows missing (del-vary)")
        return
    for ds in dataset_order(e2):
        sub = e2[e2["dataset"] == ds]
        rows = []
        for d in sorted(sub["del_pct"].unique()):
            bb = sub[sub["del_pct"] == d]
            rows.append({"Delete Percentage": d, "Mode": "H-SOSP update",
                         "Time (ms)": bb["t_dynamic_total_ms"].mean()})
            rows.append({"Delete Percentage": d, "Mode": "Static recompute",
                         "Time (ms)": bb["t_static_ms"].mean()})
        data = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.lineplot(data=data, x="Delete Percentage", y="Time (ms)",
                     hue="Mode", marker="o", markersize=8, ax=ax)
        style_axis(ax)
        ax.legend(frameon=True)
        save(fig, outdir, f"{ds}-del-vary.pdf")


# ---------------------------------------------------------------------------
# E3: scalability vs hypergraph size (bar with value labels)
# ---------------------------------------------------------------------------
def fig_size_scaling(df, outdir):
    e3 = df[df["experiment"] == "E3"].copy()
    if e3.empty:
        print("  [skip] E3 rows missing (time_vs_size)")
        return
    e3["Hyperedges"] = e3["m_hyperedges"].map(
        lambda m: f"{int(m) // 1000000}M" if m >= 1000000
        else f"{int(m) // 1000}K")
    order = [f"{int(m) // 1000000}M" if m >= 1000000 else f"{int(m) // 1000}K"
             for m in sorted(e3["m_hyperedges"].unique())]
    data = agg_mean(e3, ["Hyperedges"], ["t_dynamic_total_ms"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=data, x="Hyperedges", y="t_dynamic_total_ms",
                hue="Hyperedges", order=order, hue_order=order,
                palette=PALETTE, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", label_type="edge", padding=2,
                     fontsize=9)
    style_axis(ax)
    ax.set_xlabel("Hyperedge count")
    ax.set_ylabel("Time (ms)")
    save(fig, outdir, "time_vs_size.pdf")


# ---------------------------------------------------------------------------
# E4: cardinality sweep
# ---------------------------------------------------------------------------
def fig_cardinality(df, outdir):
    e4 = df[df["experiment"] == "E4"].copy()
    if e4.empty:
        print("  [skip] E4 rows missing (time_vs_cardinality)")
        return
    e4["Max Cardinality"] = e4["c_max"].astype(int)
    rows = []
    for c in sorted(e4["Max Cardinality"].unique()):
        bb = e4[e4["Max Cardinality"] == c]
        rows.append({"Max Cardinality": c, "Mode": "H-SOSP update",
                     "Time (ms)": bb["t_dynamic_total_ms"].mean()})
        rows.append({"Max Cardinality": c, "Mode": "Static recompute",
                     "Time (ms)": bb["t_static_ms"].mean()})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=data, x="Max Cardinality", y="Time (ms)", hue="Mode",
                palette=PALETTE, dodge=True, ax=ax)
    style_axis(ax)
    ax.set_xlabel("Maximum hyperedge cardinality")
    ax.legend(loc="upper left", frameon=True)
    save(fig, outdir, "time_vs_cardinality.pdf")


# ---------------------------------------------------------------------------
# E5: h2h density sweep (line, x = measured average degree)
# ---------------------------------------------------------------------------
def fig_density(df, outdir):
    e5 = df[df["experiment"] == "E5"].copy()
    if e5.empty:
        print("  [skip] E5 rows missing (time_vs_density)")
        return
    rows = []
    for ds in sorted(e5["dataset"].unique()):
        bb = e5[e5["dataset"] == ds]
        deg = bb["avg_h2h_deg"].mean()
        rows.append({"Avg h2h degree": deg, "Mode": "H-SOSP update",
                     "Time (ms)": bb["t_dynamic_total_ms"].mean()})
        rows.append({"Avg h2h degree": deg, "Mode": "Static recompute",
                     "Time (ms)": bb["t_static_ms"].mean()})
    data = pd.DataFrame(rows).sort_values("Avg h2h degree")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(data=data, x="Avg h2h degree", y="Time (ms)", hue="Mode",
                 marker="o", markersize=8, ax=ax)
    style_axis(ax)
    ax.legend(frameon=True)
    save(fig, outdir, "time_vs_density.pdf")


# ---------------------------------------------------------------------------
# E6 (derived): phase breakdown stacked percentage bars
# ---------------------------------------------------------------------------
def fig_stacked_phases(df, outdir):
    src = df[df["experiment"].isin(["E1", "E2"])]
    if src.empty:
        src = df
    if src.empty:
        print("  [skip] no rows for stacked_percentage")
        return
    phases = [("t_escher_ms", "ESCHER maintenance"),
              ("t_delta_ms", "Delta extraction"),
              ("t_csr_apply_ms", "CSR apply"),
              ("t_sosp_update_ms", "SOSP propagation")]
    order = dataset_order(src)
    shares = []
    for ds in order:
        bb = src[src["dataset"] == ds]
        total = sum(bb[c].mean() for c, _ in phases)
        if total <= 0:
            continue
        shares.append([bb[c].mean() / total * 100.0 for c, _ in phases])
    if not shares:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = sns.color_palette(PALETTE, len(phases))
    bottoms = [0.0] * len(shares)
    for pi, (_, label) in enumerate(phases):
        vals = [s[pi] for s in shares]
        ax.bar(order[: len(shares)], vals, bottom=bottoms, label=label,
               color=colors[pi])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    style_axis(ax)
    ax.set_ylabel("Share of update time (%)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)
    save(fig, outdir, "stacked_percentage.pdf")


# ---------------------------------------------------------------------------
# E7: change placement (random / targeted / near / far)
# ---------------------------------------------------------------------------
def fig_placement(df, outdir):
    e7 = df[df["experiment"] == "E7"].copy()
    if e7.empty:
        print("  [skip] E7 rows missing (placement)")
        return
    order = ["random", "targeted", "near", "far"]
    rows = []
    for pl in order:
        bb = e7[e7["placement"] == pl]
        if bb.empty:
            continue
        rows.append({"Placement": pl, "Mode": "H-SOSP update",
                     "Time (ms)": bb["t_dynamic_total_ms"].mean()})
        rows.append({"Placement": pl, "Mode": "Static recompute",
                     "Time (ms)": bb["t_static_ms"].mean()})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=data, x="Placement", y="Time (ms)", hue="Mode",
                palette=PALETTE, dodge=True, ax=ax)
    style_axis(ax)
    ax.set_xlabel("Changed-hyperedge placement")
    ax.legend(loc="upper left", frameon=True)
    save(fig, outdir, "placement_time.pdf")


# ---------------------------------------------------------------------------
# E8 (derived): speedup of the update over static recompute
# ---------------------------------------------------------------------------
def fig_speedup(df, outdir):
    e1 = df[(df["experiment"] == "E1") & (df["batch_kind"] == "hyperedge")]
    if e1.empty:
        print("  [skip] E1 rows missing (Speedup)")
        return
    sub = e1.copy()
    sub["Changed Edges"] = sub["batch_size"].map(human_batch)
    order = dataset_order(sub)
    hue_order = [human_batch(b) for b in sorted(sub["batch_size"].unique())]
    data = agg_mean(sub, ["dataset", "Changed Edges"], ["speedup"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=data, x="dataset", y="speedup", hue="Changed Edges",
                order=order, hue_order=hue_order, palette=PALETTE,
                dodge=True, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)
    style_axis(ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Speedup (recompute / update)")
    legend = ax.legend(title="Changed Edges", loc="upper right",
                       frameon=True)
    legend.get_frame().set_alpha(0.6)
    save(fig, outdir, "Speedup.pdf")
    avg = df[df["speedup"] > 0]["speedup"]
    if len(avg):
        print(f"  speedup: avg {avg.mean():.2f}x max {avg.max():.2f}x")


# ---------------------------------------------------------------------------
# E9 (derived): memory vs hypergraph size
# ---------------------------------------------------------------------------
def fig_memory(df, outdir):
    e3 = df[df["experiment"] == "E3"].copy()
    if e3.empty:
        print("  [skip] E3 rows missing (memory_plot)")
        return
    rows = []
    for m in sorted(e3["m_hyperedges"].unique()):
        bb = e3[e3["m_hyperedges"] == m]
        label = (f"{int(m) // 1000000}M" if m >= 1000000
                 else f"{int(m) // 1000}K")
        rows.append({"Hyperedges": label, "Component": "ESCHER structures",
                     "Memory (MB)": bb["escher_mb"].mean()})
        rows.append({"Hyperedges": label, "Component": "h2h CSR + SOSP state",
                     "Memory (MB)": bb["graph_mb"].mean()})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=data, x="Hyperedges", y="Memory (MB)", hue="Component",
                palette=PALETTE, dodge=True, ax=ax)
    style_axis(ax)
    ax.set_xlabel("Hyperedge count")
    ax.legend(loc="upper left", frameon=True)
    save(fig, outdir, "memory_plot.pdf")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    csv_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"plot_results: {len(df)} rows from {csv_path}")

    bad = df[df["correct"] == 0]
    if len(bad):
        print(f"  WARNING: {len(bad)} rows failed correctness; they are "
              "plotted anyway, fix before publishing")

    fig_time_vs_deltaE(df, outdir)
    fig_base_vs_deltaE(df, outdir)
    fig_del_vary(df, outdir)
    fig_size_scaling(df, outdir)
    fig_cardinality(df, outdir)
    fig_density(df, outdir)
    fig_stacked_phases(df, outdir)
    fig_placement(df, outdir)
    fig_speedup(df, outdir)
    fig_memory(df, outdir)
    print("plot_results: done")


if __name__ == "__main__":
    main()
