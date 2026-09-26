#!/usr/bin/env python3
"""Convert a timestamped simplicial dataset of Benson et al. to a hypergraph
file for hsospBench --hg (one hyperedge per line).

The datasets (e.g. coauth-DBLP-full, coauth-MAG-Geology-full) list every
simplex once per timestamp: PREFIX-nverts.txt holds the number of vertices
of each simplex, PREFIX-simplices.txt their vertex ids, one per line. The
paper uses each distinct simplex once, so this script

  1. reads the simplices in file order, merges repeated vertices within a
     simplex and sorts its vertices,
  2. keeps the first occurrence of every distinct vertex set,
  3. renumbers the vertices 1..V in increasing order of their original id
     (order-preserving, so it does not change what hsospBench loads), and
  4. writes one simplex per line, vertices sorted, separated by spaces.

hsospBench's loader does not remove repeated hyperedges (every line
becomes a hyperedge), so step 2 must happen here. coauth-DBLP-full:
3,700,681 timestamped simplices become 2,467,389 lines (1,930,378
vertices); hsospBench --maxcard 25 then keeps 2,466,792 hyperedges.

  benson_to_hg.py PREFIX OUT.hg     (e.g. coauth-DBLP-full/coauth-DBLP-full)
"""
import sys


def read_ints(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield int(line)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    prefix, out = sys.argv[1], sys.argv[2]
    nverts = list(read_ints(prefix + "-nverts.txt"))
    pins = read_ints(prefix + "-simplices.txt")
    seen = set()
    edges = []
    for k in nverts:
        e = tuple(sorted({next(pins) for _ in range(k)}))
        if e not in seen:
            seen.add(e)
            edges.append(e)
    if next(pins, None) is not None:
        sys.exit("more vertex ids in the simplices file than nverts lists")
    remap = {v: i + 1 for i, v in
             enumerate(sorted({v for e in edges for v in e}))}
    with open(out, "w") as fh:
        for e in edges:
            fh.write(" ".join(str(remap[v]) for v in e) + "\n")
    print(f"{len(nverts)} simplices, {len(edges)} distinct, "
          f"{len(remap)} vertices -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
