#!/usr/bin/env bash
#
# Test suite for escher-mosp (invoked by `make test`, which builds first).
#
# Every case prints PASS or FAIL; the script exits non-zero if any case
# fails. Binaries run inside a temporary directory because the MOSP
# harnesses write their graphs and change files into the working directory.
#
# Cases listed in XFAIL are known defects that a later change fixes: they
# report XFAIL while they still fail and count as a failure (XPASS) once
# they pass, so the list must shrink together with the fixes.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/escher_mosp_tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

XFAIL=" "
FAILURES=0

# run_case <name> <command...>: the command must exit with status 0.
run_case() {
    local name=$1; shift
    local log="$WORK/$name.log"
    local status=0
    # The outer subshell keeps bash's "Aborted"/"Segmentation fault" job
    # messages out of the report; the status is still that of the command.
    ( ( cd "$WORK" && "$@" ) >"$log" 2>&1; exit $? ) 2>/dev/null || status=$?
    local expected_fail=0
    [[ "$XFAIL" == *" $name "* ]] && expected_fail=1
    if [ "$status" -eq 0 ]; then
        if [ "$expected_fail" -eq 1 ]; then
            echo "XPASS $name (listed in XFAIL; remove it from the list)"
            FAILURES=$((FAILURES + 1))
        else
            echo "PASS  $name"
        fi
    elif [ "$expected_fail" -eq 1 ]; then
        echo "XFAIL $name (exit status $status)"
    else
        echo "FAIL  $name (exit status $status)"
        tail -n 15 "$log"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "=== ESCHER CBST: contents vs a host model after every operation ==="
run_case test_cbst_smoke                 "$BIN/test_cbst_smoke"
for scenario in scale reuse terminator erase surplus bestfit unfill-chain \
                random; do
    run_case "cbst_$scenario" "$BIN/test_cbst_ops" "$scenario"
done

echo "=== unit tests ==="
run_case test_dynamicgraph_roundtrip     "$BIN/test_dynamicgraph_roundtrip"
run_case test_snapshot_matches_updateCSR "$BIN/test_snapshot_matches_updateCSR"
run_case test_h2h_construction           "$BIN/test_h2h_construction"
run_case test_h2h_delta                  "$BIN/test_h2h_delta"
run_case test_hsosp_matches_dijkstra     "$BIN/test_hsosp_matches_dijkstra"
run_case test_hsosp_scale                "$BIN/test_hsosp_scale"

echo "=== H-SOSP host core without a GPU (tests/local) ==="
run_case local_tests bash -c "make -s -C '$ROOT/tests/local' && '$ROOT/tests/local/local_tests'"

echo "=== H-SOSP randomized stress (pipeline vs independent oracle) ==="
run_case hsospStress        "$BIN/hsospStress" --configs 50
run_case hsospStress_escher "$BIN/hsospStress" --configs 20 --seed 11 \
    --check-escher
run_case hsospBench_smoke   "$BIN/hsospBench" --suite smoke --reps 1 \
    --check-escher --out hsospBench_smoke.csv
# A random 30,000-hyperedge file (one hyperedge per line) through the
# real-hypergraph mode, every batch checked by the oracle and the ESCHER
# contents compared with the host model (the paper's insertions can
# repeat a vertex, which the synthetic generator never does).
awk 'BEGIN { srand(7); for (i = 0; i < 30000; i++) { k = 1 + int(rand() * 6);
     line = ""; for (j = 0; j < k; j++) line = line (j ? " " : "") \
     int(rand() * 20000); print line } }' > "$WORK/random.hg"
for kind in hyperedge vertex; do
    run_case "hsospBench_real_$kind" "$BIN/hsospBench" --hg "$WORK/random.hg" \
        --kind "$kind" --batch 2000 --batches 3 --verify all --check-escher \
        --out "hsospBench_real.csv"
done

# Usage errors: a malformed number is exit status 2 (not an abort), and a
# missing input file leaves no CSV behind.
run_case hsospBench_cli bash -c '
    for bad in "--batch abc" "--batch 12abc" "--reps 99999999999" \
               "--seed -1" "--del 5x"; do
        "$0" --hg "$1" $bad --out cli.csv; [ $? -eq 2 ] || exit 1
    done
    "$0" --hg missing.hg --out cli.csv; [ $? -eq 1 ] || exit 1
    [ ! -e cli.csv ]' "$BIN/hsospBench" "$WORK/random.hg"

echo "=== MOSP pipeline and stress tests ==="
# Fixed seeds keep the suite reproducible; seed 6 of parallelStressTest hit
# the original MOSP count-to-infinity defect (run 195).
run_case test_mosp_update        "$BIN/test_mosp_update"
run_case main                    "$BIN/main"
run_case stressTest              "$BIN/stressTest" 1 200
run_case parallelStressTest      "$BIN/parallelStressTest" 1 200
run_case parallelStressTest_seed6 "$BIN/parallelStressTest" 6 200

if [ "$FAILURES" -eq 0 ]; then
    echo "=== all tests passed ==="
else
    echo "=== $FAILURES test case(s) failed ==="
fi
[ "$FAILURES" -eq 0 ]
