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

XFAIL=" parallelStressTest_seed6 "
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

echo "=== H-SOSP randomized stress (pipeline vs independent oracle) ==="
run_case hsospStress        "$BIN/hsospStress" --configs 50
run_case hsospStress_escher "$BIN/hsospStress" --configs 20 --seed 11 \
    --check-escher

echo "=== MOSP pipeline and stress tests ==="
# Fixed seeds keep the suite reproducible; seed 6 of parallelStressTest hits
# the MOSP count-to-infinity defect (run 195).
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
