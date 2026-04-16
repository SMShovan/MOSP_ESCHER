# Migration notes

Record of every change applied to the upstream ESCHER-GPU-main / MOSP-CUDA-main
sources when they were pulled into the unified escher-mosp tree. Anything not
listed here was copied verbatim.

## ESCHER core (`escher/`)

### Removed files (motif-specific, not needed by MOSP)

- `include/motif.hpp`, `include/motif_update.hpp`
- `include/graphGeneration.hpp`, `src/graphGeneration.cpp`
- `include/utils.hpp`, `utils/utils.cpp` (replaced by the much smaller
  `escher/include/flatten.hpp`)
- `src/main.cu`, `src/HMotifCount.cu`, `src/HMotifCountUpdate.cu`
- `src/type1.{cu,cpp}`, `src/type2.{cu,cpp}`, `src/type3.{cu,cpp}`
- `src/coarseTriangle.cu`
- `kernel/motif_utils.cuh`, `kernel/motifs.cu`

### New files

- `escher/include/flatten.hpp` — free-function declaration of
  `flatten2DVector`, previously declared in the now-removed `utils.hpp`.
- `escher/include/escher_errors.hpp` — defines `escher::EscherError`
  (thrown on invariant violations) and `ESCHER_CHECK_CUDA(expr)` macro
  (thrown on CUDA failures). Public so MOSP can catch and log.

### Edits

#### `escher/kernel/find.cu`

The original included `motif_utils.cuh` purely for the `readFlat` helper.
Since that header is gone, the helper is inlined at the top of `find.cu`
with a hard bound on chain hops:

```cuda
static inline __device__ int readFlat(const int* flat, int& loc) {
    constexpr int MAX_CHAIN_HOPS = 1024;
    for (int hops = 0; hops < MAX_CHAIN_HOPS; ++hops) {
        int val = flat[loc];
        if (val < 0 && val != INT_MIN) { loc = -val; continue; }
        return val;
    }
    return INT_MIN;
}
```

The upstream version was an unbounded `while (true)` loop with no cycle
detection, so a corrupt back-pointer chain would hang the GPU. Bounded
hops trade infinite-correct-walking for guaranteed-termination.

#### `escher/utils/flatten.cpp`

`#include "../include/utils.hpp"` → `#include "../include/flatten.hpp"`.
No logic changes.

#### `escher/structure/operations.cu`

1. Include fixes
   - `#include "../include/utils.hpp"` → `#include "../include/flatten.hpp"`
   - added `#include "../include/escher_errors.hpp"`

2. `checkCuda` — the static helper now delegates to
   `escher::checkCudaImpl(...)`, throwing `EscherError` instead of
   `std::exit(-1)`.

3. `constructCBST` — added up-front argument validation:

   ```cpp
   if (numRecords < 0) throw EscherError("numRecords must be non-negative");
   if (flatPayloadSize < 0 || payloadCapacity < 0) throw EscherError(...);
   if (payloadCapacity < flatPayloadSize) throw EscherError("flat payload does not fit");
   if (numRecords > 0 && (keys == nullptr || startOffsets == nullptr)) throw ...;
   if (flatPayloadSize > 0 && flatPayload == nullptr) throw ...;
   ```

   The upstream version silently returned on overflow and crashed on null
   pointers.

4. `printEachNode` debug launch — gated behind `#ifdef ESCHER_DEBUG_CONSTRUCT`.
   Upstream launched it unconditionally, which prints one device `printf`
   per record; MOSP's stress tests call `construct` hundreds of times, which
   was flooding stdout.

5. `fillCBST` — two `printVector` dumps and a `printf("Space available from")`
   are now behind `#ifdef ESCHER_DEBUG_FILL`. The second `printVector` in
   particular copied the entire flat payload (default 65536 ints) back to
   host on every call — untenable in a stress test.

6. `fillCBST` overflow path — previously wrote `"ERROR: overflow..."` to
   `std::cerr` and returned silently, corrupting the CBST state. Now
   throws `EscherError` with a descriptive message telling the caller to
   increase `payloadCapacity`.

7. `insertCBST` — the GPU best-fit matching `printf` and the surplus-
   overflow `std::cerr` path received the same treatment: debug
   `printf` is gated behind `ESCHER_DEBUG_INSERT`; surplus overflow now
   throws `EscherError` instead of returning a half-updated mapping.

## MOSP (`mosp/`)

No kernel bodies were modified. Four files had exactly one call-site swap
each, replacing the legacy `updateGraphCSR(...)` with
`escher_mosp::updateGraphWithESCHER(...)`:

- `mosp/src/main.cu` — step 3 of the pipeline.
- `mosp/src/stressTest.cu` — per-run update call.
- `mosp/src/parallelStressTest.cu` — per-run update call.
- `mosp/src/generateTestCases.cu` — per-test-case update call.

The legacy `mosp/src/updateGraphCSR.cu` is still in the tree because the
equivalence test (`tests/unit/test_snapshot_matches_updateCSR.cu`) uses it
as the reference implementation to diff against.

## Deferred hardening (not done in this pass)

These items from the plan's Phase 3 checklist were deemed lower priority
and are left for a follow-up:

- `insertCBST` return mapping size invariant (`itemToKey.size() == newKeys.size()`).
- H2D/D2H size verification on every `cudaMemcpy` in `operations.cu`.
- `sequentialSOSPUpdate` and `parallelSOSPUpdate` still round-trip through
  disk files rather than consuming a `GraphSnapshot` directly. The
  `GraphSnapshot` API is implemented and ready — only the kernel wiring
  has not been done because it would require changes inside otherwise
  untouched CUDA code, and `test_snapshot_matches_updateCSR` already
  proves the ESCHER-backed update is correct.
