# =============================================================================
# escher-mosp unified build
# =============================================================================
#
# Targets:
#   all                 Build libescher_core.a + main + stressTest +
#                       parallelStressTest + unit tests.
#   clean               Remove build/ and bin/.
#   run                 Build and run the main pipeline.
#   stressTest          Build the sequential stress test.
#   parallelStressTest  Build the CUDA parallel stress test.
#   tests               Build the unit test binaries.
#   docs                Generate Doxygen HTML into docs/html.
#   syntax-check        Preprocess every TU without linking (works on macOS
#                       without a GPU, provided nvcc is installed).
#   test                Build everything and run tests/run_tests.sh (unit
#                       tests, randomized stress harnesses, MOSP pipeline);
#                       exits non-zero on any failure.
#
# Overridable on the command line:
#   CUDA_ARCH  target GPU architecture (default sm_86, RTX A5000 / A40;
#              use sm_80 for A100, sm_90 for H100)
#   NVCC       nvcc to use (default: nvcc on PATH, else /usr/local/cuda)
#   OPT        optimization level for host and device code (default -O3)
#
# Every object depends on the headers it includes (-MMD -MP dependency
# files) and on the compiler and flags in use (build/.flags), so changing a
# header, CUDA_ARCH or OPT rebuilds exactly what is affected.
#
# =============================================================================

NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_ARCH ?= sm_86
OPT       ?= -O3

INCLUDES  := -Iescher/include -Iescher/kernel -Igraph/include -Imosp/headers \
             -Ihypergraph/include -Ihsosp/include

# -lineinfo keeps source correlation for compute-sanitizer / Nsight without
# changing code generation. Host code uses OpenMP (test oracles, bulk
# construction).
NVFLAGS   := -std=c++17 $(OPT) -lineinfo --extended-lambda -arch=$(CUDA_ARCH) \
             -Xcompiler -fopenmp $(INCLUDES)
DEPFLAGS  := -MMD -MP

BUILDDIR  := build
BINDIR    := bin
LIBDIR    := $(BUILDDIR)/lib

# -----------------------------------------------------------------------------
# Source lists
# -----------------------------------------------------------------------------

ESCHER_CU_SRCS  := $(wildcard escher/structure/*.cu) $(wildcard escher/kernel/*.cu)
ESCHER_CPP_SRCS := $(wildcard escher/utils/*.cpp)
ESCHER_OBJS     := $(ESCHER_CU_SRCS:%=$(BUILDDIR)/%.o) \
                   $(ESCHER_CPP_SRCS:%=$(BUILDDIR)/%.o)

GRAPH_CU_SRCS   := $(wildcard graph/src/*.cu)
# Adapter-only C++ sources — no MOSP dependencies. Usable by tests that
# need DynamicGraph / GraphSnapshot but not updateGraphWithESCHER.
GRAPH_CORE_CPP_SRCS := graph/src/DynamicGraph.cpp
# C++ sources that DO depend on MOSP (read.cuh etc.). Kept in a separate
# object set so tests don't pull in unresolved MOSP symbols.
GRAPH_MOSP_CPP_SRCS := graph/src/updateGraphWithESCHER.cpp

GRAPH_CORE_OBJS := $(GRAPH_CU_SRCS:%=$(BUILDDIR)/%.o) \
                   $(GRAPH_CORE_CPP_SRCS:%=$(BUILDDIR)/%.o)
GRAPH_MOSP_OBJS := $(GRAPH_MOSP_CPP_SRCS:%=$(BUILDDIR)/%.o)
GRAPH_OBJS      := $(GRAPH_CORE_OBJS) $(GRAPH_MOSP_OBJS)

# MOSP base sources used by every binary.
MOSP_BASE := \
    mosp/src/generateGraph.cu       \
    mosp/src/generateGraphCSR.cu    \
    mosp/src/generateChangedEdges.cu \
    mosp/src/updateGraphCSR.cu      \
    mosp/src/generateTestCases.cu   \
    mosp/src/Dijkstra.cu            \
    mosp/src/read.cu

MOSP_MAIN    := $(MOSP_BASE) \
                mosp/src/main.cu \
                mosp/src/sequentialSOSPUpdate.cu \
                mosp/src/parallelSOSPUpdate.cu \
                mosp/src/sospUpdateGpu.cu \
                mosp/src/parallelCombinedGraph.cu
MOSP_STRESS  := $(MOSP_BASE) \
                mosp/src/stressTest.cu \
                mosp/src/sequentialSOSPUpdate.cu
MOSP_PSTRESS := $(MOSP_BASE) \
                mosp/src/parallelStressTest.cu \
                mosp/src/parallelSOSPUpdate.cu \
                mosp/src/sospUpdateGpu.cu \
                mosp/src/sequentialSOSPUpdate.cu

MOSP_MAIN_OBJS    := $(MOSP_MAIN:%=$(BUILDDIR)/%.o)
MOSP_STRESS_OBJS  := $(MOSP_STRESS:%=$(BUILDDIR)/%.o)
MOSP_PSTRESS_OBJS := $(MOSP_PSTRESS:%=$(BUILDDIR)/%.o)

# -----------------------------------------------------------------------------
# H-SOSP: dynamic hypergraph SOSP (hypergraph/ + hsosp/)
# -----------------------------------------------------------------------------

HYPERGRAPH_SRCS := \
    hypergraph/src/HostHypergraph.cpp \
    hypergraph/src/HypergraphGen.cpp \
    hypergraph/src/HypergraphOracle.cpp \
    hypergraph/src/DynamicHypergraph.cpp
HSOSP_DEV_SRCS  := hsosp/src/hsospDevice.cu hsosp/src/hsospDelta.cu

HYPERGRAPH_OBJS := $(HYPERGRAPH_SRCS:%=$(BUILDDIR)/%.o)
HSOSP_DEV_OBJS  := $(HSOSP_DEV_SRCS:%=$(BUILDDIR)/%.o)
HSOSP_CORE_OBJS := $(HYPERGRAPH_OBJS) $(HSOSP_DEV_OBJS)

# Unit tests
UNIT_TESTS := \
    $(BINDIR)/test_cbst_smoke \
    $(BINDIR)/test_cbst_ops \
    $(BINDIR)/test_dynamicgraph_roundtrip \
    $(BINDIR)/test_snapshot_matches_updateCSR \
    $(BINDIR)/test_h2h_construction \
    $(BINDIR)/test_h2h_delta \
    $(BINDIR)/test_hsosp_matches_dijkstra \
    $(BINDIR)/test_hsosp_scale \
    $(BINDIR)/test_mosp_update

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------

.PHONY: all clean run tests test docs syntax-check stressTest parallelStressTest

all: $(BINDIR)/main $(BINDIR)/stressTest $(BINDIR)/parallelStressTest \
     $(BINDIR)/hsospBench $(BINDIR)/hsospStress tests

stressTest:         $(BINDIR)/stressTest
parallelStressTest: $(BINDIR)/parallelStressTest
hsospBench:         $(BINDIR)/hsospBench
hsospStress:        $(BINDIR)/hsospStress
tests:              $(UNIT_TESTS)

test: all
	./tests/run_tests.sh

run: $(BINDIR)/main
	./$(BINDIR)/main

clean:
	rm -rf $(BUILDDIR) $(BINDIR)

# -----------------------------------------------------------------------------
# Library archive
# -----------------------------------------------------------------------------

LIBESCHER := $(LIBDIR)/libescher_core.a

$(LIBESCHER): $(ESCHER_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

# -----------------------------------------------------------------------------
# Pattern rules
# -----------------------------------------------------------------------------

# build/.flags records the compiler and flags; it is rewritten (and every
# object rebuilt) only when they change.
FLAGS_STAMP := $(BUILDDIR)/.flags
FLAGS_NOW   := $(NVCC) $(NVFLAGS)
$(shell mkdir -p $(BUILDDIR); \
        [ "$$(cat $(FLAGS_STAMP) 2>/dev/null)" = '$(FLAGS_NOW)' ] || \
        echo '$(FLAGS_NOW)' > $(FLAGS_STAMP))

$(BUILDDIR)/%.cu.o: %.cu $(FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) $(DEPFLAGS) -c -o $@ $<

$(BUILDDIR)/%.cpp.o: %.cpp $(FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) $(DEPFLAGS) -c -o $@ $<

-include $(shell find $(BUILDDIR) -name '*.d' 2>/dev/null)

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

$(BINDIR)/main: $(MOSP_MAIN_OBJS) $(GRAPH_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(MOSP_MAIN_OBJS) $(GRAPH_OBJS) $(LIBESCHER)

$(BINDIR)/stressTest: $(MOSP_STRESS_OBJS) $(GRAPH_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(MOSP_STRESS_OBJS) $(GRAPH_OBJS) $(LIBESCHER)

$(BINDIR)/parallelStressTest: $(MOSP_PSTRESS_OBJS) $(GRAPH_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(MOSP_PSTRESS_OBJS) $(GRAPH_OBJS) $(LIBESCHER)

$(BINDIR)/test_cbst_smoke: $(BUILDDIR)/tests/unit/test_cbst_smoke.cu.o $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_cbst_smoke.cu.o $(LIBESCHER)

$(BINDIR)/test_cbst_ops: $(BUILDDIR)/tests/unit/test_cbst_ops.cu.o $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_cbst_ops.cu.o $(LIBESCHER)

$(BINDIR)/test_dynamicgraph_roundtrip: $(BUILDDIR)/tests/unit/test_dynamicgraph_roundtrip.cu.o $(GRAPH_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_dynamicgraph_roundtrip.cu.o $(GRAPH_CORE_OBJS) $(LIBESCHER)

# H-SOSP binaries: benchmark driver + randomized stress harness.
$(BINDIR)/hsospBench: $(BUILDDIR)/hsosp/src/hsospBench.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/hsosp/src/hsospBench.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)

$(BINDIR)/hsospStress: $(BUILDDIR)/hsosp/src/hsospStress.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/hsosp/src/hsospStress.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)

$(BINDIR)/test_h2h_construction: $(BUILDDIR)/tests/unit/test_h2h_construction.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_h2h_construction.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)

$(BINDIR)/test_h2h_delta: $(BUILDDIR)/tests/unit/test_h2h_delta.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_h2h_delta.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)

$(BINDIR)/test_hsosp_matches_dijkstra: $(BUILDDIR)/tests/unit/test_hsosp_matches_dijkstra.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_hsosp_matches_dijkstra.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)

$(BINDIR)/test_hsosp_scale: $(BUILDDIR)/tests/unit/test_hsosp_scale.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_hsosp_scale.cu.o $(HSOSP_CORE_OBJS) $(LIBESCHER)

# The equivalence test links in the MOSP base (needed for generateGraphCSR,
# generateChangedEdges, updateGraphCSR, readCSR). generateTestCases.cu in the
# base additionally references the SOSP kernels and the combined-graph step,
# so those objects are linked too (this was missing upstream and made the
# test binary fail to link).
MOSP_BASE_OBJS := $(MOSP_BASE:%=$(BUILDDIR)/%.o) \
                  $(BUILDDIR)/mosp/src/sequentialSOSPUpdate.cu.o \
                  $(BUILDDIR)/mosp/src/parallelSOSPUpdate.cu.o \
                  $(BUILDDIR)/mosp/src/sospUpdateGpu.cu.o \
                  $(BUILDDIR)/mosp/src/parallelCombinedGraph.cu.o

$(BINDIR)/test_snapshot_matches_updateCSR: $(BUILDDIR)/tests/unit/test_snapshot_matches_updateCSR.cu.o $(GRAPH_OBJS) $(MOSP_BASE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_snapshot_matches_updateCSR.cu.o $(GRAPH_OBJS) $(MOSP_BASE_OBJS) $(LIBESCHER)

$(BINDIR)/test_mosp_update: $(BUILDDIR)/tests/unit/test_mosp_update.cu.o $(GRAPH_OBJS) $(MOSP_BASE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_mosp_update.cu.o $(GRAPH_OBJS) $(MOSP_BASE_OBJS) $(LIBESCHER)

# -----------------------------------------------------------------------------
# Doxygen
# -----------------------------------------------------------------------------

docs:
	doxygen Doxyfile

# -----------------------------------------------------------------------------
# Syntax check (host-only; no link, no GPU needed)
# -----------------------------------------------------------------------------
#
# Runs @c nvcc -E on every .cu/.cpp translation unit so the user can verify
# includes and templates on a MacBook before syncing to the cluster. Requires
# nvcc to be installed (which it can be without a GPU) or use @c clang-check.

ALL_SRCS := $(ESCHER_CU_SRCS) $(ESCHER_CPP_SRCS) $(GRAPH_CU_SRCS) \
            $(GRAPH_CORE_CPP_SRCS) $(GRAPH_MOSP_CPP_SRCS) \
            $(MOSP_MAIN) $(MOSP_STRESS) $(MOSP_PSTRESS) \
            $(HYPERGRAPH_SRCS) $(HSOSP_DEV_SRCS) \
            hsosp/src/hsospBench.cu hsosp/src/hsospStress.cu \
            tests/unit/test_cbst_smoke.cu \
            tests/unit/test_cbst_ops.cu \
            tests/unit/test_dynamicgraph_roundtrip.cu \
            tests/unit/test_snapshot_matches_updateCSR.cu \
            tests/unit/test_h2h_construction.cu \
            tests/unit/test_h2h_delta.cu \
            tests/unit/test_hsosp_matches_dijkstra.cu \
            tests/unit/test_hsosp_scale.cu \
            tests/unit/test_mosp_update.cu

syntax-check:
	@failed=; \
	for f in $(sort $(ALL_SRCS)); do \
	  printf 'preprocess %s ... ' $$f; \
	  if $(NVCC) $(NVFLAGS) -E $$f > /dev/null; then \
	    echo ok; \
	  else \
	    echo FAILED; failed="$$failed $$f"; \
	  fi; \
	done; \
	if [ -n "$$failed" ]; then \
	  echo "syntax-check: failed to preprocess:$$failed" >&2; \
	  exit 1; \
	fi
	@echo "syntax-check: all translation units preprocessed cleanly"
