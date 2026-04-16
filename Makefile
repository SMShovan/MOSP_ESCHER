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
#
# Overridable: CUDA_ARCH  (default sm_70, V100/Volta; MOSP kernels need
#                          --extended-lambda which works on all sm_70+).
#
# =============================================================================

NVCC      := nvcc
CUDA_ARCH ?= sm_70

INCLUDES  := -Iescher/include -Iescher/kernel -Igraph/include -Imosp/headers

NVFLAGS   := -std=c++17 -O2 --extended-lambda -arch=$(CUDA_ARCH) $(INCLUDES)
CXXFLAGS  := -std=c++17 -O2 -Wall $(INCLUDES)

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
                mosp/src/parallelCombinedGraph.cu
MOSP_STRESS  := $(MOSP_BASE) \
                mosp/src/stressTest.cu \
                mosp/src/sequentialSOSPUpdate.cu
MOSP_PSTRESS := $(MOSP_BASE) \
                mosp/src/parallelStressTest.cu \
                mosp/src/parallelSOSPUpdate.cu \
                mosp/src/sequentialSOSPUpdate.cu

MOSP_MAIN_OBJS    := $(MOSP_MAIN:%=$(BUILDDIR)/%.o)
MOSP_STRESS_OBJS  := $(MOSP_STRESS:%=$(BUILDDIR)/%.o)
MOSP_PSTRESS_OBJS := $(MOSP_PSTRESS:%=$(BUILDDIR)/%.o)

# Unit tests
UNIT_TESTS := \
    $(BINDIR)/test_cbst_smoke \
    $(BINDIR)/test_dynamicgraph_roundtrip \
    $(BINDIR)/test_snapshot_matches_updateCSR

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------

.PHONY: all clean run tests docs syntax-check stressTest parallelStressTest

all: $(BINDIR)/main $(BINDIR)/stressTest $(BINDIR)/parallelStressTest tests

stressTest:         $(BINDIR)/stressTest
parallelStressTest: $(BINDIR)/parallelStressTest
tests:              $(UNIT_TESTS)

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

$(BUILDDIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

$(BUILDDIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

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

$(BINDIR)/test_dynamicgraph_roundtrip: $(BUILDDIR)/tests/unit/test_dynamicgraph_roundtrip.cu.o $(GRAPH_CORE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_dynamicgraph_roundtrip.cu.o $(GRAPH_CORE_OBJS) $(LIBESCHER)

# The equivalence test links in the MOSP base (needed for generateGraphCSR,
# generateChangedEdges, updateGraphCSR, readCSR).
MOSP_BASE_OBJS := $(MOSP_BASE:%=$(BUILDDIR)/%.o)

$(BINDIR)/test_snapshot_matches_updateCSR: $(BUILDDIR)/tests/unit/test_snapshot_matches_updateCSR.cu.o $(GRAPH_OBJS) $(MOSP_BASE_OBJS) $(LIBESCHER)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -o $@ $(BUILDDIR)/tests/unit/test_snapshot_matches_updateCSR.cu.o $(GRAPH_OBJS) $(MOSP_BASE_OBJS) $(LIBESCHER)

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

ALL_SRCS := $(ESCHER_CU_SRCS) $(ESCHER_CPP_SRCS) $(GRAPH_CU_SRCS) $(GRAPH_CPP_SRCS) \
            $(MOSP_MAIN) $(MOSP_STRESS) $(MOSP_PSTRESS) \
            tests/unit/test_cbst_smoke.cu \
            tests/unit/test_dynamicgraph_roundtrip.cu \
            tests/unit/test_snapshot_matches_updateCSR.cu

syntax-check:
	@set -e; \
	for f in $(sort $(ALL_SRCS)); do \
	  printf 'preprocess %s ... ' $$f; \
	  $(NVCC) $(NVFLAGS) -E $$f > /dev/null && echo ok; \
	done
	@echo "syntax-check: all translation units preprocessed cleanly"
