# GPU-free validation of the hypergraph host core (see local_tests.cpp).
# Usage: make -C tests/local && ./tests/local/local_tests
CXX      ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra
# Always added (an environment CXXFLAGS replaces the default above).
LOCAL_FLAGS := -std=c++17 -fopenmp -I../../hypergraph/include

SRCS := ../../hypergraph/src/HostHypergraph.cpp \
        ../../hypergraph/src/HypergraphGen.cpp \
        ../../hypergraph/src/HypergraphOracle.cpp

local_tests: local_tests.cpp $(SRCS) $(wildcard ../../hypergraph/include/*.hpp)
	$(CXX) $(CXXFLAGS) $(LOCAL_FLAGS) -o $@ local_tests.cpp $(SRCS)

clean:
	rm -f local_tests
