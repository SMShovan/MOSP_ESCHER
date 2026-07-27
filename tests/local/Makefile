# GPU-free validation of the hypergraph host core (see local_tests.cpp).
# Usage: make -C tests/local && ./tests/local/local_tests
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -I../../hypergraph/include

local_tests: local_tests.cpp ../../hypergraph/src/HostHypergraph.cpp ../../hypergraph/src/HypergraphGen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f local_tests
