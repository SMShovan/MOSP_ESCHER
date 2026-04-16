#ifndef GENERATE_GRAPH_CUH
#define GENERATE_GRAPH_CUH

#include <string>

bool generateGraph(
    int numberOfNodes,
    int numberOfEdges,
    bool directed,
    const std::string &outputFile,
    int numberOfObjectives,
    int objectiveStartRange,
    int objectiveEndRange
);

#endif
