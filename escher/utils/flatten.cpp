#include "../include/flatten.hpp"
#include <climits>
#include <vector>

static inline int nextMultipleOf4(int num) {
    if (num <= 0) return 0;
    int q = (num + 4 - 1) / 4;
    return q * 4;
}

std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::vector<int> flatValues;
    std::vector<int> startOffsets(vec2d.size());

    int index = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        startOffsets[i] = index;
        int innerSize = static_cast<int>(vec2d[i].size());
        int paddedSize = (innerSize == 0) ? 4 : nextMultipleOf4(innerSize);
        for (int j = 0; j < paddedSize; ++j) {
            if (j < innerSize) {
                flatValues.push_back(vec2d[i][j]);
            } else if (j == paddedSize - 1) {
                flatValues.push_back(INT_MIN);
            } else {
                flatValues.push_back(0);
            }
            ++index;
        }
    }
    return {flatValues, startOffsets};
}


