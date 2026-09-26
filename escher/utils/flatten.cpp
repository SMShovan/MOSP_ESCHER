#include "../include/flatten.hpp"
#include <climits>
#include <vector>

// Padded length of a row: the next multiple of 4 that leaves at least one
// slot for the INT_MIN terminator (the same rule as the device-side
// nextMultipleOf4 in kernel/device_utils.cuh). The original padded a row of
// length 4k to exactly 4k entries, so it had no terminator: readers ran into
// the next row and a fill wrote over the row's last value.
static inline int paddedRowSize(int num) {
    return ((num + 4) / 4) * 4;
}

std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::vector<int> flatValues;
    std::vector<int> startOffsets(vec2d.size());

    int index = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        startOffsets[i] = index;
        int innerSize = static_cast<int>(vec2d[i].size());
        int paddedSize = paddedRowSize(innerSize);
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


