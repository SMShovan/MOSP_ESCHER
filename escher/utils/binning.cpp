#include "../include/binning.hpp"

void binByDegree(const int* prefixSizes,
                 int K,
                 int thSmall,
                 int thLarge,
                 std::vector<int>& smallOut,
                 std::vector<int>& mediumOut,
                 std::vector<int>& largeOut) {
    smallOut.clear();
    mediumOut.clear();
    largeOut.clear();
    smallOut.reserve(static_cast<size_t>(K));
    mediumOut.reserve(static_cast<size_t>(K) / 4);
    largeOut.reserve(1024);

    for (int i = 0; i < K; ++i) {
        int degree = prefixSizes[i] - (i == 0 ? 0 : prefixSizes[i - 1]);
        if (degree < thSmall)
            smallOut.push_back(i);
        else if (degree < thLarge)
            mediumOut.push_back(i);
        else
            largeOut.push_back(i);
    }
}
