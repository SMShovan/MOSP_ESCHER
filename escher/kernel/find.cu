#include "kernels.cuh"
#include <climits>
#include <cstdio>

// Follow back-pointer chains in a flat payload array. Negative values
// (other than INT_MIN) are chain pointers: jump to -val. On return,
// @c loc points to the resolved position. Bounded to avoid infinite loops
// on a corrupt chain.
static inline __device__ int readFlat(const int* flat, int& loc) {
    constexpr int MAX_CHAIN_HOPS = 1024;
    for (int hops = 0; hops < MAX_CHAIN_HOPS; ++hops) {
        int val = flat[loc];
        if (val < 0 && val != INT_MIN) {
            loc = -val;
            continue;
        }
        return val;
    }
    return INT_MIN; // treat as end-of-record on suspected cycle
}

__global__ void findNode(CBSTNode* nodes, int* searchIndices, int searchSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != searchIndex) {
            if (current->index > searchIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Length = %d\n",
                   searchIndex, current->index, current->value, current->length);
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}

__global__ void findContents(CBSTNode* nodes, int* searchIndices, int searchSize, int* flatValues) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != searchIndex) {
            if (current->index > searchIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            int currLoc = current->value;
            printf("\n");
            while (true) {
                int val = readFlat(flatValues, currLoc);
                if (val == 0 || val == INT_MIN) break;
                printf("%d ", val);
                currLoc++;
            }
            printf("\n");
            printf("Node %d: Index = %d, Value = %d, Length = %d\n", searchIndex, current->index, current->value, current->length);
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}


