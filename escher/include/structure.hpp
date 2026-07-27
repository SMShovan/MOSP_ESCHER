#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

#include <vector>
#include <cuda_runtime.h>

// CBST node definition shared by kernels across translation units
struct CBSTNode {
    int index;
    int value;
    int length;
    int size;
    int occupancy;    // number of data elements in the active (tail) segment
    int tailBase;     // base offset of the last segment in flatPayload
    int tailCapacity; // usable capacity of the tail segment (excludes sentinel)
    CBSTNode* left;
    CBSTNode* right;
    CBSTNode* parent;
};

// Device-side context is opaque; host context holds device buffers
struct CBSTContext {
    CBSTNode* d_nodes;
    int* d_keys;
    int* d_startOffsets;
    int* d_flatPayload;
    int* d_insertKeys;
    int* d_insertPayload;
    int* d_insertPrefixSizes;
    int* d_relocationPlan;
    int* d_avail;
    int* d_subtreeAvail;
    int fixedSize;
    int numRecords;
    int initialPayloadSize;
    const char* datasetName;
    int alignment;
    // Capacities of the reusable scratch buffers (d_insertKeys /
    // d_insertPayload / d_insertPrefixSizes / d_relocationPlan). The
    // upstream code sized these once at construct time (numRecords keys,
    // numRecords*3 payload ints) and silently overflowed device memory
    // whenever a fill/insert/unfill batch was larger than that. The
    // capacities are now tracked and the buffers grow on demand.
    int scratchKeysCap;
    long long scratchPayloadCap;
};

// Mapping returned by insertCBST: for each input item i, itemToKey[i] is the
// actual CBST key the item was stored under (may differ from newKeys[i] due to
// best-fit matching reusing a deleted slot's key).
struct InsertMapping {
    std::vector<int> itemToKey;
};

// Host API for CBST operations (free functions)
//
// rowOccupancy (optional): per-record count of REAL data values in each
// record's payload segment, in the same order as keys/startOffsets. The
// upstream code left every node's occupancy at 0 after construction (the
// "fixup pass" mentioned in build_tree.cu never existed), which made the
// first fillCBST on a row overwrite the construct-time payload from the
// segment base. Pass the true per-row counts to get correct append
// positions; pass nullptr to keep the legacy behavior.
void constructCBST(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, int payloadCapacity, const char* datasetName, CBSTContext& ctx, const int* rowOccupancy = nullptr);
void fillCBST(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes, CBSTContext& ctx);
void deleteCBST(const std::vector<int>& deleteKeys, CBSTContext& ctx);
InsertMapping insertCBST(const std::vector<int>& newKeys, const std::vector<int>& newPayload, const std::vector<int>& newPrefixSizes, CBSTContext& ctx);
void unfillCBST(const std::vector<int>& keysToUnfill, const std::vector<int>& valuesToRemove, const std::vector<int>& removePrefixSizes, CBSTContext& ctx);

// OO wrapper to manage CBST lifecycle and operations
struct CBSTOperations {
    explicit CBSTOperations(const char* datasetName, int payloadCapacity, int alignment = 4);
    ~CBSTOperations();
    CBSTOperations(const CBSTOperations&) = delete;
    CBSTOperations& operator=(const CBSTOperations&) = delete;
    CBSTOperations(CBSTOperations&& other) noexcept;
    CBSTOperations& operator=(CBSTOperations&& other) noexcept;

    void construct(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, const int* rowOccupancy = nullptr);
    InsertMapping insert(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes);
    void fill(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes);
    void erase(const std::vector<int>& deleteKeys);
    void findAndPrint(const std::vector<int>& ids) const;

    // Accessor to underlying device-resident context (read-only)
    const CBSTContext& context() const;

  private:
    CBSTContext ctx_{};
    bool constructed_ = false;
};

#endif // STRUCTURE_HPP
