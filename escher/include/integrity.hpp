#ifndef ESCHER_INTEGRITY_HPP
#define ESCHER_INTEGRITY_HPP

#include <iosfwd>
#include <vector>

#include "structure.hpp"

/**
 * @brief CBST content checker (test oracle).
 *
 * Copies a CBST (nodes, availability flags, used payload prefix) to the host
 * and compares it with the rows a host model expects. @p expected is indexed
 * by key - 1; an empty expected row means "absent, deleted or empty".
 *
 *  - every key 1..max(expected.size(), largest live key) is searched from
 *    the root exactly as the device kernels do, and the row found (following
 *    overflow chains) must hold exactly the expected values: in order, or as
 *    a sorted multiset when @p orderInsensitive;
 *  - every live node must be reachable by a search for its own key;
 *  - every live node must satisfy the tail-segment invariants fill and
 *    unfill rely on: @c occupancy is the number of live entries of the tail
 *    segment (the one starting at @c tailBase), it does not exceed
 *    @c tailCapacity, and no live entry lies in the free part of the tail
 *    segment;
 *  - the payload segments of different rows must not overlap.
 *
 * @return number of violations; details of the first few go to @p log.
 */
long long checkTreeRows(const CBSTContext& ctx,
                        const std::vector<std::vector<int>>& expected,
                        bool orderInsensitive, const char* name,
                        std::ostream& log);

/**
 * @brief Availability bookkeeping check: @c subtreeAvail[i] must equal
 *        avail[i] + subtreeAvail[2i+1] + subtreeAvail[2i+2] for every node,
 *        so the root holds the number of reusable (deleted) slots.
 * @return number of inconsistent nodes.
 */
long long checkSubtreeAvail(const CBSTContext& ctx, const char* name,
                            std::ostream& log);

#endif // ESCHER_INTEGRITY_HPP
