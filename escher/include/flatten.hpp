#ifndef ESCHER_FLATTEN_HPP
#define ESCHER_FLATTEN_HPP

#include <string>
#include <utility>
#include <vector>

/**
 * @brief Flatten a 2D vector into a padded flat array plus per-row start offsets.
 *
 * Each row is padded to the next multiple of 4 ints (or to 4 ints if empty),
 * with an @c INT_MIN sentinel written at the last slot of the padded region and
 * zeros filling the gap between real data and the sentinel. The resulting
 * @c flatValues array is the payload format consumed by @c constructCBST.
 *
 * @param vec2d Input 2D vector; each inner vector is one record.
 * @return A pair @c (flatValues, startOffsets) where @c startOffsets[i] is the
 *         index into @c flatValues where row @c i begins.
 */
std::pair<std::vector<int>, std::vector<int>>
flatten2DVector(const std::vector<std::vector<int>>& vec2d);

#endif // ESCHER_FLATTEN_HPP
