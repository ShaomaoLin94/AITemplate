#pragma once

#include <vector>

namespace ait {
namespace cpu {

/*
 * Shared temporary buffers for sequential CPU operators.
 *
 * They are thread-local so concurrent inference threads do not
 * overwrite each other's temporary data, but all generated
 * operators running on the same thread reuse the same storage.
 */
inline std::vector<float>& shared_float_scratch_0() {
  thread_local std::vector<float> buffer;
  return buffer;
}

inline std::vector<float>& shared_float_scratch_1() {
  thread_local std::vector<float> buffer;
  return buffer;
}

}  // namespace cpu
}  // namespace ait
