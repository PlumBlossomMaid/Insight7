// backends/cpu/kernels/random/numpy_rng_state.h
/**
 * @file numpy_rng_state.h
 * @brief Per-device global NumPy-compatible RNG state.
 *
 * NumPy's legacy RandomState is a single global MT19937 state consumed in
 * call order: `seed(42)` re-seeds, every subsequent draw continues from the
 * previous state. Insight mirrors this per device: `seed(base)` stores
 * `base + device_id` per device (src/ops/random.cpp), and every random
 * kernel on that device consumes the same persistent state.
 *
 * The state map is keyed by (device_type, device_id) so that CPU (device
 * 0) and GPU (device 0) do not collide.
 *
 * Thread-safety: a mutex serialises draw sequences, mirroring NumPy's
 * GIL/lock. The lock is held for the whole fill so concurrent calls cannot
 * interleave draws.
 */

#ifndef INSIGHT_CPU_NUMPY_RNG_STATE_H
#define INSIGHT_CPU_NUMPY_RNG_STATE_H

#include "numpy_rng.h"
#include <map>
#include <mutex>
#include <stdint.h>

#define NPY_RNG_KEY(type, id) ((int64_t)(((int64_t)(type) << 32) | (int64_t)(id)))

namespace ins {
namespace numpy_rng {

struct DeviceEntry {
  numpy_rk_state state;
  bool seeded = false;
};

inline std::map<int64_t, DeviceEntry> &devices() {
  static std::map<int64_t, DeviceEntry> map;
  return map;
}

inline std::mutex &mutex() {
  static std::mutex m;
  return m;
}

/**
 * @brief Seed one device's global state (re-seed = reset stream, exactly
 * like `np.random.seed(s)`).
 */
inline void seed_device(int32_t device_type, int32_t device_id,
                        uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex());
  int64_t key = NPY_RNG_KEY(device_type, device_id);
  numpy_rk_seed(&devices()[key].state, static_cast<uint32_t>(seed));
  devices()[key].seeded = true;
}

/**
 * @brief Lock and return the device's persistent state.
 *
 * Caller MUST keep the returned pointer valid and call unlock_device()
 * after the generation loop.
 */
inline numpy_rk_state *lock_device(int32_t device_type, int32_t device_id) {
  mutex().lock();
  int64_t key = NPY_RNG_KEY(device_type, device_id);
  DeviceEntry &e = devices()[key];
  if (!e.seeded) {
    // NumPy default-seeds from entropy; Insight falls back to seed 0.
    numpy_rk_seed(&e.state, 0u);
    e.seeded = true;
  }
  return &e.state;
}

inline void unlock_device() { mutex().unlock(); }

} // namespace numpy_rng
} // namespace ins

#endif // INSIGHT_CPU_NUMPY_RNG_STATE_H