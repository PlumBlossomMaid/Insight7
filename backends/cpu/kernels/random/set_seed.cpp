// backends/cpu/kernels/random/set_seed.cpp
/**
 * @file set_seed.cpp
 * @brief CPU kernel for setting random seed.
 *
 * Sets the global seed for CPU backend RNG and re-seeds the NumPy-
 * compatible per-device MT19937 stream (device 0 on CPU).
 *
 * @param inputs  [0] = uint64_t* seed value
 * @param outputs [0] = unused
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status set_seed_kernel_cpu(void **inputs, void **outputs) {
  if (!inputs[0]) {
    cpu_set_last_error("set_seed: seed pointer is null");
    return C_FAILED;
  }

  uint64_t seed = *(uint64_t *)inputs[0];
  cpu_set_global_seed(seed);
  ins::numpy_rng::seed_device(0 /* INSIGHT_DEVICE_CPU */, 0 /* device */, seed);

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(set_seed, INSIGHT_DTYPE_U64, set_seed_kernel_cpu);