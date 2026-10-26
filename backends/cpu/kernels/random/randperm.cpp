// backends/cpu/kernels/random/randperm.cpp
/**
 * @file randperm.cpp
 * @brief CPU kernel for random permutation [0, n).
 *
 * Bit-exact with `np.random.permutation(n)` for the same seed: NumPy's
 * permutation(int) builds arange(n) then shuffles with the reversed-range
 * Fisher-Yates using random_interval (distributions.c:1078,
 * mtrand.pyx _shuffle_raw).
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status randperm_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("randperm: output array is null");
    return C_FAILED;
  }

  // randperm only works on contiguous arrays (1D)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("randperm: output array must be contiguous");
    return C_FAILED;
  }

  int32_t device_id = out->device_id;
  numpy_rk_state *st = ins::numpy_rng::lock_device(out->device_type, device_id);
  int64_t n = out->numel;

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32: {
    int32_t *out_data = (int32_t *)out->data;
    for (int64_t i = 0; i < n; ++i) {
      out_data[i] = static_cast<int32_t>(i);
    }
    // for i in reversed(range(1, n)): j = random_interval(i); swap
    for (int64_t i = n - 1; i > 0; --i) {
      int64_t j = (int64_t)numpy_random_interval(st, (uint64_t)i);
      int32_t tmp = out_data[i];
      out_data[i] = out_data[j];
      out_data[j] = tmp;
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *out_data = (int64_t *)out->data;
    for (int64_t i = 0; i < n; ++i) {
      out_data[i] = i;
    }
    for (int64_t i = n - 1; i > 0; --i) {
      int64_t j = (int64_t)numpy_random_interval(st, (uint64_t)i);
      int64_t tmp = out_data[i];
      out_data[i] = out_data[j];
      out_data[j] = tmp;
    }
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("randperm: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(randperm, INSIGHT_DTYPE_I32, randperm_kernel_cpu);
REGISTER_CPU_KERNEL(randperm, INSIGHT_DTYPE_I64, randperm_kernel_cpu);