// backends/cpu/kernels/random/randint.cpp
/**
 * @file randint.cpp
 * @brief CPU kernel for random integers [low, high).
 *
 * Bit-exact with `np.random.randint(low, high, size, dtype)` for the same
 * seed: uses NumPy's masked-rejection bounded integer generator
 * (distributions.c random_bounded_uint64 with use_masked=True).
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

static C_Status randint_impl(InsightArray *out, int64_t low, int64_t high,
                             int is32) {
  // NumPy computes rng = (high - 1) - low (open interval), off = low,
  // mask = gen_mask(rng); scalar path computes mask once.
  uint64_t rng;
  if (low >= high) {
    cpu_set_last_error("randint: low must be < high");
    return C_FAILED;
  }
  rng = (uint64_t)((high - 1) - low);
  uint64_t mask = numpy_gen_mask(rng);
  uint64_t off = (uint64_t)low;

  int32_t device_id = out->device_id;
  numpy_rk_state *st = ins::numpy_rng::lock_device(out->device_type, device_id);

  int64_t ndim = out->ndim;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dims[i] = out->dims[i];
    out_strides[i] = out->strides[i];
  }
  int64_t n = out->numel;

  if (is32) {
    int32_t *out_data = (int32_t *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out =
          cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = (int32_t)numpy_random_bounded_uint64(st, off, rng,
                                                               mask);
    }
  } else {
    int64_t *out_data = (int64_t *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out =
          cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = (int64_t)numpy_random_bounded_uint64(st, off, rng,
                                                               mask);
    }
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

C_Status randint_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("randint: output array is null");
    return C_FAILED;
  }

  int64_t low = *(int64_t *)inputs[1];
  int64_t high = *(int64_t *)inputs[2];

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32:
    return randint_impl(out, low, high, 1);
  case INSIGHT_DTYPE_I64:
    return randint_impl(out, low, high, 0);
  default:
    cpu_set_last_error("randint: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(randint, INSIGHT_DTYPE_I32, randint_kernel_cpu);
REGISTER_CPU_KERNEL(randint, INSIGHT_DTYPE_I64, randint_kernel_cpu);