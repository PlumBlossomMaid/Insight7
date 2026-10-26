// backends/cpu/kernels/random/exponential.cpp
/**
 * @file exponential.cpp
 * @brief CPU kernel for exponential distribution.
 *
 * Bit-exact with `np.random.exponential(scale, size)` for the same seed:
 * NumPy computes `scale * (-log(1 - U))` with U = next_double
 * (legacy-distributions.c legacy_standard_exponential / legacy_exponential).
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status exponential_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("exponential: output array is null");
    return C_FAILED;
  }

  double scale = *(double *)inputs[1];

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

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *out_data = (float *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] =
          (float)(scale * numpy_legacy_standard_exponential(st));
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *out_data = (double *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = scale * numpy_legacy_standard_exponential(st);
    }
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("exponential: unsupported dtype");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(exponential, INSIGHT_DTYPE_F32, exponential_kernel_cpu);
REGISTER_CPU_KERNEL(exponential, INSIGHT_DTYPE_F64, exponential_kernel_cpu);