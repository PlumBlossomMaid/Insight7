// backends/cpu/kernels/random/gamma.cpp
/**
 * @file gamma.cpp
 * @brief CPU kernel for gamma distribution.
 *
 * Bit-exact with `np.random.gamma(shape, scale)` for the same seed, where
 * Insight's `rate` parameter maps to NumPy scale = 1/rate:
 *   scale * legacy_standard_gamma(shape)
 * (legacy-distributions.c legacy_gamma / legacy_standard_gamma).
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status gamma_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("gamma: output array is null");
    return C_FAILED;
  }

  double shape_param = *(double *)inputs[1];
  double rate = *(double *)inputs[2];
  double scale = 1.0 / rate;

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
      out_data[off_out] = (float)numpy_legacy_gamma(st, shape_param, scale);
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *out_data = (double *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = numpy_legacy_gamma(st, shape_param, scale);
    }
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("gamma: unsupported dtype");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(gamma, INSIGHT_DTYPE_F32, gamma_kernel_cpu);
REGISTER_CPU_KERNEL(gamma, INSIGHT_DTYPE_F64, gamma_kernel_cpu);