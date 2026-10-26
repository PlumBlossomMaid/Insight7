// backends/cpu/kernels/random/randn.cpp
/**
 * @file randn.cpp
 * @brief CPU kernel for standard normal distribution.
 *
 * Bit-exact with `np.random.randn(size)` for the same seed: uses NumPy's
 * legacy polar-method gauss with a persistent cached second value
 * (randomkit.c rk_gauss / legacy-distributions.c legacy_gauss).
 */

#include "../common/half_utils.h"
#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status randn_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("randn: output array is null");
    return C_FAILED;
  }

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
      int64_t off_out =
          cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = (float)numpy_rk_gauss(st);
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *out_data = (double *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out =
          cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = numpy_rk_gauss(st);
    }
    break;
  }
  case INSIGHT_DTYPE_F16: {
    uint16_t *out_data = (uint16_t *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out =
          cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = insight::f32_to_f16((float)numpy_rk_gauss(st));
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    uint16_t *out_data = (uint16_t *)out->data;
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_out =
          cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = insight::f32_to_bf16((float)numpy_rk_gauss(st));
    }
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("randn: unsupported dtype");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_F32, randn_kernel_cpu);
REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_F64, randn_kernel_cpu);
REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_F16, randn_kernel_cpu);
REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_BF16, randn_kernel_cpu);
