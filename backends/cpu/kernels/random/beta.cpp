// backends/cpu/kernels/random/beta.cpp
/**
 * @file beta.cpp
 * @brief CPU kernel for beta distribution.
 *
 * Bit-exact with `np.random.beta(a, b)` for the same seed
 * (legacy-distributions.c legacy_beta: Johnk's algorithm when a<=1 && b<=1,
 * gamma-ratio otherwise).
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

static void beta_fill_float(float *out_data, int64_t n, int64_t ndim,
                            const int64_t *dims, const int64_t *out_strides,
                            numpy_rk_state *st, double a, double b) {
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
    out_data[off_out] = (float)numpy_legacy_beta(st, a, b);
  }
}

static void beta_fill_double(double *out_data, int64_t n, int64_t ndim,
                             const int64_t *dims, const int64_t *out_strides,
                             numpy_rk_state *st, double a, double b) {
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
    out_data[off_out] = numpy_legacy_beta(st, a, b);
  }
}

C_Status beta_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("beta: output array is null");
    return C_FAILED;
  }

  double a = *(double *)inputs[1];
  double b = *(double *)inputs[2];

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
  case INSIGHT_DTYPE_F32:
    beta_fill_float((float *)out->data, n, ndim, dims, out_strides, st, a, b);
    break;
  case INSIGHT_DTYPE_F64:
    beta_fill_double((double *)out->data, n, ndim, dims, out_strides, st, a,
                     b);
    break;
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("beta: unsupported dtype");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(beta, INSIGHT_DTYPE_F32, beta_kernel_cpu);
REGISTER_CPU_KERNEL(beta, INSIGHT_DTYPE_F64, beta_kernel_cpu);