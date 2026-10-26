// backends/cpu/kernels/random/binomial.cpp
/**
 * @file binomial.cpp
 * @brief CPU kernel for binomial distribution.
 *
 * Bit-exact with `np.random.binomial(n, p)` for the same seed
 * (legacy-distributions.c legacy_random_binomial: inversion for
 * small n*p, BTPE otherwise, symmetric for p > 0.5).
 */

#include "common.h"
#include "numpy_rng_state.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status binomial_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("binomial: output array is null");
    return C_FAILED;
  }

  int64_t n = *(int64_t *)inputs[1];
  double p = *(double *)inputs[2];

  int32_t device_id = out->device_id;
  numpy_rk_state *st = ins::numpy_rng::lock_device(out->device_type, device_id);

  int64_t ndim = out->ndim;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dims[i] = out->dims[i];
    out_strides[i] = out->strides[i];
  }
  int64_t cnt = out->numel;

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32: {
    int32_t *out_data = (int32_t *)out->data;
    for (int64_t linear = 0; linear < cnt; ++linear) {
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = (int32_t)numpy_legacy_random_binomial(st, n, p);
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *out_data = (int64_t *)out->data;
    for (int64_t linear = 0; linear < cnt; ++linear) {
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = numpy_legacy_random_binomial(st, n, p);
    }
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("binomial: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(binomial, INSIGHT_DTYPE_I32, binomial_kernel_cpu);
REGISTER_CPU_KERNEL(binomial, INSIGHT_DTYPE_I64, binomial_kernel_cpu);