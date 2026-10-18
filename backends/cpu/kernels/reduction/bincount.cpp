// backends/cpu/kernels/reduction/bincount.cpp
/**
 * @file bincount.cpp
 * @brief CPU kernel for bincount (unweighted).
 *
 * Counts occurrences of each integer value in the input array.
 * Output array is pre-initialized to zeros.
 *
 * @param inputs  [0] = InsightArray* output (pre-zeroed)
 *                [1] = InsightArray* input (1D integer array)
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status bincount_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("bincount: null array pointer");
    return C_FAILED;
  }

  int64_t n = x->numel;
  int64_t out_len = out->numel;
  int64_t *dst = (int64_t *)out->data;
  int out_of_range = 0;

  switch (x->dtype) {
  case INSIGHT_DTYPE_I32: {
    const int32_t *src = (const int32_t *)x->data;
    _Pragma("omp parallel for reduction(| : out_of_range)") for (int64_t i = 0;
                                                                i < n; ++i) {
      int64_t idx = src[i];
      if (idx < 0 || idx >= out_len) {
        out_of_range = 1;
        continue;
      }
      _Pragma("omp atomic") dst[idx]++;
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    const int64_t *src = (const int64_t *)x->data;
    _Pragma("omp parallel for reduction(| : out_of_range)") for (int64_t i = 0;
                                                                i < n; ++i) {
      int64_t idx = src[i];
      if (idx < 0 || idx >= out_len) {
        out_of_range = 1;
        continue;
      }
      _Pragma("omp atomic") dst[idx]++;
    }
    break;
  }
  default:
    cpu_set_last_error("bincount: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }

  if (out_of_range) {
    cpu_set_last_error("bincount: value out of range for output buffer");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(bincount, INSIGHT_DTYPE_I32, bincount_kernel_cpu);
REGISTER_CPU_KERNEL(bincount, INSIGHT_DTYPE_I64, bincount_kernel_cpu);
