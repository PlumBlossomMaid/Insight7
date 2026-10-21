// backends/cpu/kernels/cast/cast_c32.cpp
/**
 * @file cast_c32.cpp
 * @brief CPU kernel for casting from c32 to all types.
 *
 * Copies layout from source to destination and converts data.
 *
 * @param inputs  [0] = InsightArray* source
 *                [1] = int32_t* target_dtype
 * @param outputs [0] = InsightArray* destination
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status cast_c32_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *src = static_cast<InsightArray *>(inputs[0]);
  InsightArray *dst = static_cast<InsightArray *>(outputs[0]);
  int32_t target_dtype = *static_cast<int32_t *>(inputs[1]);

  if (!src || !dst) {
    cpu_set_last_error("cast_c32: null array pointer");
    return C_FAILED;
  }

  // Copy layout from source to destination
  copy_layout(dst, src);
  dst->dtype = target_dtype;

  // Data conversion
  int64_t n = src->numel;
  const std::complex<float> *s =
      static_cast<const std::complex<float> *>(src->data);

  switch (target_dtype) {
  case INSIGHT_DTYPE_BOOL: {
    bool *d = static_cast<bool *>(dst->data);
    CAST_LOOP(n, d[i] = s[si].real() != 0.0f || s[si].imag() != 0.0f;);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    float *d = static_cast<float *>(dst->data);
    CAST_LOOP(n, d[i] = s[si].real(););
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *d = static_cast<double *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<double>(s[si].real()););
    break;
  }
  case INSIGHT_DTYPE_C64: {
    std::complex<double> *d = static_cast<std::complex<double> *>(dst->data);
    CAST_LOOP(n,
              d[i] = std::complex<double>(static_cast<double>(s[si].real()),
                                          static_cast<double>(s[si].imag())););
    break;
  }
  default:
    cpu_set_last_error("cast_c32: unsupported target type");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_C32, cast_c32_kernel_cpu);
