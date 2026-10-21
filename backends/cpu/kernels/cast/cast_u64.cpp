// backends/cpu/kernels/cast/cast_u64.cpp
/**
 * @file cast_u64.cpp
 * @brief CPU kernel for casting from u64 to all types.
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

C_Status cast_u64_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *src = static_cast<InsightArray *>(inputs[0]);
  InsightArray *dst = static_cast<InsightArray *>(outputs[0]);
  int32_t target_dtype = *static_cast<int32_t *>(inputs[1]);

  if (!src || !dst) {
    cpu_set_last_error("cast_u64: null array pointer");
    return C_FAILED;
  }

  // Copy layout from source to destination
  copy_layout(dst, src);
  dst->dtype = target_dtype;

  // Data conversion
  int64_t n = src->numel;
  const uint64_t *s = static_cast<const uint64_t *>(src->data);

  switch (target_dtype) {
  case INSIGHT_DTYPE_BOOL: {
    bool *d = static_cast<bool *>(dst->data);
    CAST_LOOP(n, d[i] = s[si] != 0;);
    break;
  }
  case INSIGHT_DTYPE_U8: {
    uint8_t *d = static_cast<uint8_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint8_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_I8: {
    int8_t *d = static_cast<int8_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int8_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_I16: {
    int16_t *d = static_cast<int16_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int16_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *d = static_cast<int32_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int32_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *d = static_cast<int64_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int64_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_U16: {
    uint16_t *d = static_cast<uint16_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint16_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_U32: {
    uint32_t *d = static_cast<uint32_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint32_t>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_F32: {
    float *d = static_cast<float *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<float>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *d = static_cast<double *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<double>(s[si]););
    break;
  }
  case INSIGHT_DTYPE_C32: {
    std::complex<float> *d = static_cast<std::complex<float> *>(dst->data);
    CAST_LOOP(n, d[i] = std::complex<float>(static_cast<float>(s[si]), 0.0f););
    break;
  }
  case INSIGHT_DTYPE_C64: {
    std::complex<double> *d = static_cast<std::complex<double> *>(dst->data);
    CAST_LOOP(n, d[i] = std::complex<double>(static_cast<double>(s[si]), 0.0););
    break;
  }
  default:
    cpu_set_last_error("cast_u64: unsupported target type");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_U64, cast_u64_kernel_cpu);
