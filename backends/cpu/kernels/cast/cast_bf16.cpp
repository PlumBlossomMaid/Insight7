// backends/cpu/kernels/cast/cast_bf16.cpp
/**
 * @file cast_bf16.cpp
 * @brief CPU kernel for casting from bf16 to all types.
 *
 * Converts bfloat16 (stored as uint16_t) to target dtype via f32 intermediate.
 */

#include "../common/half_utils.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status cast_bf16_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *src = static_cast<InsightArray *>(inputs[0]);
  InsightArray *dst = static_cast<InsightArray *>(outputs[0]);
  int32_t target_dtype = *static_cast<int32_t *>(inputs[1]);

  if (!src || !dst) {
    cpu_set_last_error("cast_bf16: null array pointer");
    return C_FAILED;
  }

  copy_layout(dst, src);
  dst->dtype = target_dtype;

  int64_t n = src->numel;
  const uint16_t *s = static_cast<const uint16_t *>(src->data);

  switch (target_dtype) {
  case INSIGHT_DTYPE_BOOL: {
    bool *d = static_cast<bool *>(dst->data);
    CAST_LOOP(n, d[i] = insight::bf16_to_f32(s[si]) != 0.0f;);
    break;
  }
  case INSIGHT_DTYPE_U8: {
    uint8_t *d = static_cast<uint8_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint8_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_I8: {
    int8_t *d = static_cast<int8_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int8_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_I16: {
    int16_t *d = static_cast<int16_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int16_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *d = static_cast<int32_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int32_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *d = static_cast<int64_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<int64_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_U16: {
    uint16_t *d = static_cast<uint16_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint16_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_U32: {
    uint32_t *d = static_cast<uint32_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint32_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_U64: {
    uint64_t *d = static_cast<uint64_t *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<uint64_t>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_F16: {
    uint16_t *d = static_cast<uint16_t *>(dst->data);
    CAST_LOOP(n, d[i] = insight::f32_to_f16(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_F32: {
    float *d = static_cast<float *>(dst->data);
    CAST_LOOP(n, d[i] = insight::bf16_to_f32(s[si]););
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *d = static_cast<double *>(dst->data);
    CAST_LOOP(n, d[i] = static_cast<double>(insight::bf16_to_f32(s[si])););
    break;
  }
  case INSIGHT_DTYPE_C32: {
    std::complex<float> *d = static_cast<std::complex<float> *>(dst->data);
    CAST_LOOP(n, d[i] = std::complex<float>(insight::bf16_to_f32(s[si]), 0.0f););
    break;
  }
  case INSIGHT_DTYPE_C64: {
    std::complex<double> *d = static_cast<std::complex<double> *>(dst->data);
    CAST_LOOP(n, d[i] = std::complex<double>(
                     static_cast<double>(insight::bf16_to_f32(s[si])), 0.0););
    break;
  }
  default:
    cpu_set_last_error("cast_bf16: unsupported target type");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_BF16, cast_bf16_kernel_cpu);
