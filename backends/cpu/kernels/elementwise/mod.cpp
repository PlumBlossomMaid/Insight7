// backends/cpu/kernels/elementwise/mod.cpp
/**
 * @file mod.cpp
 * @brief CPU kernel for modulo operation.
 *
 * Computes elementwise a % b (remainder of a divided by b) with stride support.
 * Uses floored division semantics (the result takes the sign of the divisor),
 * matching NumPy's mod/remainder for both integer and floating point types.
 * Complex numbers are not supported for modulo operation.
 *
 * @param inputs  [0] = InsightArray* dividend
 *                [1] = InsightArray* divisor
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../common/half_utils.h"
#include "common.h"
#include <math.h>
#include <type_traits>

/**
 * @brief Floored modulo for integer types.
 *
 * A zero divisor yields 0 (NumPy behaviour) instead of raising SIGFPE.
 */
template <typename T> static inline T cpu_mod_int(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T r = (T)(a % b);
  if constexpr (std::is_signed<T>::value) {
    if (r != 0 && ((r < 0) != (b < 0))) {
      r = (T)(r + b);
    }
  }
  return r;
}

/**
 * @brief Floored modulo for single-precision float.
 */
static inline float cpu_mod_float(float a, float b) {
  float r = fmodf(a, b);
  if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) {
    r += b;
  }
  return r;
}

/**
 * @brief Floored modulo for double-precision float.
 */
static inline double cpu_mod_double(double a, double b) {
  double r = fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) {
    r += b;
  }
  return r;
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status mod_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("mod: null array pointer");
    return C_FAILED;
  }

  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("mod: shape mismatch");
    return C_FAILED;
  }

  switch (a->dtype) {
  case INSIGHT_DTYPE_U8:
    ELEMENTWISE_KERNEL_LOOP_Func(uint8_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_I8:
    ELEMENTWISE_KERNEL_LOOP_Func(int8_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_I16:
    ELEMENTWISE_KERNEL_LOOP_Func(int16_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_I32:
    ELEMENTWISE_KERNEL_LOOP_Func(int32_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_I64:
    ELEMENTWISE_KERNEL_LOOP_Func(int64_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_U16:
    ELEMENTWISE_KERNEL_LOOP_Func(uint16_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_U32:
    ELEMENTWISE_KERNEL_LOOP_Func(uint32_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_U64:
    ELEMENTWISE_KERNEL_LOOP_Func(uint64_t, cpu_mod_int);
    break;
  case INSIGHT_DTYPE_F32:
    ELEMENTWISE_KERNEL_LOOP_Func(float, cpu_mod_float);
    break;
  case INSIGHT_DTYPE_F64:
    ELEMENTWISE_KERNEL_LOOP_Func(double, cpu_mod_double);
    break;
  case INSIGHT_DTYPE_F16: {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *out_data = (uint16_t *)out->data;
    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t a_strides[INSIGHT_MAX_NDIM];
    int64_t b_strides[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = out->dims[i];
      a_strides[i] = a->strides[i];
      b_strides[i] = b->strides[i];
      out_strides[i] = out->strides[i];
    }
    int64_t n = out->numel;
#pragma omp parallel for
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_a =
          a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides);
      int64_t off_b =
          b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides);
      int64_t off_out =
          out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides);
      float va = insight::f16_to_f32(a_data[off_a]);
      float vb = insight::f16_to_f32(b_data[off_b]);
      out_data[off_out] = insight::f32_to_f16(cpu_mod_float(va, vb));
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *out_data = (uint16_t *)out->data;
    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t a_strides[INSIGHT_MAX_NDIM];
    int64_t b_strides[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = out->dims[i];
      a_strides[i] = a->strides[i];
      b_strides[i] = b->strides[i];
      out_strides[i] = out->strides[i];
    }
    int64_t n = out->numel;
#pragma omp parallel for
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_a =
          a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides);
      int64_t off_b =
          b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides);
      int64_t off_out =
          out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides);
      float va = insight::bf16_to_f32(a_data[off_a]);
      float vb = insight::bf16_to_f32(b_data[off_b]);
      out_data[off_out] = insight::f32_to_bf16(cpu_mod_float(va, vb));
    }
    break;
  }
  default:
    cpu_set_last_error("mod: unsupported dtype (complex not supported)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U8, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I8, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I16, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I32, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I64, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U16, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U32, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U64, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_F32, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_F64, mod_kernel_cpu);
