// backends/cpu/kernels/elementwise/pow.cpp
/**
 * @file pow.cpp
 * @brief CPU kernel for power operation.
 *
 * Computes elementwise a^b (a raised to power b) with stride support.
 * For integer types, uses fast binary exponentiation.
 * For floating point types, uses standard powf/pow functions.
 * For complex types, uses cpowf/cpow.
 *
 * @param inputs  [0] = InsightArray* base
 *                [1] = InsightArray* exponent
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../common/half_utils.h"
#include "common.h"
#include <complex.h>

// ============================================================================
// Integer Power Function (fast exponentiation)
// ============================================================================

/**
 * @brief Fast integer exponentiation using binary exponentiation.
 *
 * Computes base^exp for integer types. Note that overflow is not checked
 * and follows C's unsigned/signed overflow behavior.
 *
 * @param base Base value
 * @param exp  Exponent (non-negative)
 * @return base^exp
 */
static inline int64_t cpu_int_pow(int64_t base, int64_t exp) {
  int64_t result = 1;
  int64_t b = base;
  int64_t e = exp;

  while (e > 0) {
    if (e & 1) {
      result *= b;
    }
    b *= b;
    e >>= 1;
  }

  return result;
}

#ifdef __cplusplus
extern "C" {
#endif

static void pow_loop_bool(const bool *a_data, const int64_t *a_strides,
                          const bool *b_data, const int64_t *b_strides,
                          bool *out_data, const int64_t *out_strides,
                          int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    bool a_val = a_data[off_a];
    bool b_val = b_data[off_b];
    // bool^bool: false^false = true? Actually 0^0 = 1, but use integer logic
    out_data[off_out] = (b_val == 0) ? 1 : a_val;
  }
}

static void pow_loop_uint64(const uint64_t *a_data, const int64_t *a_strides,
                            const uint64_t *b_data, const int64_t *b_strides,
                            uint64_t *out_data, const int64_t *out_strides,
                            int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    uint64_t base = a_data[off_a];
    uint64_t exp = b_data[off_b];
    uint64_t result = 1;
    uint64_t b_temp = base;
    uint64_t e_temp = exp;

    while (e_temp > 0) {
      if (e_temp & 1) {
        result *= b_temp;
      }
      b_temp *= b_temp;
      e_temp >>= 1;
    }
    out_data[off_out] = result;
  }
}

static void pow_loop_int64(const int64_t *a_data, const int64_t *a_strides,
                           const int64_t *b_data, const int64_t *b_strides,
                           int64_t *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = cpu_int_pow(a_data[off_a], b_data[off_b]);
  }
}

static void pow_loop_float(const float *a_data, const int64_t *a_strides,
                           const float *b_data, const int64_t *b_strides,
                           float *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = powf(a_data[off_a], b_data[off_b]);
  }
}

static void pow_loop_double(const double *a_data, const int64_t *a_strides,
                            const double *b_data, const int64_t *b_strides,
                            double *out_data, const int64_t *out_strides,
                            int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = pow(a_data[off_a], b_data[off_b]);
  }
}

static void pow_loop_float_complex(const std::complex<float> *a_data,
                                   const int64_t *a_strides,
                                   const std::complex<float> *b_data,
                                   const int64_t *b_strides,
                                   std::complex<float> *out_data,
                                   const int64_t *out_strides, int64_t ndim,
                                   const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = std::pow(a_data[off_a], b_data[off_b]);
  }
}

static void pow_loop_double_complex(const std::complex<double> *a_data,
                                    const int64_t *a_strides,
                                    const std::complex<double> *b_data,
                                    const int64_t *b_strides,
                                    std::complex<double> *out_data,
                                    const int64_t *out_strides, int64_t ndim,
                                    const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = std::pow(a_data[off_a], b_data[off_b]);
  }
}

C_Status pow_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("pow: null array pointer");
    return C_FAILED;
  }

  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("pow: shape mismatch");
    return C_FAILED;
  }

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

  switch (a->dtype) {
  case INSIGHT_DTYPE_BOOL:
    pow_loop_bool((const bool *)a->data + a->offset, a_strides,
                  (const bool *)b->data + b->offset, b_strides,
                  (bool *)out->data + out->offset, out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_U8:
    pow_loop_uint64((const uint64_t *)a->data + a->offset, a_strides,
                    (const uint64_t *)b->data + b->offset, b_strides,
                    (uint64_t *)out->data + out->offset, out_strides, ndim,
                    dims, n);
    break;
  case INSIGHT_DTYPE_I8:
  case INSIGHT_DTYPE_I16:
  case INSIGHT_DTYPE_I32:
  case INSIGHT_DTYPE_I64:
    pow_loop_int64((const int64_t *)a->data + a->offset, a_strides,
                   (const int64_t *)b->data + b->offset, b_strides,
                   (int64_t *)out->data + out->offset, out_strides, ndim, dims,
                   n);
    break;
  case INSIGHT_DTYPE_U16:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_U64:
    pow_loop_uint64((const uint64_t *)a->data + a->offset, a_strides,
                    (const uint64_t *)b->data + b->offset, b_strides,
                    (uint64_t *)out->data + out->offset, out_strides, ndim,
                    dims, n);
    break;
  case INSIGHT_DTYPE_F32:
    pow_loop_float((const float *)a->data + a->offset, a_strides,
                   (const float *)b->data + b->offset, b_strides,
                   (float *)out->data + out->offset, out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_F64:
    pow_loop_double((const double *)a->data + a->offset, a_strides,
                    (const double *)b->data + b->offset, b_strides,
                    (double *)out->data + out->offset, out_strides, ndim, dims,
                    n);
    break;
  case INSIGHT_DTYPE_C32:
    pow_loop_float_complex((const std::complex<float> *)a->data + a->offset,
                           a_strides,
                           (const std::complex<float> *)b->data + b->offset,
                           b_strides,
                           (std::complex<float> *)out->data + out->offset,
                           out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_C64:
    pow_loop_double_complex((const std::complex<double> *)a->data + a->offset,
                            a_strides,
                            (const std::complex<double> *)b->data + b->offset,
                            b_strides,
                            (std::complex<double> *)out->data + out->offset,
                            out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_F16: {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *out_data = (uint16_t *)out->data;
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
      out_data[off_out] = insight::f32_to_f16(powf(va, vb));
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *out_data = (uint16_t *)out->data;
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
      out_data[off_out] = insight::f32_to_bf16(powf(va, vb));
    }
    break;
  }
  default:
    cpu_set_last_error("pow: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_BOOL, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_U8, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_I8, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_I16, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_I32, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_I64, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_U16, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_U32, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_U64, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_F32, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_F64, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_C32, pow_kernel_cpu);
REGISTER_CPU_KERNEL(pow, INSIGHT_DTYPE_C64, pow_kernel_cpu);
