// backends/cpu/kernels/unary/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU unary kernels.
 *
 * Provides stride-aware iteration macros for implementing unary operations
 * on arrays with arbitrary memory layouts.
 */

#ifndef CPU_UNARY_COMMON_H
#define CPU_UNARY_COMMON_H

#include "../../registry/cpu_registry.h"
#include "../common/half_utils.h"
#include "insight/c_api/array.h"
#include <complex>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert linear index to element offset using dimension strides.
 */
static inline int64_t cpu_offset_from_linear(int64_t linear, int64_t ndim,
                                             const int64_t *dims,
                                             const int64_t *strides) {

  int64_t offset = 0;
  int64_t remaining = linear;

  for (int64_t d = ndim - 1; d >= 0; --d) {
    int64_t idx = remaining % dims[d];
    remaining /= dims[d];
    offset += idx * strides[d];
  }

  return offset;
}

/**
 * @brief Macro for unary operations with stride support.
 *
 * @param CTYPE    C data type (e.g., float, int32_t)
 * @param Func     Function to apply to each element (e.g., std::sin, std::abs)
 */
#define UNARY_KERNEL_LOOP(CTYPE, Func)                                         \
  do {                                                                         \
    const CTYPE *x_data = (const CTYPE *)x->data;                              \
    CTYPE *out_data = (CTYPE *)out->data;                                      \
    int64_t ndim = out->ndim;                                                  \
    int64_t dims[INSIGHT_MAX_NDIM];                                            \
    int64_t x_strides[INSIGHT_MAX_NDIM];                                       \
    int64_t out_strides[INSIGHT_MAX_NDIM];                                     \
    for (int i = 0; i < ndim; ++i) {                                           \
      dims[i] = out->dims[i];                                                  \
      x_strides[i] = x->strides[i];                                            \
      out_strides[i] = out->strides[i];                                        \
    }                                                                          \
    int64_t n = out->numel;                                                    \
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n;           \
                                     ++linear) {                               \
      int64_t off_x =                                                          \
          x->offset + cpu_offset_from_linear(linear, ndim, dims, x_strides);   \
      int64_t off_out = out->offset + cpu_offset_from_linear(                  \
                                          linear, ndim, dims, out_strides);    \
      out_data[off_out] = Func(x_data[off_x]);                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for unary operations returning bool (e.g., logical_not).
 */
#define UNARY_CMP_LOOP(CTYPE, Func)                                            \
  do {                                                                         \
    const CTYPE *x_data = (const CTYPE *)x->data;                              \
    bool *out_data = (bool *)out->data;                                        \
    int64_t ndim = out->ndim;                                                  \
    int64_t dims[INSIGHT_MAX_NDIM];                                            \
    int64_t x_strides[INSIGHT_MAX_NDIM];                                       \
    int64_t out_strides[INSIGHT_MAX_NDIM];                                     \
    for (int i = 0; i < ndim; ++i) {                                           \
      dims[i] = out->dims[i];                                                  \
      x_strides[i] = x->strides[i];                                            \
      out_strides[i] = out->strides[i];                                        \
    }                                                                          \
    int64_t n = out->numel;                                                    \
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n;           \
                                     ++linear) {                               \
      int64_t off_x =                                                          \
          x->offset + cpu_offset_from_linear(linear, ndim, dims, x_strides);   \
      int64_t off_out = out->offset + cpu_offset_from_linear(                  \
                                          linear, ndim, dims, out_strides);    \
      out_data[off_out] = Func(x_data[off_x]);                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for unary half-precision (fp16/bf16) operations.
 *
 * Converts each element to float32, applies Func, and converts back.
 *
 * @param CTYPE      uint16_t storage type
 * @param to_f32     Conversion function: storage -> float
 * @param from_f32   Conversion function: float -> storage
 * @param Func       float -> float unary function
 */
#define UNARY_HALF_LOOP(CTYPE, to_f32, from_f32, Func)                         \
  do {                                                                         \
    const CTYPE *x_data = (const CTYPE *)x->data;                              \
    CTYPE *out_data = (CTYPE *)out->data;                                      \
    int64_t ndim = out->ndim;                                                  \
    int64_t dims[INSIGHT_MAX_NDIM];                                            \
    int64_t x_strides[INSIGHT_MAX_NDIM];                                       \
    int64_t out_strides[INSIGHT_MAX_NDIM];                                     \
    for (int i = 0; i < ndim; ++i) {                                           \
      dims[i] = out->dims[i];                                                  \
      x_strides[i] = x->strides[i];                                            \
      out_strides[i] = out->strides[i];                                        \
    }                                                                          \
    int64_t n = out->numel;                                                    \
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n;           \
                                     ++linear) {                               \
      int64_t off_x =                                                          \
          x->offset + cpu_offset_from_linear(linear, ndim, dims, x_strides);   \
      int64_t off_out = out->offset + cpu_offset_from_linear(                  \
                                          linear, ndim, dims, out_strides);    \
      float v = to_f32(x_data[off_x]);                                         \
      out_data[off_out] = from_f32(Func(v));                                   \
    }                                                                          \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif // CPU_UNARY_COMMON_H
