// backends/cpu/kernels/reduction/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU reduction kernels.
 *
 * Provides helper functions and macros for reduction operations.
 * Input arrays are guaranteed to be contiguous (batch_size 脳 reduce_size
 * layout) by the frontend preparation step.
 */

#ifndef CPU_REDUCTION_COMMON_H
#define CPU_REDUCTION_COMMON_H

#include "../../registry/cpu_registry.h"
#include "../common/half_utils.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cmath>
#include <float.h>
#include <limits>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NaN detection helpers
// ============================================================================

static inline bool cpu_is_nan_f32(float x) { return isnan(x); }

static inline bool cpu_is_nan_f64(double x) { return isnan(x); }

static inline bool cpu_is_nan_other(const void *x) {
  (void)x;
  return false;
}

// ============================================================================
// Reduction loop macros
// ============================================================================

/**
 * @brief Macro for sum reduction.
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_SUM_LOOP(CTYPE)                                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE sum = 0;                                                           \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        sum += src[i * reduce_size + j];                                       \
      }                                                                        \
      dst[i] = sum;                                                            \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for max reduction.
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_MAX_LOOP(CTYPE)                                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE max_val = src[i * reduce_size];                                    \
      for (int64_t j = 1; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (std::isnan(val)) {                                                 \
          max_val = val;                                                       \
          break;                                                               \
        }                                                                      \
        if (val > max_val)                                                     \
          max_val = val;                                                       \
      }                                                                        \
      dst[i] = max_val;                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for min reduction.
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_MIN_LOOP(CTYPE)                                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE min_val = src[i * reduce_size];                                    \
      for (int64_t j = 1; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (std::isnan(val)) {                                                 \
          min_val = val;                                                       \
          break;                                                               \
        }                                                                      \
        if (val < min_val)                                                     \
          min_val = val;                                                       \
      }                                                                        \
      dst[i] = min_val;                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for prod reduction.
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_PROD_LOOP(CTYPE)                                                \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE prod = 1;                                                          \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        prod *= src[i * reduce_size + j];                                      \
      }                                                                        \
      dst[i] = prod;                                                           \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for any reduction (logical OR, returns bool).
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_ANY_LOOP(CTYPE)                                                 \
  do {                                                                         \
    bool *dst = (bool *)out->data;                                             \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      bool any = false;                                                        \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        if (src[i * reduce_size + j] != 0) {                                   \
          any = true;                                                          \
          break;                                                               \
        }                                                                      \
      }                                                                        \
      dst[i] = any;                                                            \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for all reduction (logical AND, returns bool).
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_ALL_LOOP(CTYPE)                                                 \
  do {                                                                         \
    bool *dst = (bool *)out->data;                                             \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      bool all_true = true;                                                    \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        if (src[i * reduce_size + j] == 0) {                                   \
          all_true = false;                                                    \
          break;                                                               \
        }                                                                      \
      }                                                                        \
      dst[i] = all_true;                                                       \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for argmax reduction (returns index).
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_ARGMAX_LOOP(CTYPE)                                              \
  do {                                                                         \
    int64_t *dst = (int64_t *)out->data;                                       \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      int64_t max_idx = 0;                                                     \
      CTYPE max_val = src[i * reduce_size];                                    \
      if (std::isnan(max_val)) {                                               \
        dst[i] = 0;                                                            \
        continue;                                                              \
      }                                                                        \
      for (int64_t j = 1; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (std::isnan(val)) {                                                 \
          max_idx = j;                                                         \
          break;                                                               \
        }                                                                      \
        if (val > max_val) {                                                   \
          max_val = val;                                                       \
          max_idx = j;                                                         \
        }                                                                      \
      }                                                                        \
      dst[i] = max_idx;                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for argmin reduction (returns index).
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_ARGMIN_LOOP(CTYPE)                                              \
  do {                                                                         \
    int64_t *dst = (int64_t *)out->data;                                       \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      int64_t min_idx = 0;                                                     \
      CTYPE min_val = src[i * reduce_size];                                    \
      if (std::isnan(min_val)) {                                               \
        dst[i] = 0;                                                            \
        continue;                                                              \
      }                                                                        \
      for (int64_t j = 1; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (std::isnan(val)) {                                                 \
          min_idx = j;                                                         \
          break;                                                               \
        }                                                                      \
        if (val < min_val) {                                                   \
          min_val = val;                                                       \
          min_idx = j;                                                         \
        }                                                                      \
      }                                                                        \
      dst[i] = min_idx;                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for count_nonzero reduction.
 *
 * @param CTYPE C data type (e.g., float, int32_t)
 */
#define REDUCE_COUNT_NONZERO_LOOP(CTYPE)                                       \
  do {                                                                         \
    int64_t *dst = (int64_t *)out->data;                                       \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      int64_t count = 0;                                                       \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        if (src[i * reduce_size + j] != 0) {                                   \
          ++count;                                                             \
        }                                                                      \
      }                                                                        \
      dst[i] = count;                                                          \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for nanmax reduction (ignore NaN).
 *
 * @param CTYPE C data type (float or double only)
 */
#define REDUCE_NANMAX_LOOP(CTYPE, IS_NAN_FUNC)                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE max_val = 0;                                                       \
      bool found = false;                                                      \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (!IS_NAN_FUNC(val)) {                                               \
          if (!found) {                                                        \
            max_val = val;                                                     \
            found = true;                                                      \
          } else if (val > max_val) {                                          \
            max_val = val;                                                     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      /* NumPy nanmax of an all-NaN slice is NaN, not 0. */                    \
      dst[i] = found ? max_val : std::numeric_limits<CTYPE>::quiet_NaN();      \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for nanmin reduction (ignore NaN).
 *
 * @param CTYPE C data type (float or double only)
 */
#define REDUCE_NANMIN_LOOP(CTYPE, IS_NAN_FUNC)                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE min_val = 0;                                                       \
      bool found = false;                                                      \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (!IS_NAN_FUNC(val)) {                                               \
          if (!found) {                                                        \
            min_val = val;                                                     \
            found = true;                                                      \
          } else if (val < min_val) {                                          \
            min_val = val;                                                     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      /* NumPy nanmin of an all-NaN slice is NaN, not 0. */                    \
      dst[i] = found ? min_val : std::numeric_limits<CTYPE>::quiet_NaN();      \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for nansum reduction (sum ignoring NaN).
 *
 * @param CTYPE C data type (float or double only)
 */
#define REDUCE_NANSUM_LOOP(CTYPE, IS_NAN_FUNC)                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int64_t total_out = batch_size;                                            \
    _Pragma("omp parallel for") for (int64_t i = 0; i < total_out; ++i) {      \
      CTYPE sum = 0;                                                           \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (!IS_NAN_FUNC(val)) {                                               \
          sum += val;                                                          \
        }                                                                      \
      }                                                                        \
      dst[i] = sum;                                                            \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for nanvar reduction (variance ignoring NaN).
 *
 * @param CTYPE C data type (float or double only)
 */
#define REDUCE_NANVAR_LOOP(CTYPE, IS_NAN_FUNC)                                 \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    int ddof = *(int *)inputs[4];                                              \
    int64_t total_out = batch_size;                                            \
    for (int64_t i = 0; i < total_out; ++i) {                                  \
      int64_t count = 0;                                                       \
      CTYPE sum = 0;                                                           \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (!IS_NAN_FUNC(val)) {                                               \
          sum += val;                                                          \
          ++count;                                                             \
        }                                                                      \
      }                                                                        \
      if (count == 0) {                                                        \
        dst[i] = 0;                                                            \
        continue;                                                              \
      }                                                                        \
      CTYPE mean = sum / (CTYPE)count;                                         \
      CTYPE sq_sum = 0;                                                        \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        CTYPE val = src[i * reduce_size + j];                                  \
        if (!IS_NAN_FUNC(val)) {                                               \
          CTYPE diff = val - mean;                                             \
          sq_sum += diff * diff;                                               \
        }                                                                      \
      }                                                                        \
      CTYPE divisor = (CTYPE)(count - ddof);                                   \
      if (divisor <= 0)                                                        \
        divisor = 1;                                                           \
      dst[i] = sq_sum / divisor;                                               \
    }                                                                          \
  } while (0)

// ============================================================================
// Half-precision reduction macros
// ============================================================================

/**
 * @brief Macro for sum reduction on half-precision types.
 * Accumulates in float32, converts back to storage type.
 */
#define REDUCE_HALF_SUM_LOOP(CTYPE, to_f32, from_f32)                          \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    _Pragma("omp parallel for") for (int64_t i = 0; i < batch_size; ++i) {     \
      float sum = 0.0f;                                                        \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        sum += to_f32(src[i * reduce_size + j]);                               \
      }                                                                        \
      dst[i] = from_f32(sum);                                                  \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for max reduction on half-precision types.
 */
#define REDUCE_HALF_MAX_LOOP(CTYPE, to_f32, from_f32)                          \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    _Pragma("omp parallel for") for (int64_t i = 0; i < batch_size; ++i) {     \
      float max_val = to_f32(src[i * reduce_size]);                            \
      for (int64_t j = 1; j < reduce_size; ++j) {                              \
        float val = to_f32(src[i * reduce_size + j]);                          \
        if (val > max_val)                                                     \
          max_val = val;                                                       \
      }                                                                        \
      dst[i] = from_f32(max_val);                                              \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for min reduction on half-precision types.
 */
#define REDUCE_HALF_MIN_LOOP(CTYPE, to_f32, from_f32)                          \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    _Pragma("omp parallel for") for (int64_t i = 0; i < batch_size; ++i) {     \
      float min_val = to_f32(src[i * reduce_size]);                            \
      for (int64_t j = 1; j < reduce_size; ++j) {                              \
        float val = to_f32(src[i * reduce_size + j]);                          \
        if (val < min_val)                                                     \
          min_val = val;                                                       \
      }                                                                        \
      dst[i] = from_f32(min_val);                                              \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for prod reduction on half-precision types.
 */
#define REDUCE_HALF_PROD_LOOP(CTYPE, to_f32, from_f32)                         \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src = (const CTYPE *)prepared->data;                          \
    int64_t batch_size = *(int64_t *)inputs[2];                                \
    int64_t reduce_size = *(int64_t *)inputs[3];                               \
    _Pragma("omp parallel for") for (int64_t i = 0; i < batch_size; ++i) {     \
      float prod = 1.0f;                                                       \
      for (int64_t j = 0; j < reduce_size; ++j) {                              \
        prod *= to_f32(src[i * reduce_size + j]);                              \
      }                                                                        \
      dst[i] = from_f32(prod);                                                 \
    }                                                                          \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif // CPU_REDUCTION_COMMON_H
