// backends/cpu/kernels/indexing/unique.cpp
/**
 * @file unique.cpp
 * @brief CPU kernel for unique operation.
 *
 * Returns unique elements and optionally indices, inverse, counts.
 * The output arrays are allocated by this kernel.
 */

#include "common.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <insight/c_api/place.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Helper functions for special types
// ============================================================================

static inline bool is_half_zero(uint16_t val) {
  // Half-precision zero: exponent = 0, mantissa = 0, sign bit ignored
  return (val & 0x7FFF) == 0;
}

static inline bool is_bfloat16_zero(uint16_t val) {
  return (val & 0x7FFF) == 0;
}

static inline bool is_float8_e4m3_zero(uint8_t val) {
  return val == 0 || val == 0x80;
}

static inline bool is_float8_e5m2_zero(uint8_t val) {
  return val == 0 || val == 0x80;
}

// ============================================================================
// Helper: allocate and set output array metadata
// ============================================================================

static void *allocate_output(InsightArray *out, size_t elem_size, int32_t dtype,
                             int64_t numel) {
  size_t bytes = elem_size * numel;
  void *data = NULL;
  if (bytes > 0) {
    data = malloc(bytes);
    if (!data)
      return NULL;
  }

  out->data = data;
  out->storage_nbytes = bytes;
  out->ndim = 1;
  out->dims[0] = numel;
  out->numel = numel;
  out->dtype = dtype;
  out->strides[0] = 1;
  out->offset = 0;
  out->is_view = 0;
  out->device_type = INSIGHT_DEVICE_CPU;
  out->device_id = 0;

  if (!out->ref_count) {
    out->ref_count = new int32_t;
    if (out->ref_count)
      *out->ref_count = 1;
  }

  return data;
}

// ============================================================================
// Macros for unique implementation
// ============================================================================

#define UNIQUE_IMPL(CTYPE)                                                     \
  do {                                                                         \
    const CTYPE *src = (const CTYPE *)x->data;                                 \
    std::vector<std::pair<CTYPE, int64_t>> pairs(n);                           \
    for (int64_t i = 0; i < n; ++i) {                                          \
      pairs[i] = {src[i], i};                                                  \
    }                                                                          \
    std::sort(pairs.begin(), pairs.end(),                                      \
              [](const auto &a, const auto &b) { return a.first < b.first; }); \
                                                                               \
    unique_count = 0;                                                          \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0 || pairs[i].first != pairs[i - 1].first) {                    \
        unique_count++;                                                        \
      }                                                                        \
    }                                                                          \
                                                                               \
    std::vector<int64_t> inverse(n);                                           \
    int64_t unique_idx = 0;                                                    \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i > 0 && pairs[i].first != pairs[i - 1].first)                       \
        unique_idx++;                                                          \
      inverse[pairs[i].second] = unique_idx;                                   \
    }                                                                          \
                                                                               \
    std::vector<int64_t> first_occ(unique_count);                              \
    unique_idx = 0;                                                            \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0 || pairs[i].first != pairs[i - 1].first) {                    \
        first_occ[unique_idx] = pairs[i].second;                               \
        unique_idx++;                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    std::vector<int64_t> counts(unique_count, 0);                              \
    for (int64_t i = 0; i < n; ++i) {                                          \
      counts[inverse[i]]++;                                                    \
    }                                                                          \
                                                                               \
    CTYPE *dst_unique = (CTYPE *)allocate_output(unique_out, sizeof(CTYPE),    \
                                                 x->dtype, unique_count);      \
    if (!dst_unique && unique_count > 0) {                                     \
      cpu_set_last_error("unique: memory allocation failed");                  \
      return C_FAILED;                                                         \
    }                                                                          \
    unique_idx = 0;                                                            \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0 || pairs[i].first != pairs[i - 1].first) {                    \
        dst_unique[unique_idx] = pairs[i].first;                               \
        unique_idx++;                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    int out_idx = 1;                                                           \
    if (return_indices) {                                                      \
      int64_t *dst_indices = (int64_t *)allocate_output(                       \
          (InsightArray *)outputs[out_idx++], sizeof(int64_t),                 \
          INSIGHT_DTYPE_I64, unique_count);                                    \
      if (!dst_indices && unique_count > 0) {                                  \
        cpu_set_last_error("unique: memory allocation failed for indices");    \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < unique_count; ++i)                               \
        dst_indices[i] = first_occ[i];                                         \
    }                                                                          \
                                                                               \
    if (return_inverse) {                                                      \
      int64_t *dst_inverse =                                                   \
          (int64_t *)allocate_output((InsightArray *)outputs[out_idx++],       \
                                     sizeof(int64_t), INSIGHT_DTYPE_I64, n);   \
      if (!dst_inverse && n > 0) {                                             \
        cpu_set_last_error("unique: memory allocation failed for inverse");    \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < n; ++i)                                          \
        dst_inverse[i] = inverse[i];                                           \
    }                                                                          \
                                                                               \
    if (return_counts) {                                                       \
      int64_t *dst_counts = (int64_t *)allocate_output(                        \
          (InsightArray *)outputs[out_idx++], sizeof(int64_t),                 \
          INSIGHT_DTYPE_I64, unique_count);                                    \
      if (!dst_counts && unique_count > 0) {                                   \
        cpu_set_last_error("unique: memory allocation failed for counts");     \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < unique_count; ++i)                               \
        dst_counts[i] = counts[i];                                             \
    }                                                                          \
  } while (0)

// Special macro for 16-bit float types (need zero handling for comparison)
#define UNIQUE_IMPL_FLOAT16(CTYPE, IS_ZERO_FUNC)                               \
  do {                                                                         \
    const CTYPE *src = (const CTYPE *)x->data;                                 \
    std::vector<std::pair<CTYPE, int64_t>> pairs(n);                           \
    for (int64_t i = 0; i < n; ++i) {                                          \
      pairs[i] = {src[i], i};                                                  \
    }                                                                          \
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {   \
      uint16_t a_norm = (a.first & 0x7FFF);                                    \
      uint16_t b_norm = (b.first & 0x7FFF);                                    \
      if (a_norm == 0 && b_norm == 0)                                          \
        return false;                                                          \
      return a.first < b.first;                                                \
    });                                                                        \
                                                                               \
    unique_count = 0;                                                          \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0) {                                                            \
        unique_count++;                                                        \
      } else {                                                                 \
        uint16_t prev_norm = (pairs[i - 1].first & 0x7FFF);                    \
        uint16_t curr_norm = (pairs[i].first & 0x7FFF);                        \
        if (prev_norm == 0 && curr_norm == 0) {                                \
          continue;                                                            \
        }                                                                      \
        if (pairs[i].first != pairs[i - 1].first) {                            \
          unique_count++;                                                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    std::vector<int64_t> inverse(n);                                           \
    int64_t unique_idx = 0;                                                    \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i > 0) {                                                             \
        uint16_t prev_norm = (pairs[i - 1].first & 0x7FFF);                    \
        uint16_t curr_norm = (pairs[i].first & 0x7FFF);                        \
        if (!(prev_norm == 0 && curr_norm == 0) &&                             \
            pairs[i].first != pairs[i - 1].first) {                            \
          unique_idx++;                                                        \
        }                                                                      \
      }                                                                        \
      inverse[pairs[i].second] = unique_idx;                                   \
    }                                                                          \
                                                                               \
    std::vector<int64_t> first_occ(unique_count);                              \
    unique_idx = 0;                                                            \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0) {                                                            \
        first_occ[unique_idx] = pairs[i].second;                               \
        unique_idx++;                                                          \
      } else {                                                                 \
        uint16_t prev_norm = (pairs[i - 1].first & 0x7FFF);                    \
        uint16_t curr_norm = (pairs[i].first & 0x7FFF);                        \
        if (!(prev_norm == 0 && curr_norm == 0) &&                             \
            pairs[i].first != pairs[i - 1].first) {                            \
          first_occ[unique_idx] = pairs[i].second;                             \
          unique_idx++;                                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    std::vector<int64_t> counts(unique_count, 0);                              \
    for (int64_t i = 0; i < n; ++i) {                                          \
      counts[inverse[i]]++;                                                    \
    }                                                                          \
                                                                               \
    CTYPE *dst_unique = (CTYPE *)allocate_output(unique_out, sizeof(CTYPE),    \
                                                 x->dtype, unique_count);      \
    if (!dst_unique && unique_count > 0) {                                     \
      cpu_set_last_error("unique: memory allocation failed");                  \
      return C_FAILED;                                                         \
    }                                                                          \
    unique_idx = 0;                                                            \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0) {                                                            \
        dst_unique[unique_idx] = pairs[i].first;                               \
        unique_idx++;                                                          \
      } else {                                                                 \
        uint16_t prev_norm = (pairs[i - 1].first & 0x7FFF);                    \
        uint16_t curr_norm = (pairs[i].first & 0x7FFF);                        \
        if (!(prev_norm == 0 && curr_norm == 0) &&                             \
            pairs[i].first != pairs[i - 1].first) {                            \
          dst_unique[unique_idx] = pairs[i].first;                             \
          unique_idx++;                                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    int out_idx = 1;                                                           \
    if (return_indices) {                                                      \
      int64_t *dst_indices = (int64_t *)allocate_output(                       \
          (InsightArray *)outputs[out_idx++], sizeof(int64_t),                 \
          INSIGHT_DTYPE_I64, unique_count);                                    \
      if (!dst_indices && unique_count > 0) {                                  \
        cpu_set_last_error("unique: memory allocation failed for indices");    \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < unique_count; ++i)                               \
        dst_indices[i] = first_occ[i];                                         \
    }                                                                          \
                                                                               \
    if (return_inverse) {                                                      \
      int64_t *dst_inverse =                                                   \
          (int64_t *)allocate_output((InsightArray *)outputs[out_idx++],       \
                                     sizeof(int64_t), INSIGHT_DTYPE_I64, n);   \
      if (!dst_inverse && n > 0) {                                             \
        cpu_set_last_error("unique: memory allocation failed for inverse");    \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < n; ++i)                                          \
        dst_inverse[i] = inverse[i];                                           \
    }                                                                          \
                                                                               \
    if (return_counts) {                                                       \
      int64_t *dst_counts = (int64_t *)allocate_output(                        \
          (InsightArray *)outputs[out_idx++], sizeof(int64_t),                 \
          INSIGHT_DTYPE_I64, unique_count);                                    \
      if (!dst_counts && unique_count > 0) {                                   \
        cpu_set_last_error("unique: memory allocation failed for counts");     \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < unique_count; ++i)                               \
        dst_counts[i] = counts[i];                                             \
    }                                                                          \
  } while (0)

// Special macro for complex types
#define UNIQUE_IMPL_COMPLEX(CTYPE)                                             \
  do {                                                                         \
    const CTYPE *src = (const CTYPE *)x->data;                                 \
    std::vector<std::pair<CTYPE, int64_t>> pairs(n);                           \
    for (int64_t i = 0; i < n; ++i) {                                          \
      pairs[i] = {src[i], i};                                                  \
    }                                                                          \
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {   \
      if (a.first.real() != b.first.real())                                    \
        return a.first.real() < b.first.real();                                \
      return a.first.imag() < b.first.imag();                                  \
    });                                                                        \
                                                                               \
    unique_count = 0;                                                          \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0 || pairs[i].first.real() != pairs[i - 1].first.real() ||      \
          pairs[i].first.imag() != pairs[i - 1].first.imag()) {                \
        unique_count++;                                                        \
      }                                                                        \
    }                                                                          \
                                                                               \
    std::vector<int64_t> inverse(n);                                           \
    int64_t unique_idx = 0;                                                    \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i > 0 && (pairs[i].first.real() != pairs[i - 1].first.real() ||      \
                    pairs[i].first.imag() != pairs[i - 1].first.imag())) {     \
        unique_idx++;                                                          \
      }                                                                        \
      inverse[pairs[i].second] = unique_idx;                                   \
    }                                                                          \
                                                                               \
    std::vector<int64_t> first_occ(unique_count);                              \
    unique_idx = 0;                                                            \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0 || pairs[i].first.real() != pairs[i - 1].first.real() ||      \
          pairs[i].first.imag() != pairs[i - 1].first.imag()) {                \
        first_occ[unique_idx] = pairs[i].second;                               \
        unique_idx++;                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    std::vector<int64_t> counts(unique_count, 0);                              \
    for (int64_t i = 0; i < n; ++i) {                                          \
      counts[inverse[i]]++;                                                    \
    }                                                                          \
                                                                               \
    CTYPE *dst_unique = (CTYPE *)allocate_output(unique_out, sizeof(CTYPE),    \
                                                 x->dtype, unique_count);      \
    if (!dst_unique && unique_count > 0) {                                     \
      cpu_set_last_error("unique: memory allocation failed");                  \
      return C_FAILED;                                                         \
    }                                                                          \
    unique_idx = 0;                                                            \
    for (int64_t i = 0; i < n; ++i) {                                          \
      if (i == 0 || pairs[i].first.real() != pairs[i - 1].first.real() ||      \
          pairs[i].first.imag() != pairs[i - 1].first.imag()) {                \
        dst_unique[unique_idx] = pairs[i].first;                               \
        unique_idx++;                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    int out_idx = 1;                                                           \
    if (return_indices) {                                                      \
      int64_t *dst_indices = (int64_t *)allocate_output(                       \
          (InsightArray *)outputs[out_idx++], sizeof(int64_t),                 \
          INSIGHT_DTYPE_I64, unique_count);                                    \
      if (!dst_indices && unique_count > 0) {                                  \
        cpu_set_last_error("unique: memory allocation failed for indices");    \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < unique_count; ++i)                               \
        dst_indices[i] = first_occ[i];                                         \
    }                                                                          \
                                                                               \
    if (return_inverse) {                                                      \
      int64_t *dst_inverse =                                                   \
          (int64_t *)allocate_output((InsightArray *)outputs[out_idx++],       \
                                     sizeof(int64_t), INSIGHT_DTYPE_I64, n);   \
      if (!dst_inverse && n > 0) {                                             \
        cpu_set_last_error("unique: memory allocation failed for inverse");    \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < n; ++i)                                          \
        dst_inverse[i] = inverse[i];                                           \
    }                                                                          \
                                                                               \
    if (return_counts) {                                                       \
      int64_t *dst_counts = (int64_t *)allocate_output(                        \
          (InsightArray *)outputs[out_idx++], sizeof(int64_t),                 \
          INSIGHT_DTYPE_I64, unique_count);                                    \
      if (!dst_counts && unique_count > 0) {                                   \
        cpu_set_last_error("unique: memory allocation failed for counts");     \
        return C_FAILED;                                                       \
      }                                                                        \
      for (int64_t i = 0; i < unique_count; ++i)                               \
        dst_counts[i] = counts[i];                                             \
    }                                                                          \
  } while (0)

// ============================================================================
// Main kernel
// ============================================================================

C_Status unique_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  bool return_indices = *(bool *)inputs[1];
  bool return_inverse = *(bool *)inputs[2];
  bool return_counts = *(bool *)inputs[3];

  if (!x) {
    cpu_set_last_error("unique: null input array");
    return C_FAILED;
  }

  int64_t n = x->numel;
  int64_t unique_count = 0;

  InsightArray *unique_out = (InsightArray *)outputs[0];

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    UNIQUE_IMPL(bool);
    break;
  case INSIGHT_DTYPE_U8:
    UNIQUE_IMPL(uint8_t);
    break;
  case INSIGHT_DTYPE_U16:
    UNIQUE_IMPL(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    UNIQUE_IMPL(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    UNIQUE_IMPL(uint64_t);
    break;
  case INSIGHT_DTYPE_I8:
    UNIQUE_IMPL(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    UNIQUE_IMPL(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    UNIQUE_IMPL(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    UNIQUE_IMPL(int64_t);
    break;
  case INSIGHT_DTYPE_F16:
    UNIQUE_IMPL_FLOAT16(uint16_t, is_half_zero);
    break;
  case INSIGHT_DTYPE_BF16:
    UNIQUE_IMPL_FLOAT16(uint16_t, is_bfloat16_zero);
    break;
  case INSIGHT_DTYPE_F32:
    UNIQUE_IMPL(float);
    break;
  case INSIGHT_DTYPE_F64:
    UNIQUE_IMPL(double);
    break;
  case INSIGHT_DTYPE_F8_E4M3:
    UNIQUE_IMPL(uint8_t);
    break;
  case INSIGHT_DTYPE_F8_E5M2:
    UNIQUE_IMPL(uint8_t);
    break;
  case INSIGHT_DTYPE_C32:
    UNIQUE_IMPL_COMPLEX(std::complex<float>);
    break;
  case INSIGHT_DTYPE_C64:
    UNIQUE_IMPL_COMPLEX(std::complex<double>);
    break;
  default:
    cpu_set_last_error("unique: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_BOOL, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_U8, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_U16, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_U32, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_U64, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_I8, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_I16, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_I32, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_I64, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_F16, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_BF16, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_F32, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_F64, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_F8_E4M3, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_F8_E5M2, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_C32, unique_kernel_cpu);
REGISTER_CPU_KERNEL(unique, INSIGHT_DTYPE_C64, unique_kernel_cpu);