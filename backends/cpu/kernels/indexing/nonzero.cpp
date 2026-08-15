// backends/cpu/kernels/indexing/nonzero.cpp
/**
 * @file nonzero.cpp
 * @brief CPU kernel for nonzero operation.
 *
 * Returns indices of non-zero elements.
 * The output array is allocated by this kernel.
 *
 * Input layout:
 *   inputs[0] = InsightArray* input
 *   inputs[1] = InsightArray* output (placeholder, will be overwritten)
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result (same as inputs[1])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <insight/c_api/place.h>
#include <vector>

// ============================================================================
// Helper functions for special types
// ============================================================================

static inline bool is_zero_f16(uint16_t val) {
  // Half-precision zero: exponent = 0, mantissa = 0, sign bit ignored
  return (val & 0x7FFF) == 0;
}

static inline bool is_zero_bf16(uint16_t val) { return (val & 0x7FFF) == 0; }

static inline bool is_zero_f8_e4m3(uint8_t val) {
  return val == 0 || val == 0x80;
}

static inline bool is_zero_f8_e5m2(uint8_t val) {
  return val == 0 || val == 0x80;
}

template <typename T> static inline bool is_zero(const T &val) {
  return val == T(0);
}

template <> inline bool is_zero(const std::complex<float> &val) {
  return val.real() == 0.0f && val.imag() == 0.0f;
}

template <> inline bool is_zero(const std::complex<double> &val) {
  return val.real() == 0.0 && val.imag() == 0.0;
}

// ============================================================================
// Helper: allocate and set output array metadata
// ============================================================================

static void *allocate_output(InsightArray *out, size_t elem_size, int32_t dtype,
                             int64_t ndim, const int64_t *dims, int64_t numel) {
  size_t bytes = elem_size * numel;
  void *data = NULL;
  if (bytes > 0) {
    data = malloc(bytes);
    if (!data)
      return NULL;
  }

  out->data = data;
  out->storage_nbytes = bytes;
  out->ndim = ndim;
  for (int i = 0; i < ndim; ++i) {
    out->dims[i] = dims[i];
  }
  out->numel = numel;
  out->dtype = dtype;

  // Compute strides for row-major layout
  if (ndim > 0) {
    out->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
      out->strides[i] = out->strides[i + 1] * out->dims[i + 1];
    }
  }

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
// Template implementation
// ============================================================================

template <typename T>
static void nonzero_count_impl(const T *src, int64_t total, int64_t ndim,
                               const int64_t *dims, const int64_t *strides,
                               int64_t &nz_count) {

  nz_count = 0;
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t indices[INSIGHT_MAX_NDIM];
    int64_t tmp = linear;
    for (int d = ndim - 1; d >= 0; --d) {
      indices[d] = tmp % dims[d];
      tmp /= dims[d];
    }

    int64_t offset = 0;
    for (int d = 0; d < ndim; ++d) {
      offset += indices[d] * strides[d];
    }

    if (!is_zero(src[offset])) {
      ++nz_count;
    }
  }
}

template <typename T>
static void nonzero_fill_impl(const T *src, int64_t *dst, int64_t total,
                              int64_t ndim, const int64_t *dims,
                              const int64_t *strides, int64_t nz_count) {

  int64_t out_idx = 0;
  for (int64_t linear = 0; linear < total && out_idx < nz_count; ++linear) {
    int64_t indices[INSIGHT_MAX_NDIM];
    int64_t tmp = linear;
    for (int d = ndim - 1; d >= 0; --d) {
      indices[d] = tmp % dims[d];
      tmp /= dims[d];
    }

    int64_t offset = 0;
    for (int d = 0; d < ndim; ++d) {
      offset += indices[d] * strides[d];
    }

    if (!is_zero(src[offset])) {
      for (int d = 0; d < ndim; ++d) {
        dst[d * nz_count + out_idx] = indices[d];
      }
      ++out_idx;
    }
  }
}

// Macro: Generate type distribution code
#define NONZERO_DISPATCH(CTYPE)                                                \
  do {                                                                         \
    nonzero_count_impl<CTYPE>((const CTYPE *)x->data, total, ndim, dims,       \
                              strides, nz_count);                              \
    if (nz_count > 0) {                                                        \
      int64_t out_dims[2] = {ndim, nz_count};                                  \
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims,    \
                      ndim *nz_count);                                         \
      nonzero_fill_impl<CTYPE>((const CTYPE *)x->data, (int64_t *)out->data,   \
                               total, ndim, dims, strides, nz_count);          \
    } else {                                                                   \
      int64_t out_dims[2] = {ndim, 0};                                         \
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims,    \
                      0);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Main kernel
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

C_Status nonzero_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("nonzero: null array pointer");
    return C_FAILED;
  }

  int64_t ndim = x->ndim;
  int64_t total = x->numel;

  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t strides[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dims[i] = x->dims[i];
    strides[i] = x->strides[i];
  }

  int64_t nz_count = 0;

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    NONZERO_DISPATCH(bool);
    break;
  case INSIGHT_DTYPE_U8:
    NONZERO_DISPATCH(uint8_t);
    break;
  case INSIGHT_DTYPE_U16:
    NONZERO_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    NONZERO_DISPATCH(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    NONZERO_DISPATCH(uint64_t);
    break;
  case INSIGHT_DTYPE_I8:
    NONZERO_DISPATCH(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    NONZERO_DISPATCH(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    NONZERO_DISPATCH(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    NONZERO_DISPATCH(int64_t);
    break;
  case INSIGHT_DTYPE_F16: {
    const uint16_t *src = (const uint16_t *)x->data;
    nz_count = 0;
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t tmp = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = tmp % dims[d];
        tmp /= dims[d];
      }
      int64_t offset = 0;
      for (int d = 0; d < ndim; ++d) {
        offset += indices[d] * strides[d];
      }
      if (!is_zero_f16(src[offset]))
        ++nz_count;
    }
    if (nz_count > 0) {
      int64_t out_dims[2] = {ndim, nz_count};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims,
                      ndim * nz_count);
      int64_t out_idx = 0;
      for (int64_t linear = 0; linear < total && out_idx < nz_count; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
          indices[d] = tmp % dims[d];
          tmp /= dims[d];
        }
        int64_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
          offset += indices[d] * strides[d];
        }
        if (!is_zero_f16(src[offset])) {
          for (int d = 0; d < ndim; ++d) {
            ((int64_t *)out->data)[d * nz_count + out_idx] = indices[d];
          }
          ++out_idx;
        }
      }
    } else {
      int64_t out_dims[2] = {ndim, 0};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims, 0);
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    // Like F16, use is_zero_bf16
    const uint16_t *src = (const uint16_t *)x->data;
    nz_count = 0;
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t tmp = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = tmp % dims[d];
        tmp /= dims[d];
      }
      int64_t offset = 0;
      for (int d = 0; d < ndim; ++d) {
        offset += indices[d] * strides[d];
      }
      if (!is_zero_bf16(src[offset]))
        ++nz_count;
    }
    // The filling logic is similar...
    if (nz_count > 0) {
      int64_t out_dims[2] = {ndim, nz_count};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims,
                      ndim * nz_count);
      int64_t out_idx = 0;
      for (int64_t linear = 0; linear < total && out_idx < nz_count; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
          indices[d] = tmp % dims[d];
          tmp /= dims[d];
        }
        int64_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
          offset += indices[d] * strides[d];
        }
        if (!is_zero_bf16(src[offset])) {
          for (int d = 0; d < ndim; ++d) {
            ((int64_t *)out->data)[d * nz_count + out_idx] = indices[d];
          }
          ++out_idx;
        }
      }
    } else {
      int64_t out_dims[2] = {ndim, 0};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims, 0);
    }
    break;
  }
  case INSIGHT_DTYPE_F32:
    NONZERO_DISPATCH(float);
    break;
  case INSIGHT_DTYPE_F64:
    NONZERO_DISPATCH(double);
    break;
  case INSIGHT_DTYPE_F8_E4M3: {
    const uint8_t *src = (const uint8_t *)x->data;
    nz_count = 0;
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t tmp = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = tmp % dims[d];
        tmp /= dims[d];
      }
      int64_t offset = 0;
      for (int d = 0; d < ndim; ++d) {
        offset += indices[d] * strides[d];
      }
      if (!is_zero_f8_e4m3(src[offset]))
        ++nz_count;
    }
    if (nz_count > 0) {
      int64_t out_dims[2] = {ndim, nz_count};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims,
                      ndim * nz_count);
      int64_t out_idx = 0;
      for (int64_t linear = 0; linear < total && out_idx < nz_count; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
          indices[d] = tmp % dims[d];
          tmp /= dims[d];
        }
        int64_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
          offset += indices[d] * strides[d];
        }
        if (!is_zero_f8_e4m3(src[offset])) {
          for (int d = 0; d < ndim; ++d) {
            ((int64_t *)out->data)[d * nz_count + out_idx] = indices[d];
          }
          ++out_idx;
        }
      }
    } else {
      int64_t out_dims[2] = {ndim, 0};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims, 0);
    }
    break;
  }
  case INSIGHT_DTYPE_F8_E5M2: {
    // Similar to F8_E4M3
    const uint8_t *src = (const uint8_t *)x->data;
    nz_count = 0;
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t tmp = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = tmp % dims[d];
        tmp /= dims[d];
      }
      int64_t offset = 0;
      for (int d = 0; d < ndim; ++d) {
        offset += indices[d] * strides[d];
      }
      if (!is_zero_f8_e5m2(src[offset]))
        ++nz_count;
    }
    if (nz_count > 0) {
      int64_t out_dims[2] = {ndim, nz_count};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims,
                      ndim * nz_count);
      int64_t out_idx = 0;
      for (int64_t linear = 0; linear < total && out_idx < nz_count; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
          indices[d] = tmp % dims[d];
          tmp /= dims[d];
        }
        int64_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
          offset += indices[d] * strides[d];
        }
        if (!is_zero_f8_e5m2(src[offset])) {
          for (int d = 0; d < ndim; ++d) {
            ((int64_t *)out->data)[d * nz_count + out_idx] = indices[d];
          }
          ++out_idx;
        }
      }
    } else {
      int64_t out_dims[2] = {ndim, 0};
      allocate_output(out, sizeof(int64_t), INSIGHT_DTYPE_I64, 2, out_dims, 0);
    }
    break;
  }
  case INSIGHT_DTYPE_C32:
    NONZERO_DISPATCH(std::complex<float>);
    break;
  case INSIGHT_DTYPE_C64:
    NONZERO_DISPATCH(std::complex<double>);
    break;
  default:
    cpu_set_last_error("nonzero: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_BOOL, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_U8, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_U16, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_U32, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_U64, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_I8, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_I16, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_I32, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_I64, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_F16, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_BF16, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_F32, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_F64, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_F8_E4M3, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_F8_E5M2, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_C32, nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(nonzero, INSIGHT_DTYPE_C64, nonzero_kernel_cpu);