// backends/cpu/kernels/indexing/take.cpp
/**
 * @file take.cpp
 * @brief CPU kernel for take operation.
 *
 * Takes elements from an array at specified indices.
 * Supports both flat (no axis) and axis-based indexing.
 * Handles non-contiguous arrays via stride-based addressing.
 */

#include "common.h"
#include <complex>

// ============================================================================
// Take implementation for 1D (flattened) case with stride support
// ============================================================================

/**
 * @brief Take from flattened array with stride support.
 *
 * @tparam T Data type
 * @param dst Destination array data pointer
 * @param dst_offset Destination array offset (elements)
 * @param dst_stride Destination array stride (elements, for output)
 * @param src Source array data pointer
 * @param src_offset Source array offset (elements)
 * @param src_stride Source array stride (elements)
 * @param idx_data Index array data
 * @param n Number of elements to take
 * @param src_size Total size of source array
 */
template <typename T>
static void take_flat_impl(T *dst, int64_t dst_offset, int64_t dst_stride,
                           const T *src, int64_t src_offset, int64_t src_stride,
                           const int64_t *idx_data, int64_t n,
                           int64_t src_size) {

#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    int64_t pos = idx_data[i];
    if (pos < 0)
      pos += src_size;

    int64_t src_idx = src_offset + pos * src_stride;
    int64_t dst_idx = dst_offset + i * dst_stride;
    dst[dst_idx] = src[src_idx];
  }
}

// ============================================================================
// Take implementation for ND case with axis and stride support
// ============================================================================

/**
 * @brief Take from ND array along an axis with stride support.
 *
 * @tparam T Data type
 * @param dst Destination array data pointer
 * @param dst_offset Destination array offset (elements)
 * @param dst_strides Destination array strides
 * @param src Source array data pointer
 * @param src_offset Source array offset (elements)
 * @param src_strides Source array strides
 * @param idx_data Index array data
 * @param total_out Total number of output elements
 * @param ndim Number of dimensions
 * @param out_dims Output dimension sizes
 * @param axis Axis along which to take
 * @param idx_len Length of index array
 */
template <typename T>
static void
take_axis_impl(T *dst, int64_t dst_offset, const int64_t *dst_strides,
               const T *src, int64_t src_offset, const int64_t *src_strides,
               const int64_t *idx_data, int64_t total_out, int64_t ndim,
               const int64_t *out_dims, int64_t axis, int64_t idx_len,
               int64_t src_axis_len) {

  // Precompute multiplier for fast axis coordinate calculation
  int64_t axis_mult = 1;
  for (int64_t d = axis + 1; d < ndim; ++d) {
    axis_mult *= out_dims[d];
  }

#pragma omp parallel for
  for (int64_t linear = 0; linear < total_out; ++linear) {
    // Compute coordinate along axis using linear index
    int64_t axis_coord = (linear / axis_mult) % out_dims[axis];

    // Get index from array (cyclic if idx_len < out_dims[axis])
    int64_t pos = idx_data[axis_coord % idx_len];
    if (pos < 0)
      pos += src_axis_len; // normalize against the SOURCE axis length

    // Compute source offset (using src_strides and src_offset)
    int64_t src_idx = src_offset;
    int64_t tmp = linear;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      int64_t coord = tmp % out_dims[d];
      tmp /= out_dims[d];
      if (d == axis) {
        src_idx += pos * src_strides[d];
      } else {
        src_idx += coord * src_strides[d];
      }
    }

    // Compute destination offset (using dst_strides and dst_offset)
    int64_t dst_idx = dst_offset;
    tmp = linear;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      int64_t coord = tmp % out_dims[d];
      tmp /= out_dims[d];
      dst_idx += coord * dst_strides[d];
    }

    dst[dst_idx] = src[src_idx];
  }
}

// ============================================================================
// Type dispatch macro
// ============================================================================

#define TAKE_DISPATCH(CTYPE)                                                   \
  do {                                                                         \
    if (!has_axis) {                                                           \
      take_flat_impl<CTYPE>((CTYPE *)out->data, out->offset, out->strides[0],  \
                            (const CTYPE *)x->data, x->offset, x->strides[0],  \
                            idx_data, n, x->numel);                            \
    } else {                                                                   \
      take_axis_impl<CTYPE>(                                                   \
          (CTYPE *)out->data, out->offset, out->strides,                       \
          (const CTYPE *)x->data, x->offset, x->strides, idx_data, n, ndim,    \
          out_dims, normalized_axis, idx->numel, x->dims[normalized_axis]);    \
    }                                                                          \
  } while (0)

// ============================================================================
// Kernel entry point
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

C_Status take_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  InsightArray *idx = (InsightArray *)inputs[2];
  int normalized_axis = *(int *)inputs[3];
  bool has_axis = *(bool *)inputs[4];

  if (!out || !x || !idx) {
    cpu_set_last_error("take: null array pointer");
    return C_FAILED;
  }

  // Ensure index array is int64
  if (idx->dtype != INSIGHT_DTYPE_I64) {
    cpu_set_last_error("take: index array must be int64");
    return C_FAILED;
  }

  int64_t n = out->numel;
  const int64_t *idx_data = (const int64_t *)idx->data;

  // Bound checking (optional, can be disabled for performance)
  if (!has_axis) {
    int64_t src_size = x->numel;
    for (int64_t i = 0; i < idx->numel; ++i) {
      int64_t pos = idx_data[i];
      if (pos < 0)
        pos += src_size;
      if (pos < 0 || pos >= src_size) {
        cpu_set_last_error("take: index out of bounds");
        return C_FAILED;
      }
    }
  } else {
    int64_t axis_size = x->dims[normalized_axis];
    for (int64_t i = 0; i < idx->numel; ++i) {
      int64_t pos = idx_data[i];
      if (pos < 0)
        pos += axis_size;
      if (pos < 0 || pos >= axis_size) {
        cpu_set_last_error("take: index out of bounds");
        return C_FAILED;
      }
    }
  }

  // Prepare output dimensions
  int64_t ndim = out->ndim;
  int64_t out_dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    out_dims[i] = out->dims[i];
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    TAKE_DISPATCH(bool);
    break;
  case INSIGHT_DTYPE_U8:
    TAKE_DISPATCH(uint8_t);
    break;
  case INSIGHT_DTYPE_I8:
    TAKE_DISPATCH(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    TAKE_DISPATCH(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    TAKE_DISPATCH(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    TAKE_DISPATCH(int64_t);
    break;
  case INSIGHT_DTYPE_U16:
    TAKE_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    TAKE_DISPATCH(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    TAKE_DISPATCH(uint64_t);
    break;
  case INSIGHT_DTYPE_F16:
    TAKE_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_BF16:
    TAKE_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_F32:
    TAKE_DISPATCH(float);
    break;
  case INSIGHT_DTYPE_F64:
    TAKE_DISPATCH(double);
    break;
  case INSIGHT_DTYPE_C32:
    TAKE_DISPATCH(std::complex<float>);
    break;
  case INSIGHT_DTYPE_C64:
    TAKE_DISPATCH(std::complex<double>);
    break;
  case INSIGHT_DTYPE_F8_E4M3:
    TAKE_DISPATCH(uint8_t);
    break;
  case INSIGHT_DTYPE_F8_E5M2:
    TAKE_DISPATCH(uint8_t);
    break;
  default:
    cpu_set_last_error("take: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// ============================================================================
// Kernel registration for all supported types
// ============================================================================

REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_BOOL, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_U8, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_I8, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_I16, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_I32, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_I64, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_U16, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_U32, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_U64, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_F16, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_BF16, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_F32, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_F64, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_C32, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_C64, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_F8_E4M3, take_kernel_cpu);
REGISTER_CPU_KERNEL(take, INSIGHT_DTYPE_F8_E5M2, take_kernel_cpu);