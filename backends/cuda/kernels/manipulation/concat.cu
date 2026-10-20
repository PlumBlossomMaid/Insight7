// backends/cuda/kernels/manipulation/concat.cu
/**
 * @file concat.cu
 * @brief CUDA kernel for concatenation operation.
 *
 * Optimized: replaces O(N) element-by-element cudaMemcpy loop
 * with O(num_inputs) cudaMemcpy2D calls (covers all axis cases).
 * Inspired by PaddlePaddle's concat kernel pattern.
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" {

C_Status concat_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  if (!out) {
    gpu_set_last_error("concat: output is null");
    return C_FAILED;
  }

  if (!inputs[1]) {
    gpu_set_last_error("concat: num_inputs is null");
    return C_FAILED;
  }
  int num_inputs = *static_cast<int *>(inputs[1]);
  if (num_inputs < 1) {
    gpu_set_last_error("concat: no input arrays");
    return C_FAILED;
  }

  // Extract input arrays
  std::vector<InsightArray *> in_arrays;
  in_arrays.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    if (!inputs[2 + i]) {
      gpu_set_last_error("concat: null input array");
      return C_FAILED;
    }
    in_arrays.push_back(static_cast<InsightArray *>(inputs[2 + i]));
  }

  // Extract axis
  if (!inputs[2 + num_inputs]) {
    gpu_set_last_error("concat: axis is null");
    return C_FAILED;
  }
  int axis = *static_cast<int *>(inputs[2 + num_inputs]);

  int64_t ndim = out->ndim;
  int64_t elem_size;
  switch (out->dtype) {
  case INSIGHT_DTYPE_BOOL:
  case INSIGHT_DTYPE_U8:
  case INSIGHT_DTYPE_I8:
    elem_size = 1;
    break;
  case INSIGHT_DTYPE_I16:
  case INSIGHT_DTYPE_U16:
    elem_size = 2;
    break;
  case INSIGHT_DTYPE_I32:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_F32:
    elem_size = 4;
    break;
  case INSIGHT_DTYPE_I64:
  case INSIGHT_DTYPE_U64:
  case INSIGHT_DTYPE_F64:
  case INSIGHT_DTYPE_C32:
    elem_size = 8;
    break;
  case INSIGHT_DTYPE_C64:
    elem_size = 16;
    break;
  default:
    gpu_set_last_error("concat: unsupported dtype");
    return C_FAILED;
  }

  // inner = elements after concat axis, outer = elements before
  int64_t inner = 1;
  for (int d = axis + 1; d < ndim; ++d)
    inner *= out->dims[d];
  int64_t outer = 1;
  for (int d = 0; d < axis; ++d)
    outer *= out->dims[d];

  if (inner == 0 || outer == 0) {
    gpu_set_last_error("concat: zero-dim tensor not supported");
    return C_FAILED;
  }

  // cudaMemcpy2D copies each input as a 2D array:
  //   width = axis_dim * inner * elem_size  (bytes per "row")
  //   height = outer                       (number of "rows")
  //   src_pitch = bytes between consecutive rows in the source
  //   dst_pitch = out_axis_dim * inner * elem_size (strided in output)
  //
  // The source may be a strided view (e.g. a slice from fftshift), so
  // src_pitch must respect src->strides instead of assuming contiguity.
  // When the source layout cannot be expressed as a uniform 2D pitch
  // (outer dims not row-contiguous), fall back to row-by-row copies.
  int64_t axis_offset = 0;
  for (int i = 0; i < num_inputs; ++i) {
    InsightArray *src = in_arrays[i];
    int64_t src_axis_dim = src->dims[axis];
    if (src_axis_dim <= 0) {
      axis_offset += src_axis_dim;
      continue;
    }

    size_t width = static_cast<size_t>(src_axis_dim * inner) *
                   static_cast<size_t>(elem_size);
    size_t height = static_cast<size_t>(outer);
    size_t dst_pitch = static_cast<size_t>(out->dims[axis] * inner) *
                       static_cast<size_t>(elem_size);
    size_t dst_off = static_cast<size_t>(axis_offset * inner) *
                     static_cast<size_t>(elem_size);
    size_t src_off =
        static_cast<size_t>(src->offset) * static_cast<size_t>(elem_size);

    // Determine whether the source is expressible as a uniform 2D block:
    // dims [axis, ndim) must be contiguous within a row, and dims
    // [0, axis) must be row-contiguous (uniform pitch = strides[axis-1]).
    bool src_2d_ok = true;
    int64_t expect = 1;
    for (int64_t d = ndim - 1; d >= axis; --d) {
      if (src->strides[d] != expect) {
        src_2d_ok = false;
        break;
      }
      expect *= src->dims[d];
    }
    for (int64_t d = axis - 1; d > 0; --d) {
      if (src->strides[d - 1] != src->strides[d] * src->dims[d]) {
        src_2d_ok = false;
        break;
      }
    }

    if (!src_2d_ok) {
      // Fallback: one cudaMemcpy per row, honoring per-dim strides.
      // Decode the flat row index over dims [0, axis) in row-major order.
      for (int64_t row = 0; row < outer; ++row) {
        int64_t rem = row;
        int64_t src_row_off = src->offset;
        for (int64_t d = axis - 1; d >= 0; --d) {
          int64_t dim = src->dims[d];
          int64_t idx = (dim > 0) ? rem % dim : 0;
          if (dim > 0) {
            rem /= dim;
          }
          src_row_off += idx * src->strides[d];
        }
        cudaError_t err = cudaMemcpy(
            static_cast<char *>(out->data) + dst_off +
                row * dst_pitch,
            static_cast<const char *>(src->data) +
                src_row_off * static_cast<int64_t>(elem_size),
            width, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
          char buf[256];
          snprintf(buf, sizeof(buf),
                   "concat: cudaMemcpy failed for input %d row %lld: %s", i,
                   (long long)row, cudaGetErrorString(err));
          gpu_set_last_error(buf);
          return C_FAILED;
        }
      }
      axis_offset += src_axis_dim;
      continue;
    }

    size_t src_pitch =
        (axis > 0)
            ? static_cast<size_t>(src->strides[axis - 1]) *
                  static_cast<size_t>(elem_size)
            : width;

    cudaError_t err =
        cudaMemcpy2D(static_cast<char *>(out->data) + dst_off, dst_pitch,
                     static_cast<const char *>(src->data) + src_off, src_pitch,
                     width, height, cudaMemcpyDeviceToDevice);

    if (err != cudaSuccess) {
      char buf[256];
      snprintf(buf, sizeof(buf), "concat: cudaMemcpy2D failed for input %d: %s",
               i, cudaGetErrorString(err));
      gpu_set_last_error(buf);
      return C_FAILED;
    }
    axis_offset += src_axis_dim;
  }

  // Check for async errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_BOOL, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U8, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I8, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I16, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I64, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U16, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U64, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_F32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_F64, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_C32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_C64, concat_kernel_gpu);
