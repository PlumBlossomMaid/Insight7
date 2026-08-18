// backends/cuda/kernels/manipulation/concat.cu
/**
 * @file concat.cu
 * @brief CUDA kernel for concatenation operation.
 *
 * Optimized: replaces O(N) element-by-element cudaMemcpy loop
 * with O(num_inputs) cudaMemcpy2D calls (covers all axis cases).
 * Inspired by PaddlePaddle's concat kernel pattern.
 */

#include "../../registry/ixuca_registry.h"
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
  //   src_pitch = width                    (contiguous in source)
  //   dst_pitch = out_axis_dim * inner * elem_size (strided in output)
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
    size_t src_pitch = width;
    size_t dst_pitch = static_cast<size_t>(out->dims[axis] * inner) *
                       static_cast<size_t>(elem_size);
    size_t dst_off = static_cast<size_t>(axis_offset * inner) *
                     static_cast<size_t>(elem_size);
    size_t src_off =
        static_cast<size_t>(src->offset) * static_cast<size_t>(elem_size);

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

REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_BOOL, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_U8, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_I8, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_I16, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_I32, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_I64, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_U16, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_U32, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_U64, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_F32, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_F64, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_C32, concat_kernel_gpu);
REGISTER_IXUCA_KERNEL(concat, INSIGHT_DTYPE_C64, concat_kernel_gpu);
