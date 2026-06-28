// backends/cuda/kernels/indexing/take.cu
/**
 * @file take.cu
 * @brief CUDA kernel for take operation.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void take_kernel(const T *src, const int64_t *indices, T *dst,
                            int64_t n, int64_t inner_size, int64_t axis_size,
                            bool has_axis) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (has_axis) {
      // Axis mode: indices apply to the first dimension
      int64_t outer_idx = idx / inner_size;
      int64_t inner_idx = idx % inner_size;
      int64_t src_idx = indices[outer_idx] * inner_size + inner_idx;
      dst[idx] = src[src_idx];
    } else {
      // Flattened mode
      dst[idx] = src[indices[idx]];
    }
  }
}

extern "C" {

C_Status take_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  InsightArray *indices = static_cast<InsightArray *>(inputs[2]);

  if (!out || !x || !indices) {
    gpu_set_last_error("take: null array pointer");
    return C_FAILED;
  }

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int64_t inner_size = 1;
  if (x->ndim > 1) {
    for (int d = 1; d < x->ndim; ++d) {
      inner_size *= x->dims[d];
    }
  }

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Note: inputs[3] is &normalized_axis, inputs[4] is &has_axis
  // int normalized_axis = *static_cast<int *>(inputs[3]); // Not used in this
  // logic
  bool has_axis = *static_cast<bool *>(inputs[4]);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    take_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data),
        static_cast<const int64_t *>(indices->data),
        static_cast<float *>(out->data), n, inner_size, x->dims[0], has_axis);
    break;
  case INSIGHT_DTYPE_F64:
    take_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data),
        static_cast<const int64_t *>(indices->data),
        static_cast<double *>(out->data), n, inner_size, x->dims[0], has_axis);
    break;
  case INSIGHT_DTYPE_I32:
    take_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<const int64_t *>(indices->data),
        static_cast<int32_t *>(out->data), n, inner_size, x->dims[0], has_axis);
    break;
  case INSIGHT_DTYPE_I64:
    take_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<const int64_t *>(indices->data),
        static_cast<int64_t *>(out->data), n, inner_size, x->dims[0], has_axis);
    break;
  default:
    gpu_set_last_error("take: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(take, INSIGHT_DTYPE_F32, take_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(take, INSIGHT_DTYPE_F64, take_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(take, INSIGHT_DTYPE_I32, take_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(take, INSIGHT_DTYPE_I64, take_kernel_gpu);
