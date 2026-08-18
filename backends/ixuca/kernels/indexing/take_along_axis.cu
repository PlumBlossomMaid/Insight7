// backends/cuda/kernels/indexing/take_along_axis.cu
/**
 * @file take_along_axis.cu
 * @brief CUDA kernel for take_along_axis operation.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void
take_along_axis_kernel(const T *src, const int64_t *indices, T *dst, int64_t n,
                       int64_t ndim, const int64_t *dims,
                       const int64_t *src_strides, const int64_t *dst_strides,
                       int64_t axis) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Convert linear index to multi-dimensional indices
    int64_t out_indices[10];
    int64_t remaining = idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      out_indices[d] = remaining % dims[d];
      remaining /= dims[d];
    }

    // Get index value
    int64_t index_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      index_offset += out_indices[d] * dst_strides[d];
    }
    int64_t src_idx = indices[index_offset];

    // Compute source offset
    int64_t src_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      int64_t idx_d = (d == axis) ? src_idx : out_indices[d];
      src_offset += idx_d * src_strides[d];
    }

    dst[idx] = src[src_offset];
  }
}

extern "C" {

C_Status take_along_axis_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  InsightArray *indices = static_cast<InsightArray *>(inputs[2]);

  if (!out || !x || !indices) {
    gpu_set_last_error("take_along_axis: null array pointer");
    return C_FAILED;
  }

  int axis = *static_cast<int *>(inputs[3]);
  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  int64_t *d_dims, *d_src_strides, *d_dst_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_src_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_dst_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, out->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src_strides, x->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dst_strides, out->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    take_along_axis_kernel<float>
        <<<blocks, threads>>>(static_cast<const float *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<float *>(out->data), n, ndim, d_dims,
                              d_src_strides, d_dst_strides, axis);
    break;
  case INSIGHT_DTYPE_F64:
    take_along_axis_kernel<double>
        <<<blocks, threads>>>(static_cast<const double *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<double *>(out->data), n, ndim, d_dims,
                              d_src_strides, d_dst_strides, axis);
    break;
  case INSIGHT_DTYPE_I32:
    take_along_axis_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<const int32_t *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<int32_t *>(out->data), n, ndim,
                              d_dims, d_src_strides, d_dst_strides, axis);
    break;
  case INSIGHT_DTYPE_I64:
    take_along_axis_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<const int64_t *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<int64_t *>(out->data), n, ndim,
                              d_dims, d_src_strides, d_dst_strides, axis);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_src_strides);
    cudaFree(d_dst_strides);
    gpu_set_last_error("take_along_axis: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(d_dims);
  cudaFree(d_src_strides);
  cudaFree(d_dst_strides);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(take_along_axis, INSIGHT_DTYPE_F32,
                         take_along_axis_kernel_gpu);
REGISTER_IXUCA_KERNEL(take_along_axis, INSIGHT_DTYPE_F64,
                         take_along_axis_kernel_gpu);
REGISTER_IXUCA_KERNEL(take_along_axis, INSIGHT_DTYPE_I32,
                         take_along_axis_kernel_gpu);
REGISTER_IXUCA_KERNEL(take_along_axis, INSIGHT_DTYPE_I64,
                         take_along_axis_kernel_gpu);
