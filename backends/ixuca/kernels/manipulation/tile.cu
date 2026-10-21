// backends/cuda/kernels/manipulation/tile.cu
/**
 * @file tile.cu
 * @brief CUDA kernel for tile operation.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void tile_kernel(const T *src, T *dst, int64_t n, int64_t ndim,
                            const int64_t *in_dims, const int64_t *out_dims) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Convert linear index to multi-dimensional indices
    int64_t indices[10];
    int64_t remaining = idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      indices[d] = remaining % out_dims[d];
      remaining /= out_dims[d];
    }

    // Map output indices to input indices
    int64_t in_indices[10];
    for (int64_t d = 0; d < ndim; ++d) {
      in_indices[d] = indices[d] % in_dims[d];
    }

    // Compute input strides
    int64_t in_strides[10];
    in_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
      in_strides[d] = in_strides[d + 1] * in_dims[d + 1];
    }

    // Compute input offset
    int64_t in_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      in_offset += in_indices[d] * in_strides[d];
    }

    dst[idx] = src[in_offset];
  }
}

extern "C" {

C_Status tile_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("tile: null array pointer");
    return C_FAILED;
  }

  int64_t ndim = x->ndim;
  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  int64_t *d_in_dims, *d_out_dims;
  cudaMalloc(&d_in_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_out_dims, ndim * sizeof(int64_t));
  cudaMemcpy(d_in_dims, x->dims, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_dims, out->dims, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    tile_kernel<float><<<blocks, threads>>>(static_cast<const float *>(x->data),
                                            static_cast<float *>(out->data), n,
                                            ndim, d_in_dims, d_out_dims);
    break;
  case INSIGHT_DTYPE_F64:
    tile_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_in_dims, d_out_dims);
    break;
  case INSIGHT_DTYPE_I32:
    tile_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<int32_t *>(out->data), n, ndim, d_in_dims, d_out_dims);
    break;
  case INSIGHT_DTYPE_I64:
    tile_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<int64_t *>(out->data), n, ndim, d_in_dims, d_out_dims);
    break;
  default:
    cudaFree(d_in_dims);
    cudaFree(d_out_dims);
    gpu_set_last_error("tile: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(d_in_dims);
  cudaFree(d_out_dims);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(tile, INSIGHT_DTYPE_F32, tile_kernel_gpu);
REGISTER_IXUCA_KERNEL(tile, INSIGHT_DTYPE_F64, tile_kernel_gpu);
REGISTER_IXUCA_KERNEL(tile, INSIGHT_DTYPE_I32, tile_kernel_gpu);
REGISTER_IXUCA_KERNEL(tile, INSIGHT_DTYPE_I64, tile_kernel_gpu);
