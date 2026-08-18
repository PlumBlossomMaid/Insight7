// backends/cuda/kernels/unary/bitwise_not.cu
/**
 * @file bitwise_not.cu
 * @brief CUDA kernel for bitwise NOT operation.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void
bitwise_not_kernel(const T *x, T *out, int64_t n, int ndim, const int64_t *dims,
                   const int64_t *x_strides, const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    out[off_out] = ~x[off_x];
  }
}

// BOOL specialization: ~bool maps to logical not (NumPy/PyTorch/Paddle
// behavior)
__global__ void bitwise_not_bool_kernel(const bool *x, bool *out, int64_t n,
                                        int ndim, const int64_t *dims,
                                        const int64_t *x_strides,
                                        const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    out[off_out] = !x[off_x];
  }
}

extern "C" {

C_Status bitwise_not_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("bitwise_not: null array pointer");
    return C_FAILED;
  }

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int ndim = out->ndim;
  int threads = unary_threads();
  int blocks = unary_blocks(n);

  int64_t *d_dims, *d_x_strides, *d_out_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, out->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_strides, x->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_strides, out->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  switch (out->dtype) {
  case INSIGHT_DTYPE_BOOL:
    bitwise_not_bool_kernel<<<blocks, threads>>>(
        static_cast<const bool *>(x->data), static_cast<bool *>(out->data), n,
        ndim, d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I8:
    bitwise_not_kernel<int8_t><<<blocks, threads>>>(
        static_cast<const int8_t *>(x->data), static_cast<int8_t *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I16:
    bitwise_not_kernel<int16_t>
        <<<blocks, threads>>>(static_cast<const int16_t *>(x->data),
                              static_cast<int16_t *>(out->data), n, ndim,
                              d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I32:
    bitwise_not_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<const int32_t *>(x->data),
                              static_cast<int32_t *>(out->data), n, ndim,
                              d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I64:
    bitwise_not_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<const int64_t *>(x->data),
                              static_cast<int64_t *>(out->data), n, ndim,
                              d_dims, d_x_strides, d_out_strides);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("bitwise_not: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(d_dims);
  cudaFree(d_x_strides);
  cudaFree(d_out_strides);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(bitwise_not, INSIGHT_DTYPE_BOOL,
                         bitwise_not_kernel_gpu);
REGISTER_IXUCA_KERNEL(bitwise_not, INSIGHT_DTYPE_I8, bitwise_not_kernel_gpu);
REGISTER_IXUCA_KERNEL(bitwise_not, INSIGHT_DTYPE_I16,
                         bitwise_not_kernel_gpu);
REGISTER_IXUCA_KERNEL(bitwise_not, INSIGHT_DTYPE_I32,
                         bitwise_not_kernel_gpu);
REGISTER_IXUCA_KERNEL(bitwise_not, INSIGHT_DTYPE_I64,
                         bitwise_not_kernel_gpu);
