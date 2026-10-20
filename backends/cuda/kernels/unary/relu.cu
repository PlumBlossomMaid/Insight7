// backends/cuda/kernels/unary/relu.cu
/**
 * @file relu.cu
 * @brief CUDA kernel for ReLU activation.
 *
 * Computes relu(x) = max(x, 0) elementwise.
 * Supports integer and floating point types.
 */

#include "../../registry/cuda_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void relu_kernel(const T *x, T *out, int64_t n, int ndim,
                            const int64_t *dims, const int64_t *x_strides,
                            const int64_t *out_strides, int64_t x_offset,
                            int64_t out_offset) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = x_offset + unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = out_offset + unary_offset(idx, ndim, dims, out_strides);
    T v = x[off_x];
    out[off_out] = v > static_cast<T>(0) ? v : static_cast<T>(0);
  }
}

extern "C" {

C_Status relu_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("relu: null array pointer");
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

  int64_t x_offset = x->offset;
  int64_t out_offset = out->offset;

  switch (out->dtype) {
  case INSIGHT_DTYPE_I8:
    relu_kernel<int8_t><<<blocks, threads>>>(
        static_cast<const int8_t *>(x->data), static_cast<int8_t *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset);
    break;
  case INSIGHT_DTYPE_I16:
    relu_kernel<int16_t><<<blocks, threads>>>(
        static_cast<const int16_t *>(x->data),
        static_cast<int16_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset);
    break;
  case INSIGHT_DTYPE_I32:
    relu_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<int32_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset);
    break;
  case INSIGHT_DTYPE_I64:
    relu_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<int64_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset);
    break;
  case INSIGHT_DTYPE_F32:
    relu_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset);
    break;
  case INSIGHT_DTYPE_F64:
    relu_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("relu: unsupported dtype");
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

REGISTER_GPU_KERNEL(relu, INSIGHT_DTYPE_I8, relu_kernel_gpu);
REGISTER_GPU_KERNEL(relu, INSIGHT_DTYPE_I16, relu_kernel_gpu);
REGISTER_GPU_KERNEL(relu, INSIGHT_DTYPE_I32, relu_kernel_gpu);
REGISTER_GPU_KERNEL(relu, INSIGHT_DTYPE_I64, relu_kernel_gpu);
REGISTER_GPU_KERNEL(relu, INSIGHT_DTYPE_F32, relu_kernel_gpu);
REGISTER_GPU_KERNEL(relu, INSIGHT_DTYPE_F64, relu_kernel_gpu);