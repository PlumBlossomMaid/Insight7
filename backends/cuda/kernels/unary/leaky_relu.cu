// backends/cuda/kernels/unary/leaky_relu.cu
/**
 * @file leaky_relu.cu
 * @brief CUDA kernel for Leaky ReLU activation.
 *
 * Computes leaky_relu(x) = max(x, alpha * x) elementwise.
 * The alpha slope is passed as a double* in inputs[1].
 * Supports float32 and float64.
 */

#include "../../registry/cuda_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void leaky_relu_kernel(const T *x, T *out, int64_t n, int ndim,
                                  const int64_t *dims, const int64_t *x_strides,
                                  const int64_t *out_strides, int64_t x_offset,
                                  int64_t out_offset, T alpha) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = x_offset + unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = out_offset + unary_offset(idx, ndim, dims, out_strides);
    T v = x[off_x];
    out[off_out] = v > static_cast<T>(0) ? v : alpha * v;
  }
}

extern "C" {

C_Status leaky_relu_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("leaky_relu: null array pointer");
    return C_FAILED;
  }
  if (!inputs[1]) {
    gpu_set_last_error("leaky_relu: alpha is null");
    return C_FAILED;
  }

  double alpha_host = *static_cast<double *>(inputs[1]);

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
  case INSIGHT_DTYPE_F32:
    leaky_relu_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset,
        static_cast<float>(alpha_host));
    break;
  case INSIGHT_DTYPE_F64:
    leaky_relu_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset,
        alpha_host);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("leaky_relu: unsupported dtype");
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

REGISTER_GPU_KERNEL(leaky_relu, INSIGHT_DTYPE_F32, leaky_relu_kernel_gpu);
REGISTER_GPU_KERNEL(leaky_relu, INSIGHT_DTYPE_F64, leaky_relu_kernel_gpu);