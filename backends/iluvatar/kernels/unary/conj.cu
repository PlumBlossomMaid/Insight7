// backends/cuda/kernels/unary/conj.cu
/**
 * @file conj.cu
 * @brief CUDA kernel for complex conjugate.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void conj_c32_kernel(const cuFloatComplex *x, cuFloatComplex *out,
                                int64_t n, int ndim, const int64_t *dims,
                                const int64_t *x_strides,
                                const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    cuFloatComplex v = x[off_x];
    out[off_out] = make_cuFloatComplex(cuCrealf(v), -cuCimagf(v));
  }
}

__global__ void conj_c64_kernel(const cuDoubleComplex *x, cuDoubleComplex *out,
                                int64_t n, int ndim, const int64_t *dims,
                                const int64_t *x_strides,
                                const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    cuDoubleComplex v = x[off_x];
    out[off_out] = make_cuDoubleComplex(cuCreal(v), -cuCimag(v));
  }
}

extern "C" {

C_Status conj_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("conj: null array pointer");
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
  case INSIGHT_DTYPE_C32:
    conj_c32_kernel<<<blocks, threads>>>(
        static_cast<const cuFloatComplex *>(x->data),
        static_cast<cuFloatComplex *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides);
    break;
  case INSIGHT_DTYPE_C64:
    conj_c64_kernel<<<blocks, threads>>>(
        static_cast<const cuDoubleComplex *>(x->data),
        static_cast<cuDoubleComplex *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("conj: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(conj, INSIGHT_DTYPE_C32, conj_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(conj, INSIGHT_DTYPE_C64, conj_kernel_gpu);
