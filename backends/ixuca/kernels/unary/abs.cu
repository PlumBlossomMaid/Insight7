// backends/cuda/kernels/unary/abs.cu
/**
 * @file abs.cu
 * @brief CUDA kernel for absolute value.
 *
 * Computes element-wise absolute value.
 * Supports signed integers, floats, and complex types.
 *
 * @param inputs  [0] = InsightArray* input
 * @param outputs [0] = InsightArray* output
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>

// Signed integer abs
template <typename T>
__global__ void abs_signed_kernel(const T *x, T *out, int64_t n, int ndim,
                                  const int64_t *dims, const int64_t *x_strides,
                                  const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    T v = x[off_x];
    out[off_out] = v < 0 ? -v : v;
  }
}

// Float abs
__global__ void abs_f32_kernel(const float *x, float *out, int64_t n, int ndim,
                               const int64_t *dims, const int64_t *x_strides,
                               const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    out[off_out] = fabsf(x[off_x]);
  }
}

__global__ void abs_f64_kernel(const double *x, double *out, int64_t n,
                               int ndim, const int64_t *dims,
                               const int64_t *x_strides,
                               const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    out[off_out] = fabs(x[off_x]);
  }
}

// Complex abs
__global__ void abs_c32_kernel(const cuFloatComplex *x, float *out, int64_t n,
                               int ndim, const int64_t *dims,
                               const int64_t *x_strides,
                               const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    cuFloatComplex v = x[off_x];
    float r = cuCrealf(v);
    float i = cuCimagf(v);
    out[off_out] = sqrtf(r * r + i * i);
  }
}

__global__ void abs_c64_kernel(const cuDoubleComplex *x, double *out, int64_t n,
                               int ndim, const int64_t *dims,
                               const int64_t *x_strides,
                               const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    cuDoubleComplex v = x[off_x];
    double r = cuCreal(v);
    double i = cuCimag(v);
    out[off_out] = sqrt(r * r + i * i);
  }
}

extern "C" {

C_Status abs_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("abs: null array pointer");
    return C_FAILED;
  }

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int ndim = out->ndim;
  int threads = unary_threads();
  int blocks = unary_blocks(n);

  // Allocate device memory for dims and strides
  int64_t *d_dims, *d_x_strides, *d_out_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, out->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_strides, x->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_strides, out->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  // For complex input, abs returns real output. Dispatch on input dtype
  // for complex cases, output dtype for everything else.
  int32_t dispatch_dtype = out->dtype;
  if (x->dtype == INSIGHT_DTYPE_C32 || x->dtype == INSIGHT_DTYPE_C64)
    dispatch_dtype = x->dtype;

  switch (dispatch_dtype) {
  case INSIGHT_DTYPE_I8:
    abs_signed_kernel<int8_t><<<blocks, threads>>>(
        static_cast<const int8_t *>(x->data), static_cast<int8_t *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I16:
    abs_signed_kernel<int16_t>
        <<<blocks, threads>>>(static_cast<const int16_t *>(x->data),
                              static_cast<int16_t *>(out->data), n, ndim,
                              d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I32:
    abs_signed_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<const int32_t *>(x->data),
                              static_cast<int32_t *>(out->data), n, ndim,
                              d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_I64:
    abs_signed_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<const int64_t *>(x->data),
                              static_cast<int64_t *>(out->data), n, ndim,
                              d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_F32:
    abs_f32_kernel<<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        ndim, d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_F64:
    abs_f64_kernel<<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_C32:
    abs_c32_kernel<<<blocks, threads>>>(
        static_cast<const cuFloatComplex *>(x->data),
        static_cast<float *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides);
    break;
  case INSIGHT_DTYPE_C64:
    abs_c64_kernel<<<blocks, threads>>>(
        static_cast<const cuDoubleComplex *>(x->data),
        static_cast<double *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("abs: unsupported dtype");
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

// Register for all supported types
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_I8, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_I16, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_I32, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_I64, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_F32, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_F64, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_C32, abs_kernel_gpu);
REGISTER_IXUCA_KERNEL(abs, INSIGHT_DTYPE_C64, abs_kernel_gpu);
