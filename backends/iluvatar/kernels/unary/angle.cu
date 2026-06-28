// backends/cuda/kernels/unary/angle.cu
/**
 * @file angle.cu
 * @brief CUDA kernel for angle/phase computation.
 *
 * For complex inputs: computes atan2(imag, real).
 * For real inputs: computes atan2(0, x) which returns 0 for x >= 0, pi for x <
 * 0.
 *
 * C32 -> F32, C64 -> F64, F32 -> F32, F64 -> F64.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>

// --- Real kernels (same dtype in/out) ---

__global__ void angle_f32_kernel(const float *x, float *out, int64_t n,
                                 int ndim, const int64_t *dims,
                                 const int64_t *x_strides,
                                 const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    out[off_out] = atan2f(0.0f, x[off_x]);
  }
}

__global__ void angle_f64_kernel(const double *x, double *out, int64_t n,
                                 int ndim, const int64_t *dims,
                                 const int64_t *x_strides,
                                 const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    out[off_out] = atan2(0.0, x[off_x]);
  }
}

// --- Complex kernels (complex in, real out) ---

__global__ void angle_c32_kernel(const cuFloatComplex *x, float *out, int64_t n,
                                 int ndim, const int64_t *dims,
                                 const int64_t *x_strides,
                                 const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    cuFloatComplex v = x[off_x];
    out[off_out] = atan2f(cuCimagf(v), cuCrealf(v));
  }
}

__global__ void angle_c64_kernel(const cuDoubleComplex *x, double *out,
                                 int64_t n, int ndim, const int64_t *dims,
                                 const int64_t *x_strides,
                                 const int64_t *out_strides) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = unary_offset(idx, ndim, dims, out_strides);
    cuDoubleComplex v = x[off_x];
    out[off_out] = atan2(cuCimag(v), cuCreal(v));
  }
}

extern "C" {

C_Status angle_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("angle: null array pointer");
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

  switch (x->dtype) {
  case INSIGHT_DTYPE_C32:
    angle_c32_kernel<<<blocks, threads>>>(
        static_cast<const cuFloatComplex *>(x->data),
        static_cast<float *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides);
    break;
  case INSIGHT_DTYPE_C64:
    angle_c64_kernel<<<blocks, threads>>>(
        static_cast<const cuDoubleComplex *>(x->data),
        static_cast<double *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides);
    break;
  case INSIGHT_DTYPE_F32:
    angle_f32_kernel<<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        ndim, d_dims, d_x_strides, d_out_strides);
    break;
  case INSIGHT_DTYPE_F64:
    angle_f64_kernel<<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("angle: unsupported dtype");
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

// Register for all supported types (keyed by input dtype)
REGISTER_ILUVATAR_KERNEL(angle, INSIGHT_DTYPE_C32, angle_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(angle, INSIGHT_DTYPE_C64, angle_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(angle, INSIGHT_DTYPE_F32, angle_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(angle, INSIGHT_DTYPE_F64, angle_kernel_gpu);
