// backends/cuda/kernels/signal/bsplines/gauss_spline.cu
// CUDA kernel for Gaussian approximation to B-spline
// gauss_spline(x, n) = 1/sqrt(2*pi*sigma^2) * exp(-x^2 / (2*sigma^2))
// where sigma^2 = (n+1)/12
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void gauss_spline_kernel_f64(double *data, double coeff,
                                        double r_sigma_sq, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double x2 = data[i] * data[i];
  data[i] = coeff * exp(-x2 * r_sigma_sq);
}

__global__ void gauss_spline_kernel_f32(float *data, float coeff,
                                        float r_sigma_sq, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float x2 = data[i] * data[i];
  data[i] = coeff * expf(-x2 * r_sigma_sq);
}

__global__ void gauss_spline_kernel_f16(uint16_t *data, float coeff,
                                        float r_sigma_sq, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = __half2float(*(const __half *)&data[i]);
  float x2 = val * val;
  float result = coeff * expf(-x2 * r_sigma_sq);
  __half res = __float2half(result);
  data[i] = *(uint16_t *)&res;
}

__global__ void gauss_spline_kernel_bf16(uint16_t *data, float coeff,
                                         float r_sigma_sq, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&data[i]);
  float x2 = val * val;
  float result = coeff * expf(-x2 * r_sigma_sq);
  __nv_bfloat16 res = __float2bfloat16(result);
  data[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status gauss_spline_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *r_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!r_arr || !out) {
    gpu_set_last_error("gauss_spline: null array pointer");
    return C_FAILED;
  }

  if (!r_arr->data || r_arr->numel < 1) {
    gpu_set_last_error("gauss_spline: order array is empty");
    return C_FAILED;
  }

  // Read order parameter
  double n;
  if (r_arr->dtype == INSIGHT_DTYPE_F64) {
    n = *(const double *)r_arr->data;
  } else if (r_arr->dtype == INSIGHT_DTYPE_I32) {
    n = (double)(*(const int32_t *)r_arr->data);
  } else if (r_arr->dtype == INSIGHT_DTYPE_I64) {
    n = (double)(*(const int64_t *)r_arr->data);
  } else if (r_arr->dtype == INSIGHT_DTYPE_F32) {
    n = (double)(*(const float *)r_arr->data);
  } else {
    gpu_set_last_error("gauss_spline: unsupported order dtype");
    return C_FAILED;
  }

  double sigma_sq = (n + 1.0) / 12.0;
  double r_sigma_sq = 0.5 / sigma_sq;
  double coeff = 1.0 / sqrt(2.0 * M_PI * sigma_sq);

  int64_t M = out->numel;
  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    gauss_spline_kernel_f64<<<blocks, threads>>>((double *)out->data, coeff,
                                                 r_sigma_sq, M);
    break;
  case INSIGHT_DTYPE_F32:
    gauss_spline_kernel_f32<<<blocks, threads>>>(
        (float *)out->data, (float)coeff, (float)r_sigma_sq, M);
    break;
  case INSIGHT_DTYPE_F16:
    gauss_spline_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)out->data, (float)coeff, (float)r_sigma_sq, M);
    break;
  case INSIGHT_DTYPE_BF16:
    gauss_spline_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)out->data, (float)coeff, (float)r_sigma_sq, M);
    break;
  default:
    gpu_set_last_error(
        "gauss_spline: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(gauss_spline, INSIGHT_DTYPE_F64, gauss_spline_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(gauss_spline, INSIGHT_DTYPE_F32, gauss_spline_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(gauss_spline, INSIGHT_DTYPE_F16, gauss_spline_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(gauss_spline, INSIGHT_DTYPE_BF16, gauss_spline_kernel_gpu);
