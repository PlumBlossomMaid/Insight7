// backends/cuda/kernels/signal/waveforms/chirp.cu
// CUDA kernel for linear chirp signal
// w[n] = cos(2*pi * (f0 + (f1-f0)*t/2) * t) where t = n/(M-1) * T
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void chirp_kernel_f64(double *out, double f0, double f1, double T,
                                 int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double inv_M1 = (M > 1) ? 1.0 / (double)(M - 1) : 1.0;
  double t = (double)i * inv_M1 * T;
  double beta = (f1 - f0) / T;
  double inst_freq = f0 + beta * t;
  out[i] = cos(2.0 * M_PI * inst_freq * t);
}

__global__ void chirp_kernel_f32(float *out, float f0, float f1, float T,
                                 int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float t = (float)i * inv_M1 * T;
  float beta = (f1 - f0) / T;
  float inst_freq = f0 + beta * t;
  out[i] = cosf(2.0f * (float)M_PI * inst_freq * t);
}

__global__ void chirp_kernel_f16(uint16_t *out, float f0, float f1, float T,
                                 int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float t = (float)i * inv_M1 * T;
  float beta = (f1 - f0) / T;
  float inst_freq = f0 + beta * t;
  float result = cosf(2.0f * (float)M_PI * inst_freq * t);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void chirp_kernel_bf16(uint16_t *out, float f0, float f1, float T,
                                  int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float t = (float)i * inv_M1 * T;
  float beta = (f1 - f0) / T;
  float inst_freq = f0 + beta * t;
  float result = cosf(2.0f * (float)M_PI * inst_freq * t);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status chirp_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *f0_arr = (InsightArray *)inputs[0];
  InsightArray *f1_arr = (InsightArray *)inputs[1];
  InsightArray *T_arr = (InsightArray *)inputs[2];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!f0_arr || !f1_arr || !T_arr || !out) {
    gpu_set_last_error("chirp: null array pointer");
    return C_FAILED;
  }

  if (!f0_arr->data || !f1_arr->data || !T_arr->data) {
    gpu_set_last_error("chirp: parameter array data is null");
    return C_FAILED;
  }

  double f0 = *(const double *)f0_arr->data;
  double f1 = *(const double *)f1_arr->data;
  double T = *(const double *)T_arr->data;
  int64_t M = out->numel;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    chirp_kernel_f64<<<blocks, threads>>>((double *)out->data, f0, f1, T, M);
    break;
  case INSIGHT_DTYPE_F32:
    chirp_kernel_f32<<<blocks, threads>>>((float *)out->data, (float)f0,
                                          (float)f1, (float)T, M);
    break;
  case INSIGHT_DTYPE_F16:
    chirp_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, (float)f0,
                                          (float)f1, (float)T, M);
    break;
  case INSIGHT_DTYPE_BF16:
    chirp_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, (float)f0,
                                           (float)f1, (float)T, M);
    break;
  default:
    gpu_set_last_error("chirp: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(chirp, INSIGHT_DTYPE_F64, chirp_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(chirp, INSIGHT_DTYPE_F32, chirp_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(chirp, INSIGHT_DTYPE_F16, chirp_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(chirp, INSIGHT_DTYPE_BF16, chirp_kernel_gpu);
