// backends/cuda/kernels/signal/waveforms/gausspulse.cu
// CUDA kernel for Gaussian-modulated sinusoidal pulse
// env = exp(a*t^2), I = env * cos(2*pi*fc*t)
// where a = -4*ln(2)*pi^2*fc^2*bw^2 (from -6dB bandwidth reference)
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void gausspulse_kernel_f64(double *out, double a, double two_pi_fc,
                                      int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double inv_M1 = (M > 1) ? 1.0 / (double)(M - 1) : 1.0;
  double t = (double)i * inv_M1;
  double t2 = t * t;
  double env = exp(a * t2);
  out[i] = env * cos(two_pi_fc * t);
}

__global__ void gausspulse_kernel_f32(float *out, float a, float two_pi_fc,
                                      int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float t = (float)i * inv_M1;
  float t2 = t * t;
  float env = expf(a * t2);
  out[i] = env * cosf(two_pi_fc * t);
}

__global__ void gausspulse_kernel_f16(uint16_t *out, float a, float two_pi_fc,
                                      int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float t = (float)i * inv_M1;
  float t2 = t * t;
  float env = expf(a * t2);
  float result = env * cosf(two_pi_fc * t);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void gausspulse_kernel_bf16(uint16_t *out, float a, float two_pi_fc,
                                       int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float t = (float)i * inv_M1;
  float t2 = t * t;
  float env = expf(a * t2);
  float result = env * cosf(two_pi_fc * t);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status gausspulse_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *fc_arr = (InsightArray *)inputs[0];
  InsightArray *bw_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!fc_arr || !bw_arr || !out) {
    gpu_set_last_error("gausspulse: null array pointer");
    return C_FAILED;
  }

  if (!fc_arr->data || !bw_arr->data) {
    gpu_set_last_error("gausspulse: parameter array data is null");
    return C_FAILED;
  }

  double fc = *(const double *)fc_arr->data;
  double bw = *(const double *)bw_arr->data;
  int64_t M = out->numel;

  // Envelope parameter: a = -4*ln(2)*pi^2*fc^2*bw^2
  double a = -4.0 * log(2.0) * M_PI * M_PI * fc * fc * bw * bw;
  double two_pi_fc = 2.0 * M_PI * fc;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    gausspulse_kernel_f64<<<blocks, threads>>>((double *)out->data, a,
                                               two_pi_fc, M);
    break;
  case INSIGHT_DTYPE_F32:
    gausspulse_kernel_f32<<<blocks, threads>>>((float *)out->data, (float)a,
                                               (float)two_pi_fc, M);
    break;
  case INSIGHT_DTYPE_F16:
    gausspulse_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, (float)a,
                                               (float)two_pi_fc, M);
    break;
  case INSIGHT_DTYPE_BF16:
    gausspulse_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, (float)a,
                                                (float)two_pi_fc, M);
    break;
  default:
    gpu_set_last_error(
        "gausspulse: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_IXUCA_KERNEL(gausspulse, INSIGHT_DTYPE_F64, gausspulse_kernel_gpu);
REGISTER_IXUCA_KERNEL(gausspulse, INSIGHT_DTYPE_F32, gausspulse_kernel_gpu);
REGISTER_IXUCA_KERNEL(gausspulse, INSIGHT_DTYPE_F16, gausspulse_kernel_gpu);
REGISTER_IXUCA_KERNEL(gausspulse, INSIGHT_DTYPE_BF16, gausspulse_kernel_gpu);
