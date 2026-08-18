// backends/cuda/kernels/signal/wavelets/ricker.cu
// CUDA kernel for Ricker (Mexican hat) wavelet.
// w[n] = (2/(sqrt(3*a)*pi^0.25)) * (1 - (t/a)^2) * exp(-t^2/(2*a^2))
// where t = n - (M-1)/2
// inputs[0]: a (width parameter, F64, 1-element)
// outputs[0]: real wavelet (F64)
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void signal_ricker_kernel(double *out, int64_t M, double a) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  double half = (double)(M - 1) / 2.0;
  double t = (double)i - half;
  double inv_a = 1.0 / a;
  double ta = t * inv_a;
  double norm = 2.0 / (sqrt(3.0 * a) * pow(M_PI, 0.25));
  out[i] = norm * (1.0 - ta * ta) * exp(-0.5 * t * t * inv_a * inv_a);
}

__global__ void signal_ricker_kernel_f32(float *out, int64_t M, float a) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  float half = (float)(M - 1) / 2.0f;
  float t = (float)i - half;
  float inv_a = 1.0f / a;
  float ta = t * inv_a;
  float norm = 2.0f / (sqrtf(3.0f * a) * powf((float)M_PI, 0.25f));
  out[i] = norm * (1.0f - ta * ta) * expf(-0.5f * t * t * inv_a * inv_a);
}

__global__ void signal_ricker_kernel_f16(uint16_t *out, int64_t M, float a) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  float half = (float)(M - 1) / 2.0f;
  float t = (float)i - half;
  float inv_a = 1.0f / a;
  float ta = t * inv_a;
  float norm = 2.0f / (sqrtf(3.0f * a) * powf((float)M_PI, 0.25f));
  float result = norm * (1.0f - ta * ta) * expf(-0.5f * t * t * inv_a * inv_a);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_ricker_kernel_bf16(uint16_t *out, int64_t M, float a) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  float half = (float)(M - 1) / 2.0f;
  float t = (float)i - half;
  float inv_a = 1.0f / a;
  float ta = t * inv_a;
  float norm = 2.0f / (sqrtf(3.0f * a) * powf((float)M_PI, 0.25f));
  float result = norm * (1.0f - ta * ta) * expf(-0.5f * t * t * inv_a * inv_a);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_ricker_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *a_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a_arr || !out) {
    gpu_set_last_error("signal_ricker: null array pointer");
    return C_FAILED;
  }

  if (!a_arr->data || !out->data) {
    gpu_set_last_error("signal_ricker: null data pointer");
    return C_FAILED;
  }

  double a;
  cudaMemcpy(&a, a_arr->data, sizeof(double), cudaMemcpyDeviceToHost);

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_ricker_kernel<<<blocks, threads>>>((double *)out->data, M, a);
    break;
  case INSIGHT_DTYPE_F32:
    signal_ricker_kernel_f32<<<blocks, threads>>>((float *)out->data, M,
                                                  (float)a);
    break;
  case INSIGHT_DTYPE_F16:
    signal_ricker_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, M,
                                                  (float)a);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_ricker_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M,
                                                   (float)a);
    break;
  default:
    gpu_set_last_error(
        "signal_ricker: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_IXUCA_KERNEL(signal_ricker, INSIGHT_DTYPE_F64,
                         signal_ricker_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_ricker, INSIGHT_DTYPE_F32,
                         signal_ricker_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_ricker, INSIGHT_DTYPE_F16,
                         signal_ricker_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_ricker, INSIGHT_DTYPE_BF16,
                         signal_ricker_kernel_gpu);
