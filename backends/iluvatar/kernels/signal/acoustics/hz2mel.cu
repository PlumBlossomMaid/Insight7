// backends/cuda/kernels/signal/acoustics/hz2mel.cu
// CUDA kernel for Hz to Mel conversion.
// mel = 2595 * log10(1 + hz/700)
// inputs[0]: hz array (F64 or F32)
// outputs[0]: mel array (same type)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void signal_hz2mel_kernel(T *out, const T *in, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = T(2595.0) * log10(T(1.0) + in[i] / T(700.0));
}

__global__ void signal_hz2mel_kernel_f16(uint16_t *out, const uint16_t *in,
                                         int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __half2float(*(const __half *)&in[i]);
  float result = 2595.0f * log10f(1.0f + val / 700.0f);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_hz2mel_kernel_bf16(uint16_t *out, const uint16_t *in,
                                          int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&in[i]);
  float result = 2595.0f * log10f(1.0f + val / 700.0f);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_hz2mel_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *hz_arr = (InsightArray *)inputs[0];
  InsightArray *mel_arr = (InsightArray *)outputs[0];

  if (!hz_arr || !mel_arr) {
    gpu_set_last_error("signal_hz2mel: null array pointer");
    return C_FAILED;
  }

  if (!hz_arr->data || !mel_arr->data) {
    gpu_set_last_error("signal_hz2mel: null data pointer");
    return C_FAILED;
  }

  int64_t n = hz_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (hz_arr->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_hz2mel_kernel<double><<<blocks, threads>>>(
        (double *)mel_arr->data, (const double *)hz_arr->data, n);
    break;
  case INSIGHT_DTYPE_F32:
    signal_hz2mel_kernel<float><<<blocks, threads>>>(
        (float *)mel_arr->data, (const float *)hz_arr->data, n);
    break;
  case INSIGHT_DTYPE_F16:
    signal_hz2mel_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)mel_arr->data, (const uint16_t *)hz_arr->data, n);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_hz2mel_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)mel_arr->data, (const uint16_t *)hz_arr->data, n);
    break;
  default:
    gpu_set_last_error(
        "signal_hz2mel: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_hz2mel, INSIGHT_DTYPE_F64,
                         signal_hz2mel_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_hz2mel, INSIGHT_DTYPE_F32,
                         signal_hz2mel_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_hz2mel, INSIGHT_DTYPE_F16,
                         signal_hz2mel_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_hz2mel, INSIGHT_DTYPE_BF16,
                         signal_hz2mel_kernel_gpu);
