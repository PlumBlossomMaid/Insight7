// backends/cuda/kernels/signal/acoustics/mel2hz.cu
// CUDA kernel for Mel to Hz conversion.
// hz = 700 * (10^(mel/2595) - 1) for standard
// inputs[0]: mel array (F64 or F32)
// outputs[0]: hz array (same type)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void signal_mel2hz_kernel(T *out, const T *in, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = T(700.0) * (pow(T(10.0), in[i] / T(2595.0)) - T(1.0));
}

__global__ void signal_mel2hz_kernel_f16(uint16_t *out, const uint16_t *in,
                                         int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __half2float(*(const __half *)&in[i]);
  float result = 700.0f * (powf(10.0f, val / 2595.0f) - 1.0f);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_mel2hz_kernel_bf16(uint16_t *out, const uint16_t *in,
                                          int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&in[i]);
  float result = 700.0f * (powf(10.0f, val / 2595.0f) - 1.0f);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_mel2hz_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *mel_arr = (InsightArray *)inputs[0];
  InsightArray *hz_arr = (InsightArray *)outputs[0];

  if (!mel_arr || !hz_arr) {
    gpu_set_last_error("signal_mel2hz: null array pointer");
    return C_FAILED;
  }

  if (!mel_arr->data || !hz_arr->data) {
    gpu_set_last_error("signal_mel2hz: null data pointer");
    return C_FAILED;
  }

  int64_t n = mel_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (mel_arr->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_mel2hz_kernel<double><<<blocks, threads>>>(
        (double *)hz_arr->data, (const double *)mel_arr->data, n);
    break;
  case INSIGHT_DTYPE_F32:
    signal_mel2hz_kernel<float><<<blocks, threads>>>(
        (float *)hz_arr->data, (const float *)mel_arr->data, n);
    break;
  case INSIGHT_DTYPE_F16:
    signal_mel2hz_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)hz_arr->data, (const uint16_t *)mel_arr->data, n);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_mel2hz_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)hz_arr->data, (const uint16_t *)mel_arr->data, n);
    break;
  default:
    gpu_set_last_error(
        "signal_mel2hz: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_mel2hz, INSIGHT_DTYPE_F64, signal_mel2hz_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_mel2hz, INSIGHT_DTYPE_F32, signal_mel2hz_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_mel2hz, INSIGHT_DTYPE_F16, signal_mel2hz_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_mel2hz, INSIGHT_DTYPE_BF16,
                    signal_mel2hz_kernel_gpu);
