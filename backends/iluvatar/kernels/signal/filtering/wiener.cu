// backends/cuda/kernels/signal/filtering/wiener.cu
// CUDA kernel for Wiener filter using local statistics.
// For each element: compute local mean and variance in a neighborhood,
// then apply y[i] = x[i] - noise * (x[i] - mean) / max(var, eps)
// inputs[0]: data array (F64 or F32)
// inputs[1]: kernel_size (I32, 1-element, neighborhood half-width)
// outputs[0]: filtered array
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void signal_wiener_kernel(T *out, const T *x, int64_t n,
                                     int kernel_size, T noise) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int64_t lo = i - (int64_t)kernel_size;
  if (lo < 0)
    lo = 0;
  int64_t hi = i + (int64_t)kernel_size;
  if (hi >= n)
    hi = n - 1;
  int64_t count = hi - lo + 1;

  T sum = T(0.0);
  T sum2 = T(0.0);
  for (int64_t j = lo; j <= hi; ++j) {
    sum += x[j];
    sum2 += x[j] * x[j];
  }

  T mean = sum / (T)count;
  T var = sum2 / (T)count - mean * mean;
  if (var < T(1e-15))
    var = T(1e-15);

  out[i] = x[i] - noise * (x[i] - mean) / var;
}

__global__ void signal_wiener_kernel_f16(uint16_t *out, const uint16_t *in,
                                         int64_t n, int kernel_size) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int64_t lo = i - (int64_t)kernel_size;
  if (lo < 0)
    lo = 0;
  int64_t hi = i + (int64_t)kernel_size;
  if (hi >= n)
    hi = n - 1;
  int64_t count = hi - lo + 1;

  float sum = 0.0f;
  float sum2 = 0.0f;
  for (int64_t j = lo; j <= hi; ++j) {
    float val = __half2float(*(const __half *)&in[j]);
    sum += val;
    sum2 += val * val;
  }

  float mean = sum / (float)count;
  float var = sum2 / (float)count - mean * mean;
  if (var < 1e-15f)
    var = 1e-15f;

  float x_i = __half2float(*(const __half *)&in[i]);
  float result = x_i - 0.0f * (x_i - mean) / var;
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_wiener_kernel_bf16(uint16_t *out, const uint16_t *in,
                                          int64_t n, int kernel_size) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int64_t lo = i - (int64_t)kernel_size;
  if (lo < 0)
    lo = 0;
  int64_t hi = i + (int64_t)kernel_size;
  if (hi >= n)
    hi = n - 1;
  int64_t count = hi - lo + 1;

  float sum = 0.0f;
  float sum2 = 0.0f;
  for (int64_t j = lo; j <= hi; ++j) {
    float val = __bfloat162float(*(const __nv_bfloat16 *)&in[j]);
    sum += val;
    sum2 += val * val;
  }

  float mean = sum / (float)count;
  float var = sum2 / (float)count - mean * mean;
  if (var < 1e-15f)
    var = 1e-15f;

  float x_i = __bfloat162float(*(const __nv_bfloat16 *)&in[i]);
  float result = x_i - 0.0f * (x_i - mean) / var;
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_wiener_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *ks_arr = (InsightArray *)inputs[1];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !ks_arr || !y_arr) {
    gpu_set_last_error("signal_wiener: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !ks_arr->data || !y_arr->data) {
    gpu_set_last_error("signal_wiener: null data pointer");
    return C_FAILED;
  }

  int32_t kernel_size;
  cudaMemcpy(&kernel_size, ks_arr->data, sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  if (kernel_size < 1)
    kernel_size = 1;

  int64_t n = x_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_wiener_kernel<double>
        <<<blocks, threads>>>((double *)y_arr->data,
                              (const double *)x_arr->data, n, kernel_size, 0.0);
    break;
  case INSIGHT_DTYPE_F32:
    signal_wiener_kernel<float><<<blocks, threads>>>(
        (float *)y_arr->data, (const float *)x_arr->data, n, kernel_size, 0.0f);
    break;
  case INSIGHT_DTYPE_F16:
    signal_wiener_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)y_arr->data, (const uint16_t *)x_arr->data, n, kernel_size);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_wiener_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)y_arr->data, (const uint16_t *)x_arr->data, n, kernel_size);
    break;
  default:
    gpu_set_last_error(
        "signal_wiener: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_wiener, INSIGHT_DTYPE_F64,
                         signal_wiener_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_wiener, INSIGHT_DTYPE_F32,
                         signal_wiener_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_wiener, INSIGHT_DTYPE_F16,
                         signal_wiener_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_wiener, INSIGHT_DTYPE_BF16,
                         signal_wiener_kernel_gpu);
