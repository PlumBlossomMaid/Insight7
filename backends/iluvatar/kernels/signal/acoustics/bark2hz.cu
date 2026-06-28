// backends/cuda/kernels/signal/acoustics/bark2hz.cu
// CUDA kernel for Bark to Hz conversion.
// Uses Newton's method to invert: bark = 13*atan(0.00076*hz)
// + 3.5*atan((hz/7500)^2) Initial guess: hz = 600 * sinh(bark / 3.0)
// (Traunmuller approximation) Then refine with 5 Newton iterations. inputs[0]:
// bark array (F64 or F32) outputs[0]: hz array (same type)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ double bark_func_dev(double hz) {
  double ratio = hz / 7500.0;
  return 13.0 * atan(0.00076 * hz) + 3.5 * atan(ratio * ratio);
}

__device__ double bark_deriv_dev(double hz) {
  double t1 = 0.00076 * hz;
  double t2 = hz / 7500.0;
  double t2sq = t2 * t2;
  return 13.0 * 0.00076 / (1.0 + t1 * t1) +
         3.5 * 2.0 * t2 * (1.0 / 7500.0) / (1.0 + t2sq * t2sq);
}

__global__ void signal_bark2hz_kernel_f64(double *out, const double *in,
                                          int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  double b = in[i];
  double h = 600.0 * sinh(b / 3.0);

  for (int iter = 0; iter < 5; ++iter) {
    double f = bark_func_dev(h) - b;
    double df = bark_deriv_dev(h);
    if (fabs(df) < 1e-30)
      break;
    h -= f / df;
    if (h < 0.0)
      h = 0.0;
  }
  out[i] = h;
}

__global__ void signal_bark2hz_kernel_f32(float *out, const float *in,
                                          int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float b = in[i];
  float h = 600.0f * sinhf(b / 3.0f);

  for (int iter = 0; iter < 5; ++iter) {
    double hd = (double)h;
    double f = bark_func_dev(hd) - (double)b;
    double df = bark_deriv_dev(hd);
    if (fabs(df) < 1e-30)
      break;
    h -= (float)(f / df);
    if (h < 0.0f)
      h = 0.0f;
  }
  out[i] = h;
}

__global__ void signal_bark2hz_kernel_f16(uint16_t *out, const uint16_t *in,
                                          int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __half2float(*(const __half *)&in[i]);
  float b = val;
  float h = 600.0f * sinhf(b / 3.0f);
  for (int iter = 0; iter < 5; ++iter) {
    double hd = (double)h;
    double f = bark_func_dev(hd) - (double)b;
    double df = bark_deriv_dev(hd);
    if (fabs(df) < 1e-30)
      break;
    h -= (float)(f / df);
    if (h < 0.0f)
      h = 0.0f;
  }
  __half res = __float2half(h);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_bark2hz_kernel_bf16(uint16_t *out, const uint16_t *in,
                                           int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&in[i]);
  float b = val;
  float h = 600.0f * sinhf(b / 3.0f);
  for (int iter = 0; iter < 5; ++iter) {
    double hd = (double)h;
    double f = bark_func_dev(hd) - (double)b;
    double df = bark_deriv_dev(hd);
    if (fabs(df) < 1e-30)
      break;
    h -= (float)(f / df);
    if (h < 0.0f)
      h = 0.0f;
  }
  __nv_bfloat16 res = __float2bfloat16(h);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_bark2hz_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *bark_arr = (InsightArray *)inputs[0];
  InsightArray *hz_arr = (InsightArray *)outputs[0];

  if (!bark_arr || !hz_arr) {
    gpu_set_last_error("signal_bark2hz: null array pointer");
    return C_FAILED;
  }

  if (!bark_arr->data || !hz_arr->data) {
    gpu_set_last_error("signal_bark2hz: null data pointer");
    return C_FAILED;
  }

  int64_t n = bark_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (bark_arr->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_bark2hz_kernel_f64<<<blocks, threads>>>(
        (double *)hz_arr->data, (const double *)bark_arr->data, n);
    break;
  case INSIGHT_DTYPE_F32:
    signal_bark2hz_kernel_f32<<<blocks, threads>>>(
        (float *)hz_arr->data, (const float *)bark_arr->data, n);
    break;
  case INSIGHT_DTYPE_F16:
    signal_bark2hz_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)hz_arr->data, (const uint16_t *)bark_arr->data, n);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_bark2hz_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)hz_arr->data, (const uint16_t *)bark_arr->data, n);
    break;
  default:
    gpu_set_last_error(
        "signal_bark2hz: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_bark2hz, INSIGHT_DTYPE_F64,
                         signal_bark2hz_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_bark2hz, INSIGHT_DTYPE_F32,
                         signal_bark2hz_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_bark2hz, INSIGHT_DTYPE_F16,
                         signal_bark2hz_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_bark2hz, INSIGHT_DTYPE_BF16,
                         signal_bark2hz_kernel_gpu);
