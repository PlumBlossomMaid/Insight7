// backends/cuda/kernels/signal/filter_design/firwin.cu
// CUDA kernel for FIR filter windowed sinc design.
// h[n] = sinc(2*fc*(n - (M-1)/2)) * window[n] where fc is cutoff frequency
// The window function is a Hann window applied element-wise.
// inputs[0]: fc (cutoff frequency, F64, 1-element)
// inputs[1]: M (filter length, I64, 1-element)
// outputs[0]: FIR coefficients (F64)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ double sinc_dev(double x) {
  if (fabs(x) < 1e-15)
    return 1.0;
  return sin(M_PI * x) / (M_PI * x);
}

__global__ void signal_firwin_kernel(double *out, int64_t M_len, double fc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M_len)
    return;

  double half = (double)(M_len - 1) / 2.0;
  double n_centered = (double)i - half;
  double h = sinc_dev(2.0 * fc * n_centered);
  double win = 0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (double)(M_len - 1)));
  out[i] = h * win;
}

__global__ void signal_firwin_kernel_f32(float *out, int64_t M_len, float fc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M_len)
    return;

  float half = (float)(M_len - 1) / 2.0f;
  float n_centered = (float)i - half;
  double arg = 2.0 * (double)fc * (double)n_centered;
  float h =
      (fabs(arg) < 1e-15) ? 1.0f : (float)(sin(M_PI * arg) / (M_PI * arg));
  float win =
      0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(M_len - 1)));
  out[i] = h * win;
}

__global__ void signal_firwin_kernel_f16(uint16_t *out, int64_t M_len,
                                         float fc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M_len)
    return;

  float half = (float)(M_len - 1) / 2.0f;
  float n_centered = (float)i - half;
  double arg = 2.0 * (double)fc * (double)n_centered;
  float h =
      (fabs(arg) < 1e-15) ? 1.0f : (float)(sin(M_PI * arg) / (M_PI * arg));
  float win =
      0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(M_len - 1)));
  __half res = __float2half(h * win);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_firwin_kernel_bf16(uint16_t *out, int64_t M_len,
                                          float fc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M_len)
    return;

  float half = (float)(M_len - 1) / 2.0f;
  float n_centered = (float)i - half;
  double arg = 2.0 * (double)fc * (double)n_centered;
  float h =
      (fabs(arg) < 1e-15) ? 1.0f : (float)(sin(M_PI * arg) / (M_PI * arg));
  float win =
      0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(M_len - 1)));
  __nv_bfloat16 res = __float2bfloat16(h * win);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_firwin_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *fc_arr = (InsightArray *)inputs[0];
  InsightArray *M_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!fc_arr || !M_arr || !out) {
    gpu_set_last_error("signal_firwin: null array pointer");
    return C_FAILED;
  }

  if (!fc_arr->data || !M_arr->data || !out->data) {
    gpu_set_last_error("signal_firwin: null data pointer");
    return C_FAILED;
  }

  double fc;
  int64_t M_len;
  cudaMemcpy(&fc, fc_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&M_len, M_arr->data, sizeof(int64_t), cudaMemcpyDeviceToHost);

  if (M_len <= 0) {
    gpu_set_last_error("signal_firwin: M must be positive");
    return C_FAILED;
  }

  int threads = 256;
  int blocks = (int)((M_len + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_firwin_kernel<<<blocks, threads>>>((double *)out->data, M_len, fc);
    break;
  case INSIGHT_DTYPE_F32:
    signal_firwin_kernel_f32<<<blocks, threads>>>((float *)out->data, M_len,
                                                  (float)fc);
    break;
  case INSIGHT_DTYPE_F16:
    signal_firwin_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, M_len,
                                                  (float)fc);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_firwin_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M_len,
                                                   (float)fc);
    break;
  default:
    gpu_set_last_error(
        "signal_firwin: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_firwin, INSIGHT_DTYPE_F64,
                         signal_firwin_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_firwin, INSIGHT_DTYPE_F32,
                         signal_firwin_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_firwin, INSIGHT_DTYPE_F16,
                         signal_firwin_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_firwin, INSIGHT_DTYPE_BF16,
                         signal_firwin_kernel_gpu);
