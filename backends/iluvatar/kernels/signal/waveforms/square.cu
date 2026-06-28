// backends/cuda/kernels/signal/waveforms/square.cu
// CUDA kernel for square wave waveform
// w[n] = 1 if (n/(M-1)) mod 1.0 < duty else -1
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void square_kernel_f64(double *out, double duty, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double inv_M1 = (M > 1) ? 1.0 / (double)(M - 1) : 1.0;
  double phase = fmod((double)i * inv_M1, 1.0);
  out[i] = (phase < duty) ? 1.0 : -1.0;
}

__global__ void square_kernel_f32(float *out, float duty, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float phase = fmodf((float)i * inv_M1, 1.0f);
  out[i] = (phase < duty) ? 1.0f : -1.0f;
}

__global__ void square_kernel_f16(uint16_t *out, float duty, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float phase = fmodf((float)i * inv_M1, 1.0f);
  float result = (phase < duty) ? 1.0f : -1.0f;
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void square_kernel_bf16(uint16_t *out, float duty, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
  float phase = fmodf((float)i * inv_M1, 1.0f);
  float result = (phase < duty) ? 1.0f : -1.0f;
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

// signal_square kernel (renamed to avoid collision with unary square)
C_Status signal_square_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *duty_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!duty_arr || !out) {
    gpu_set_last_error("signal_square: null array pointer");
    return C_FAILED;
  }

  if (!duty_arr->data || duty_arr->numel < 1) {
    gpu_set_last_error("square: duty array is empty");
    return C_FAILED;
  }

  double duty = *(const double *)duty_arr->data;
  int64_t M = out->numel;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    square_kernel_f64<<<blocks, threads>>>((double *)out->data, duty, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    square_kernel_f32<<<blocks, threads>>>((float *)out->data, (float)duty, M);
  } else if (out->dtype == INSIGHT_DTYPE_F16) {
    square_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, (float)duty,
                                           M);
  } else if (out->dtype == INSIGHT_DTYPE_BF16) {
    square_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, (float)duty,
                                            M);
  } else {
    gpu_set_last_error(
        "square: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_square, INSIGHT_DTYPE_F64,
                         signal_square_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_square, INSIGHT_DTYPE_F32,
                         signal_square_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_square, INSIGHT_DTYPE_F16,
                         signal_square_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_square, INSIGHT_DTYPE_BF16,
                         signal_square_kernel_gpu);
