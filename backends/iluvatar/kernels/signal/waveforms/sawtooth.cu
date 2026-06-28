// backends/cuda/kernels/signal/waveforms/sawtooth.cu
// CUDA kernel for sawtooth/triangle waveform
// w[n] = 2 * (t*n - floor(0.5 + t*n)) where t = width parameter
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sawtooth_kernel_f64(double *out, double width, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double tn = width * (double)i;
  out[i] = 2.0 * (tn - floor(0.5 + tn));
}

__global__ void sawtooth_kernel_f32(float *out, float width, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float tn = width * (float)i;
  out[i] = 2.0f * (tn - floorf(0.5f + tn));
}

__global__ void sawtooth_kernel_f16(uint16_t *out, float width, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float tn = width * (float)i;
  float result = 2.0f * (tn - floorf(0.5f + tn));
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void sawtooth_kernel_bf16(uint16_t *out, float width, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float tn = width * (float)i;
  float result = 2.0f * (tn - floorf(0.5f + tn));
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status sawtooth_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *width_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!width_arr || !out) {
    gpu_set_last_error("sawtooth: null array pointer");
    return C_FAILED;
  }

  if (!width_arr->data || width_arr->numel < 1) {
    gpu_set_last_error("sawtooth: width array is empty");
    return C_FAILED;
  }

  double width = *(const double *)width_arr->data;
  int64_t M = out->numel;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    sawtooth_kernel_f64<<<blocks, threads>>>((double *)out->data, width, M);
    break;
  case INSIGHT_DTYPE_F32:
    sawtooth_kernel_f32<<<blocks, threads>>>((float *)out->data, (float)width,
                                             M);
    break;
  case INSIGHT_DTYPE_F16:
    sawtooth_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data,
                                             (float)width, M);
    break;
  case INSIGHT_DTYPE_BF16:
    sawtooth_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data,
                                              (float)width, M);
    break;
  default:
    gpu_set_last_error(
        "sawtooth: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(sawtooth, INSIGHT_DTYPE_F64, sawtooth_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(sawtooth, INSIGHT_DTYPE_F32, sawtooth_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(sawtooth, INSIGHT_DTYPE_F16, sawtooth_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(sawtooth, INSIGHT_DTYPE_BF16, sawtooth_kernel_gpu);
