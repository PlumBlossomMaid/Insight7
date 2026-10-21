// backends/cuda/kernels/signal/windows/bartlett.cu
// CUDA kernel for Bartlett window generation.
// w[n] = 1 - |2*n/(M-1) - 1|
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T> __global__ void bartlett_kernel(T *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  if (M == 1) {
    out[0] = T(1.0);
    return;
  }
  T denom = T(M - 1);
  out[i] = T(1.0) - fabs(T(2.0) * T(i) / denom - T(1.0));
}

__global__ void bartlett_kernel_fp16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  if (M == 1) {
    out[0] = __float2half(1.0f);
    return;
  }
  float denom = (float)(M - 1);
  float val = 1.0f - fabsf(2.0f * (float)i / denom - 1.0f);
  out[i] = __float2half(val);
}

__global__ void bartlett_kernel_bf16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  if (M == 1) {
    out[0] = __float2bfloat16(1.0f);
    return;
  }
  float denom = (float)(M - 1);
  float val = 1.0f - fabsf(2.0f * (float)i / denom - 1.0f);
  out[i] = __float2bfloat16(val);
}

extern "C" {

C_Status bartlett_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("bartlett: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    bartlett_kernel<double><<<blocks, threads>>>((double *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    bartlett_kernel<float><<<blocks, threads>>>((float *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_F16) {
    bartlett_kernel_fp16<<<blocks, threads>>>((uint16_t *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_BF16) {
    bartlett_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M);
  } else {
    gpu_set_last_error("bartlett: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(bartlett, INSIGHT_DTYPE_F64, bartlett_kernel_gpu);
REGISTER_IXUCA_KERNEL(bartlett, INSIGHT_DTYPE_F32, bartlett_kernel_gpu);
REGISTER_IXUCA_KERNEL(bartlett, INSIGHT_DTYPE_F16, bartlett_kernel_gpu);
REGISTER_IXUCA_KERNEL(bartlett, INSIGHT_DTYPE_BF16, bartlett_kernel_gpu);
