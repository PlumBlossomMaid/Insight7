// backends/cuda/kernels/signal/windows/triang.cu
// CUDA kernel for triangular window generation.
// Matches scipy.signal.windows.triang implementation.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T> __global__ void triang_kernel_odd(T *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  T denom = T(M / 2 + 1);
  if (n <= (M - 1) / 2) {
    out[n] = T(n + 1) / denom;
  } else {
    out[n] = T(M - n) / denom;
  }
}

template <typename T> __global__ void triang_kernel_even(T *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  T denom = T(M / 2);
  if (n < M / 2) {
    out[n] = T(n + 1) / denom;
  } else {
    out[n] = T(M - n) / denom;
  }
}

__global__ void triang_kernel_odd_fp16(uint16_t *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  float denom = (float)(M / 2 + 1);
  if (n <= (M - 1) / 2) {
    out[n] = __float2half((float)(n + 1) / denom);
  } else {
    out[n] = __float2half((float)(M - n) / denom);
  }
}

__global__ void triang_kernel_even_fp16(uint16_t *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  float denom = (float)(M / 2);
  if (n < M / 2) {
    out[n] = __float2half((float)(n + 1) / denom);
  } else {
    out[n] = __float2half((float)(M - n) / denom);
  }
}

__global__ void triang_kernel_odd_bf16(uint16_t *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  float denom = (float)(M / 2 + 1);
  if (n <= (M - 1) / 2) {
    out[n] = __float2bfloat16((float)(n + 1) / denom);
  } else {
    out[n] = __float2bfloat16((float)(M - n) / denom);
  }
}

__global__ void triang_kernel_even_bf16(uint16_t *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  float denom = (float)(M / 2);
  if (n < M / 2) {
    out[n] = __float2bfloat16((float)(n + 1) / denom);
  } else {
    out[n] = __float2bfloat16((float)(M - n) / denom);
  }
}

extern "C" {

C_Status triang_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("triang: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    if (M % 2 == 1) {
      triang_kernel_odd<double><<<blocks, threads>>>((double *)out->data, M);
    } else {
      triang_kernel_even<double><<<blocks, threads>>>((double *)out->data, M);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    if (M % 2 == 1) {
      triang_kernel_odd<float><<<blocks, threads>>>((float *)out->data, M);
    } else {
      triang_kernel_even<float><<<blocks, threads>>>((float *)out->data, M);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F16) {
    if (M % 2 == 1) {
      triang_kernel_odd_fp16<<<blocks, threads>>>((uint16_t *)out->data, M);
    } else {
      triang_kernel_even_fp16<<<blocks, threads>>>((uint16_t *)out->data, M);
    }
  } else if (out->dtype == INSIGHT_DTYPE_BF16) {
    if (M % 2 == 1) {
      triang_kernel_odd_bf16<<<blocks, threads>>>((uint16_t *)out->data, M);
    } else {
      triang_kernel_even_bf16<<<blocks, threads>>>((uint16_t *)out->data, M);
    }
  } else {
    gpu_set_last_error("triang: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(triang, INSIGHT_DTYPE_F64, triang_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(triang, INSIGHT_DTYPE_F32, triang_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(triang, INSIGHT_DTYPE_F16, triang_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(triang, INSIGHT_DTYPE_BF16, triang_kernel_gpu);
