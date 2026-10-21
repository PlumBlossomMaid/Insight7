// backends/cuda/kernels/signal/windows/cosine.cu
// CUDA kernel for cosine window generation.
// w[n] = sin(pi*(n+0.5)/M)
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T> __global__ void cosine_kernel(T *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = sin(T(M_PI) * (T(i) + T(0.5)) / T(M));
}

__global__ void cosine_kernel_fp16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = sinf((float)M_PI * ((float)i + 0.5f) / (float)M);
  out[i] = __float2half(val);
}

__global__ void cosine_kernel_bf16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = sinf((float)M_PI * ((float)i + 0.5f) / (float)M);
  out[i] = __float2bfloat16(val);
}

extern "C" {

C_Status cosine_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("cosine: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    cosine_kernel<double><<<blocks, threads>>>((double *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    cosine_kernel<float><<<blocks, threads>>>((float *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_F16) {
    cosine_kernel_fp16<<<blocks, threads>>>((uint16_t *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_BF16) {
    cosine_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M);
  } else {
    gpu_set_last_error("cosine: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(cosine, INSIGHT_DTYPE_F64, cosine_kernel_gpu);
REGISTER_IXUCA_KERNEL(cosine, INSIGHT_DTYPE_F32, cosine_kernel_gpu);
REGISTER_IXUCA_KERNEL(cosine, INSIGHT_DTYPE_F16, cosine_kernel_gpu);
REGISTER_IXUCA_KERNEL(cosine, INSIGHT_DTYPE_BF16, cosine_kernel_gpu);
