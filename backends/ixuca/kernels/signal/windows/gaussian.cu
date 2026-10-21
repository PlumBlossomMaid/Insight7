// backends/cuda/kernels/signal/windows/gaussian.cu
// CUDA kernel for Gaussian window generation.
// w[n] = exp(-0.5 * ((n - (M-1)/2) / (sigma * (M-1)/2))^2)
// inputs[0]: sigma (1-element F64 array on device)
// outputs[0]: window output
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void gaussian_kernel(T *out, int64_t M, T sigma) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  T half = T(M - 1) / T(2.0);
  T denom = sigma * half;
  if (denom == T(0.0)) {
    out[i] = (i == (int64_t)half) ? T(1.0) : T(0.0);
    return;
  }
  T diff = T(i) - half;
  T scale = T(-0.5) / (denom * denom);
  out[i] = exp(scale * diff * diff);
}

__global__ void gaussian_kernel_fp16(uint16_t *out, int64_t M, float sigma) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float half = (float)(M - 1) / 2.0f;
  float denom_f = sigma * half;
  if (denom_f == 0.0f) {
    out[i] = ((float)i == half) ? __float2half(1.0f) : __float2half(0.0f);
    return;
  }
  float diff = (float)i - half;
  float scale = -0.5f / (denom_f * denom_f);
  float val = expf(scale * diff * diff);
  out[i] = __float2half(val);
}

__global__ void gaussian_kernel_bf16(uint16_t *out, int64_t M, float sigma) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float half = (float)(M - 1) / 2.0f;
  float denom_f = sigma * half;
  if (denom_f == 0.0f) {
    out[i] =
        ((float)i == half) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    return;
  }
  float diff = (float)i - half;
  float scale = -0.5f / (denom_f * denom_f);
  float val = expf(scale * diff * diff);
  out[i] = __float2bfloat16(val);
}

extern "C" {

C_Status gaussian_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *sigma_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!sigma_arr || !out) {
    gpu_set_last_error("gaussian: null array pointer");
    return C_FAILED;
  }

  // Read sigma from device memory
  double sigma_host;
  cudaMemcpy(&sigma_host, sigma_arr->data, sizeof(double),
             cudaMemcpyDeviceToHost);

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    gaussian_kernel<double>
        <<<blocks, threads>>>((double *)out->data, M, sigma_host);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    gaussian_kernel<float>
        <<<blocks, threads>>>((float *)out->data, M, (float)sigma_host);
  } else if (out->dtype == INSIGHT_DTYPE_F16) {
    gaussian_kernel_fp16<<<blocks, threads>>>((uint16_t *)out->data, M,
                                              (float)sigma_host);
  } else if (out->dtype == INSIGHT_DTYPE_BF16) {
    gaussian_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M,
                                              (float)sigma_host);
  } else {
    gpu_set_last_error("gaussian: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(gaussian, INSIGHT_DTYPE_F64, gaussian_kernel_gpu);
REGISTER_IXUCA_KERNEL(gaussian, INSIGHT_DTYPE_F32, gaussian_kernel_gpu);
REGISTER_IXUCA_KERNEL(gaussian, INSIGHT_DTYPE_F16, gaussian_kernel_gpu);
REGISTER_IXUCA_KERNEL(gaussian, INSIGHT_DTYPE_BF16, gaussian_kernel_gpu);
