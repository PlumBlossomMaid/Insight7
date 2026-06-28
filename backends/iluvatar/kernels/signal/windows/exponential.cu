// backends/cuda/kernels/signal/windows/exponential.cu
// CUDA kernel for exponential window generation.
// w[n] = exp(-|n - center| / tau)
// inputs[0]: tau   (1-element F64 array on device)
// inputs[1]: center (1-element F64 array on device)
// outputs[0]: window output
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void exponential_kernel(T *out, int64_t M, T inv_tau, T center) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = exp(-fabs(T(i) - center) * inv_tau);
}

__global__ void exponential_kernel_f16(uint16_t *out, int64_t M, float inv_tau,
                                       float center) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float result = expf(-fabsf((float)i - center) * inv_tau);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void exponential_kernel_bf16(uint16_t *out, int64_t M, float inv_tau,
                                        float center) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float result = expf(-fabsf((float)i - center) * inv_tau);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_exponential_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *tau_arr = (InsightArray *)inputs[0];
  InsightArray *center_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!tau_arr || !center_arr || !out) {
    gpu_set_last_error("exponential: null array pointer");
    return C_FAILED;
  }

  // Read tau and center from device memory
  double tau_host, center_host;
  cudaMemcpy(&tau_host, tau_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&center_host, center_arr->data, sizeof(double),
             cudaMemcpyDeviceToHost);

  if (tau_host <= 0.0) {
    gpu_set_last_error("exponential: tau must be positive");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  double inv_tau = 1.0 / tau_host;

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    exponential_kernel<double>
        <<<blocks, threads>>>((double *)out->data, M, inv_tau, center_host);
    break;
  case INSIGHT_DTYPE_F32:
    exponential_kernel<float><<<blocks, threads>>>(
        (float *)out->data, M, (float)inv_tau, (float)center_host);
    break;
  case INSIGHT_DTYPE_F16:
    exponential_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)out->data, M, (float)inv_tau, (float)center_host);
    break;
  case INSIGHT_DTYPE_BF16:
    exponential_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)out->data, M, (float)inv_tau, (float)center_host);
    break;
  default:
    gpu_set_last_error(
        "exponential: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_exponential, INSIGHT_DTYPE_F64,
                         signal_exponential_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_exponential, INSIGHT_DTYPE_F32,
                         signal_exponential_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_exponential, INSIGHT_DTYPE_F16,
                         signal_exponential_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_exponential, INSIGHT_DTYPE_BF16,
                         signal_exponential_kernel_gpu);
