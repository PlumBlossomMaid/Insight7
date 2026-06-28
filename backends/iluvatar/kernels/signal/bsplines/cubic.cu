// backends/cuda/kernels/signal/bsplines/cubic.cu
// CUDA kernel for cubic B-spline basis function
// |x| < 1:       2/3 - |x|^2*(2-|x|)/2
// 1 <= |x| < 2:  (2-|x|)^3 / 6
// |x| >= 2:      0
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cubic_kernel_f64(double *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double ax = fabs(data[i]);
  if (ax < 1.0) {
    double ax2 = ax * ax;
    data[i] = (2.0 / 3.0) - 0.5 * ax2 * (2.0 - ax);
  } else if (ax < 2.0) {
    double t = 2.0 - ax;
    data[i] = (t * t * t) / 6.0;
  } else {
    data[i] = 0.0;
  }
}

__global__ void cubic_kernel_f32(float *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float ax = fabsf(data[i]);
  if (ax < 1.0f) {
    float ax2 = ax * ax;
    data[i] = (2.0f / 3.0f) - 0.5f * ax2 * (2.0f - ax);
  } else if (ax < 2.0f) {
    float t = 2.0f - ax;
    data[i] = (t * t * t) / 6.0f;
  } else {
    data[i] = 0.0f;
  }
}

__global__ void cubic_kernel_f16(uint16_t *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = __half2float(*(const __half *)&data[i]);
  float ax = fabsf(val);
  float result;
  if (ax < 1.0f) {
    float ax2 = ax * ax;
    result = (2.0f / 3.0f) - 0.5f * ax2 * (2.0f - ax);
  } else if (ax < 2.0f) {
    float t = 2.0f - ax;
    result = (t * t * t) / 6.0f;
  } else {
    result = 0.0f;
  }
  __half res = __float2half(result);
  data[i] = *(uint16_t *)&res;
}

__global__ void cubic_kernel_bf16(uint16_t *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&data[i]);
  float ax = fabsf(val);
  float result;
  if (ax < 1.0f) {
    float ax2 = ax * ax;
    result = (2.0f / 3.0f) - 0.5f * ax2 * (2.0f - ax);
  } else if (ax < 2.0f) {
    float t = 2.0f - ax;
    result = (t * t * t) / 6.0f;
  } else {
    result = 0.0f;
  }
  __nv_bfloat16 res = __float2bfloat16(result);
  data[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status cubic_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("cubic: null array pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    cubic_kernel_f64<<<blocks, threads>>>((double *)out->data, M);
    break;
  case INSIGHT_DTYPE_F32:
    cubic_kernel_f32<<<blocks, threads>>>((float *)out->data, M);
    break;
  case INSIGHT_DTYPE_F16:
    cubic_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, M);
    break;
  case INSIGHT_DTYPE_BF16:
    cubic_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M);
    break;
  default:
    gpu_set_last_error("cubic: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(cubic, INSIGHT_DTYPE_F64, cubic_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(cubic, INSIGHT_DTYPE_F32, cubic_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(cubic, INSIGHT_DTYPE_F16, cubic_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(cubic, INSIGHT_DTYPE_BF16, cubic_kernel_gpu);
