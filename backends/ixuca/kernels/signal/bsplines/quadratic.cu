// backends/cuda/kernels/signal/bsplines/quadratic.cu
// CUDA kernel for quadratic B-spline basis function
// |x| < 0.5:     3/4 - x^2
// 0.5 <= |x| < 1.5: (3-2*|x|)^2 / 8
// |x| >= 1.5:    0
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void quadratic_kernel_f64(double *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double ax = fabs(data[i]);
  if (ax < 0.5) {
    data[i] = 0.75 - ax * ax;
  } else if (ax < 1.5) {
    double t = 3.0 - 2.0 * ax;
    data[i] = (t * t) / 8.0;
  } else {
    data[i] = 0.0;
  }
}

__global__ void quadratic_kernel_f32(float *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float ax = fabsf(data[i]);
  if (ax < 0.5f) {
    data[i] = 0.75f - ax * ax;
  } else if (ax < 1.5f) {
    float t = 3.0f - 2.0f * ax;
    data[i] = (t * t) / 8.0f;
  } else {
    data[i] = 0.0f;
  }
}

__global__ void quadratic_kernel_f16(uint16_t *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = __half2float(*(const __half *)&data[i]);
  float ax = fabsf(val);
  float result;
  if (ax < 0.5f) {
    result = 0.75f - ax * ax;
  } else if (ax < 1.5f) {
    float t = 3.0f - 2.0f * ax;
    result = (t * t) / 8.0f;
  } else {
    result = 0.0f;
  }
  __half res = __float2half(result);
  data[i] = *(uint16_t *)&res;
}

__global__ void quadratic_kernel_bf16(uint16_t *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&data[i]);
  float ax = fabsf(val);
  float result;
  if (ax < 0.5f) {
    result = 0.75f - ax * ax;
  } else if (ax < 1.5f) {
    float t = 3.0f - 2.0f * ax;
    result = (t * t) / 8.0f;
  } else {
    result = 0.0f;
  }
  __nv_bfloat16 res = __float2bfloat16(result);
  data[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status quadratic_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("quadratic: null array pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    quadratic_kernel_f64<<<blocks, threads>>>((double *)out->data, M);
    break;
  case INSIGHT_DTYPE_F32:
    quadratic_kernel_f32<<<blocks, threads>>>((float *)out->data, M);
    break;
  case INSIGHT_DTYPE_F16:
    quadratic_kernel_f16<<<blocks, threads>>>((uint16_t *)out->data, M);
    break;
  case INSIGHT_DTYPE_BF16:
    quadratic_kernel_bf16<<<blocks, threads>>>((uint16_t *)out->data, M);
    break;
  default:
    gpu_set_last_error(
        "quadratic: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_IXUCA_KERNEL(quadratic, INSIGHT_DTYPE_F64, quadratic_kernel_gpu);
REGISTER_IXUCA_KERNEL(quadratic, INSIGHT_DTYPE_F32, quadratic_kernel_gpu);
REGISTER_IXUCA_KERNEL(quadratic, INSIGHT_DTYPE_F16, quadratic_kernel_gpu);
REGISTER_IXUCA_KERNEL(quadratic, INSIGHT_DTYPE_BF16, quadratic_kernel_gpu);
