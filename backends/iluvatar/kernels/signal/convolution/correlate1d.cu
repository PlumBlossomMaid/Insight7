// backends/cuda/kernels/signal/convolution/correlate1d.cu
// CUDA kernel for 1D direct correlation
// y[n] = sum_{k=0}^{nh-1} x[n+k-offset] * h[nh-1-k]
// Same as convolution but with kernel h reversed.
// mode: 0=valid, 1=same, 2=full
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void correlate1d_f64(const double *x, const double *h, double *y,
                                int64_t nx, int64_t nh, int64_t out_len,
                                int64_t offset) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len)
    return;
  double sum = 0.0;
  for (int64_t j = 0; j < nh; ++j) {
    int64_t idx = i + j - offset;
    if (idx >= 0 && idx < nx) {
      sum += x[idx] * h[nh - 1 - j];
    }
  }
  y[i] = sum;
}

__global__ void correlate1d_f32(const float *x, const float *h, float *y,
                                int64_t nx, int64_t nh, int64_t out_len,
                                int64_t offset) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len)
    return;
  float sum = 0.0f;
  for (int64_t j = 0; j < nh; ++j) {
    int64_t idx = i + j - offset;
    if (idx >= 0 && idx < nx) {
      sum += x[idx] * h[nh - 1 - j];
    }
  }
  y[i] = sum;
}

__global__ void correlate1d_f16(const uint16_t *x, const uint16_t *h,
                                uint16_t *y, int64_t nx, int64_t nh,
                                int64_t out_len, int64_t offset) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len)
    return;
  float sum = 0.0f;
  for (int64_t j = 0; j < nh; ++j) {
    int64_t idx = i + j - offset;
    if (idx >= 0 && idx < nx) {
      float xv = __half2float(*(const __half *)&x[idx]);
      float hv = __half2float(*(const __half *)&h[nh - 1 - j]);
      sum += xv * hv;
    }
  }
  __half res = __float2half(sum);
  y[i] = *(uint16_t *)&res;
}

__global__ void correlate1d_bf16(const uint16_t *x, const uint16_t *h,
                                 uint16_t *y, int64_t nx, int64_t nh,
                                 int64_t out_len, int64_t offset) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len)
    return;
  float sum = 0.0f;
  for (int64_t j = 0; j < nh; ++j) {
    int64_t idx = i + j - offset;
    if (idx >= 0 && idx < nx) {
      float xv = __bfloat162float(*(const __nv_bfloat16 *)&x[idx]);
      float hv = __bfloat162float(*(const __nv_bfloat16 *)&h[nh - 1 - j]);
      sum += xv * hv;
    }
  }
  __nv_bfloat16 res = __float2bfloat16(sum);
  y[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status correlate1d_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *h = (InsightArray *)inputs[1];
  int32_t mode = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !h || !out) {
    gpu_set_last_error("correlate1d: null array pointer");
    return C_FAILED;
  }

  int64_t nx = x->numel;
  int64_t nh = h->numel;
  int64_t out_len, offset;

  if (mode == 0) { // valid
    out_len = nx - nh + 1;
    offset = 0;
  } else if (mode == 1) { // same
    out_len = nx;
    offset = nh / 2;
  } else { // full
    out_len = nx + nh - 1;
    offset = 0;
  }

  int threads = 256;
  int blocks = (int)((out_len + threads - 1) / threads);

  switch (x->dtype) {
  case INSIGHT_DTYPE_F64:
    correlate1d_f64<<<blocks, threads>>>(
        (const double *)x->data, (const double *)h->data, (double *)out->data,
        nx, nh, out_len, offset);
    break;
  case INSIGHT_DTYPE_F32:
    correlate1d_f32<<<blocks, threads>>>(
        (const float *)x->data, (const float *)h->data, (float *)out->data, nx,
        nh, out_len, offset);
    break;
  case INSIGHT_DTYPE_F16:
    correlate1d_f16<<<blocks, threads>>>(
        (const uint16_t *)x->data, (const uint16_t *)h->data,
        (uint16_t *)out->data, nx, nh, out_len, offset);
    break;
  case INSIGHT_DTYPE_BF16:
    correlate1d_bf16<<<blocks, threads>>>(
        (const uint16_t *)x->data, (const uint16_t *)h->data,
        (uint16_t *)out->data, nx, nh, out_len, offset);
    break;
  default:
    gpu_set_last_error(
        "correlate1d: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(correlate1d, INSIGHT_DTYPE_F32,
                         correlate1d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(correlate1d, INSIGHT_DTYPE_F64,
                         correlate1d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(correlate1d, INSIGHT_DTYPE_F16,
                         correlate1d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(correlate1d, INSIGHT_DTYPE_BF16,
                         correlate1d_kernel_gpu);
