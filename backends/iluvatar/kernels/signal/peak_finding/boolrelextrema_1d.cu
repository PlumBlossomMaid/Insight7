// backends/cuda/kernels/signal/peak_finding/boolrelextrema_1d.cu
// CUDA kernel for 1D relative extrema detection
// For each element, checks if it is a local max/min by comparing with
// 'order' neighbors on each side.
// inputs: [0]=data (1D), [1]=order (I32 scalar), [2]=comparator (I32:
// 0=greater, 1=less) outputs: [0]=mask (bool, same shape as data)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void boolrelextrema_1d_f64(const double *data, bool *mask, int64_t n,
                                      int32_t order, int32_t cmp) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  bool is_extremum = true;
  for (int32_t j = 1; j <= order; ++j) {
    if (i - j >= 0) {
      if (cmp == 0 && data[i] < data[i - j]) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && data[i] > data[i - j]) {
        is_extremum = false;
        break;
      }
    }
    if (i + j < n) {
      if (cmp == 0 && data[i] < data[i + j]) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && data[i] > data[i + j]) {
        is_extremum = false;
        break;
      }
    }
  }
  mask[i] = is_extremum;
}

__global__ void boolrelextrema_1d_f32(const float *data, bool *mask, int64_t n,
                                      int32_t order, int32_t cmp) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  bool is_extremum = true;
  for (int32_t j = 1; j <= order; ++j) {
    if (i - j >= 0) {
      if (cmp == 0 && data[i] < data[i - j]) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && data[i] > data[i - j]) {
        is_extremum = false;
        break;
      }
    }
    if (i + j < n) {
      if (cmp == 0 && data[i] < data[i + j]) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && data[i] > data[i + j]) {
        is_extremum = false;
        break;
      }
    }
  }
  mask[i] = is_extremum;
}

__global__ void boolrelextrema_1d_f16(const uint16_t *data, bool *mask,
                                      int64_t n, int32_t order, int32_t cmp) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float val = __half2float(*(const __half *)&data[i]);
  bool is_extremum = true;
  for (int32_t j = 1; j <= order; ++j) {
    if (i - j >= 0) {
      float nb = __half2float(*(const __half *)&data[i - j]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (i + j < n) {
      float nb = __half2float(*(const __half *)&data[i + j]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
  }
  mask[i] = is_extremum;
}

__global__ void boolrelextrema_1d_bf16(const uint16_t *data, bool *mask,
                                       int64_t n, int32_t order, int32_t cmp) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float val = __bfloat162float(*(const __nv_bfloat16 *)&data[i]);
  bool is_extremum = true;
  for (int32_t j = 1; j <= order; ++j) {
    if (i - j >= 0) {
      float nb = __bfloat162float(*(const __nv_bfloat16 *)&data[i - j]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (i + j < n) {
      float nb = __bfloat162float(*(const __nv_bfloat16 *)&data[i + j]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
  }
  mask[i] = is_extremum;
}

extern "C" {

C_Status boolrelextrema_1d_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *data = (InsightArray *)inputs[0];
  int32_t order = *(int32_t *)((InsightArray *)inputs[1])->data;
  int32_t cmp = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!data || !out) {
    gpu_set_last_error("boolrelextrema_1d: null array pointer");
    return C_FAILED;
  }

  int64_t n = data->numel;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (data->dtype) {
  case INSIGHT_DTYPE_F64:
    boolrelextrema_1d_f64<<<blocks, threads>>>(
        (const double *)data->data, (bool *)out->data, n, order, cmp);
    break;
  case INSIGHT_DTYPE_F32:
    boolrelextrema_1d_f32<<<blocks, threads>>>(
        (const float *)data->data, (bool *)out->data, n, order, cmp);
    break;
  case INSIGHT_DTYPE_F16:
    boolrelextrema_1d_f16<<<blocks, threads>>>(
        (const uint16_t *)data->data, (bool *)out->data, n, order, cmp);
    break;
  case INSIGHT_DTYPE_BF16:
    boolrelextrema_1d_bf16<<<blocks, threads>>>(
        (const uint16_t *)data->data, (bool *)out->data, n, order, cmp);
    break;
  default:
    gpu_set_last_error(
        "boolrelextrema_1d: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(boolrelextrema_1d, INSIGHT_DTYPE_F32,
                    boolrelextrema_1d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(boolrelextrema_1d, INSIGHT_DTYPE_F64,
                    boolrelextrema_1d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(boolrelextrema_1d, INSIGHT_DTYPE_F16,
                    boolrelextrema_1d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(boolrelextrema_1d, INSIGHT_DTYPE_BF16,
                    boolrelextrema_1d_kernel_gpu);
