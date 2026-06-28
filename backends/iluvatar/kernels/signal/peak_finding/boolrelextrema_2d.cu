// backends/cuda/kernels/signal/peak_finding/boolrelextrema_2d.cu
// CUDA kernel for 2D relative extrema detection
// Operates along the specified axis for a 2D array.
// Each thread handles one element.
// inputs: [0]=data (2D, rows x cols), [1]=order (I32 scalar),
//         [2]=comparator (I32: 0=greater, 1=less),
//         [3]=axis (I32: 0=along rows, 1=along cols)
// outputs: [0]=mask (bool, same shape as data)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void boolrelextrema_2d_f64_cols(const double *data, bool *mask,
                                           int64_t rows, int64_t cols,
                                           int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  double val = data[idx];
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (c - j >= 0) {
      double nb = data[r * cols + c - j];
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (c + j < cols) {
      double nb = data[r * cols + c + j];
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_f64_rows(const double *data, bool *mask,
                                           int64_t rows, int64_t cols,
                                           int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  double val = data[idx];
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (r - j >= 0) {
      double nb = data[(r - j) * cols + c];
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (r + j < rows) {
      double nb = data[(r + j) * cols + c];
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_f32_cols(const float *data, bool *mask,
                                           int64_t rows, int64_t cols,
                                           int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  float val = data[idx];
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (c - j >= 0) {
      float nb = data[r * cols + c - j];
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (c + j < cols) {
      float nb = data[r * cols + c + j];
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_f32_rows(const float *data, bool *mask,
                                           int64_t rows, int64_t cols,
                                           int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  float val = data[idx];
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (r - j >= 0) {
      float nb = data[(r - j) * cols + c];
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (r + j < rows) {
      float nb = data[(r + j) * cols + c];
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_f16_cols(const uint16_t *data, bool *mask,
                                           int64_t rows, int64_t cols,
                                           int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  float val = __half2float(*(const __half *)&data[idx]);
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (c - j >= 0) {
      float nb = __half2float(*(const __half *)&data[r * cols + c - j]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (c + j < cols) {
      float nb = __half2float(*(const __half *)&data[r * cols + c + j]);
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_f16_rows(const uint16_t *data, bool *mask,
                                           int64_t rows, int64_t cols,
                                           int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  float val = __half2float(*(const __half *)&data[idx]);
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (r - j >= 0) {
      float nb = __half2float(*(const __half *)&data[(r - j) * cols + c]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (r + j < rows) {
      float nb = __half2float(*(const __half *)&data[(r + j) * cols + c]);
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_bf16_cols(const uint16_t *data, bool *mask,
                                            int64_t rows, int64_t cols,
                                            int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&data[idx]);
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (c - j >= 0) {
      float nb =
          __bfloat162float(*(const __nv_bfloat16 *)&data[r * cols + c - j]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (c + j < cols) {
      float nb =
          __bfloat162float(*(const __nv_bfloat16 *)&data[r * cols + c + j]);
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
  mask[idx] = is_extremum;
}

__global__ void boolrelextrema_2d_bf16_rows(const uint16_t *data, bool *mask,
                                            int64_t rows, int64_t cols,
                                            int32_t order, int32_t cmp) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total)
    return;

  int64_t r = idx / cols;
  int64_t c = idx % cols;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&data[idx]);
  bool is_extremum = true;

  for (int32_t j = 1; j <= order; ++j) {
    if (r - j >= 0) {
      float nb =
          __bfloat162float(*(const __nv_bfloat16 *)&data[(r - j) * cols + c]);
      if (cmp == 0 && val < nb) {
        is_extremum = false;
        break;
      }
      if (cmp == 1 && val > nb) {
        is_extremum = false;
        break;
      }
    }
    if (r + j < rows) {
      float nb =
          __bfloat162float(*(const __nv_bfloat16 *)&data[(r + j) * cols + c]);
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
  mask[idx] = is_extremum;
}

extern "C" {

C_Status boolrelextrema_2d_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *data = (InsightArray *)inputs[0];
  int32_t order = *(int32_t *)((InsightArray *)inputs[1])->data;
  int32_t cmp = *(int32_t *)((InsightArray *)inputs[2])->data;
  int32_t axis = *(int32_t *)((InsightArray *)inputs[3])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!data || !out) {
    gpu_set_last_error("boolrelextrema_2d: null array pointer");
    return C_FAILED;
  }

  int64_t rows = data->dims[0];
  int64_t cols = data->dims[1];
  int64_t total = rows * cols;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);

  switch (data->dtype) {
  case INSIGHT_DTYPE_F64:
    if (axis == 1) {
      boolrelextrema_2d_f64_cols<<<blocks, threads>>>(
          (const double *)data->data, (bool *)out->data, rows, cols, order,
          cmp);
    } else {
      boolrelextrema_2d_f64_rows<<<blocks, threads>>>(
          (const double *)data->data, (bool *)out->data, rows, cols, order,
          cmp);
    }
    break;
  case INSIGHT_DTYPE_F32:
    if (axis == 1) {
      boolrelextrema_2d_f32_cols<<<blocks, threads>>>(
          (const float *)data->data, (bool *)out->data, rows, cols, order, cmp);
    } else {
      boolrelextrema_2d_f32_rows<<<blocks, threads>>>(
          (const float *)data->data, (bool *)out->data, rows, cols, order, cmp);
    }
    break;
  case INSIGHT_DTYPE_F16:
    if (axis == 1) {
      boolrelextrema_2d_f16_cols<<<blocks, threads>>>(
          (const uint16_t *)data->data, (bool *)out->data, rows, cols, order,
          cmp);
    } else {
      boolrelextrema_2d_f16_rows<<<blocks, threads>>>(
          (const uint16_t *)data->data, (bool *)out->data, rows, cols, order,
          cmp);
    }
    break;
  case INSIGHT_DTYPE_BF16:
    if (axis == 1) {
      boolrelextrema_2d_bf16_cols<<<blocks, threads>>>(
          (const uint16_t *)data->data, (bool *)out->data, rows, cols, order,
          cmp);
    } else {
      boolrelextrema_2d_bf16_rows<<<blocks, threads>>>(
          (const uint16_t *)data->data, (bool *)out->data, rows, cols, order,
          cmp);
    }
    break;
  default:
    gpu_set_last_error(
        "boolrelextrema_2d: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(boolrelextrema_2d, INSIGHT_DTYPE_F32,
                    boolrelextrema_2d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(boolrelextrema_2d, INSIGHT_DTYPE_F64,
                    boolrelextrema_2d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(boolrelextrema_2d, INSIGHT_DTYPE_F16,
                    boolrelextrema_2d_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(boolrelextrema_2d, INSIGHT_DTYPE_BF16,
                    boolrelextrema_2d_kernel_gpu);
