// backends/cuda/kernels/signal/convolution/convolve2d.cu
// CUDA kernel for 2D direct convolution
// y[i,j] = sum_{k,l} x[i+k-offset_r, j+l-offset_c] * h[k,l]
// mode: 0=valid, 1=same, 2=full
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void convolve2d_f64(const double *x, const double *h, double *y,
                               int64_t rows, int64_t cols, int64_t hrows,
                               int64_t hcols, int64_t out_rows,
                               int64_t out_cols, int64_t offset_r,
                               int64_t offset_c) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = out_rows * out_cols;
  if (idx >= total)
    return;
  int64_t i = idx / out_cols;
  int64_t j = idx % out_cols;

  double sum = 0.0;
  for (int64_t k = 0; k < hrows; ++k) {
    int64_t ri = i + k - offset_r;
    if (ri < 0 || ri >= rows)
      continue;
    for (int64_t l = 0; l < hcols; ++l) {
      int64_t ci = j + l - offset_c;
      if (ci >= 0 && ci < cols) {
        sum += x[ri * cols + ci] * h[k * hcols + l];
      }
    }
  }
  y[idx] = sum;
}

__global__ void convolve2d_f32(const float *x, const float *h, float *y,
                               int64_t rows, int64_t cols, int64_t hrows,
                               int64_t hcols, int64_t out_rows,
                               int64_t out_cols, int64_t offset_r,
                               int64_t offset_c) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = out_rows * out_cols;
  if (idx >= total)
    return;
  int64_t i = idx / out_cols;
  int64_t j = idx % out_cols;

  float sum = 0.0f;
  for (int64_t k = 0; k < hrows; ++k) {
    int64_t ri = i + k - offset_r;
    if (ri < 0 || ri >= rows)
      continue;
    for (int64_t l = 0; l < hcols; ++l) {
      int64_t ci = j + l - offset_c;
      if (ci >= 0 && ci < cols) {
        sum += x[ri * cols + ci] * h[k * hcols + l];
      }
    }
  }
  y[idx] = sum;
}

__global__ void convolve2d_f16(const uint16_t *x, const uint16_t *h,
                               uint16_t *y, int64_t rows, int64_t cols,
                               int64_t hrows, int64_t hcols, int64_t out_rows,
                               int64_t out_cols, int64_t offset_r,
                               int64_t offset_c) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = out_rows * out_cols;
  if (idx >= total)
    return;
  int64_t i = idx / out_cols;
  int64_t j = idx % out_cols;

  float sum = 0.0f;
  for (int64_t k = 0; k < hrows; ++k) {
    int64_t ri = i + k - offset_r;
    if (ri < 0 || ri >= rows)
      continue;
    for (int64_t l = 0; l < hcols; ++l) {
      int64_t ci = j + l - offset_c;
      if (ci >= 0 && ci < cols) {
        float xv = __half2float(*(const __half *)&x[ri * cols + ci]);
        float hv = __half2float(*(const __half *)&h[k * hcols + l]);
        sum += xv * hv;
      }
    }
  }
  __half res = __float2half(sum);
  y[idx] = *(uint16_t *)&res;
}

__global__ void convolve2d_bf16(const uint16_t *x, const uint16_t *h,
                                uint16_t *y, int64_t rows, int64_t cols,
                                int64_t hrows, int64_t hcols, int64_t out_rows,
                                int64_t out_cols, int64_t offset_r,
                                int64_t offset_c) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = out_rows * out_cols;
  if (idx >= total)
    return;
  int64_t i = idx / out_cols;
  int64_t j = idx % out_cols;

  float sum = 0.0f;
  for (int64_t k = 0; k < hrows; ++k) {
    int64_t ri = i + k - offset_r;
    if (ri < 0 || ri >= rows)
      continue;
    for (int64_t l = 0; l < hcols; ++l) {
      int64_t ci = j + l - offset_c;
      if (ci >= 0 && ci < cols) {
        float xv = __bfloat162float(*(const __nv_bfloat16 *)&x[ri * cols + ci]);
        float hv = __bfloat162float(*(const __nv_bfloat16 *)&h[k * hcols + l]);
        sum += xv * hv;
      }
    }
  }
  __nv_bfloat16 res = __float2bfloat16(sum);
  y[idx] = *(uint16_t *)&res;
}

extern "C" {

C_Status convolve2d_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *h = (InsightArray *)inputs[1];
  int32_t mode = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !h || !out) {
    gpu_set_last_error("convolve2d: null array pointer");
    return C_FAILED;
  }

  int64_t rows = x->dims[0];
  int64_t cols = x->dims[1];
  int64_t hrows = h->dims[0];
  int64_t hcols = h->dims[1];

  int64_t out_rows, out_cols, offset_r, offset_c;

  if (mode == 0) { // valid
    out_rows = rows - hrows + 1;
    out_cols = cols - hcols + 1;
    offset_r = 0;
    offset_c = 0;
  } else if (mode == 1) { // same
    out_rows = rows;
    out_cols = cols;
    offset_r = hrows / 2;
    offset_c = hcols / 2;
  } else { // full
    out_rows = rows + hrows - 1;
    out_cols = cols + hcols - 1;
    offset_r = 0;
    offset_c = 0;
  }

  int64_t total = out_rows * out_cols;
  int threads = 256;
  int blocks = (int)((total + threads - 1) / threads);

  switch (x->dtype) {
  case INSIGHT_DTYPE_F64:
    convolve2d_f64<<<blocks, threads>>>(
        (const double *)x->data, (const double *)h->data, (double *)out->data,
        rows, cols, hrows, hcols, out_rows, out_cols, offset_r, offset_c);
    break;
  case INSIGHT_DTYPE_F32:
    convolve2d_f32<<<blocks, threads>>>(
        (const float *)x->data, (const float *)h->data, (float *)out->data,
        rows, cols, hrows, hcols, out_rows, out_cols, offset_r, offset_c);
    break;
  case INSIGHT_DTYPE_F16:
    convolve2d_f16<<<blocks, threads>>>(
        (const uint16_t *)x->data, (const uint16_t *)h->data,
        (uint16_t *)out->data, rows, cols, hrows, hcols, out_rows, out_cols,
        offset_r, offset_c);
    break;
  case INSIGHT_DTYPE_BF16:
    convolve2d_bf16<<<blocks, threads>>>(
        (const uint16_t *)x->data, (const uint16_t *)h->data,
        (uint16_t *)out->data, rows, cols, hrows, hcols, out_rows, out_cols,
        offset_r, offset_c);
    break;
  default:
    gpu_set_last_error(
        "convolve2d: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_IXUCA_KERNEL(convolve2d, INSIGHT_DTYPE_F32, convolve2d_kernel_gpu);
REGISTER_IXUCA_KERNEL(convolve2d, INSIGHT_DTYPE_F64, convolve2d_kernel_gpu);
REGISTER_IXUCA_KERNEL(convolve2d, INSIGHT_DTYPE_F16, convolve2d_kernel_gpu);
REGISTER_IXUCA_KERNEL(convolve2d, INSIGHT_DTYPE_BF16, convolve2d_kernel_gpu);
