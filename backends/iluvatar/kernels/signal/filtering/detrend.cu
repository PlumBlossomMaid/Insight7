// backends/cuda/kernels/signal/filtering/detrend.cu
// CUDA kernel for linear detrending.
// Fits y = a + b*x via least squares, subtracts trend from data.
// inputs[0]: data array (F64 or F32)
// outputs[0]: detrended array
//
// Note: The slope/intercept computation uses a two-pass approach:
// first compute sums on host (fast for single array), then subtract
// trend on GPU. For very large arrays, a reduction kernel would be
// more efficient, but the sums are O(n) with no parallelism issues.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
__global__ void signal_detrend_kernel(T *out, const T *x, int64_t n, T a, T b) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = x[i] - (a + b * (T)i);
}

__global__ void signal_detrend_kernel_f16(uint16_t *out, const uint16_t *in,
                                          int64_t n, float a, float b) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __half2float(*(const __half *)&in[i]);
  float result = val - (a + b * (float)i);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_detrend_kernel_bf16(uint16_t *out, const uint16_t *in,
                                           int64_t n, float a, float b) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&in[i]);
  float result = val - (a + b * (float)i);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_detrend_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr) {
    gpu_set_last_error("signal_detrend: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !y_arr->data) {
    gpu_set_last_error("signal_detrend: null data pointer");
    return C_FAILED;
  }

  int64_t n = x_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  // Copy data to host for sum computation
  double sum_x = (double)n * (double)(n - 1) * 0.5;
  double sum_x2 = (double)(n - 1) * (double)n * (double)(2 * n - 1) / 6.0;
  double nd = (double)n;
  double denom = nd * sum_x2 - sum_x * sum_x;

  double sum_y = 0.0, sum_xy = 0.0;

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    // Copy to host for sum computation
    std::vector<double> host_x(n);
    cudaMemcpy(host_x.data(), x_arr->data, n * sizeof(double),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) {
      sum_y += host_x[i];
      sum_xy += (double)i * host_x[i];
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    signal_detrend_kernel<double><<<blocks, threads>>>(
        (double *)y_arr->data, (const double *)x_arr->data, n, a, b);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    std::vector<float> host_x(n);
    cudaMemcpy(host_x.data(), x_arr->data, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) {
      sum_y += (double)host_x[i];
      sum_xy += (double)i * (double)host_x[i];
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    signal_detrend_kernel<float>
        <<<blocks, threads>>>((float *)y_arr->data, (const float *)x_arr->data,
                              n, (float)a, (float)b);
    break;
  }
  case INSIGHT_DTYPE_F16: {
    std::vector<uint16_t> host_x(n);
    cudaMemcpy(host_x.data(), x_arr->data, n * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) {
      float val = __half2float(*(const __half *)&host_x[i]);
      sum_y += (double)val;
      sum_xy += (double)i * (double)val;
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    signal_detrend_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)y_arr->data, (const uint16_t *)x_arr->data, n, (float)a,
        (float)b);
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    std::vector<uint16_t> host_x(n);
    cudaMemcpy(host_x.data(), x_arr->data, n * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) {
      float val = __bfloat162float(*(const __nv_bfloat16 *)&host_x[i]);
      sum_y += (double)val;
      sum_xy += (double)i * (double)val;
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    signal_detrend_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)y_arr->data, (const uint16_t *)x_arr->data, n, (float)a,
        (float)b);
    break;
  }
  default:
    gpu_set_last_error(
        "signal_detrend: unsupported dtype, need F32, F64, F16, or BF16");
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

REGISTER_ILUVATAR_KERNEL(signal_detrend, INSIGHT_DTYPE_F64,
                         signal_detrend_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_detrend, INSIGHT_DTYPE_F32,
                         signal_detrend_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_detrend, INSIGHT_DTYPE_F16,
                         signal_detrend_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_detrend, INSIGHT_DTYPE_BF16,
                         signal_detrend_kernel_gpu);
