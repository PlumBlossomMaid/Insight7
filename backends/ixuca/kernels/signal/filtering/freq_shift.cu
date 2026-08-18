// backends/cuda/kernels/signal/filtering/freq_shift.cu
// CUDA kernel for frequency shift.
// y[n] = x[n] * exp(j * 2 * pi * freq * n / fs)
// inputs[0]: data array (F64, F32, C64, or C32)
// inputs[1]: freq (F64, 1-element)
// inputs[2]: fs (F64, 1-element)
// outputs[0]: shifted array
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void signal_freq_shift_kernel_f64(double *out, const double *x,
                                             int64_t n, double phase_inc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double phase = phase_inc * (double)i;
  out[i] = x[i] * cos(phase);
}

__global__ void signal_freq_shift_kernel_f32(float *out, const float *x,
                                             int64_t n, float phase_inc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float phase = phase_inc * (float)i;
  out[i] = x[i] * cosf(phase);
}

__global__ void signal_freq_shift_kernel_c64(cuDoubleComplex *out,
                                             const cuDoubleComplex *in,
                                             int64_t n, double phase_inc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double phase = phase_inc * (double)i;
  double c = cos(phase);
  double s = sin(phase);
  double re = cuCreal(in[i]);
  double im = cuCimag(in[i]);
  out[i] = make_cuDoubleComplex(re * c - im * s, re * s + im * c);
}

__global__ void signal_freq_shift_kernel_c32(cuFloatComplex *out,
                                             const cuFloatComplex *in,
                                             int64_t n, float phase_inc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float phase = phase_inc * (float)i;
  float c = cosf(phase);
  float s = sinf(phase);
  float re = cuCrealf(in[i]);
  float im = cuCimagf(in[i]);
  out[i] = make_cuFloatComplex(re * c - im * s, re * s + im * c);
}

__global__ void signal_freq_shift_kernel_f16(uint16_t *out, const uint16_t *in,
                                             int64_t n, float phase_inc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __half2float(*(const __half *)&in[i]);
  float phase = phase_inc * (float)i;
  float result = val * cosf(phase);
  __half res = __float2half(result);
  out[i] = *(uint16_t *)&res;
}

__global__ void signal_freq_shift_kernel_bf16(uint16_t *out, const uint16_t *in,
                                              int64_t n, float phase_inc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float val = __bfloat162float(*(const __nv_bfloat16 *)&in[i]);
  float phase = phase_inc * (float)i;
  float result = val * cosf(phase);
  __nv_bfloat16 res = __float2bfloat16(result);
  out[i] = *(uint16_t *)&res;
}

extern "C" {

C_Status signal_freq_shift_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *freq_arr = (InsightArray *)inputs[1];
  InsightArray *fs_arr = (InsightArray *)inputs[2];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !freq_arr || !fs_arr || !y_arr) {
    gpu_set_last_error("signal_freq_shift: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !freq_arr->data || !fs_arr->data || !y_arr->data) {
    gpu_set_last_error("signal_freq_shift: null data pointer");
    return C_FAILED;
  }

  double freq, fs;
  cudaMemcpy(&freq, freq_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&fs, fs_arr->data, sizeof(double), cudaMemcpyDeviceToHost);

  if (fs == 0.0) {
    gpu_set_last_error("signal_freq_shift: fs must be non-zero");
    return C_FAILED;
  }

  int64_t n = x_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  double phase_inc = 2.0 * M_PI * freq / fs;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_freq_shift_kernel_f64<<<blocks, threads>>>(
        (double *)y_arr->data, (const double *)x_arr->data, n, phase_inc);
    break;
  case INSIGHT_DTYPE_F32:
    signal_freq_shift_kernel_f32<<<blocks, threads>>>(
        (float *)y_arr->data, (const float *)x_arr->data, n, (float)phase_inc);
    break;
  case INSIGHT_DTYPE_C64:
    signal_freq_shift_kernel_c64<<<blocks, threads>>>(
        (cuDoubleComplex *)y_arr->data, (const cuDoubleComplex *)x_arr->data, n,
        phase_inc);
    break;
  case INSIGHT_DTYPE_C32:
    signal_freq_shift_kernel_c32<<<blocks, threads>>>(
        (cuFloatComplex *)y_arr->data, (const cuFloatComplex *)x_arr->data, n,
        (float)phase_inc);
    break;
  case INSIGHT_DTYPE_F16:
    signal_freq_shift_kernel_f16<<<blocks, threads>>>(
        (uint16_t *)y_arr->data, (const uint16_t *)x_arr->data, n,
        (float)phase_inc);
    break;
  case INSIGHT_DTYPE_BF16:
    signal_freq_shift_kernel_bf16<<<blocks, threads>>>(
        (uint16_t *)y_arr->data, (const uint16_t *)x_arr->data, n,
        (float)phase_inc);
    break;
  default:
    gpu_set_last_error("signal_freq_shift: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(signal_freq_shift, INSIGHT_DTYPE_F64,
                         signal_freq_shift_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_freq_shift, INSIGHT_DTYPE_F32,
                         signal_freq_shift_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_freq_shift, INSIGHT_DTYPE_C64,
                         signal_freq_shift_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_freq_shift, INSIGHT_DTYPE_C32,
                         signal_freq_shift_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_freq_shift, INSIGHT_DTYPE_F16,
                         signal_freq_shift_kernel_gpu);
REGISTER_IXUCA_KERNEL(signal_freq_shift, INSIGHT_DTYPE_BF16,
                         signal_freq_shift_kernel_gpu);
