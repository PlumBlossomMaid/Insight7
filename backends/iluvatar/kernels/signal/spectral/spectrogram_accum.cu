// NOTE: These kernels (spectrogram_accum, spectrogram_power) are currently
// not called from the frontend. The frontend uses composite ops for these
// operations. These kernels are preserved for future backend dispatch
// optimization.
//
// backends/cuda/kernels/signal/spectral/spectrogram_accum.cu
// CUDA kernel for extracting real/imag parts from complex STFT output
// and computing power spectrum |Xf|^2.
// Each thread handles one element.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

// Extract real part: out[i] = real(in[i])
__global__ void spectrogram_accum_c64(const cuDoubleComplex *in, double *out,
                                      int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;
  out[i] = cuCreal(in[i]);
}

__global__ void spectrogram_accum_c32(const cuFloatComplex *in, float *out,
                                      int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;
  out[i] = cuCrealf(in[i]);
}

// Power spectrum: out[i] = |in[i]|^2 = real^2 + imag^2
__global__ void spectrogram_power_c64(const cuDoubleComplex *in, double *out,
                                      int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;
  double r = cuCreal(in[i]);
  double im = cuCimag(in[i]);
  out[i] = r * r + im * im;
}

__global__ void spectrogram_power_c32(const cuFloatComplex *in, float *out,
                                      int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;
  float r = cuCrealf(in[i]);
  float im = cuCimagf(in[i]);
  out[i] = r * r + im * im;
}

extern "C" {

C_Status spectrogram_accum_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *in = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!in || !out) {
    gpu_set_last_error("spectrogram_accum: null array pointer");
    return C_FAILED;
  }

  int64_t numel = in->numel;
  int threads = 256;
  int blocks = (int)((numel + threads - 1) / threads);

  if (in->dtype == INSIGHT_DTYPE_C64) {
    spectrogram_accum_c64<<<blocks, threads>>>(
        (const cuDoubleComplex *)in->data, (double *)out->data, numel);
  } else if (in->dtype == INSIGHT_DTYPE_C32) {
    spectrogram_accum_c32<<<blocks, threads>>>((const cuFloatComplex *)in->data,
                                               (float *)out->data, numel);
  } else {
    gpu_set_last_error("spectrogram_accum: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

C_Status spectrogram_power_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *in = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!in || !out) {
    gpu_set_last_error("spectrogram_power: null array pointer");
    return C_FAILED;
  }

  int64_t numel = in->numel;
  int threads = 256;
  int blocks = (int)((numel + threads - 1) / threads);

  if (in->dtype == INSIGHT_DTYPE_C64) {
    spectrogram_power_c64<<<blocks, threads>>>(
        (const cuDoubleComplex *)in->data, (double *)out->data, numel);
  } else if (in->dtype == INSIGHT_DTYPE_C32) {
    spectrogram_power_c32<<<blocks, threads>>>((const cuFloatComplex *)in->data,
                                               (float *)out->data, numel);
  } else {
    gpu_set_last_error("spectrogram_power: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(spectrogram_accum, INSIGHT_DTYPE_C64,
                         spectrogram_accum_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(spectrogram_accum, INSIGHT_DTYPE_C32,
                         spectrogram_accum_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(spectrogram_power, INSIGHT_DTYPE_C64,
                         spectrogram_power_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(spectrogram_power, INSIGHT_DTYPE_C32,
                         spectrogram_power_kernel_gpu);
