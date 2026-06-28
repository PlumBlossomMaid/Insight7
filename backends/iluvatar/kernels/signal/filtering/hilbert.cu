// backends/cuda/kernels/signal/filtering/hilbert.cu
// CUDA kernel for Hilbert transform phase shifting.
// For each frequency bin: positive freq -> multiply by j,
// DC and Nyquist unchanged. Constructs analytic signal in frequency domain.
// inputs[0]: complex spectrum array (C64 or C32)
// inputs[1]: n (signal length, I64, 1-element)
// outputs[0]: modified spectrum (same type)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void signal_hilbert_kernel_c64(cuDoubleComplex *out,
                                          const cuDoubleComplex *in,
                                          int64_t numel, int64_t nfft,
                                          int even) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;

  int64_t k = i % nfft;
  if (k == 0) {
    out[i] = in[i]; // DC unchanged
  } else if (even && k == nfft - 1) {
    out[i] = in[i]; // Nyquist unchanged
  } else {
    // Multiply by j: (a + bi) * j = -b + ai
    double re = cuCreal(in[i]);
    double im = cuCimag(in[i]);
    out[i] = make_cuDoubleComplex(-im, re);
  }
}

__global__ void signal_hilbert_kernel_c32(cuFloatComplex *out,
                                          const cuFloatComplex *in,
                                          int64_t numel, int64_t nfft,
                                          int even) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;

  int64_t k = i % nfft;
  if (k == 0) {
    out[i] = in[i];
  } else if (even && k == nfft - 1) {
    out[i] = in[i];
  } else {
    float re = cuCrealf(in[i]);
    float im = cuCimagf(in[i]);
    out[i] = make_cuFloatComplex(-im, re);
  }
}

extern "C" {

C_Status signal_hilbert_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *spec_arr = (InsightArray *)inputs[0];
  InsightArray *n_arr = (InsightArray *)inputs[1];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!spec_arr || !n_arr || !out_arr) {
    gpu_set_last_error("signal_hilbert: null array pointer");
    return C_FAILED;
  }

  if (!spec_arr->data || !n_arr->data || !out_arr->data) {
    gpu_set_last_error("signal_hilbert: null data pointer");
    return C_FAILED;
  }

  int64_t n;
  cudaMemcpy(&n, n_arr->data, sizeof(int64_t), cudaMemcpyDeviceToHost);

  int64_t numel = spec_arr->numel;
  int even = (n % 2 == 0) ? 1 : 0;
  int64_t nfft = even ? (n / 2 + 1) : ((n + 1) / 2);

  int threads = 256;
  int blocks = (int)((numel + threads - 1) / threads);

  switch (spec_arr->dtype) {
  case INSIGHT_DTYPE_C64:
    signal_hilbert_kernel_c64<<<blocks, threads>>>(
        (cuDoubleComplex *)out_arr->data,
        (const cuDoubleComplex *)spec_arr->data, numel, nfft, even);
    break;
  case INSIGHT_DTYPE_C32:
    signal_hilbert_kernel_c32<<<blocks, threads>>>(
        (cuFloatComplex *)out_arr->data, (const cuFloatComplex *)spec_arr->data,
        numel, nfft, even);
    break;
  default:
    gpu_set_last_error("signal_hilbert: unsupported dtype, need C32 or C64");
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

REGISTER_ILUVATAR_KERNEL(signal_hilbert, INSIGHT_DTYPE_C64,
                    signal_hilbert_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(signal_hilbert, INSIGHT_DTYPE_C32,
                    signal_hilbert_kernel_gpu);
