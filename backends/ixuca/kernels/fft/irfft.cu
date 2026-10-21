// backends/cuda/kernels/fft/irfft.cu
/**
 * @file irfft.cu
 * @brief CUDA kernel for inverse real FFT (complex-to-real).
 *
 * Performs 1D inverse real FFT on contiguous data with batch processing.
 * Input is complex of length (fft_len/2 + 1), output is real of length fft_len.
 * Uses cuFFT library with thread-local plan caching.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (real)
 *   inputs[1] = InsightArray* input (complex)
 *   inputs[2] = int64_t* fft_len (output real length)
 *   inputs[3] = int64_t* batch_size
 *   inputs[4] = int* inverse (unused, always inverse)
 *   inputs[5] = int* real_input (unused, for compatibility)
 *   inputs[6] = int* norm_code (unused, normalization handled by frontend)
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

extern "C" {

/**
 * @brief GPU entry point for inverse real FFT.
 *
 * @param inputs  See file header for layout
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status irfft_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  int64_t fft_len = *static_cast<int64_t *>(inputs[2]);
  int64_t batch_size = *static_cast<int64_t *>(inputs[3]);

  if (!out || !x) {
    gpu_set_last_error("irfft: null array pointer");
    return C_FAILED;
  }

  // Double precision (C64 -> F64)
  if (x->dtype == INSIGHT_DTYPE_C64) {
    cufftHandle plan = cufft_ensure_plan(fft_len, batch_size, CUFFT_INVERSE,
                                         CuFFTKind::C2R, false);
    if (plan == 0) {
      gpu_set_last_error("irfft: failed to create cuFFT plan (C64)");
      return C_FAILED;
    }

    cufftResult result =
        cufftExecZ2D(plan, static_cast<cufftDoubleComplex *>(x->data),
                     static_cast<cufftDoubleReal *>(out->data));

    if (result != CUFFT_SUCCESS) {
      cufft_set_error("irfft cufftExecZ2D", result);
      return C_FAILED;
    }
  }
  // Single precision (C32 -> F32)
  else if (x->dtype == INSIGHT_DTYPE_C32) {
    cufftHandle plan = cufft_ensure_plan(fft_len, batch_size, CUFFT_INVERSE,
                                         CuFFTKind::C2R, true);
    if (plan == 0) {
      gpu_set_last_error("irfft: failed to create cuFFT plan (C32)");
      return C_FAILED;
    }

    cufftResult result =
        cufftExecC2R(plan, static_cast<cufftComplex *>(x->data),
                     static_cast<cufftReal *>(out->data));

    if (result != CUFFT_SUCCESS) {
      cufft_set_error("irfft cufftExecC2R", result);
      return C_FAILED;
    }
  } else {
    gpu_set_last_error("irfft: unsupported dtype (must be C32 or C64)");
    return C_FAILED;
  }

  // cuFFT is asynchronous — synchronize before returning
  cudaError_t sync_err = cudaStreamSynchronize(0);
  if (sync_err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(sync_err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(irfft, INSIGHT_DTYPE_C32, irfft_kernel_gpu);
REGISTER_IXUCA_KERNEL(irfft, INSIGHT_DTYPE_C64, irfft_kernel_gpu);
