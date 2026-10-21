// backends/cuda/kernels/fft/rfft.cu
/**
 * @file rfft.cu
 * @brief CUDA kernel for real-to-complex FFT.
 *
 * Performs 1D real FFT on contiguous data with batch processing.
 * Output is complex with length (fft_len/2 + 1).
 * Uses cuFFT library with thread-local plan caching.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (complex)
 *   inputs[1] = InsightArray* input (real)
 *   inputs[2] = int64_t* fft_len
 *   inputs[3] = int64_t* batch_size
 *   inputs[4] = int* inverse (unused, always forward)
 *   inputs[5] = int* real_input (unused, for compatibility)
 *   inputs[6] = int* norm_code (unused, normalization handled by frontend)
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

extern "C" {

/**
 * @brief GPU entry point for real-to-complex FFT.
 *
 * @param inputs  See file header for layout
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status rfft_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  int64_t fft_len = *static_cast<int64_t *>(inputs[2]);
  int64_t batch_size = *static_cast<int64_t *>(inputs[3]);

  if (!out || !x) {
    gpu_set_last_error("rfft: null array pointer");
    return C_FAILED;
  }

  // Double precision (F64 -> C64)
  if (x->dtype == INSIGHT_DTYPE_F64) {
    cufftHandle plan = cufft_ensure_plan(fft_len, batch_size, CUFFT_FORWARD,
                                         CuFFTKind::R2C, false);
    if (plan == 0) {
      gpu_set_last_error("rfft: failed to create cuFFT plan (F64)");
      return C_FAILED;
    }

    cufftResult result =
        cufftExecD2Z(plan, static_cast<cufftDoubleReal *>(x->data),
                     static_cast<cufftDoubleComplex *>(out->data));

    if (result != CUFFT_SUCCESS) {
      cufft_set_error("rfft cufftExecD2Z", result);
      return C_FAILED;
    }
  }
  // Single precision (F32 -> C32)
  else if (x->dtype == INSIGHT_DTYPE_F32) {
    cufftHandle plan = cufft_ensure_plan(fft_len, batch_size, CUFFT_FORWARD,
                                         CuFFTKind::R2C, true);
    if (plan == 0) {
      gpu_set_last_error("rfft: failed to create cuFFT plan (F32)");
      return C_FAILED;
    }

    cufftResult result = cufftExecR2C(plan, static_cast<cufftReal *>(x->data),
                                      static_cast<cufftComplex *>(out->data));

    if (result != CUFFT_SUCCESS) {
      cufft_set_error("rfft cufftExecR2C", result);
      return C_FAILED;
    }
  } else {
    gpu_set_last_error("rfft: unsupported dtype (must be F32 or F64)");
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

REGISTER_IXUCA_KERNEL(rfft, INSIGHT_DTYPE_F32, rfft_kernel_gpu);
REGISTER_IXUCA_KERNEL(rfft, INSIGHT_DTYPE_F64, rfft_kernel_gpu);
