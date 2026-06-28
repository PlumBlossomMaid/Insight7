// backends/cuda/kernels/fft/fft_c2c.cu
/**
 * @file fft_c2c.cu
 * @brief CUDA kernel for complex-to-complex FFT.
 *
 * Performs 1D complex FFT on contiguous data with batch processing.
 * Uses cuFFT library with thread-local plan caching.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int64_t* fft_len
 *   inputs[3] = int64_t* batch_size
 *   inputs[4] = int* inverse (0=forward, 1=backward)
 *   inputs[5] = int* real_input (unused, for compatibility)
 *   inputs[6] = int* norm_code (unused, normalization handled by frontend)
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {

/**
 * @brief GPU entry point for complex-to-complex FFT.
 *
 * @param inputs  See file header for layout
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status fft_c2c_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  int64_t fft_len = *static_cast<int64_t *>(inputs[2]);
  int64_t batch_size = *static_cast<int64_t *>(inputs[3]);
  int inverse = *static_cast<int *>(inputs[4]);

  if (!out || !x) {
    gpu_set_last_error("fft_c2c: null array pointer");
    return C_FAILED;
  }

  int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
  int64_t total_complex = fft_len * batch_size;

  // Double precision (C64)
  if (x->dtype == INSIGHT_DTYPE_C64) {
    // Copy input to output (in-place transform)
    cudaMemcpy(out->data, x->data, total_complex * 2 * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cufftHandle plan = cufft_ensure_plan(fft_len, batch_size, direction,
                                         CuFFTKind::C2C, false);
    if (plan == 0) {
      gpu_set_last_error("fft_c2c: failed to create cuFFT plan (C64)");
      return C_FAILED;
    }

    cufftResult result =
        cufftExecZ2Z(plan, static_cast<cufftDoubleComplex *>(out->data),
                     static_cast<cufftDoubleComplex *>(out->data), direction);

    if (result != CUFFT_SUCCESS) {
      cufft_set_error("fft_c2c cufftExecZ2Z", result);
      return C_FAILED;
    }
  }
  // Single precision (C32)
  else if (x->dtype == INSIGHT_DTYPE_C32) {
    cudaMemcpy(out->data, x->data, total_complex * 2 * sizeof(float),
               cudaMemcpyDeviceToDevice);

    cufftHandle plan =
        cufft_ensure_plan(fft_len, batch_size, direction, CuFFTKind::C2C, true);
    if (plan == 0) {
      gpu_set_last_error("fft_c2c: failed to create cuFFT plan (C32)");
      return C_FAILED;
    }

    cufftResult result =
        cufftExecC2C(plan, static_cast<cufftComplex *>(out->data),
                     static_cast<cufftComplex *>(out->data), direction);

    if (result != CUFFT_SUCCESS) {
      cufft_set_error("fft_c2c cufftExecC2C", result);
      return C_FAILED;
    }
  } else {
    gpu_set_last_error("fft_c2c: unsupported dtype (must be C32 or C64)");
    return C_FAILED;
  }

  // cuFFT is asynchronous — synchronize before returning so callers can read
  // the result immediately (concat/transpose in fftshift, element-wise in
  // apply_norm, etc.)
  cudaError_t sync_err = cudaStreamSynchronize(0);
  if (sync_err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(sync_err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(fft, INSIGHT_DTYPE_C32, fft_c2c_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(fft, INSIGHT_DTYPE_C64, fft_c2c_kernel_gpu);
