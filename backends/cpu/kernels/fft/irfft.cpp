// backends/cpu/kernels/fft/irfft.cpp
/**
 * @file irfft.cpp
 * @brief CPU kernel for inverse real FFT (complex-to-real).
 *
 * Performs 1D inverse real FFT on contiguous data with batch processing.
 * Input is complex of length (fft_len/2 + 1), output is real of length fft_len.
 * Uses shared thread-local FFTW plan cache.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (real)
 *   inputs[1] = InsightArray* input (complex)
 *   inputs[2] = int64_t* fft_len (output real length)
 *   inputs[3] = int64_t* batch_size
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <string.h>

C_Status irfft_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int64_t fft_len = *(int64_t *)inputs[2];
  int64_t batch_size = *(int64_t *)inputs[3];

  if (!out || !x) {
    cpu_set_last_error("irfft: null array pointer");
    return C_FAILED;
  }

  // Double precision
  if (x->dtype == INSIGHT_DTYPE_F64 || x->dtype == INSIGHT_DTYPE_C64) {
    double *src = (double *)x->data;
    double *dst = (double *)out->data;

    fftw_plan plan =
        fft_ensure_plan_f64(fft_len, batch_size, FFTW_BACKWARD, FFT_KIND_C2R);
    fftw_execute_dft_c2r(plan, (fftw_complex *)src, dst);
  }
  // Single precision
  else if (x->dtype == INSIGHT_DTYPE_F32 || x->dtype == INSIGHT_DTYPE_C32) {
    float *src = (float *)x->data;
    float *dst = (float *)out->data;

    fftwf_plan plan =
        fft_ensure_plan_f32(fft_len, batch_size, FFTW_BACKWARD, FFT_KIND_C2R);
    fftwf_execute_dft_c2r(plan, (fftwf_complex *)src, dst);
  } else {
    cpu_set_last_error("irfft: input must be complex (C32 or C64)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(irfft, INSIGHT_DTYPE_C32, irfft_kernel_cpu);
REGISTER_CPU_KERNEL(irfft, INSIGHT_DTYPE_C64, irfft_kernel_cpu);

#endif // INSIGHT_USE_FFTW3
