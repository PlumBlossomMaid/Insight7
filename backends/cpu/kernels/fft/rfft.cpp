// backends/cpu/kernels/fft/rfft.cpp
/**
 * @file rfft.cpp
 * @brief CPU kernel for real-to-complex FFT.
 *
 * Performs 1D real FFT on contiguous data with batch processing.
 * Output is complex with length (fft_len/2 + 1).
 * Uses shared thread-local FFTW plan cache.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (complex)
 *   inputs[1] = InsightArray* input (real)
 *   inputs[2] = int64_t* fft_len
 *   inputs[3] = int64_t* batch_size
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <string.h>

C_Status rfft_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int64_t fft_len = *(int64_t *)inputs[2];
  int64_t batch_size = *(int64_t *)inputs[3];

  if (!out || !x) {
    cpu_set_last_error("rfft: null array pointer");
    return C_FAILED;
  }

  // Double precision
  if (x->dtype == INSIGHT_DTYPE_F64) {
    double *src = (double *)x->data;
    double *dst = (double *)out->data;

    fftw_plan plan =
        fft_ensure_plan_f64(fft_len, batch_size, FFTW_FORWARD, FFT_KIND_R2C);
    fftw_execute_dft_r2c(plan, src, (fftw_complex *)dst);
  }
  // Single precision
  else if (x->dtype == INSIGHT_DTYPE_F32) {
    float *src = (float *)x->data;
    float *dst = (float *)out->data;

    fftwf_plan plan =
        fft_ensure_plan_f32(fft_len, batch_size, FFTW_FORWARD, FFT_KIND_R2C);
    fftwf_execute_dft_r2c(plan, src, (fftwf_complex *)dst);
  } else {
    cpu_set_last_error("rfft: input must be float32 or float64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(rfft, INSIGHT_DTYPE_F32, rfft_kernel_cpu);
REGISTER_CPU_KERNEL(rfft, INSIGHT_DTYPE_F64, rfft_kernel_cpu);

#endif // INSIGHT_USE_FFTW3
