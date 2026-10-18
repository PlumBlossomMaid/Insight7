// backends/cpu/kernels/fft/fft_c2c.cpp
/**
 * @file fft_c2c.cpp
 * @brief CPU kernel for complex-to-complex FFT.
 *
 * Performs 1D complex FFT on contiguous data with batch processing.
 * Uses shared thread-local FFTW plan cache.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int64_t* fft_len
 *   inputs[3] = int64_t* batch_size
 *   inputs[4] = int* inverse (0=forward, 1=backward)
 *   inputs[5] = int* real_input (unused, for compatibility)
 *   inputs[6] = int* norm_code (unused, normalization handled by frontend)
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <string.h>

C_Status fft_c2c_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int64_t fft_len = *(int64_t *)inputs[2];
  int64_t batch_size = *(int64_t *)inputs[3];
  int inverse = *(int *)inputs[4];

  if (!out || !x) {
    cpu_set_last_error("fft_c2c: null array pointer");
    return C_FAILED;
  }

  int direction = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
  int64_t total_complex = fft_len * batch_size;

  // Double precision
  if (x->dtype == INSIGHT_DTYPE_F64 || x->dtype == INSIGHT_DTYPE_C64) {
    double *data = (double *)out->data;
    memcpy(data, x->data, total_complex * 2 * sizeof(double));

    fftw_plan plan =
        fft_ensure_plan_f64(fft_len, batch_size, direction, FFT_KIND_C2C);
    fftw_execute_dft(plan, (fftw_complex *)data, (fftw_complex *)data);
  }
  // Single precision
  else if (x->dtype == INSIGHT_DTYPE_F32 || x->dtype == INSIGHT_DTYPE_C32) {
    float *data = (float *)out->data;
    memcpy(data, x->data, total_complex * 2 * sizeof(float));

    fftwf_plan plan =
        fft_ensure_plan_f32(fft_len, batch_size, direction, FFT_KIND_C2C);
    fftwf_execute_dft(plan, (fftwf_complex *)data, (fftwf_complex *)data);
  } else {
    cpu_set_last_error("fft_c2c: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(fft, INSIGHT_DTYPE_F32, fft_c2c_kernel_cpu);
REGISTER_CPU_KERNEL(fft, INSIGHT_DTYPE_F64, fft_c2c_kernel_cpu);
REGISTER_CPU_KERNEL(fft, INSIGHT_DTYPE_C32, fft_c2c_kernel_cpu);
REGISTER_CPU_KERNEL(fft, INSIGHT_DTYPE_C64, fft_c2c_kernel_cpu);

#endif // INSIGHT_USE_FFTW3
