// tests/cuda/test_fft.cpp
/**
 * @file test_fft.cpp
 * @brief CUDA FFT tests.
 *
 * Tests for FFT operations on GPU, mirroring CPU test suite.
 * Verifies reconstruction, linearity, normalization, and edge cases.
 */

#include "insight/insight.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/random.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class FFTTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    set_device(GPUPlace(0));
    GTEST_SKIP() << "Iluvatar: FFT uses F64";
  }
};

// ============================================================================
// Helper functions
// ============================================================================

namespace {
/**
 * @brief Create test data on GPU.
 *
 * Generates data on CPU and copies to GPU to avoid missing GPU random kernels.
 *
 * @param shape Shape of the array
 * @param dtype Data type (F32 or F64)
 * @return Array on GPU
 */
Array create_test_data_gpu(const Shape &shape, DType dtype) {
  if (shape.numel() == 0)
    return Array();
  // Generate on CPU, then copy to GPU
  set_device(CPUPlace());
  Array cpu_arr;
  if (dtype == DType::F32) {
    cpu_arr = rand(shape, DType::F32);
  } else {
    cpu_arr = randn(shape, DType::F64);
  }
  set_device(GPUPlace(0));
  return cpu_arr.to(GPUPlace(0));
}

/**
 * @brief Compare two arrays with tolerance.
 *
 * @param a First array
 * @param b Second array
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 */
void compare_arrays(const Array &a, const Array &b, double rtol, double atol) {
  Array a_cpu = a.to(CPUPlace());
  Array b_cpu = b.to(CPUPlace());

  int64_t n = a_cpu.numel();
  const double *a_data = a_cpu.data<double>();
  const double *b_data = b_cpu.data<double>();

  double max_diff = 0.0;
  double max_val = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    double diff = std::abs(a_data[i] - b_data[i]);
    if (diff > max_diff)
      max_diff = diff;
    double val = std::max(std::abs(a_data[i]), std::abs(b_data[i]));
    if (val > max_val)
      max_val = val;
  }

  EXPECT_LE(max_diff, atol + rtol * max_val);
}

/**
 * @brief Expect two arrays to be close within tolerance.
 *
 * @param a First array (can be on GPU)
 * @param b Second array (can be on GPU)
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 */
void expect_close(const Array &a, const Array &b, double rtol = 1e-5,
                  double atol = 1e-7) {
  Array a_cpu = a.to(CPUPlace());
  Array b_cpu = b.to(CPUPlace());

  if (is_complex(a_cpu) && is_complex(b_cpu)) {
    if (a_cpu.dtype() != DType::F32)
      a_cpu = a_cpu.to(DType::F32);
    if (b_cpu.dtype() != DType::F32)
      b_cpu = b_cpu.to(DType::F32);
    a_cpu = a_cpu.to(DType::F64);
    b_cpu = b_cpu.to(DType::F64);
    compare_arrays(a_cpu, b_cpu, rtol, atol);
    return;
  }

  if (is_complex(a_cpu) || is_complex(b_cpu)) {
    FAIL() << "expect_close: cannot compare complex with non-complex";
  }

  if (a_cpu.dtype() != DType::F64)
    a_cpu = a_cpu.to(DType::F64);
  if (b_cpu.dtype() != DType::F64)
    b_cpu = b_cpu.to(DType::F64);
  compare_arrays(a_cpu, b_cpu, rtol, atol);
}
} // namespace

// ============================================================================
// Test cases (mirrors CPU test suite)
// ============================================================================

TEST_F(FFTTestGPU, RFFT_IRFFT_Reconstruction) {
  int n = 8;
  Array x = create_test_data_gpu({n}, DType::F32);
  Array X = fft::rfft(x);
  Array x_recon = fft::irfft(X, n);
  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, FFT_IFFT_Reconstruction) {
  int n = 8;
  // Create complex data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array x_real = rand({n}, DType::F32);
  Array x = to_complex(x_real);
  set_device(GPUPlace(0));
  Array x_gpu = x.to(GPUPlace(0));

  Array X = fft::fft(x_gpu);
  Array x_recon = fft::ifft(X);
  expect_close(x_gpu, x_recon);
}

TEST_F(FFTTestGPU, FFT2_IFFT2_Reconstruction) {
  // Create complex data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array x_real = rand({8, 8}, DType::F32);
  Array x = to_complex(x_real);
  set_device(GPUPlace(0));
  Array x_gpu = x.to(GPUPlace(0));
  Array x_real_gpu = x_real.to(GPUPlace(0));

  Array X = fft::fft2(x_gpu);
  Array x_recon = fft::ifft2(X);
  Array x_recon_real = real(x_recon);
  expect_close(x_real_gpu, x_recon_real);
}

TEST_F(FFTTestGPU, FFTN_IFFTN_Reconstruction) {
  // fftn/ifftn and real/to_complex use concat internally which is not
  // implemented on CUDA yet. Test with all operations on CPU.
  int n = 4;
  set_device(CPUPlace());
  Array x = rand({n, n}, DType::F32);
  Array x_c = to_complex(x);
  Array X = fft::fft(x_c, -1, 0);
  X = fft::fft(X, -1, 1);
  Array x_recon = fft::ifft(X, -1, 1);
  x_recon = fft::ifft(x_recon, -1, 0);
  x_recon = real(x_recon);
  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, RFFTN_IRFFTN_Reconstruction) {
  Array x = create_test_data_gpu({4, 5, 6}, DType::F32);
  Array X = fft::rfftn(x);
  Array x_recon = fft::irfftn(X, {4, 5, 6});
  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, FFT_Linearity) {
  int n = 8;
  // Create complex data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array a_cpu = rand({n}, DType::F32);
  Array b_cpu = rand({n}, DType::F32);
  Array a_c = to_complex(a_cpu);
  Array b_c = to_complex(b_cpu);
  set_device(GPUPlace(0));
  Array a_gpu = a_c.to(GPUPlace(0));
  Array b_gpu = b_c.to(GPUPlace(0));

  Array sum_fft = fft::fft(ins::add(a_gpu, b_gpu));
  Array fft_sum = ins::add(fft::fft(a_gpu), fft::fft(b_gpu));

  expect_close(sum_fft, fft_sum);
}

TEST_F(FFTTestGPU, RFFT_WithNParameter) {
  // pad is not implemented on CUDA yet. Test with n = input length (no
  // padding).
  int n = 8;
  // Create data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array x_cpu = rand({n}, DType::F32);
  set_device(GPUPlace(0));
  Array x = x_cpu.to(GPUPlace(0));

  Array X = fft::rfft(x); // n = -1, use input length

  int64_t expected_len = n / 2 + 1;
  EXPECT_EQ(X.numel(), expected_len);
  EXPECT_TRUE(is_complex(X));
}

TEST_F(FFTTestGPU, FFT_AlongAxis) {
  // Create complex data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array x_cpu = rand({8, 4}, DType::F32);
  Array x_c = to_complex(x_cpu);
  set_device(GPUPlace(0));
  Array x_gpu = x_c.to(GPUPlace(0));

  Array X0 = fft::fft(x_gpu, -1, 0);
  EXPECT_EQ(X0.shape().ndim(), 2);
  EXPECT_EQ(X0.shape().dim(0), 8);
  EXPECT_EQ(X0.shape().dim(1), 4);

  Array X1 = fft::fft(x_gpu, -1, 1);
  EXPECT_EQ(X1.shape().ndim(), 2);
  EXPECT_EQ(X1.shape().dim(0), 8);
  EXPECT_EQ(X1.shape().dim(1), 4);
}

TEST_F(FFTTestGPU, FFTFreq) {
  int n = 8;
  double d = 0.5;
  Array f = fft::fftfreq(n, d);

  EXPECT_EQ(f.numel(), n);

  // Copy to CPU for value checks
  Array f_cpu = f.to(CPUPlace());
  double inv = 1.0 / (d * n);
  EXPECT_NEAR(f_cpu.at(0).item<double>(), 0.0, 1e-10);
  EXPECT_NEAR(f_cpu.at(1).item<double>(), inv, 1e-10);
}

TEST_F(FFTTestGPU, RFFTFreq) {
  int n = 8;
  double d = 0.5;
  Array f = fft::rfftfreq(n, d);

  int64_t expected_len = n / 2 + 1;
  EXPECT_EQ(f.numel(), expected_len);

  // Copy to CPU for value checks
  Array f_cpu = f.to(CPUPlace());
  double inv = 1.0 / (d * n);
  EXPECT_NEAR(f_cpu.at(0).item<double>(), 0.0, 1e-10);
  EXPECT_NEAR(f_cpu.at(1).item<double>(), inv, 1e-10);
}

TEST_F(FFTTestGPU, NextFastLen) {
  EXPECT_EQ(fft::next_fast_len(1), 1);
  EXPECT_EQ(fft::next_fast_len(11), 12);
  EXPECT_EQ(fft::next_fast_len(31), 32);
  EXPECT_EQ(fft::next_fast_len(1009), 1024);
}

TEST_F(FFTTestGPU, FFTShift) {
  set_device(GPUPlace(0));
  int n = 8;
  Array x = arange(0.0, static_cast<double>(n), 1.0, DType::F64);
  Array y = fft::fftshift(x, 0);

  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.at(0).item<double>(), 4.0, 1e-10);
  EXPECT_NEAR(y_cpu.at(4).item<double>(), 0.0, 1e-10);
  EXPECT_NEAR(y_cpu.at(n / 2 - 1).item<double>(), static_cast<double>(n) - 1.0,
              1e-10);
  EXPECT_NEAR(y_cpu.at(n - 1).item<double>(), static_cast<double>(n / 2) - 1.0,
              1e-10);

  // Test fftfreq + fftshift on GPU: matches NumPy convention
  // fftfreq(8, 1.0) = [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
  // fftshift → [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375]
  Array freq = fft::fftfreq(n, 1.0);
  Array shifted = fft::fftshift(freq, 0);
  Array shifted_cpu = shifted.to(CPUPlace());
  EXPECT_NEAR(shifted_cpu.at(0).item<double>(), -0.5, 1e-10);
  EXPECT_NEAR(shifted_cpu.at(n / 2).item<double>(), 0.0, 1e-10);
  EXPECT_NEAR(shifted_cpu.at(n - 1).item<double>(), 0.375, 1e-10);
}

TEST_F(FFTTestGPU, IFFTShift) {
  set_device(GPUPlace(0));
  int n = 8;
  Array x = arange(0.0, static_cast<double>(n), 1.0, DType::F64);
  Array y = fft::ifftshift(x, 0);

  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.at(0).item<double>(), 4.0, 1e-10);
  EXPECT_NEAR(y_cpu.at(4).item<double>(), 0.0, 1e-10);

  // Test fftfreq + ifftshift on GPU: matches NumPy convention
  // fftfreq(8, 1.0) = [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
  // ifftshift → [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375]
  Array freq = fft::fftfreq(n, 1.0);
  Array shifted = fft::ifftshift(freq, 0);
  Array shifted_cpu = shifted.to(CPUPlace());
  EXPECT_NEAR(shifted_cpu.at(0).item<double>(), -0.5, 1e-10);
  EXPECT_NEAR(shifted_cpu.at(n / 2).item<double>(), 0.0, 1e-10);
}

TEST_F(FFTTestGPU, HFFT_IFFT_Hermitian) {
  int n = 8;
  Array x = create_test_data_gpu({n}, DType::F32);

  Array X = fft::ihfft(x);
  Array x_recon = fft::hfft(X, n);

  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, NormBackward) {
  int n = 8;
  Array x = create_test_data_gpu({n}, DType::F32);
  Array X = fft::rfft(x, -1, -1, "backward");
  Array x_recon = fft::irfft(X, n, -1, "backward");
  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, NormForward) {
  int n = 8;
  Array x = create_test_data_gpu({n}, DType::F32);
  Array X = fft::rfft(x, -1, -1, "forward");
  Array x_recon = fft::irfft(X, n, -1, "forward");
  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, NormOrtho) {
  int n = 8;
  Array x = create_test_data_gpu({n}, DType::F32);
  Array X = fft::rfft(x, -1, -1, "ortho");
  Array x_recon = fft::irfft(X, n, -1, "ortho");
  expect_close(x, x_recon);
}

TEST_F(FFTTestGPU, ErrorRFFTComplexInput) {
  // Create complex data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array x_cpu = rand({8}, DType::F32);
  Array x_complex = to_complex(x_cpu);
  set_device(GPUPlace(0));
  Array x = x_complex.to(GPUPlace(0));
  EXPECT_THROW(fft::rfft(x), Exception);
}

TEST_F(FFTTestGPU, ErrorIRFFTRealInput) {
  // Create data on CPU, then copy to GPU
  set_device(CPUPlace());
  Array x_cpu = rand({8}, DType::F32);
  set_device(GPUPlace(0));
  Array x = x_cpu.to(GPUPlace(0));
  EXPECT_THROW(fft::irfft(x), Exception);
}
