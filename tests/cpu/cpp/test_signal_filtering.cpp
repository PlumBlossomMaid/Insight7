// tests/cpu/test_signal_filtering.cpp
#include "insight/insight.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalFilteringTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
#ifndef INSIGHT_USE_FFTW3
    GTEST_SKIP() << "FFTW3 not available, skipping FFT-dependent tests";
#endif
  }
};

namespace {
template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}
} // namespace

// ========== hilbert ==========

TEST_F(SignalFilteringTestCPU, HilbertBasic) {
  // Hilbert transform produces a complex analytic signal
  int64_t N = 64;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::cos(2.0 * M_PI * i / N);
  }
  Array x = to_array(x_data);
  Array y = signal::hilbert(x, N);

  ASSERT_EQ(y.dtype(), DType::C64);
  ASSERT_EQ(y.numel(), N);

  // The analytic signal should have non-zero magnitude everywhere
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_GT(std::abs(data[i]), 0.01) << "near-zero magnitude at " << i;
  }
}

TEST_F(SignalFilteringTestCPU, HilbertDC) {
  // Hilbert transform of DC should be zero (imaginary part)
  std::vector<double> x_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  Array x = to_array(x_data);
  Array y = signal::hilbert(x, 8);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  // Real part should be 1.0 (DC preserved)
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(data[i].real(), 1.0, 1e-10);
    EXPECT_NEAR(data[i].imag(), 0.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, HilbertDefaultN) {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
  Array x = to_array(x_data);
  Array y = signal::hilbert(x); // N defaults to x.numel()
  EXPECT_EQ(y.numel(), 4);
  EXPECT_EQ(y.dtype(), DType::C64);
}

// ========== detrend ==========

TEST_F(SignalFilteringTestCPU, DetrendConstant) {
  // Remove constant (mean)
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::detrend(x, -1, "constant");

  // Mean = 3.0, so detrended = [-2, -1, 0, 1, 2]
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], -2.0, 1e-10);
  EXPECT_NEAR(d[1], -1.0, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 1.0, 1e-10);
  EXPECT_NEAR(d[4], 2.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, DetrendLinear) {
  // Remove linear trend from [0, 1, 2, 3, 4] + trend = [0, 2, 4, 6, 8]
  // Input: [0, 3, 6, 9, 12] = linear trend with slope 3
  // After linear detrend, should be all zeros
  std::vector<double> x_data = {0.0, 3.0, 6.0, 9.0, 12.0};
  Array x = to_array(x_data);
  Array y = signal::detrend(x, -1, "linear");
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10) << "mismatch at " << i;
  }
}

TEST_F(SignalFilteringTestCPU, DetrendLinearWithNoise) {
  // Add small perturbation to linear signal
  std::vector<double> x_data = {0.1, 3.0, 5.9, 9.1, 12.0};
  Array x = to_array(x_data);
  Array y = signal::detrend(x, -1, "linear");
  // After detrend, residuals should be small
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 0.0, 0.2) << "mismatch at " << i;
  }
}

// ========== lfilter (FIR case) ==========

TEST_F(SignalFilteringTestCPU, LfilterFIRBasic) {
  // b = [1, 1], a = [1], x = [1, 0, 0, 0, 0]
  // scipy.signal.lfilter returns len(x) samples: [1, 1, 0, 0, 0]
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  EXPECT_EQ(y.numel(), 5);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 1.0, 1e-10);
  for (int i = 2; i < 5; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, LfilterFIRMovingAvg) {
  // 3-point moving average: b = [1/3, 1/3, 1/3], a = [1]
  std::vector<double> b_data = {1.0 / 3, 1.0 / 3, 1.0 / 3};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  // scipy.signal.lfilter: [1/3, 1, 2, 3, 4]
  EXPECT_EQ(y.numel(), 5);
  EXPECT_NEAR(y.data<double>()[0], 1.0 / 3, 1e-10);
  EXPECT_NEAR(y.data<double>()[1], 1.0, 1e-10);
  EXPECT_NEAR(y.data<double>()[2], 2.0, 1e-10);
  EXPECT_NEAR(y.data<double>()[3], 3.0, 1e-10);
  EXPECT_NEAR(y.data<double>()[4], 4.0, 1e-10);
}

// ========== lfilter (IIR case) ==========

TEST_F(SignalFilteringTestCPU, LfilterIIRBasic) {
  // Simple first-order IIR: y[n] = x[n] + 0.5*y[n-1]
  // b = [1], a = [1, -0.5]
  std::vector<double> b_data = {1.0};
  std::vector<double> a_data = {1.0, -0.5};
  // x = [1, 0, 0, 0, 0] (impulse)
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  // y = [1, 0.5, 0.25, 0.125, 0.0625]
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[2], 0.25, 1e-10);
  EXPECT_NEAR(d[3], 0.125, 1e-10);
  EXPECT_NEAR(d[4], 0.0625, 1e-10);
}

TEST_F(SignalFilteringTestCPU, LfilterIIRNonUnitB0) {
  // 2nd-order lowpass with b[0] != 1
  // b = [0.0675, 0.135, 0.0675], a = [1, -1.143, 0.4128]
  std::vector<double> b_data = {0.0675, 0.135, 0.0675};
  std::vector<double> a_data = {1.0, -1.143, 0.4128};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  ASSERT_EQ(y.numel(), 8);
  // Impulse response from the difference equation
  std::vector<double> expected = {
      0.0675,          0.2121525,       0.2821263075,    0.2348938174725,
      0.1520218936351, 0.0767968565722, 0.0250241693695, -0.0030991168037};
  expect_float_equal(y, expected, 1e-10);
}

TEST_F(SignalFilteringTestCPU, LfilterIIRNonUnitB0Ramp) {
  std::vector<double> b_data = {0.0675, 0.135, 0.0675};
  std::vector<double> a_data = {1.0, -1.143, 0.4128};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  ASSERT_EQ(y.numel(), 8);
  std::vector<double> expected = {
      0.0675,          0.3471525,       0.9089313075,    1.7056039324725,
      2.6542984510801, 3.5447898262599, 4.0360003708092, 3.9598591835548};
  expect_float_equal(y, expected, 1e-10);
}

TEST_F(SignalFilteringTestCPU, LfilterPureGain) {
  // b = [2], a = [1] -> y = 2*x, same length as x
  std::vector<double> b_data = {2.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, -2.0, 3.0, 0.5, 4.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  ASSERT_EQ(y.numel(), 5);
  std::vector<double> expected = {2.0, -4.0, 6.0, 1.0, 8.0};
  expect_float_equal(y, expected, 1e-10);
}

// ========== lfilter_zi ==========

TEST_F(SignalFilteringTestCPU, LfilterZiFIR) {
#ifndef INSIGHT_USE_OPENBLAS
  GTEST_SKIP() << "OpenBLAS not available, skipping LAPACK-dependent test";
#endif
  // FIR: b=[1,1], a=[1] -> zi should be empty or trivial
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array zi = signal::lfilter_zi(b, a);
  // For FIR (na=1), zi length = max(nb,na)-1 = 1
  // zi = b[1]/a[0] = 1.0
  EXPECT_EQ(zi.numel(), 1);
  EXPECT_NEAR(zi.data<double>()[0], 1.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, LfilterZiIIR) {
#ifndef INSIGHT_USE_OPENBLAS
  GTEST_SKIP() << "OpenBLAS not available, skipping LAPACK-dependent test";
#endif
  // b=[1], a=[1, -0.5] -> zi = b[1]-a[1]*b[0] / (1-a[1]) = 0.5/0.5 = 1.0
  std::vector<double> b_data = {1.0};
  std::vector<double> a_data = {1.0, -0.5};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array zi = signal::lfilter_zi(b, a);
  EXPECT_EQ(zi.numel(), 1);
  EXPECT_NEAR(zi.data<double>()[0], 1.0, 1e-10);
}

// ========== filtfilt ==========

TEST_F(SignalFilteringTestCPU, FiltfiltFIR) {
  // Zero-phase filtering with b=[1,1], a=[1]
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::filtfilt(b, a, x);
  // filtfilt should have zero phase distortion
  // The output should be symmetric if input is symmetric
  EXPECT_EQ(y.numel(), x.numel());
}

// ========== freq_shift ==========

TEST_F(SignalFilteringTestCPU, FreqShiftBasic) {
  // Shift a DC signal by some frequency
  int64_t N = 64;
  std::vector<double> x_data(N, 1.0);
  Array x = to_array(x_data);
  Array y = signal::freq_shift(x, 0.0, 1.0);
  // freq_shift of 0 should return the original signal (as complex)
  ASSERT_EQ(y.dtype(), DType::C64);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_NEAR(std::abs(data[i]), 1.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, FreqShiftPositive) {
  // Shift a signal by fs/4 (quarter sample rate)
  int64_t N = 16;
  std::vector<double> x_data(N, 1.0);
  Array x = to_array(x_data);
  double fs = 1.0;
  Array y = signal::freq_shift(x, fs / 4.0, fs);
  // Should produce a complex sinusoid at fs/4
  ASSERT_EQ(y.dtype(), DType::C64);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  // At freq=fs/4, phase increments by pi/2 per sample
  for (int64_t i = 0; i < N; ++i) {
    double expected_real = std::cos(2.0 * M_PI * 0.25 * i);
    double expected_imag = std::sin(2.0 * M_PI * 0.25 * i);
    EXPECT_NEAR(data[i].real(), expected_real, 1e-10);
    EXPECT_NEAR(data[i].imag(), expected_imag, 1e-10);
  }
}

// ========== decimate ==========

TEST_F(SignalFilteringTestCPU, DecimateBasic) {
  // Create a smooth signal and decimate by 2
  int64_t N = 100;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::sin(2.0 * M_PI * i / N);
  }
  Array x = to_array(x_data);
  Array y = signal::decimate(x, 2, -1, false);
  // Output length should be ceil(N/2) = 50
  EXPECT_EQ(y.numel(), 50);
}

TEST_F(SignalFilteringTestCPU, DecimateZeroPhase) {
  int64_t N = 100;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::sin(2.0 * M_PI * i / N);
  }
  Array x = to_array(x_data);
  Array y = signal::decimate(x, 4, -1, true);
  EXPECT_EQ(y.numel(), 25);
}

// ========== resample ==========

TEST_F(SignalFilteringTestCPU, ResampleIdentity) {
  // Resample to same length -> should return same signal
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::resample(x, 5);
  EXPECT_EQ(y.numel(), 5);
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-6);
  }
}

TEST_F(SignalFilteringTestCPU, ResampleUpsample) {
  // Upsample a 4-point signal to 8 points
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
  Array x = to_array(x_data);
  Array y = signal::resample(x, 8);
  EXPECT_EQ(y.numel(), 8);
}

TEST_F(SignalFilteringTestCPU, ResampleDownsample) {
  // Downsample an 8-point signal to 4 points
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  Array x = to_array(x_data);
  Array y = signal::resample(x, 4);
  EXPECT_EQ(y.numel(), 4);
}

// ========== resample_poly ==========

TEST_F(SignalFilteringTestCPU, ResamplePolyIdentity) {
  // up=1, down=1 -> identity
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 1, 1);
  EXPECT_EQ(y.numel(), 5);
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, ResamplePolyUpsample) {
  // up=2, down=1 -> double the length
  std::vector<double> x_data = {1.0, 0.0, 1.0, 0.0};
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 2, 1);
  // Output length should be approximately 2*4 = 8
  EXPECT_GE(y.numel(), 7);
  EXPECT_LE(y.numel(), 9);
}

TEST_F(SignalFilteringTestCPU, ResamplePolyDownsample) {
  // up=1, down=2 -> halve the length
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 1, 2);
  // Output length should be approximately 8/2 = 4
  EXPECT_GE(y.numel(), 3);
  EXPECT_LE(y.numel(), 5);
}

TEST_F(SignalFilteringTestCPU, ResamplePolyPreservesEnergy) {
  // A constant signal must not be resampled into zeros
  int64_t N = 32;
  std::vector<double> x_data(N, 1.0);
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 2, 1);
  ASSERT_EQ(y.numel(), 64);

  const double *d = y.data<double>();
  double max_abs = 0.0;
  double energy = 0.0;
  for (int64_t i = 0; i < y.numel(); ++i) {
    ASSERT_TRUE(std::isfinite(d[i])) << "non-finite at " << i;
    max_abs = std::max(max_abs, std::abs(d[i]));
    energy += d[i] * d[i];
  }
  EXPECT_GT(max_abs, 0.5) << "resample_poly returned an all-zero signal";
  EXPECT_GT(energy / y.numel(), 0.25);

  // Interior of a constant signal stays constant and non-zero
  for (int64_t i = 20; i < 44; ++i) {
    EXPECT_GT(std::abs(d[i]), 0.5) << "zero interior sample at " << i;
    EXPECT_NEAR(d[i], d[32], 0.05) << "interior not flat at " << i;
  }
}

TEST_F(SignalFilteringTestCPU, ResamplePolySineNotZero) {
  int64_t N = 64;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i)
    x_data[i] = std::sin(2.0 * M_PI * i / N);
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 3, 2);

  const double *d = y.data<double>();
  double max_abs = 0.0;
  for (int64_t i = 0; i < y.numel(); ++i)
    max_abs = std::max(max_abs, std::abs(d[i]));
  EXPECT_GT(max_abs, 0.1) << "resample_poly returned an all-zero signal";
}

// ========== firfilter ==========

TEST_F(SignalFilteringTestCPU, FirfilterBasic) {
  // b = [1, 2, 1], x = [1, 0, 0, 0, 0]
  // y = convolve([1,2,1], [1,0,0,0,0]) = [1, 2, 1, 0, 0, 0, 0]
  std::vector<double> b_data = {1.0, 2.0, 1.0};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array x = to_array(x_data);
  Array y = signal::firfilter(b, x);
  EXPECT_EQ(y.numel(), 7);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 2.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  for (int i = 3; i < 7; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10);
  }
}

// ========== sosfilt ==========

TEST_F(SignalFilteringTestCPU, SosfiltSingleSection) {
  // Single biquad: y[n] = x[n] (identity)
  // sos = [1, 0, 0, 1, 0, 0] (b0=1, b1=0, b2=0, a0=1, a1=0, a2=0)
  std::vector<double> sos_data = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  Array sos = to_array(sos_data, DType::F64, CPUPlace());
  sos = reshape(sos, {1, 6});

  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);

  Array y = signal::sosfilt(x, sos);

  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, SosfiltLowpass) {
  // Simple lowpass biquad: y[n] = 0.5*x[n] + 0.5*x[n-1]
  // b=[0.5, 0.5, 0], a=[1, 0, 0]
  std::vector<double> sos_data = {0.5, 0.5, 0.0, 1.0, 0.0, 0.0};
  Array sos = to_array(sos_data, DType::F64, CPUPlace());
  sos = reshape(sos, {1, 6});

  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array x = to_array(x_data);

  Array y = signal::sosfilt(x, sos);

  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 0.5, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, SosfiltCascade) {
  // Two cascaded sections
  // Section 1: gain=1, no feedback
  // Section 2: gain=1, no feedback
  std::vector<double> sos_data = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  Array sos = to_array(sos_data, DType::F64, CPUPlace());
  sos = reshape(sos, {2, 6});

  std::vector<double> x_data = {1.0, 2.0, 3.0};
  Array x = to_array(x_data);

  Array y = signal::sosfilt(x, sos);

  const double *d = y.data<double>();
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-10);
  }
}

// ========== upfirdn ==========

TEST_F(SignalFilteringTestCPU, UpfirdnUpsampleOnly) {
  // p=2, q=1, h=[1] -> upsample by 2
  std::vector<double> h_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0};
  Array h = to_array(h_data);
  Array x = to_array(x_data);

  Array y = signal::upfirdn(h, x, 2, 1);

  // Upsampled: [1, 0, 2, 0, 3, 0]
  EXPECT_EQ(y.numel(), 6);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 2.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
  EXPECT_NEAR(d[4], 3.0, 1e-10);
  EXPECT_NEAR(d[5], 0.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, UpfirdnDownsampleOnly) {
  // p=1, q=2, h=[1] -> downsample by 2
  std::vector<double> h_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Array h = to_array(h_data);
  Array x = to_array(x_data);

  Array y = signal::upfirdn(h, x, 1, 2);

  // Downsampled: [1, 3, 5]
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 3.0, 1e-10);
  EXPECT_NEAR(d[2], 5.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, UpfirdnFloat32) {
  // F32 input must be read as float, not reinterpreted as double
  std::vector<float> h_data = {0.5f, 0.5f};
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Array h = to_array(h_data);
  Array x = to_array(x_data);

  Array y = signal::upfirdn(h, x, 1, 1);

  ASSERT_EQ(y.dtype(), DType::F32);
  std::vector<float> expected = {1.5f, 2.5f, 3.5f, 2.0f};
  expect_float_equal(y, expected, 1e-5f);
}

TEST_F(SignalFilteringTestCPU, UpfirdnFloat32Upsample) {
  std::vector<float> h_data = {1.0f};
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
  Array h = to_array(h_data);
  Array x = to_array(x_data);

  Array y = signal::upfirdn(h, x, 2, 1);

  ASSERT_EQ(y.dtype(), DType::F32);
  std::vector<float> expected = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};
  expect_float_equal(y, expected, 1e-5f);
}

TEST_F(SignalFilteringTestCPU, ChannelizePolyFloat32) {
  std::vector<float> h_data = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Array h = to_array(h_data);
  Array x = to_array(x_data);

  Array y = signal::channelize_poly(x, h, 2);

  ASSERT_EQ(y.dtype(), DType::F32);
  ASSERT_EQ(y.numel(), 4);
  // channel 0 taps [1,1] over x[0],x[2]; channel 1 taps [1,1] over x[1],x[3]
  std::vector<float> expected = {1.0f, 4.0f, 2.0f, 6.0f};
  expect_float_equal(y, expected, 1e-5f);
}

// ========== channelize_poly ==========

TEST_F(SignalFilteringTestCPU, ChannelizePolyBasic) {
  // Simple test: 2 channels, 4 samples
  std::vector<double> h_data = {1.0, 1.0, 1.0, 1.0}; // 4-tap filter
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0}; // 4 samples

  Array h = to_array(h_data);
  Array x = to_array(x_data);

  Array y = signal::channelize_poly(x, h, 2);

  // Should produce 2x2 output
  EXPECT_EQ(y.shape().ndim(), 2);
  EXPECT_EQ(y.shape().dims()[0], 2);
  EXPECT_EQ(y.shape().dims()[1], 2);
}

TEST_F(SignalFilteringTestCPU, ChannelizePolyInvalidInput) {
  Array h = to_array(std::vector<double>{1.0, 1.0});
  Array x = to_array(std::vector<double>{1.0, 2.0, 3.0});

  // n_channels must be >= 1
  EXPECT_THROW(signal::channelize_poly(x, h, 0), std::exception);
}
