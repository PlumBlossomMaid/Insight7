// tests/cuda/test_signal_filter_design.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

namespace {

class SignalFilterDesignTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    // Iluvatar: signal API uses F64
    GTEST_SKIP() << "Iluvatar: signal API uses F64 (hardware limit)";
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}

// ========== kaiser_beta (scalar, no GPU/CPU difference) ==========

TEST_F(SignalFilterDesignTestGPU, KaiserBetaLow) {
  EXPECT_DOUBLE_EQ(signal::kaiser_beta(0.0), 0.0);
  EXPECT_DOUBLE_EQ(signal::kaiser_beta(21.0), 0.0);
}

TEST_F(SignalFilterDesignTestGPU, KaiserBetaMid) {
  double a = 40.0;
  double expected = 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  EXPECT_NEAR(signal::kaiser_beta(a), expected, 1e-10);
}

TEST_F(SignalFilterDesignTestGPU, KaiserBetaHigh) {
  double a = 60.0;
  double expected = 0.1102 * (a - 8.7);
  EXPECT_NEAR(signal::kaiser_beta(a), expected, 1e-10);
}

TEST_F(SignalFilterDesignTestGPU, KaiserBetaBoundary) {
  double a = 50.0;
  double beta = signal::kaiser_beta(a);
  EXPECT_GT(beta, 4.0);
  EXPECT_LT(beta, 6.0);
}

// ========== kaiser_atten (scalar, no GPU/CPU difference) ==========

TEST_F(SignalFilterDesignTestGPU, KaiserAttenBasic) {
  double atten = signal::kaiser_atten(101, 0.1);
  double expected = 2.285 * 100.0 * M_PI * 0.1 + 7.95;
  EXPECT_NEAR(atten, expected, 1e-10);
}

TEST_F(SignalFilterDesignTestGPU, KaiserAttenNarrow) {
  double atten_wide = signal::kaiser_atten(101, 0.2);
  double atten_narrow = signal::kaiser_atten(101, 0.05);
  EXPECT_GT(atten_wide, atten_narrow);
}

// ========== firwin ==========

TEST_F(SignalFilterDesignTestGPU, FirwinLowpassBasic) {
  // firwin returns CPU array (composite ops use CPU data internally)
  Array h = signal::firwin(11, {0.3}, "hamming", "lowpass", true);
  Array h_gpu = h.to(GPUPlace(0));
  Array h_cpu = h_gpu.to(CPUPlace());
  EXPECT_EQ(h_cpu.numel(), 11);
  const double *d = h_cpu.data<double>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10) << "symmetry failed at index " << i;
  }
}

TEST_F(SignalFilterDesignTestGPU, FirwinLowpassDCGain) {
  Array h = signal::firwin(21, {0.25}, "hamming", "lowpass", true);
  Array h_gpu = h.to(GPUPlace(0));
  Array dc_gpu = sum(h_gpu);
  Array dc_cpu = dc_gpu.to(CPUPlace());
  EXPECT_NEAR(dc_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalFilterDesignTestGPU, FirwinHighpassBasic) {
  Array h = signal::firwin(11, {0.3}, "hamming", "highpass", true);
  Array h_gpu = h.to(GPUPlace(0));
  Array h_cpu = h_gpu.to(CPUPlace());
  EXPECT_EQ(h_cpu.numel(), 11);
  const double *d = h_cpu.data<double>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10);
  }
}

TEST_F(SignalFilterDesignTestGPU, FirwinHighpassNyquistGain) {
  Array h = signal::firwin(21, {0.25}, "hamming", "highpass", true);
  const double *d = h.data<double>();
  double nyquist_gain = 0.0;
  for (int i = 0; i < 21; ++i) {
    nyquist_gain += d[i] * (i % 2 == 0 ? 1.0 : -1.0);
  }
  EXPECT_NEAR(nyquist_gain, 1.0, 1e-6);
}

TEST_F(SignalFilterDesignTestGPU, FirwinBandpass) {
  Array h = signal::firwin(21, {0.2, 0.5}, "hamming", "bandpass", false);
  Array h_gpu = h.to(GPUPlace(0));
  Array dc_gpu = sum(h_gpu);
  Array dc_cpu = dc_gpu.to(CPUPlace());
  EXPECT_NEAR(dc_cpu.data<double>()[0], 0.0, 0.01);
}

TEST_F(SignalFilterDesignTestGPU, FirwinBandstop) {
  Array h = signal::firwin(21, {0.2, 0.5}, "hamming", "bandstop", false);
  EXPECT_EQ(h.numel(), 21);
}

TEST_F(SignalFilterDesignTestGPU, FirwinDifferentWindows) {
  Array h1 = signal::firwin(11, {0.3}, "hann", "lowpass", false);
  Array h2 = signal::firwin(11, {0.3}, "blackman", "lowpass", false);
  Array h3 = signal::firwin(11, {0.3}, "kaiser", "lowpass", false);
  EXPECT_EQ(h1.numel(), 11);
  EXPECT_EQ(h2.numel(), 11);
  EXPECT_EQ(h3.numel(), 11);
  EXPECT_NE(h1.data<double>()[0], h2.data<double>()[0]);
}

TEST_F(SignalFilterDesignTestGPU, FirwinUnscaled) {
  Array h = signal::firwin(21, {0.25}, "hamming", "lowpass", false);
  Array dc = sum(h);
  double dc_val = dc.data<double>()[0];
  EXPECT_NE(dc_val, 1.0);
}

// ========== firwin2 ==========

TEST_F(SignalFilterDesignTestGPU, Firwin2Lowpass) {
  std::vector<double> freq = {0.0, 0.5, 0.5, 1.0};
  std::vector<double> gain = {1.0, 1.0, 0.0, 0.0};
  Array h = signal::firwin2(15, freq, gain, 0, "hamming", false);
  Array h_gpu = h.to(GPUPlace(0));
  Array dc_gpu = sum(h_gpu);
  Array dc_cpu = dc_gpu.to(CPUPlace());
  EXPECT_EQ(h.numel(), 15);
  EXPECT_NEAR(dc_cpu.data<double>()[0], 1.0, 0.1);
}

TEST_F(SignalFilterDesignTestGPU, Firwin2Highpass) {
  std::vector<double> freq = {0.0, 0.5, 0.5, 1.0};
  std::vector<double> gain = {0.0, 0.0, 1.0, 1.0};
  Array h = signal::firwin2(15, freq, gain, 0, "hamming", false);
  EXPECT_EQ(h.numel(), 15);
}

TEST_F(SignalFilterDesignTestGPU, Firwin2Bandpass) {
  std::vector<double> freq = {0.0, 0.2, 0.2, 0.5, 0.5, 1.0};
  std::vector<double> gain = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  Array h = signal::firwin2(21, freq, gain, 0, "hamming", false);
  EXPECT_EQ(h.numel(), 21);
}

TEST_F(SignalFilterDesignTestGPU, Firwin2DifferentLengths) {
  std::vector<double> freq = {0.0, 0.5, 1.0};
  std::vector<double> gain = {1.0, 1.0, 0.0};
  Array h_short = signal::firwin2(7, freq, gain);
  Array h_long = signal::firwin2(31, freq, gain);
  EXPECT_EQ(h_short.numel(), 7);
  EXPECT_EQ(h_long.numel(), 31);
}

// ========== cmplx_sort ==========

TEST_F(SignalFilterDesignTestGPU, CmplxSortReal) {
  std::vector<double> data = {3.0, -1.0, 2.0, -4.0};
  Array arr = to_array(data);
  Array sorted = signal::cmplx_sort(arr);
  const double *d = sorted.data<double>();
  EXPECT_NEAR(d[0], -1.0, 1e-10);
  EXPECT_NEAR(d[1], 2.0, 1e-10);
  EXPECT_NEAR(d[2], 3.0, 1e-10);
  EXPECT_NEAR(d[3], -4.0, 1e-10);
}

TEST_F(SignalFilterDesignTestGPU, CmplxSortComplex) {
  std::vector<std::complex<double>> data = {{0, 5}, {1, 0}, {0, 3}, {4, 0}};
  Array arr = to_array(data, DType::C64);
  Array sorted = signal::cmplx_sort(arr);
  const std::complex<double> *d =
      reinterpret_cast<const std::complex<double> *>(sorted.data<char>());
  EXPECT_NEAR(std::abs(d[0]), 1.0, 1e-10);
  EXPECT_NEAR(std::abs(d[1]), 3.0, 1e-10);
  EXPECT_NEAR(std::abs(d[2]), 4.0, 1e-10);
  EXPECT_NEAR(std::abs(d[3]), 5.0, 1e-10);
}

TEST_F(SignalFilterDesignTestGPU, CmplxSortSingle) {
  std::vector<double> data = {42.0};
  Array arr = to_array(data);
  Array sorted = signal::cmplx_sort(arr);
  EXPECT_NEAR(sorted.data<double>()[0], 42.0, 1e-10);
}

TEST_F(SignalFilterDesignTestGPU, CmplxSortZeros) {
  std::vector<double> data = {0.0, 0.0, 0.0};
  Array arr = to_array(data);
  Array sorted = signal::cmplx_sort(arr);
  EXPECT_NEAR(sorted.data<double>()[0], 0.0, 1e-10);
  EXPECT_NEAR(sorted.data<double>()[1], 0.0, 1e-10);
  EXPECT_NEAR(sorted.data<double>()[2], 0.0, 1e-10);
}

} // namespace
