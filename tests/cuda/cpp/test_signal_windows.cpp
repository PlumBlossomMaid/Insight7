// tests/cuda/test_signal_windows.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

namespace {

class SignalWindowsTestGPU : public ::testing::Test {
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

// ========== boxcar ==========

TEST_F(SignalWindowsTestGPU, BoxcarBasic) {
  Array w = signal::boxcar(5);
  Array w_cpu = w.to(CPUPlace());
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_DOUBLE_EQ(w_cpu.data<double>()[i], 1.0);
  }
}

TEST_F(SignalWindowsTestGPU, BoxcarSize1) {
  Array w = signal::boxcar(1);
  Array w_cpu = w.to(CPUPlace());
  EXPECT_EQ(w_cpu.numel(), 1);
  EXPECT_DOUBLE_EQ(w_cpu.data<double>()[0], 1.0);
}

TEST_F(SignalWindowsTestGPU, BoxcarPeriodic) {
  Array w = signal::boxcar(5, false);
  Array w_cpu = w.to(CPUPlace());
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_DOUBLE_EQ(w_cpu.data<double>()[i], 1.0);
  }
}

// ========== triang ==========

TEST_F(SignalWindowsTestGPU, TriangOdd) {
  Array w = signal::triang(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0 / 3.0, 1e-10);
  EXPECT_NEAR(d[1], 2.0 / 3.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[3], 2.0 / 3.0, 1e-10);
  EXPECT_NEAR(d[4], 1.0 / 3.0, 1e-10);
}

TEST_F(SignalWindowsTestGPU, TriangEven) {
  Array w = signal::triang(4);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.25, 1e-10);
  EXPECT_NEAR(d[1], 0.75, 1e-10);
  EXPECT_NEAR(d[2], 0.75, 1e-10);
  EXPECT_NEAR(d[3], 0.25, 1e-10);
}

// ========== parzen ==========

TEST_F(SignalWindowsTestGPU, ParzenBasic) {
  Array w = signal::parzen(8);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.00390625, 1e-6);
  EXPECT_NEAR(d[1], 0.10546875, 1e-6);
  EXPECT_NEAR(d[2], 0.47265625, 1e-6);
  EXPECT_NEAR(d[3], 0.91796875, 1e-6);
  EXPECT_NEAR(d[4], d[3], 1e-10);
  EXPECT_NEAR(d[5], d[2], 1e-10);
  EXPECT_NEAR(d[6], d[1], 1e-10);
  EXPECT_NEAR(d[7], d[0], 1e-10);
}

// ========== bohman ==========

TEST_F(SignalWindowsTestGPU, BohmanEndpoints) {
  Array w = signal::bohman(11);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[10], 0.0, 1e-10);
  EXPECT_NEAR(d[5], 1.0, 1e-6);
}

// ========== bartlett ==========

TEST_F(SignalWindowsTestGPU, BartlettBasic) {
  Array w = signal::bartlett(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[3], 0.5, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
}

// ========== cosine ==========

TEST_F(SignalWindowsTestGPU, CosineBasic) {
  Array w = signal::cosine(4);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], std::sin(M_PI / 8.0), 1e-10);
  EXPECT_NEAR(d[1], std::sin(3.0 * M_PI / 8.0), 1e-10);
  EXPECT_NEAR(d[2], std::sin(5.0 * M_PI / 8.0), 1e-10);
  EXPECT_NEAR(d[3], std::sin(7.0 * M_PI / 8.0), 1e-10);
}

// ========== blackman ==========

TEST_F(SignalWindowsTestGPU, BlackmanBasic) {
  Array w = signal::blackman(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

// ========== hann ==========

TEST_F(SignalWindowsTestGPU, HannBasic) {
  Array w = signal::hann(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[3], 0.5, 1e-10);
}

// ========== hamming ==========

TEST_F(SignalWindowsTestGPU, HammingBasic) {
  Array w = signal::hamming(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.08, 1e-10);
  EXPECT_NEAR(d[4], 0.08, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

// ========== kaiser ==========

TEST_F(SignalWindowsTestGPU, KaiserBasic) {
  Array w_rect = signal::kaiser(5, 0.0);
  Array w_cpu = w_rect.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 1.0, 1e-10);
  }
}

TEST_F(SignalWindowsTestGPU, KaiserSymmetry) {
  Array w = signal::kaiser(11, 5.0);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10);
  }
  EXPECT_NEAR(d[5], 1.0, 1e-10);
  EXPECT_GT(d[0], 0.0);
}

// ========== gaussian ==========

TEST_F(SignalWindowsTestGPU, GaussianBasic) {
  Array w = signal::gaussian(5, 1.0);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
  EXPECT_NEAR(d[0], std::exp(-2.0), 1e-10);
}

// ========== general_gaussian ==========

TEST_F(SignalWindowsTestGPU, GeneralGaussianBasic) {
  Array w = signal::general_gaussian(5, 1.0, 1.0);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[0], std::exp(-2.0), 1e-10);
}

// ========== general_cosine ==========

TEST_F(SignalWindowsTestGPU, GeneralCosineHFT90D) {
  std::vector<double> HFT90D = {1, 1.942604, 1.340318, 0.440811, 0.043097};
  Array w = signal::general_cosine(64, HFT90D, false);
  EXPECT_EQ(w.numel(), 64);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  double max_val = d[0];
  for (int64_t i = 1; i < 64; ++i) {
    if (d[i] > max_val)
      max_val = d[i];
  }
  EXPECT_GT(max_val, 4.0);
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[63], 0.0, 1e-3);
}

// ========== tukey ==========

TEST_F(SignalWindowsTestGPU, TukeyAlpha0) {
  Array w = signal::tukey(10, 0.0);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(d[i], 1.0, 1e-10);
  }
}

TEST_F(SignalWindowsTestGPU, TukeyAlpha1) {
  Array w_tukey = signal::tukey(10, 1.0);
  Array w_hann = signal::hann(10);
  Array t_cpu = w_tukey.to(CPUPlace());
  Array h_cpu = w_hann.to(CPUPlace());
  const double *dt = t_cpu.data<double>();
  const double *dh = h_cpu.data<double>();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(dt[i], dh[i], 1e-10);
  }
}

// ========== barthann ==========

TEST_F(SignalWindowsTestGPU, BarthannBasic) {
  Array w = signal::barthann(11);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[5], 1.0, 0.02);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10);
  }
}

// ========== exponential ==========

TEST_F(SignalWindowsTestGPU, ExponentialBasic) {
  Array w = signal::exponential(5, -1.0, 1.0);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
  EXPECT_NEAR(d[0], std::exp(-2.0), 1e-10);
}

// ========== chebwin ==========

TEST_F(SignalWindowsTestGPU, ChebwinBasic) {
  Array w = signal::chebwin(11, 50.0);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_EQ(w_cpu.numel(), 11);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-6);
  }
  for (int64_t i = 0; i < 11; ++i) {
    EXPECT_GE(d[i], 0.0);
    EXPECT_LE(d[i], 1.0 + 1e-6);
  }
  double max_val = 0;
  for (int64_t i = 0; i < 11; ++i) {
    if (d[i] > max_val)
      max_val = d[i];
  }
  EXPECT_NEAR(max_val, 1.0, 1e-6);
}

// ========== taylor ==========

TEST_F(SignalWindowsTestGPU, TaylorBasic) {
  Array w = signal::taylor(32, 4, -30.0, true);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_EQ(w_cpu.numel(), 32);
  for (int64_t i = 0; i < 16; ++i) {
    EXPECT_NEAR(d[i], d[31 - i], 1e-6);
  }
  for (int64_t i = 0; i < 32; ++i) {
    EXPECT_GE(d[i], 0.0);
    EXPECT_LE(d[i], 1.0 + 1e-6);
  }
  double max_val = 0;
  for (int64_t i = 0; i < 32; ++i) {
    if (d[i] > max_val)
      max_val = d[i];
  }
  EXPECT_NEAR(max_val, 1.0, 1e-6);
}

// ========== get_window ==========

TEST_F(SignalWindowsTestGPU, GetWindowBoxcar) {
  Array w = signal::get_window("boxcar", 8);
  Array w_cpu = w.to(CPUPlace());
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(w_cpu.data<double>()[i], 1.0);
  }
}

TEST_F(SignalWindowsTestGPU, GetWindowHann) {
  Array w = signal::get_window("hann", 5);
  Array ref = signal::hann(5, false);
  Array w_cpu = w.to(CPUPlace());
  Array r_cpu = ref.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  const double *r = r_cpu.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], r[i], 1e-10);
  }
}

TEST_F(SignalWindowsTestGPU, GetWindowKaiser) {
  Array w = signal::get_window("kaiser", 5.0, 11);
  Array ref = signal::kaiser(11, 5.0, false);
  Array w_cpu = w.to(CPUPlace());
  Array r_cpu = ref.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  const double *r = r_cpu.data<double>();
  for (int64_t i = 0; i < 11; ++i) {
    EXPECT_NEAR(d[i], r[i], 1e-10);
  }
}

// ========== nuttall / blackmanharris / flattop ==========

TEST_F(SignalWindowsTestGPU, NuttallBasic) {
  Array w = signal::nuttall(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0003628, 1e-4);
  EXPECT_NEAR(d[4], 0.0003628, 1e-4);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWindowsTestGPU, BlackmanharrisBasic) {
  Array w = signal::blackmanharris(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_NEAR(d[0], 6.0e-5, 1e-4);
  EXPECT_NEAR(d[4], 6.0e-5, 1e-4);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWindowsTestGPU, FlattopBasic) {
  Array w = signal::flattop(5);
  Array w_cpu = w.to(CPUPlace());
  const double *d = w_cpu.data<double>();
  EXPECT_EQ(w_cpu.numel(), 5);
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

// ========== periodic ==========

TEST_F(SignalWindowsTestGPU, HannPeriodic) {
  Array w_sym = signal::hann(10, true);
  Array w_per = signal::hann(10, false);
  Array s_cpu = w_sym.to(CPUPlace());
  Array p_cpu = w_per.to(CPUPlace());
  EXPECT_EQ(s_cpu.numel(), 10);
  EXPECT_EQ(p_cpu.numel(), 10);
  const double *ds = s_cpu.data<double>();
  const double *dp = p_cpu.data<double>();
  EXPECT_NE(ds[9], dp[9]);
}

} // namespace
