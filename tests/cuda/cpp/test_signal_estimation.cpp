// tests/cuda/test_signal_estimation.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

namespace {

class EstimationTestGPU : public ::testing::Test {
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

// ========== KalmanFilter construction ==========

TEST_F(EstimationTestGPU, KalmanFilterConstruction) {
  KalmanFilter kf(4, 2, 0, 1, DType::F64);

  EXPECT_EQ(kf.dim_x, 4);
  EXPECT_EQ(kf.dim_z, 2);
  EXPECT_EQ(kf.dim_u, 0);
  EXPECT_EQ(kf.points, 1);

  EXPECT_EQ(kf.x.numel(), 4);
  EXPECT_EQ(kf.P.numel(), 16);
  EXPECT_EQ(kf.z.numel(), 2);
  EXPECT_EQ(kf.R.numel(), 4);
  EXPECT_EQ(kf.Q.numel(), 16);
  EXPECT_EQ(kf.F.numel(), 16);
  EXPECT_EQ(kf.H.numel(), 8);
}

TEST_F(EstimationTestGPU, KalmanFilterMultiPoint) {
  KalmanFilter kf(4, 2, 0, 10, DType::F64);

  EXPECT_EQ(kf.x.numel(), 40);
  EXPECT_EQ(kf.P.numel(), 160);
}

TEST_F(EstimationTestGPU, KalmanFilterInvalidDims) {
  EXPECT_THROW(KalmanFilter(0, 2, 0, 1, DType::F64), std::exception);
  EXPECT_THROW(KalmanFilter(4, 0, 0, 1, DType::F64), std::exception);
  EXPECT_THROW(KalmanFilter(4, 2, 0, 0, DType::F64), std::exception);
}

// ========== KalmanFilter predict ==========

TEST_F(EstimationTestGPU, KalmanFilterPredict) {
  // KalmanFilter uses CPU internally (backend kernel), so create on CPU
  KalmanFilter kf(2, 1, 0, 1, DType::F64);

  double *xd = kf.x.data<double>();
  xd[0] = 1.0;
  xd[1] = 0.5;

  kf.predict();

  EXPECT_NEAR(kf.x.data<double>()[0], 1.0, 1e-10);
  EXPECT_NEAR(kf.x.data<double>()[1], 0.5, 1e-10);
}

TEST_F(EstimationTestGPU, KalmanFilterPredictWithF) {
  KalmanFilter kf(2, 1, 0, 1, DType::F64);

  double *xd = kf.x.data<double>();
  xd[0] = 1.0;
  xd[1] = 0.0;

  double *fd = kf.F.data<double>();
  fd[0] = 1.0;
  fd[1] = 1.0;
  fd[2] = 0.0;
  fd[3] = 1.0;

  kf.predict();

  EXPECT_NEAR(kf.x.data<double>()[0], 1.0, 1e-10);
  EXPECT_NEAR(kf.x.data<double>()[1], 0.0, 1e-10);
}

// ========== KalmanFilter update ==========

TEST_F(EstimationTestGPU, KalmanFilterUpdate) {
  KalmanFilter kf(2, 1, 0, 1, DType::F64);

  double *xd = kf.x.data<double>();
  xd[0] = 0.0;
  xd[1] = 0.0;

  double *pd = kf.P.data<double>();
  pd[0] = 1.0;
  pd[1] = 0.0;
  pd[2] = 0.0;
  pd[3] = 1.0;

  double *hd = kf.H.data<double>();
  hd[0] = 1.0;
  hd[1] = 0.0;

  double *rd = kf.R.data<double>();
  rd[0] = 0.1;

  std::vector<double> z_data = {5.0};
  Array z = to_array(z_data, Shape({1, 1, 1}), DType::F64, CPUPlace());

  kf.predict();
  kf.update(z);

  EXPECT_GT(kf.x.data<double>()[0], 0.0);
  EXPECT_TRUE(std::isfinite(kf.x.data<double>()[0]));
  EXPECT_TRUE(std::isfinite(kf.x.data<double>()[1]));
}

// ========== KalmanFilter F32 ==========

TEST_F(EstimationTestGPU, KalmanFilterF32) {
  KalmanFilter kf(2, 1, 0, 1, DType::F32);

  EXPECT_EQ(kf.x.dtype(), DType::F32);
  EXPECT_EQ(kf.P.dtype(), DType::F32);

  float *xd = kf.x.data<float>();
  xd[0] = 1.0f;
  xd[1] = 0.5f;

  kf.predict();

  EXPECT_NEAR(kf.x.data<float>()[0], 1.0f, 1e-5f);
}

// ========== KalmanFilter predict-update cycle ==========

TEST_F(EstimationTestGPU, KalmanFilterCycle) {
  KalmanFilter kf(2, 1, 0, 1, DType::F64);

  kf.x.data<double>()[0] = 0.0;
  kf.x.data<double>()[1] = 1.0;

  double *pd = kf.P.data<double>();
  pd[0] = 1.0;
  pd[1] = 0.0;
  pd[2] = 0.0;
  pd[3] = 1.0;

  double *hd = kf.H.data<double>();
  hd[0] = 1.0;
  hd[1] = 0.0;

  double *rd = kf.R.data<double>();
  rd[0] = 0.5;

  double *fd = kf.F.data<double>();
  fd[0] = 1.0;
  fd[1] = 1.0;
  fd[2] = 0.0;
  fd[3] = 1.0;

  for (int i = 0; i < 5; ++i) {
    kf.predict();
    std::vector<double> z_data = {static_cast<double>(i + 1)};
    Array z = to_array(z_data, Shape({1, 1, 1}), DType::F64, CPUPlace());
    kf.update(z);
  }

  EXPECT_TRUE(std::isfinite(kf.x.data<double>()[0]));
  EXPECT_TRUE(std::isfinite(kf.x.data<double>()[1]));
}

} // namespace
