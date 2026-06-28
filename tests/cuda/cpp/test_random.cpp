// tests/cuda/test_random.cpp
#include "insight/insight.h"
#include "insight/ops/random.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ins;

namespace {
template <typename T> void fill_sequential(Array &arr) {
  T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i)
    data[i] = static_cast<T>(i);
}

template <typename T>
void expect_float_equal_gpu(const Array &gpu_arr,
                            const std::vector<T> &expected, T tol = 1e-6) {
  Array cpu_arr = gpu_arr.to(CPUPlace());
  const T *data = cpu_arr.data<T>();
  for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol);
  }
}
} // namespace

class RandomTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
    seed(42);
  }
};

// ========== Seed Management ==========

TEST_F(RandomTestGPU, DifferentSeedsProduceDifferentSequences) {
  seed(123);
  Array a1 = rand({100}, DType::F32, GPUPlace(0));

  seed(456);
  Array a2 = rand({100}, DType::F32, GPUPlace(0));

  Array cpu_a1 = a1.to(CPUPlace());
  Array cpu_a2 = a2.to(CPUPlace());

  const float *a1_data = cpu_a1.data<float>();
  const float *a2_data = cpu_a2.data<float>();

  bool all_same = true;
  for (int i = 0; i < 100; ++i) {
    if (a1_data[i] != a2_data[i]) {
      all_same = false;
      break;
    }
  }
  EXPECT_FALSE(all_same);
}

// ========== rand ==========

TEST_F(RandomTestGPU, RandShape) {
  Array a = rand({3, 4, 5}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({3, 4, 5}));
  EXPECT_EQ(a.dtype(), DType::F32);
  EXPECT_EQ(a.place(), GPUPlace(0));
}

TEST_F(RandomTestGPU, RandValuesInRange) {
  Array a = rand({10000}, DType::F32, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], 0.0f);
    EXPECT_LT(data[i], 1.0f);
  }
}

TEST_F(RandomTestGPU, RandStatisticalMean) {
  Array a = rand({100000}, DType::F64, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const double *data = cpu_a.data<double>();

  double sum = 0.0;
  for (int i = 0; i < 100000; ++i) {
    sum += data[i];
  }
  double mean = sum / 100000.0;

  EXPECT_NEAR(mean, 0.5, 0.005);
}

TEST_F(RandomTestGPU, RandWithExplicitPlace) {
  Array a = rand({10, 10}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.place(), GPUPlace(0));
}

// ========== randn ==========

TEST_F(RandomTestGPU, RandnShape) {
  Array a = randn({3, 4}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({3, 4}));
  EXPECT_EQ(a.dtype(), DType::F32);
}

TEST_F(RandomTestGPU, RandnStatisticalMeanAndStd) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

// ========== randint ==========

TEST_F(RandomTestGPU, RandintShape) {
  Array a = randint(0, 10, {3, 4}, DType::I64, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({3, 4}));
  EXPECT_EQ(a.dtype(), DType::I64);
}

TEST_F(RandomTestGPU, RandintValuesInRange) {
  Array a = randint(5, 10, {10000}, DType::I32, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const int32_t *data = cpu_a.data<int32_t>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], 5);
    EXPECT_LT(data[i], 10);
  }
}

TEST_F(RandomTestGPU, RandintStatisticalMean) {
  seed(42);
  Array a = randint(0, 100, {50000}, DType::I64, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const int64_t *data = cpu_a.data<int64_t>();

  double sum = 0.0;
  for (int i = 0; i < 50000; ++i) {
    sum += static_cast<double>(data[i]);
  }
  double mean = sum / 50000.0;

  EXPECT_NEAR(mean, 49.5, 1);
}

// ========== normal ==========

TEST_F(RandomTestGPU, NormalShape) {
  Array a = normal(5.0, 2.0, {3, 4}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({3, 4}));
}

TEST_F(RandomTestGPU, NormalStatisticalMeanAndStd) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

// ========== uniform ==========

TEST_F(RandomTestGPU, UniformShape) {
  Array a = uniform(-3.0, 7.0, {3, 4}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({3, 4}));
}

TEST_F(RandomTestGPU, UniformValuesInRange) {
  Array a = uniform(-2.5, 3.5, {10000}, DType::F32, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], -2.5f);
    EXPECT_LT(data[i], 3.5f);
  }
}

TEST_F(RandomTestGPU, UniformStatisticalMean) {
  Array a = uniform(-10.0, 10.0, {100000}, DType::F64, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const double *data = cpu_a.data<double>();

  double sum = 0.0;
  for (int i = 0; i < 100000; ++i) {
    sum += data[i];
  }
  double mean = sum / 100000.0;

  EXPECT_NEAR(mean, 0.0, 0.05);
}

// ========== randperm ==========

TEST_F(RandomTestGPU, RandpermShape) {
  Array a = randperm(20, DType::I64, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({20}));
  EXPECT_EQ(a.dtype(), DType::I64);
}

TEST_F(RandomTestGPU, RandpermContainsAllNumbers) {
  Array a = randperm(15, DType::I32, GPUPlace(0));
  Array cpu_a = a.to(CPUPlace());
  const int32_t *data = cpu_a.data<int32_t>();
  std::vector<bool> seen(15, false);
  for (int i = 0; i < 15; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LT(data[i], 15);
    seen[data[i]] = true;
  }
  for (int i = 0; i < 15; ++i) {
    EXPECT_TRUE(seen[i]);
  }
}

// ========== like functions ==========

TEST_F(RandomTestGPU, RandLike) {
  Array original({5, 5}, DType::F32, GPUPlace(0));
  Array copy = rand_like(original);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), original.dtype());
  EXPECT_EQ(copy.place(), original.place());
}

TEST_F(RandomTestGPU, RandnLike) {
  Array original({5, 5}, DType::F64, GPUPlace(0));
  Array copy = randn_like(original);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), original.dtype());
}

TEST_F(RandomTestGPU, RandintLike) {
  Array original({5, 5}, DType::F64, GPUPlace(0));
  Array copy = randint_like(original, 0, 10);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), DType::I64);
}

// ========== Additional Distributions ==========

TEST_F(RandomTestGPU, ExponentialShape) {
  Array a = exponential(1.0, {100, 100}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTestGPU, ExponentialValuesPositive) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(RandomTestGPU, ExponentialStatisticalMean) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(RandomTestGPU, GammaShape) {
  Array a = gamma(2.0, 1.0, {100, 100}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTestGPU, GammaValuesPositive) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(RandomTestGPU, ChisquareShape) {
  Array a = chisquare(5.0, {100, 100}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTestGPU, ChisquareValuesPositive) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(RandomTestGPU, PoissonShape) {
  Array a = poisson(3.0, {100, 100}, DType::I64, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTestGPU, BinomialShape) {
  Array a = binomial(10, 0.5, {100, 100}, DType::I64, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTestGPU, BetaShape) {
  Array a = beta(2.0, 5.0, {100, 100}, DType::F32, GPUPlace(0));
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTestGPU, BetaValuesInRange) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}
