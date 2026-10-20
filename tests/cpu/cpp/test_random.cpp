// tests/cpu/test_random.cpp
#include "insight/insight.h"
#include "insight/ops/random.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class RandomTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    seed(42); // Set a fixed seed for reproducibility in tests
  }
};

// ========== Seed Management ==========

TEST_F(RandomTest, DifferentSeedsProduceDifferentSequences) {
  seed(123);
  Array a1 = rand({100});

  seed(456);
  Array a2 = rand({100});

  const float *a1_data = a1.data<float>();
  const float *a2_data = a2.data<float>();

  // Check that not all elements are the same
  bool all_same = true;
  for (int i = 0; i < 100; ++i) {
    if (a1_data[i] != a2_data[i]) {
      all_same = false;
      break;
    }
  }
  EXPECT_FALSE(all_same);
}

TEST_F(RandomTest, GetSeedReturnsWhatWasSet) {
  seed(1234567);
  EXPECT_EQ(get_seed(), 1234567u);

  seed(42);
  EXPECT_EQ(get_seed(), 42u);
}

TEST_F(RandomTest, SameSeedProducesSameSequence) {
  seed(42);
  Array a = rand({100});

  seed(42);
  Array b = rand({100});

  const float *a_data = a.data<float>();
  const float *b_data = b.data<float>();
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(a_data[i], b_data[i]) << "mismatch at index " << i;
  }
}

TEST_F(RandomTest, SameSeedProducesSameSequenceRandn) {
  seed(7);
  Array a = randn({256}, DType::F64);

  seed(7);
  Array b = randn({256}, DType::F64);

  const double *a_data = a.data<double>();
  const double *b_data = b.data<double>();
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(a_data[i], b_data[i]) << "mismatch at index " << i;
  }
}

// ========== rand ==========

TEST_F(RandomTest, RandShape) {
  Array a = rand({3, 4, 5});
  EXPECT_EQ(a.shape(), Shape({3, 4, 5}));
  EXPECT_EQ(a.dtype(), DType::F32);
}

TEST_F(RandomTest, RandValuesInRange) {
  Array a = rand({10000}, DType::F32);
  const float *data = a.data<float>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], 0.0f);
    EXPECT_LT(data[i], 1.0f);
  }
}

TEST_F(RandomTest, RandStatisticalMean) {
  // Uniform [0,1) has mean = 0.5, variance = 1/12 ≈ 0.08333
  Array a = rand({100000}, DType::F64);
  const double *data = a.data<double>();

  double sum = 0.0;
  for (int i = 0; i < 100000; ++i) {
    sum += data[i];
  }
  double mean = sum / 100000.0;

  // Allow 1% error (0.5 ± 0.005)
  EXPECT_NEAR(mean, 0.5, 0.005);
}

TEST_F(RandomTest, RandWithExplicitPlace) {
  Array a = rand({10, 10}, DType::F32, CPUPlace());
  EXPECT_EQ(a.place(), CPUPlace());
}

// ========== randn ==========

TEST_F(RandomTest, RandnShape) {
  Array a = randn({3, 4});
  EXPECT_EQ(a.shape(), Shape({3, 4}));
  EXPECT_EQ(a.dtype(), DType::F32);
}

TEST_F(RandomTest, RandnStatisticalMeanAndStd) {
  // Standard normal: mean=0, std=1
  Array a = randn({200000}, DType::F64);
  const double *data = a.data<double>();

  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < 200000; ++i) {
    sum += data[i];
    sum_sq += data[i] * data[i];
  }
  double mean = sum / 200000.0;
  double var = sum_sq / 200000.0 - mean * mean;
  double std = std::sqrt(var);

  // Allow 1% error
  EXPECT_NEAR(mean, 0.0, 0.01);
  EXPECT_NEAR(std, 1.0, 0.01);
}

// ========== randint ==========

TEST_F(RandomTest, RandintShape) {
  Array a = randint(0, 10, {3, 4});
  EXPECT_EQ(a.shape(), Shape({3, 4}));
  EXPECT_EQ(a.dtype(), DType::I64);
}

TEST_F(RandomTest, RandintValuesInRange) {
  Array a = randint(5, 10, {10000}, DType::I32);
  const int32_t *data = a.data<int32_t>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], 5);
    EXPECT_LT(data[i], 10);
  }
}

TEST_F(RandomTest, RandintStatisticalMean) {
  // Uniform integer [0, 100) has mean = 49.5
  seed(42);
  Array a = randint(0, 100, {50000}, DType::I64);
  const int64_t *data = a.data<int64_t>();

  double sum = 0.0;
  for (int i = 0; i < 50000; ++i) {
    sum += static_cast<double>(data[i]);
  }
  double mean = sum / 50000.0;

  EXPECT_NEAR(mean, 49.5, 1);
}

// ========== normal ==========

TEST_F(RandomTest, NormalShape) {
  Array a = normal(5.0, 2.0, {3, 4});
  EXPECT_EQ(a.shape(), Shape({3, 4}));
}

TEST_F(RandomTest, NormalStatisticalMeanAndStd) {
  // N(5, 2^2) -> mean=5, std=2
  Array a = normal(5.0, 2.0, {200000}, DType::F64);
  const double *data = a.data<double>();

  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < 200000; ++i) {
    sum += data[i];
    sum_sq += data[i] * data[i];
  }
  double mean = sum / 200000.0;
  double var = sum_sq / 200000.0 - mean * mean;
  double std = std::sqrt(var);

  EXPECT_NEAR(mean, 5.0, 0.02);
  EXPECT_NEAR(std, 2.0, 0.02);
}

// ========== uniform ==========

TEST_F(RandomTest, UniformShape) {
  Array a = uniform(-3.0, 7.0, {3, 4});
  EXPECT_EQ(a.shape(), Shape({3, 4}));
}

TEST_F(RandomTest, UniformValuesInRange) {
  Array a = uniform(-2.5, 3.5, {10000}, DType::F32);
  const float *data = a.data<float>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], -2.5f);
    EXPECT_LT(data[i], 3.5f);
  }
}

TEST_F(RandomTest, UniformStatisticalMean) {
  // Uniform [-10, 10] has mean = 0, variance = 100/3 ≈ 33.33
  Array a = uniform(-10.0, 10.0, {100000}, DType::F64);
  const double *data = a.data<double>();

  double sum = 0.0;
  for (int i = 0; i < 100000; ++i) {
    sum += data[i];
  }
  double mean = sum / 100000.0;

  EXPECT_NEAR(mean, 0.0, 0.05);
}

// ========== randperm ==========

TEST_F(RandomTest, RandpermShape) {
  Array a = randperm(20);
  EXPECT_EQ(a.shape(), Shape({20}));
  EXPECT_EQ(a.dtype(), DType::I64);
}

TEST_F(RandomTest, RandpermContainsAllNumbers) {
  Array a = randperm(15, DType::I32);
  const int32_t *data = a.data<int32_t>();
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

TEST_F(RandomTest, RandLike) {
  Array original({5, 5}, DType::F32);
  Array copy = rand_like(original);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), original.dtype());
  EXPECT_EQ(copy.place(), original.place());
}

TEST_F(RandomTest, RandnLike) {
  Array original({5, 5}, DType::F64);
  Array copy = randn_like(original);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), original.dtype());
}

TEST_F(RandomTest, RandintLike) {
  Array original({5, 5}, DType::F64);
  // randint_like always returns integer type, default I64
  Array copy = randint_like(original, 0, 10);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), DType::I64);
}

// ========== Additional Distributions ==========

TEST_F(RandomTest, ExponentialShape) {
  Array a = exponential(1.0, {100, 100});
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTest, ExponentialValuesPositive) {
  Array a = exponential(2.0, {5000}, DType::F64);
  const double *data = a.data<double>();
  for (int i = 0; i < 5000; ++i) {
    EXPECT_GT(data[i], 0.0);
  }
}

TEST_F(RandomTest, ExponentialStatisticalMean) {
  // Exponential with scale = 2.0 has mean = 2.0
  Array a = exponential(2.0, {100000}, DType::F64);
  const double *data = a.data<double>();

  double sum = 0.0;
  for (int i = 0; i < 100000; ++i) {
    sum += data[i];
  }
  double mean = sum / 100000.0;

  EXPECT_NEAR(mean, 2.0, 0.05);
}

TEST_F(RandomTest, GammaShape) {
  Array a = gamma(2.0, 1.0, {100, 100});
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTest, GammaValuesPositive) {
  Array a = gamma(1.5, 2.0, {5000}, DType::F64);
  const double *data = a.data<double>();
  for (int i = 0; i < 5000; ++i) {
    EXPECT_GT(data[i], 0.0);
  }
}

TEST_F(RandomTest, ChisquareShape) {
  Array a = chisquare(5.0, {100, 100});
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTest, ChisquareValuesPositive) {
  Array a = chisquare(3.0, {5000}, DType::F64);
  const double *data = a.data<double>();
  for (int i = 0; i < 5000; ++i) {
    EXPECT_GT(data[i], 0.0);
  }
}

TEST_F(RandomTest, PoissonShape) {
  Array a = poisson(3.0, {100, 100});
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTest, BinomialShape) {
  Array a = binomial(10, 0.5, {100, 100});
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTest, BetaShape) {
  Array a = beta(2.0, 5.0, {100, 100});
  EXPECT_EQ(a.shape(), Shape({100, 100}));
}

TEST_F(RandomTest, BetaValuesInRange) {
  Array a = beta(2.0, 5.0, {10000}, DType::F64);
  const double *data = a.data<double>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], 0.0);
    EXPECT_LE(data[i], 1.0);
  }
}

// ========== multinomial ==========

TEST_F(RandomTest, MultinomialShape) {
  Array probs = to_array(std::vector<double>{0.1, 0.2, 0.7});
  Array out = multinomial(probs, 10, /*replacement=*/true);
  EXPECT_EQ(out.shape(), Shape({10}));
  EXPECT_EQ(out.dtype(), DType::I64);
}

TEST_F(RandomTest, MultinomialValuesInRange) {
  Array probs = to_array(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
  seed(42);
  Array out = multinomial(probs, 10000, /*replacement=*/true);
  const int64_t *data = out.data<int64_t>();
  for (int i = 0; i < 10000; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LT(data[i], 5);
  }
}

TEST_F(RandomTest, MultinomialWeightedDominant) {
  // Category 2 has 90% of the mass.
  Array probs = to_array(std::vector<double>{0.05, 0.05, 0.9});
  seed(42);
  Array out = multinomial(probs, 20000, /*replacement=*/true);
  const int64_t *data = out.data<int64_t>();
  int64_t count2 = 0;
  for (int i = 0; i < 20000; ++i) {
    if (data[i] == 2)
      ++count2;
  }
  EXPECT_GT(count2, 16000);
}

TEST_F(RandomTest, MultinomialWithoutReplacementUnique) {
  Array probs = to_array(std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0});
  seed(42);
  Array out = multinomial(probs, 5, /*replacement=*/false);
  const int64_t *data = out.data<int64_t>();
  std::vector<bool> seen(5, false);
  for (int i = 0; i < 5; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LT(data[i], 5);
    EXPECT_FALSE(seen[data[i]]) << "duplicate index " << data[i];
    seen[data[i]] = true;
  }
}

// ========== choice ==========

TEST_F(RandomTest, ChoiceShapeAndDtype) {
  Array x = arange(10, DType::F32);
  Array out = choice(x, 5);
  EXPECT_EQ(out.shape(), Shape({5}));
  EXPECT_EQ(out.dtype(), DType::F32);
}

TEST_F(RandomTest, ChoiceValuesInRange) {
  Array x = arange(10, DType::I64);
  seed(42);
  Array out = choice(x, 5000, /*replace=*/true);
  const int64_t *data = out.data<int64_t>();
  for (int i = 0; i < 5000; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LT(data[i], 10);
  }
}

TEST_F(RandomTest, ChoiceWithP) {
  Array x = arange(4, DType::I64);
  Array p = to_array(std::vector<double>{0.05, 0.05, 0.05, 0.85});
  seed(42);
  Array out = choice(x, 20000, /*replace=*/true, p);
  const int64_t *data = out.data<int64_t>();
  int64_t count3 = 0;
  for (int i = 0; i < 20000; ++i) {
    if (data[i] == 3)
      ++count3;
  }
  EXPECT_GT(count3, 15000);
}

TEST_F(RandomTest, ChoiceWithoutReplacementUnique) {
  Array x = arange(8, DType::I64);
  seed(42);
  Array out = choice(x, 8, /*replace=*/false);
  const int64_t *data = out.data<int64_t>();
  std::vector<bool> seen(8, false);
  for (int i = 0; i < 8; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LT(data[i], 8);
    EXPECT_FALSE(seen[data[i]]) << "duplicate index " << data[i];
    seen[data[i]] = true;
  }
}

// ========== shuffle ==========

TEST_F(RandomTest, ShuffleShapeAndDtype) {
  Array x = arange(50, DType::F32);
  seed(42);
  Array out = shuffle(x);
  EXPECT_EQ(out.shape(), Shape({50}));
  EXPECT_EQ(out.dtype(), DType::F32);
}

TEST_F(RandomTest, ShufflePreservesSet) {
  Array x = arange(100, DType::I64);
  seed(42);
  Array out = shuffle(x);
  std::vector<int64_t> src(x.data<int64_t>(), x.data<int64_t>() + 100);
  std::vector<int64_t> dst(out.data<int64_t>(), out.data<int64_t>() + 100);
  std::sort(src.begin(), src.end());
  std::sort(dst.begin(), dst.end());
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(src[i], dst[i]);
  }
}

TEST_F(RandomTest, ShuffleChangesOrder) {
  Array x = arange(50, DType::I64);
  seed(42);
  Array out = shuffle(x);
  const int64_t *src = x.data<int64_t>();
  const int64_t *dst = out.data<int64_t>();
  int changed = 0;
  for (int i = 0; i < 50; ++i) {
    if (src[i] != dst[i])
      ++changed;
  }
  EXPECT_GT(changed, 0);
}