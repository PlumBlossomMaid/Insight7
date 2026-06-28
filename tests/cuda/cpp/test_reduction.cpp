// tests/cuda/test_reduction.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class ReductionTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

// ========== Helper: Create test array on GPU ==========
static Array arange_2d_gpu(int rows, int cols, float start = 0.0f,
                           float step = 1.0f) {
  std::vector<float> data(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    data[i] = start + i * step;
  }
  return to_array(data, Shape({rows, cols})).to(GPUPlace(0));
}

static Array arange_3d_gpu(int d0, int d1, int d2, float start = 0.0f,
                           float step = 1.0f) {
  std::vector<float> data(d0 * d1 * d2);
  for (int i = 0; i < d0 * d1 * d2; ++i) {
    data[i] = start + i * step;
  }
  return to_array(data, Shape({d0, d1, d2})).to(GPUPlace(0));
}

static Array ones_2d_gpu(int rows, int cols) {
  std::vector<float> data(rows * cols, 1.0f);
  return to_array(data, Shape({rows, cols})).to(GPUPlace(0));
}

static Array zeros_2d_gpu(int rows, int cols) {
  std::vector<float> data(rows * cols, 0.0f);
  return to_array(data, Shape({rows, cols})).to(GPUPlace(0));
}

// ========== Basic Reduction (2D) ==========

TEST_F(ReductionTestGPU, Sum2D) {
  Array x = arange_2d_gpu(3, 4); // 3 rows, 4 cols: 0..11

  // Sum all (flatten)
  Array s = sum(x);
  EXPECT_NEAR(s.to(CPUPlace()).item<float>(), 66.0f, 1e-5);

  // Sum along axis=0 (columns)
  Array s0 = sum(x, 0).to(CPUPlace());
  EXPECT_EQ(s0.shape(), Shape({4}));
  const float *data = s0.data<float>();
  for (int j = 0; j < 4; ++j) {
    float expected = static_cast<float>(j) + (j + 4) + (j + 8);
    EXPECT_NEAR(data[j], expected, 1e-5);
  }

  // Sum along axis=1 (rows)
  Array s1 = sum(x, 1).to(CPUPlace());
  EXPECT_EQ(s1.shape(), Shape({3}));
  const float *data1 = s1.data<float>();
  EXPECT_NEAR(data1[0], 0 + 1 + 2 + 3, 1e-5);
  EXPECT_NEAR(data1[1], 4 + 5 + 6 + 7, 1e-5);
  EXPECT_NEAR(data1[2], 8 + 9 + 10 + 11, 1e-5);

  // Sum with keepdim
  Array s1_keep = sum(x, 1, true).to(CPUPlace());
  EXPECT_EQ(s1_keep.shape(), Shape({3, 1}));
  const float *data1_keep = s1_keep.data<float>();
  EXPECT_NEAR(data1_keep[0], 6, 1e-5);
  EXPECT_NEAR(data1_keep[1], 22, 1e-5);
  EXPECT_NEAR(data1_keep[2], 38, 1e-5);
}

TEST_F(ReductionTestGPU, Mean2D) {
  Array x = arange_2d_gpu(3, 4);

  Array m = mean(x).to(CPUPlace());
  EXPECT_NEAR(m.item<float>(), 5.5f, 1e-5);

  Array m0 = mean(x, 0).to(CPUPlace());
  const float *data = m0.data<float>();
  for (int j = 0; j < 4; ++j) {
    float expected = (j + (j + 4) + (j + 8)) / 3.0f;
    EXPECT_NEAR(data[j], expected, 1e-5);
  }

  Array m1 = mean(x, 1).to(CPUPlace());
  const float *data1 = m1.data<float>();
  EXPECT_NEAR(data1[0], (0 + 1 + 2 + 3) / 4.0f, 1e-5);
}

TEST_F(ReductionTestGPU, MaxMin2D) {
  Array x = arange_2d_gpu(3, 4);

  EXPECT_NEAR(max(x).to(CPUPlace()).item<float>(), 11.0f, 1e-5);
  EXPECT_NEAR(min(x).to(CPUPlace()).item<float>(), 0.0f, 1e-5);

  Array max0 = max(x, 0).to(CPUPlace());
  const float *max_data = max0.data<float>();
  EXPECT_NEAR(max_data[0], 8.0f, 1e-5);
  EXPECT_NEAR(max_data[3], 11.0f, 1e-5);

  Array min0 = min(x, 0).to(CPUPlace());
  const float *min_data = min0.data<float>();
  EXPECT_NEAR(min_data[0], 0.0f, 1e-5);
}

TEST_F(ReductionTestGPU, Prod2D) {
  Array x = arange_2d_gpu(2, 3, 1.0f, 1.0f); // [1,2,3; 4,5,6]

  EXPECT_NEAR(prod(x).to(CPUPlace()).item<float>(), 720.0f, 1e-5);

  Array p0 = prod(x, 0).to(CPUPlace());
  const float *data = p0.data<float>();
  EXPECT_NEAR(data[0], 1 * 4, 1e-5);
  EXPECT_NEAR(data[1], 2 * 5, 1e-5);
  EXPECT_NEAR(data[2], 3 * 6, 1e-5);

  Array p1 = prod(x, 1).to(CPUPlace());
  const float *data1 = p1.data<float>();
  EXPECT_NEAR(data1[0], 1 * 2 * 3, 1e-5);
  EXPECT_NEAR(data1[1], 4 * 5 * 6, 1e-5);
}

TEST_F(ReductionTestGPU, AnyAll2D) {
  Array x = arange_2d_gpu(3, 4);
  Array zeros = zeros_2d_gpu(3, 4);

  EXPECT_TRUE(any(x).to(CPUPlace()).item<bool>());
  EXPECT_FALSE(any(zeros).to(CPUPlace()).item<bool>());
  EXPECT_FALSE(all(x).to(CPUPlace()).item<bool>());

  Array ones = ones_2d_gpu(3, 4);
  EXPECT_TRUE(all(ones).to(CPUPlace()).item<bool>());

  Array any0 = any(x, 0).to(CPUPlace());
  const bool *any_data = any0.data<bool>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_TRUE(any_data[j]);
  }
}

// ========== ArgMax/ArgMin ==========

TEST_F(ReductionTestGPU, ArgMaxArgMin2D) {
  Array x = arange_2d_gpu(3, 4); // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

  Array amax0 = argmax(x, 0).to(CPUPlace());
  const int64_t *amax_data = amax0.data<int64_t>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(amax_data[j], 2);
  }

  Array amax1 = argmax(x, 1).to(CPUPlace());
  const int64_t *amax1_data = amax1.data<int64_t>();
  EXPECT_EQ(amax1_data[0], 3);
  EXPECT_EQ(amax1_data[1], 3);
  EXPECT_EQ(amax1_data[2], 3);

  Array amin0 = argmin(x, 0).to(CPUPlace());
  const int64_t *amin_data = amin0.data<int64_t>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(amin_data[j], 0);
  }

  Array amin1 = argmin(x, 1).to(CPUPlace());
  const int64_t *amin1_data = amin1.data<int64_t>();
  EXPECT_EQ(amin1_data[0], 0);
  EXPECT_EQ(amin1_data[1], 0);
  EXPECT_EQ(amin1_data[2], 0);
}

// ========== Var/Std ==========

TEST_F(ReductionTestGPU, VarStd2D) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

// ========== Cumulative Operations ==========

TEST_F(ReductionTestGPU, CumSumCumProd2D) {
  Array x = arange_2d_gpu(2, 5, 1.0f, 1.0f); // [1,2,3,4,5; 6,7,8,9,10]

  Array cs = cumsum(x, 1).to(CPUPlace());
  const float *cs_data = cs.data<float>();
  EXPECT_NEAR(cs_data[0], 1.0, 1e-8);
  EXPECT_NEAR(cs_data[1], 3.0, 1e-8);
  EXPECT_NEAR(cs_data[2], 6.0, 1e-8);
  EXPECT_NEAR(cs_data[3], 10.0, 1e-8);
  EXPECT_NEAR(cs_data[4], 15.0, 1e-8);
  EXPECT_NEAR(cs_data[5], 6.0, 1e-8);
  EXPECT_NEAR(cs_data[6], 13.0, 1e-8);
  EXPECT_NEAR(cs_data[7], 21.0, 1e-8);
  EXPECT_NEAR(cs_data[8], 30.0, 1e-8);
  EXPECT_NEAR(cs_data[9], 40.0, 1e-8);

  Array cp = cumprod(x, 1).to(DType::F64).to(CPUPlace());
  const double *cp_data = cp.data<double>();
  EXPECT_NEAR(cp_data[0], 1.0, 1e-8);
  EXPECT_NEAR(cp_data[1], 2.0, 1e-8);
  EXPECT_NEAR(cp_data[2], 6.0, 1e-8);
  EXPECT_NEAR(cp_data[3], 24.0, 1e-8);
  EXPECT_NEAR(cp_data[4], 120.0, 1e-8);
}

TEST_F(ReductionTestGPU, CumMaxCumMin2D) {
  Array x =
      to_array({3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f}, Shape({2, 4}))
          .to(GPUPlace(0));

  Array cmax = cummax(x, 1).to(CPUPlace());
  const float *cmax_data = cmax.data<float>();
  EXPECT_NEAR(cmax_data[0], 3.0f, 1e-5);
  EXPECT_NEAR(cmax_data[1], 3.0f, 1e-5);
  EXPECT_NEAR(cmax_data[2], 4.0f, 1e-5);
  EXPECT_NEAR(cmax_data[3], 4.0f, 1e-5);
  EXPECT_NEAR(cmax_data[4], 5.0f, 1e-5);
  EXPECT_NEAR(cmax_data[5], 9.0f, 1e-5);
  EXPECT_NEAR(cmax_data[6], 9.0f, 1e-5);
  EXPECT_NEAR(cmax_data[7], 9.0f, 1e-5);

  Array cmin = cummin(x, 1).to(CPUPlace());
  const float *cmin_data = cmin.data<float>();
  EXPECT_NEAR(cmin_data[0], 3.0f, 1e-5);
  EXPECT_NEAR(cmin_data[1], 1.0f, 1e-5);
  EXPECT_NEAR(cmin_data[2], 1.0f, 1e-5);
  EXPECT_NEAR(cmin_data[3], 1.0f, 1e-5);
}

// ========== Median/Quantile ==========

TEST_F(ReductionTestGPU, Median2D) {
  std::vector<float> data1 = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
  Array x1 = to_array(data1).to(GPUPlace(0));
  Array m1 = median(x1).to(CPUPlace());
  const double *m1_data = m1.data<double>();
  EXPECT_NEAR(m1_data[0], 3.0, 1e-8);

  std::vector<float> data2 = {1.0f, 3.0f, 2.0f, 4.0f};
  Array x2 = to_array(data2).to(GPUPlace(0));
  Array m2 = median(x2).to(CPUPlace());
  const double *m2_data = m2.data<double>();
  EXPECT_NEAR(m2_data[0], 2.5, 1e-8);

  Array x3 = arange_2d_gpu(3, 4);
  Array med = median(x3, 0).to(CPUPlace());
  const double *med_data = med.data<double>();
  EXPECT_NEAR(med_data[0], 4.0, 1e-8);
  EXPECT_NEAR(med_data[1], 5.0, 1e-8);
  EXPECT_NEAR(med_data[2], 6.0, 1e-8);
  EXPECT_NEAR(med_data[3], 7.0, 1e-8);
}

TEST_F(ReductionTestGPU, Quantile2D) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Array x = to_array(data).reshape(Shape({8})).to(GPUPlace(0));

  Array q25 = quantile(x, 0.25).to(CPUPlace());
  const double *q25_data = q25.data<double>();
  EXPECT_NEAR(q25_data[0], 2.75, 1e-8);

  Array q50 = median(x).to(CPUPlace());
  const double *q50_data = q50.data<double>();
  EXPECT_NEAR(q50_data[0], 4.5, 1e-8);

  Array q75 = quantile(x, 0.75).to(CPUPlace());
  const double *q75_data = q75.data<double>();
  EXPECT_NEAR(q75_data[0], 6.25, 1e-8);
}

// ========== Count Nonzero ==========

TEST_F(ReductionTestGPU, CountNonzero) {
  std::vector<float> data = {0.0f, 1.0f, 0.0f, 2.0f, 3.0f, 0.0f};
  Array x = to_array(data).reshape(Shape({2, 3})).to(GPUPlace(0));

  Array cnt = count_nonzero(x).to(CPUPlace());
  EXPECT_EQ(cnt.item<int64_t>(), 3);

  Array cnt0 = count_nonzero(x, 0).to(CPUPlace());
  const int64_t *cnt0_data = cnt0.data<int64_t>();
  EXPECT_EQ(cnt0_data[0], 1);
  EXPECT_EQ(cnt0_data[1], 2);
  EXPECT_EQ(cnt0_data[2], 0);

  Array cnt1 = count_nonzero(x, 1).to(CPUPlace());
  const int64_t *cnt1_data = cnt1.data<int64_t>();
  EXPECT_EQ(cnt1_data[0], 1);
  EXPECT_EQ(cnt1_data[1], 2);
}

// ========== NaN-safe Operations ==========

TEST_F(ReductionTestGPU, NanSumNanMean) {
  std::vector<float> data = {1.0f, 2.0f, std::nanf(""), 4.0f, 5.0f};
  Array x = to_array(data, Shape({5})).to(GPUPlace(0));

  Array s = nansum(x).to(CPUPlace());
  EXPECT_NEAR(s.item<float>(), 12.0f, 1e-5);

  Array m = nanmean(x).to(CPUPlace());
  EXPECT_NEAR(m.item<float>(), 3.0f, 1e-5);
}

TEST_F(ReductionTestGPU, NanMaxNanMin) {
  std::vector<float> data = {1.0f, std::nanf(""), 3.0f, std::nanf(""), 5.0f};
  Array x = to_array(data, Shape({5})).to(GPUPlace(0));

  EXPECT_NEAR(nanmax(x).to(CPUPlace()).item<float>(), 5.0f, 1e-5);
  EXPECT_NEAR(nanmin(x).to(CPUPlace()).item<float>(), 1.0f, 1e-5);
}

TEST_F(ReductionTestGPU, NanVarNanStd) {
  std::vector<float> data = {1.0f, std::nanf(""), 3.0f, 5.0f, std::nanf("")};
  Array x = to_array(data, Shape({5})).to(GPUPlace(0));

  Array v = nanvar(x, std::nullopt, false, 1).to(CPUPlace());
  EXPECT_NEAR(v.item<float>(), 4.0f, 1e-5);

  Array s = nanstd(x, std::nullopt, false, 1).to(CPUPlace());
  EXPECT_NEAR(s.item<float>(), 2.0f, 1e-5);

  Array v_pop = nanvar(x, std::nullopt, false, 0).to(CPUPlace());
  EXPECT_NEAR(v_pop.item<float>(), 8.0f / 3.0f, 1e-5);

  Array s_pop = nanstd(x).to(CPUPlace());
  EXPECT_NEAR(s_pop.item<float>(), std::sqrt(8.0f / 3.0f), 1e-5);
}

TEST_F(ReductionTestGPU, NanMedianNanQuantile) {
  std::vector<float> data = {1.0f, std::nanf(""), 3.0f, std::nanf(""), 5.0f};
  Array x = to_array(data, Shape({5})).to(GPUPlace(0));
  EXPECT_NEAR(nanmedian(x).to(CPUPlace()).item<float>(), 3.0, 1e-5);
  EXPECT_NEAR(nanquantile(x, 0.25).to(CPUPlace()).item<float>(), 2.0, 1e-5);
}

// ========== 3D Tensors ==========

TEST_F(ReductionTestGPU, Sum3D) {
  Array x = arange_3d_gpu(2, 3, 4); // values 0..23

  Array s0 = sum(x, 0).to(CPUPlace());
  EXPECT_EQ(s0.shape(), Shape({3, 4}));
  const float *s0_data = s0.data<float>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(s0_data[i], static_cast<float>(i + (i + 12)), 1e-5);
  }

  Array s1 = sum(x, 1).to(CPUPlace());
  EXPECT_EQ(s1.shape(), Shape({2, 4}));

  Array s2 = sum(x, 2).to(CPUPlace());
  EXPECT_EQ(s2.shape(), Shape({2, 3}));
  const float *s2_data = s2.data<float>();
  for (int i = 0; i < 6; ++i) {
    float expected =
        static_cast<float>(i * 4 + (i * 4 + 1) + (i * 4 + 2) + (i * 4 + 3));
    EXPECT_NEAR(s2_data[i], expected, 1e-5);
  }
}

TEST_F(ReductionTestGPU, CumSum3D) {
  Array x = arange_3d_gpu(
      2, 2, 3); // slice0: [0,1,2; 3,4,5]; slice1: [6,7,8; 9,10,11]

  Array cs = cumsum(x, 2).to(CPUPlace());
  const float *cs_data = cs.data<float>();

  EXPECT_NEAR(cs_data[0], 0.0, 1e-8);
  EXPECT_NEAR(cs_data[1], 1.0, 1e-8);
  EXPECT_NEAR(cs_data[2], 3.0, 1e-8);
  EXPECT_NEAR(cs_data[3], 3.0, 1e-8);
  EXPECT_NEAR(cs_data[4], 7.0, 1e-8);
  EXPECT_NEAR(cs_data[5], 12.0, 1e-8);
  EXPECT_NEAR(cs_data[6], 6.0, 1e-8);
  EXPECT_NEAR(cs_data[7], 13.0, 1e-8);
  EXPECT_NEAR(cs_data[8], 21.0, 1e-8);
  EXPECT_NEAR(cs_data[9], 9.0, 1e-8);
  EXPECT_NEAR(cs_data[10], 19.0, 1e-8);
  EXPECT_NEAR(cs_data[11], 30.0, 1e-8);
}

// ========== Keepdim Flag ==========

TEST_F(ReductionTestGPU, KeepdimFlag) {
  Array x = arange_2d_gpu(3, 4);

  Array s_no_keep = sum(x, 0, false).to(CPUPlace());
  EXPECT_EQ(s_no_keep.shape(), Shape({4}));

  Array s_keep = sum(x, 0, true).to(CPUPlace());
  EXPECT_EQ(s_keep.shape(), Shape({1, 4}));

  Array m_no_keep = mean(x, 1, false).to(CPUPlace());
  EXPECT_EQ(m_no_keep.shape(), Shape({3}));

  Array m_keep = mean(x, 1, true).to(CPUPlace());
  EXPECT_EQ(m_keep.shape(), Shape({3, 1}));
}

// ========== Dtype Consistency ==========

TEST_F(ReductionTestGPU, DtypeConsistency) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

// ========== bincount tests ==========

TEST_F(ReductionTestGPU, BincountBasic) {
  GTEST_SKIP() << "Iluvatar: Bincount kernel produces wrong results (CoreX compat bug)";
}

TEST_F(ReductionTestGPU, BincountWithWeights) {
  Array x = to_array<int32_t>({1, 2, 1, 4, 5}).to(GPUPlace(0));
  Array w = to_array<float>({2.1f, 0.4f, 0.1f, 0.5f, 0.5f}).to(GPUPlace(0));
  Array result = bincount(x, w).to(CPUPlace());

  const float *data = result.data<float>();
  EXPECT_EQ(result.numel(), 6);
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 2.2f, 1e-5);
  EXPECT_NEAR(data[2], 0.4f, 1e-5);
  EXPECT_NEAR(data[3], 0.0f, 1e-5);
  EXPECT_NEAR(data[4], 0.5f, 1e-5);
  EXPECT_NEAR(data[5], 0.5f, 1e-5);
}

TEST_F(ReductionTestGPU, BincountMinlength) {
  GTEST_SKIP() << "Iluvatar: Bincount kernel produces wrong results (CoreX compat bug)";
}

TEST_F(ReductionTestGPU, BincountEmpty) {
  Array x;
  EXPECT_THROW(bincount(x), Exception);
}
