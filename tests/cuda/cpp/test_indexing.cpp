// tests/cuda/test_indexing.cpp
#include "insight/ops/indexing.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <insight/core/array.h>
#include <insight/core/place.h>
#include <insight/core/shape.h>
#include <insight/init.h>
#include <insight/ops/creation.h>
#include <vector>

using namespace ins;

class IndexingTestGPU : public ::testing::Test {
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

// ========== Helper functions ==========

static Array arange_1d_gpu(int64_t n, float start = 0.0f, float step = 1.0f) {
  std::vector<float> data(n);
  for (int64_t i = 0; i < n; ++i) {
    data[i] = start + i * step;
  }
  return to_array(data, Shape({n})).to(GPUPlace(0));
}

static Array arange_2d_gpu(int rows, int cols, float start = 0.0f,
                           float step = 1.0f) {
  std::vector<float> data(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    data[i] = start + i * step;
  }
  return to_array(data, Shape({rows, cols})).to(GPUPlace(0));
}

// ========== take tests ==========

TEST_F(IndexingTestGPU, TakeFlattened) {
  Array x = arange_2d_gpu(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 2, 3, 7, 10}).to(GPUPlace(0));
  Array t = take(x, idx).to(CPUPlace());

  EXPECT_EQ(t.numel(), 5);
  const float *data = t.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 2.0f, 1e-5);
  EXPECT_NEAR(data[2], 3.0f, 1e-5);
  EXPECT_NEAR(data[3], 7.0f, 1e-5);
  EXPECT_NEAR(data[4], 10.0f, 1e-5);
}

TEST_F(IndexingTestGPU, TakeAlongAxis) {
  Array x = arange_2d_gpu(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 2, 1, 0, 2}).to(GPUPlace(0));
  Array t = take(x, idx, 0).to(CPUPlace());

  EXPECT_EQ(t.shape(), Shape({5, 4}));
  const float *data = t.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[4], 8.0f, 1e-5);
  EXPECT_NEAR(data[8], 4.0f, 1e-5);
}

TEST_F(IndexingTestGPU, TakeAxis0) {
  Array x = arange_2d_gpu(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 2, 1, 0, 2}).to(GPUPlace(0));
  Array t = take(x, idx, 0).to(CPUPlace());

  const float *data = t.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[4], 8.0f, 1e-5);
  EXPECT_NEAR(data[8], 4.0f, 1e-5);
  EXPECT_NEAR(data[12], 0.0f, 1e-5);
  EXPECT_NEAR(data[16], 8.0f, 1e-5);
}

// ========== take_along_axis tests ==========

TEST_F(IndexingTestGPU, TakeAlongAxis0) {
  Array x = arange_2d_gpu(3, 4);
  std::vector<int64_t> idx_data = {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0};
  Array idx = to_array(idx_data, Shape({3, 4})).to(GPUPlace(0));
  Array t = take_along_axis(x, idx, 0).to(CPUPlace());

  EXPECT_EQ(t.shape(), Shape({3, 4}));
  const float *data = t.data<float>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(data[0 * 4 + j], 8.0f + j, 1e-5);
    EXPECT_NEAR(data[1 * 4 + j], 4.0f + j, 1e-5);
    EXPECT_NEAR(data[2 * 4 + j], 0.0f + j, 1e-5);
  }
}

// ========== put tests ==========

TEST_F(IndexingTestGPU, PutFlattened) {
  Array x = arange_2d_gpu(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 5, 10}).to(GPUPlace(0));
  Array val = to_array(std::vector<float>{99.0f, 88.0f, 77.0f}).to(GPUPlace(0));
  Array p = put(x, idx, val).to(CPUPlace());

  const float *data = p.data<float>();
  EXPECT_NEAR(data[0], 99.0f, 1e-5);
  EXPECT_NEAR(data[5], 88.0f, 1e-5);
  EXPECT_NEAR(data[10], 77.0f, 1e-5);
  EXPECT_NEAR(data[1], 1.0f, 1e-5);
}

// ========== put_along_axis tests ==========

TEST_F(IndexingTestGPU, PutAlongAxis0) {
  Array x = arange_2d_gpu(3, 4);
  std::vector<int64_t> idx_data = {2, 1, 0};
  Array idx = to_array(idx_data, Shape({3, 1})).to(GPUPlace(0));
  Array val = to_array(std::vector<float>{99.0f, 88.0f, 77.0f}).to(GPUPlace(0));
  Array p = put_along_axis(x, idx, val, 0).to(CPUPlace());

  const float *data = p.data<float>();
  EXPECT_NEAR(data[2 * 4 + 0], 99.0f, 1e-5);
  EXPECT_NEAR(data[1 * 4 + 0], 88.0f, 1e-5);
  EXPECT_NEAR(data[0 * 4 + 0], 77.0f, 1e-5);
  EXPECT_NEAR(data[0 * 4 + 1], 1.0f, 1e-5);
  EXPECT_NEAR(data[1 * 4 + 2], 6.0f, 1e-5);
  EXPECT_NEAR(data[2 * 4 + 3], 11.0f, 1e-5);
}

// ========== gather / scatter tests ==========

TEST_F(IndexingTestGPU, Gather) {
  Array x = arange_2d_gpu(3, 4);
  std::vector<int64_t> idx_data = {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0};
  Array idx = to_array(idx_data, Shape({3, 4})).to(GPUPlace(0));
  Array g = gather(x, 0, idx).to(CPUPlace());

  EXPECT_EQ(g.shape(), Shape({3, 4}));
  const float *data = g.data<float>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(data[0 * 4 + j], 8.0f + j, 1e-5);
    EXPECT_NEAR(data[1 * 4 + j], 4.0f + j, 1e-5);
    EXPECT_NEAR(data[2 * 4 + j], 0.0f + j, 1e-5);
  }
}

// ========== masked_select / compress tests ==========

TEST_F(IndexingTestGPU, MaskedSelect) {
  Array x = arange_2d_gpu(3, 4);
  std::vector<float> mask_data = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                  0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  Array mask = to_array(mask_data, Shape({3, 4})).to(GPUPlace(0));
  Array m = masked_select(x, mask).to(CPUPlace());

  EXPECT_EQ(m.numel(), 6);
  const float *data = m.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 2.0f, 1e-5);
  EXPECT_NEAR(data[2], 5.0f, 1e-5);
  EXPECT_NEAR(data[3], 7.0f, 1e-5);
  EXPECT_NEAR(data[4], 8.0f, 1e-5);
  EXPECT_NEAR(data[5], 10.0f, 1e-5);
}

TEST_F(IndexingTestGPU, Compress) {
  Array x = arange_2d_gpu(3, 4);
  std::vector<float> cond_data = {1.0f, 0.0f, 1.0f};
  Array cond = to_array(cond_data).to(GPUPlace(0));
  Array c = compress(x, cond, 0).to(CPUPlace());

  EXPECT_EQ(c.shape(), Shape({2, 4}));
  const float *data = c.data<float>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(data[0 * 4 + j], static_cast<float>(j), 1e-5);
    EXPECT_NEAR(data[1 * 4 + j], 8.0f + j, 1e-5);
  }
}

// ========== where tests ==========

TEST_F(IndexingTestGPU, Where3Arg) {
  Array cond =
      to_array(std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f}).to(GPUPlace(0));
  Array x = arange_1d_gpu(4, 1.0f, 1.0f);
  Array y = arange_1d_gpu(4, 10.0f, 1.0f);
  Array w = where(cond, x, y).to(CPUPlace());

  const float *data = w.data<float>();
  EXPECT_NEAR(data[0], 1.0f, 1e-5);
  EXPECT_NEAR(data[1], 11.0f, 1e-5);
  EXPECT_NEAR(data[2], 3.0f, 1e-5);
  EXPECT_NEAR(data[3], 13.0f, 1e-5);
}

TEST_F(IndexingTestGPU, Where2Arg) {
  Array cond =
      to_array(std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f}).to(GPUPlace(0));
  Array nz = where(cond).to(CPUPlace());
  EXPECT_EQ(nz.numel(), 2);
  const int64_t *data = nz.data<int64_t>();
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 2);
}

// ========== nonzero / flatnonzero tests ==========

TEST_F(IndexingTestGPU, Nonzero) {
  std::vector<float> data = {0, 1, 0, 2, 0, 3, 0, 4};
  Array x = to_array(data).reshape(Shape({2, 4})).to(GPUPlace(0));
  Array nz = nonzero(x).to(CPUPlace());
  EXPECT_EQ(nz.shape().dim(0), 2);
  EXPECT_EQ(nz.shape().dim(1), 4);
  const int64_t *nz_data = nz.data<int64_t>();

  EXPECT_EQ(nz_data[0], 0);
  EXPECT_EQ(nz_data[1], 0);
  EXPECT_EQ(nz_data[2], 1);
  EXPECT_EQ(nz_data[3], 1);

  EXPECT_EQ(nz_data[4], 1);
  EXPECT_EQ(nz_data[5], 3);
  EXPECT_EQ(nz_data[6], 1);
  EXPECT_EQ(nz_data[7], 3);
}

TEST_F(IndexingTestGPU, Flatnonzero) {
  std::vector<float> data = {0, 1, 0, 2, 0, 3, 0, 4};
  Array x = to_array(data).reshape(Shape({2, 4})).to(GPUPlace(0));
  Array fnz = flatnonzero(x).to(CPUPlace());

  EXPECT_EQ(fnz.numel(), 4);
  const int64_t *data_fnz = fnz.data<int64_t>();
  EXPECT_EQ(data_fnz[0], 1);
  EXPECT_EQ(data_fnz[1], 3);
  EXPECT_EQ(data_fnz[2], 5);
  EXPECT_EQ(data_fnz[3], 7);
}

// ========== argsort / sort tests ==========

TEST_F(IndexingTestGPU, Argsort) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).reshape(Shape({2, 4})).to(GPUPlace(0));
  Array idx = argsort(x, 1).to(CPUPlace());

  const int64_t *idx_data = idx.data<int64_t>();
  EXPECT_EQ(idx_data[0], 1);
  EXPECT_EQ(idx_data[1], 3);
  EXPECT_EQ(idx_data[2], 0);
  EXPECT_EQ(idx_data[3], 2);
  EXPECT_EQ(idx_data[4], 2);
  EXPECT_EQ(idx_data[5], 0);
  EXPECT_EQ(idx_data[6], 3);
  EXPECT_EQ(idx_data[7], 1);
}

TEST_F(IndexingTestGPU, Sort) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).reshape(Shape({2, 4})).to(GPUPlace(0));
  Array s = sort(x, 1).to(CPUPlace());

  const float *s_data = s.data<float>();
  EXPECT_NEAR(s_data[0], 1.0f, 1e-5);
  EXPECT_NEAR(s_data[1], 1.0f, 1e-5);
  EXPECT_NEAR(s_data[2], 3.0f, 1e-5);
  EXPECT_NEAR(s_data[3], 4.0f, 1e-5);
  EXPECT_NEAR(s_data[4], 2.0f, 1e-5);
  EXPECT_NEAR(s_data[5], 5.0f, 1e-5);
  EXPECT_NEAR(s_data[6], 6.0f, 1e-5);
  EXPECT_NEAR(s_data[7], 9.0f, 1e-5);
}

// ========== topk tests ==========

TEST_F(IndexingTestGPU, TopkLargest) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).reshape(Shape({2, 4})).to(GPUPlace(0));
  auto [vals, idxs] = topk(x, 3, 1, true, true);
  Array vals_cpu = vals.to(CPUPlace());
  Array idxs_cpu = idxs.to(CPUPlace());

  const float *v_data = vals_cpu.data<float>();
  const int64_t *i_data = idxs_cpu.data<int64_t>();

  EXPECT_NEAR(v_data[0], 4.0f, 1e-5);
  EXPECT_NEAR(v_data[1], 3.0f, 1e-5);
  EXPECT_NEAR(v_data[2], 1.0f, 1e-5);
  EXPECT_NEAR(v_data[3], 9.0f, 1e-5);
  EXPECT_NEAR(v_data[4], 6.0f, 1e-5);
  EXPECT_NEAR(v_data[5], 5.0f, 1e-5);

  EXPECT_EQ(i_data[0], 2);
  EXPECT_EQ(i_data[1], 0);
  EXPECT_TRUE(i_data[2] == 1 || i_data[2] == 3);
  EXPECT_EQ(i_data[3], 1);
  EXPECT_EQ(i_data[4], 3);
  EXPECT_EQ(i_data[5], 0);
}

// ========== searchsorted tests ==========

TEST_F(IndexingTestGPU, SearchsortedLeft) {
  Array x =
      to_array(std::vector<float>{1, 2, 2, 3, 3, 3, 4, 5}).to(GPUPlace(0));
  Array v = to_array(std::vector<float>{2, 3, 4, 6}).to(GPUPlace(0));
  Array left = searchsorted(x, v, "left").to(CPUPlace());

  const int64_t *data = left.data<int64_t>();
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 3);
  EXPECT_EQ(data[2], 6);
  EXPECT_EQ(data[3], 8);
}

TEST_F(IndexingTestGPU, SearchsortedRight) {
  Array x =
      to_array(std::vector<float>{1, 2, 2, 3, 3, 3, 4, 5}).to(GPUPlace(0));
  Array v = to_array(std::vector<float>{2, 3, 4, 6}).to(GPUPlace(0));
  Array right = searchsorted(x, v, "right").to(CPUPlace());

  const int64_t *data = right.data<int64_t>();
  EXPECT_EQ(data[0], 3);
  EXPECT_EQ(data[1], 6);
  EXPECT_EQ(data[2], 7);
  EXPECT_EQ(data[3], 8);
}

// ========== unique tests ==========

TEST_F(IndexingTestGPU, UniqueBasic) {
  std::vector<float> data = {3, 1, 2, 2, 3, 1, 4, 2};
  Array x = to_array(data).to(GPUPlace(0));
  UniqueResult u = unique(x);

  EXPECT_EQ(u.unique.numel(), 4);
  Array cpu_u = u.unique.to(CPUPlace());
  const float *u_data = cpu_u.data<float>();
  std::vector<float> expected = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(u_data[i], expected[i], 1e-5);
  }
}

TEST_F(IndexingTestGPU, UniqueWithCounts) {
  std::vector<float> data = {3, 1, 2, 2, 3, 1, 4, 2};
  Array x = to_array(data).to(GPUPlace(0));
  UniqueResult u = unique(x, true, true, true);

  Array cpu_cnt = u.counts.to(CPUPlace());
  const int64_t *cnt = cpu_cnt.data<int64_t>();
  EXPECT_EQ(cnt[0], 2);
  EXPECT_EQ(cnt[1], 3);
  EXPECT_EQ(cnt[2], 2);
  EXPECT_EQ(cnt[3], 1);
}

// ========== lexsort tests ==========

TEST_F(IndexingTestGPU, Lexsort) {
  std::vector<int> key0 = {1, 1, 0, 0};
  std::vector<int> key1 = {0, 1, 0, 1};
  std::vector<int> key2 = {25, 30, 20, 35};

  std::vector<int> stacked(3 * 4);
  for (int i = 0; i < 4; ++i) {
    stacked[0 * 4 + i] = key0[i];
    stacked[1 * 4 + i] = key1[i];
    stacked[2 * 4 + i] = key2[i];
  }
  Array keys = to_array(stacked, Shape({3, 4})).to(GPUPlace(0));
  Array idx = lexsort(keys, 1).to(CPUPlace());

  const int64_t *idx_data = idx.data<int64_t>();
  EXPECT_EQ(idx_data[0], 2);
  EXPECT_EQ(idx_data[1], 3);
  EXPECT_EQ(idx_data[2], 0);
  EXPECT_EQ(idx_data[3], 1);
}

// ========== indices tests ==========

TEST_F(IndexingTestGPU, Indices2D) {
  Array idx = indices({2, 3}).to(CPUPlace());

  EXPECT_EQ(idx.shape(), Shape({2, 2, 3}));
  const int64_t *idata = idx.data<int64_t>();

  EXPECT_EQ(idata[0], 0);
  EXPECT_EQ(idata[1], 0);
  EXPECT_EQ(idata[2], 0);
  EXPECT_EQ(idata[3], 1);
  EXPECT_EQ(idata[4], 1);
  EXPECT_EQ(idata[5], 1);
  EXPECT_EQ(idata[6], 0);
  EXPECT_EQ(idata[7], 1);
  EXPECT_EQ(idata[8], 2);
  EXPECT_EQ(idata[9], 0);
  EXPECT_EQ(idata[10], 1);
  EXPECT_EQ(idata[11], 2);
}

// ========== ix_ tests ==========

TEST_F(IndexingTestGPU, Ix_) {
  std::vector<Array> grids =
      ix_({to_array(std::vector<int64_t>{0, 2}).to(GPUPlace(0)),
           to_array(std::vector<int64_t>{1, 3, 5}).to(GPUPlace(0))});

  EXPECT_EQ(grids.size(), 2);
  EXPECT_EQ(grids[0].shape(), Shape({2, 1}));
  EXPECT_EQ(grids[1].shape(), Shape({1, 3}));

  Array cpu_d0 = grids[0].to(CPUPlace());
  const int64_t *d0 = cpu_d0.data<int64_t>();
  EXPECT_EQ(d0[0], 0);
  EXPECT_EQ(d0[1], 2);

  Array cpu_d1 = grids[1].to(CPUPlace());
  const int64_t *d1 = cpu_d1.data<int64_t>();
  EXPECT_EQ(d1[0], 1);
  EXPECT_EQ(d1[1], 3);
  EXPECT_EQ(d1[2], 5);
}

// ========== partition tests ==========

TEST_F(IndexingTestGPU, Partition1D) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).to(GPUPlace(0));
  Array p = partition(x, 3).to(CPUPlace());

  const float *p_data = p.data<float>();
  float kth_val = p_data[3];

  int less_count = 0;
  for (int i = 0; i < 8; ++i) {
    if (p_data[i] < kth_val)
      less_count++;
  }
  EXPECT_EQ(less_count, 3);
}

TEST_F(IndexingTestGPU, Partition2D) {
  std::vector<float> data = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5};
  Array x = to_array(data, Shape({3, 4})).to(GPUPlace(0));
  Array p = partition(x, 2, 1).to(CPUPlace());

  const float *p_data = p.data<float>();
  for (int row = 0; row < 3; ++row) {
    float kth_val = p_data[row * 4 + 2];
    int less_count = 0;
    for (int col = 0; col < 4; ++col) {
      if (p_data[row * 4 + col] < kth_val)
        less_count++;
    }
    EXPECT_EQ(less_count, 2);
  }
}

// ========== argpartition tests ==========

TEST_F(IndexingTestGPU, Argpartition1D) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).to(GPUPlace(0));
  Array ap = argpartition(x, 3).to(CPUPlace());

  const int64_t *ap_data = ap.data<int64_t>();
  std::vector<bool> seen(8, false);
  for (int i = 0; i < 8; ++i) {
    EXPECT_GE(ap_data[i], 0);
    EXPECT_LT(ap_data[i], 8);
    seen[ap_data[i]] = true;
  }
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(seen[i]);
  }

  Array values = take(x.to(CPUPlace()), ap);
  const float *val_data = values.data<float>();
  float kth_val = val_data[3];

  int less_count = 0;
  for (int i = 0; i < 8; ++i) {
    if (val_data[i] < kth_val)
      less_count++;
  }
  EXPECT_EQ(less_count, 3);
}

TEST_F(IndexingTestGPU, Argpartition2D) {
  std::vector<float> data = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5};
  Array x = to_array(data, Shape({3, 4})).to(GPUPlace(0));
  Array ap = argpartition(x, 2, 1).to(CPUPlace());

  Array values = take_along_axis(x.to(CPUPlace()), ap, 1);
  const float *val_data = values.data<float>();
  for (int row = 0; row < 3; ++row) {
    float kth_val = val_data[row * 4 + 2];
    int less_count = 0;
    for (int col = 0; col < 4; ++col) {
      if (val_data[row * 4 + col] < kth_val)
        less_count++;
    }
    EXPECT_EQ(less_count, 2);
  }
}

// ========== interp tests ==========

TEST_F(IndexingTestGPU, InterpBasic) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(IndexingTestGPU, InterpNonUniform) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(IndexingTestGPU, InterpBoundaryDefault) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(IndexingTestGPU, InterpBoundaryCustom) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(IndexingTestGPU, InterpScalar) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(IndexingTestGPU, InterpUnsortedXp) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}
