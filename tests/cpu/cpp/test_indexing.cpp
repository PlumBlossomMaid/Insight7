// tests/test_indexing.cpp
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

class IndexingTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    set_device(CPUPlace());
  }
};

// ========== Helper functions ==========

static Array arange_1d(int64_t n, float start = 0.0f, float step = 1.0f) {
  std::vector<float> data(n);
  for (int64_t i = 0; i < n; ++i) {
    data[i] = start + i * step;
  }
  return to_array(data, Shape({n}));
}

static Array arange_2d(int rows, int cols, float start = 0.0f,
                       float step = 1.0f) {
  std::vector<float> data(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    data[i] = start + i * step;
  }
  return to_array(data, Shape({rows, cols}));
}

// ========== take tests ==========

TEST_F(IndexingTest, TakeFlattened) {
  Array x = arange_2d(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 2, 3, 7, 10});
  Array t = take(x, idx);

  EXPECT_EQ(t.numel(), 5);
  const float *data = t.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 2.0f, 1e-5);
  EXPECT_NEAR(data[2], 3.0f, 1e-5);
  EXPECT_NEAR(data[3], 7.0f, 1e-5);
  EXPECT_NEAR(data[4], 10.0f, 1e-5);
}

TEST_F(IndexingTest, TakeAlongAxis) {
  Array x = arange_2d(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 2, 1, 0, 2});
  Array t = take(x, idx, 0);

  EXPECT_EQ(t.shape(), Shape({5, 4}));
  const float *data = t.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5); // row0 from row0: [0,1,2,3]
  EXPECT_NEAR(data[4], 8.0f, 1e-5); // row1 from row2: [8,9,10,11]
  EXPECT_NEAR(data[8], 4.0f, 1e-5); // row2 from row1: [4,5,6,7]
}

TEST_F(IndexingTest, TakeAxis0) {
  Array x = arange_2d(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 2, 1, 0, 2});
  Array t = take(x, idx, 0);

  const float *data = t.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[4], 8.0f, 1e-5);
  EXPECT_NEAR(data[8], 4.0f, 1e-5);
  EXPECT_NEAR(data[12], 0.0f, 1e-5);
  EXPECT_NEAR(data[16], 8.0f, 1e-5);
}

// ========== take_along_axis tests ==========

TEST_F(IndexingTest, TakeAlongAxis0) {
  Array x = arange_2d(3, 4);
  std::vector<int64_t> idx_data = {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0};
  Array idx = to_array(idx_data, Shape({3, 4}));
  Array t = take_along_axis(x, idx, 0);

  EXPECT_EQ(t.shape(), Shape({3, 4}));
  const float *data = t.data<float>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(data[0 * 4 + j], 8.0f + j, 1e-5);
    EXPECT_NEAR(data[1 * 4 + j], 4.0f + j, 1e-5);
    EXPECT_NEAR(data[2 * 4 + j], 0.0f + j, 1e-5);
  }
}

// ========== put tests ==========

TEST_F(IndexingTest, PutFlattened) {
  Array x = arange_2d(3, 4);
  Array idx = to_array(std::vector<int64_t>{0, 5, 10});
  Array val = to_array(std::vector<float>{99.0f, 88.0f, 77.0f});
  Array p = put(x, idx, val);

  const float *data = p.data<float>();
  EXPECT_NEAR(data[0], 99.0f, 1e-5);
  EXPECT_NEAR(data[5], 88.0f, 1e-5);
  EXPECT_NEAR(data[10], 77.0f, 1e-5);
  EXPECT_NEAR(data[1], 1.0f, 1e-5);
}

// ========== put_along_axis tests ==========

TEST_F(IndexingTest, PutAlongAxis0) {
  Array x = arange_2d(3, 4);
  std::vector<int64_t> idx_data = {2, 1, 0};
  Array idx = to_array(idx_data, Shape({3, 1}));
  Array val = to_array(std::vector<float>{99.0f, 88.0f, 77.0f});
  Array p = put_along_axis(x, idx, val, 0);

  const float *data = p.data<float>();
  EXPECT_NEAR(data[2 * 4 + 0], 99.0f, 1e-5);
  EXPECT_NEAR(data[1 * 4 + 0], 88.0f, 1e-5);
  EXPECT_NEAR(data[0 * 4 + 0], 77.0f, 1e-5);
  EXPECT_NEAR(data[0 * 4 + 1], 1.0f, 1e-5);
  EXPECT_NEAR(data[1 * 4 + 2], 6.0f, 1e-5);
  EXPECT_NEAR(data[2 * 4 + 3], 11.0f, 1e-5);
}

// ========== gather / scatter tests ==========

TEST_F(IndexingTest, Gather) {
  Array x = arange_2d(3, 4);
  std::vector<int64_t> idx_data = {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0};
  Array idx = to_array(idx_data, Shape({3, 4}));
  Array g = gather(x, 0, idx);

  EXPECT_EQ(g.shape(), Shape({3, 4}));
  const float *data = g.data<float>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(data[0 * 4 + j], 8.0f + j, 1e-5);
    EXPECT_NEAR(data[1 * 4 + j], 4.0f + j, 1e-5);
    EXPECT_NEAR(data[2 * 4 + j], 0.0f + j, 1e-5);
  }
}

// ========== masked_select / compress tests ==========

TEST_F(IndexingTest, MaskedSelect) {
  Array x = arange_2d(3, 4);
  std::vector<bool> mask_data = {true,  false, true, false, false, true,
                                 false, true,  true, false, true,  false};
  Array mask = to_array(mask_data, Shape({3, 4}));
  Array m = masked_select(x, mask);

  EXPECT_EQ(m.numel(), 6);
  const float *data = m.data<float>();
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 2.0f, 1e-5);
  EXPECT_NEAR(data[2], 5.0f, 1e-5);
  EXPECT_NEAR(data[3], 7.0f, 1e-5);
  EXPECT_NEAR(data[4], 8.0f, 1e-5);
  EXPECT_NEAR(data[5], 10.0f, 1e-5);
}

TEST_F(IndexingTest, Compress) {
  Array x = arange_2d(3, 4);
  std::vector<bool> cond_data = {true, false, true};
  Array cond = to_array(cond_data);
  Array c = compress(x, cond, 0);

  EXPECT_EQ(c.shape(), Shape({2, 4}));
  const float *data = c.data<float>();
  for (int j = 0; j < 4; ++j) {
    EXPECT_NEAR(data[0 * 4 + j], static_cast<float>(j), 1e-5);
    EXPECT_NEAR(data[1 * 4 + j], 8.0f + j, 1e-5);
  }
}

// ========== where tests ==========

TEST_F(IndexingTest, Where3Arg) {
  Array cond = to_array(std::vector<bool>{true, false, true, false});
  Array x = arange_1d(4, 1.0f, 1.0f);
  Array y = arange_1d(4, 10.0f, 1.0f);
  Array w = where(cond, x, y);

  const float *data = w.data<float>();
  EXPECT_NEAR(data[0], 1.0f, 1e-5);
  EXPECT_NEAR(data[1], 11.0f, 1e-5);
  EXPECT_NEAR(data[2], 3.0f, 1e-5);
  EXPECT_NEAR(data[3], 13.0f, 1e-5);
}

TEST_F(IndexingTest, Where2Arg) {
  Array cond = to_array(std::vector<bool>{true, false, true, false});
  Array nz = where(cond);
  EXPECT_EQ(nz.numel(), 2);
  const int64_t *data = nz.data<int64_t>(); // data is a null pointer
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 2);
}

// ========== nonzero / flatnonzero tests ==========

TEST_F(IndexingTest, Nonzero) {
  std::vector<float> data = {0, 1, 0, 2, 0, 3, 0, 4};
  Array x = to_array(data).reshape(Shape({2, 4}));
  Array nz = nonzero(x);
  EXPECT_EQ(nz.shape().dim(0), 2);
  EXPECT_EQ(nz.shape().dim(1), 4);
  const int64_t *nz_data = nz.data<int64_t>();

  // Row indices (first dimension)
  EXPECT_EQ(nz_data[0], 0);
  EXPECT_EQ(nz_data[1], 0);
  EXPECT_EQ(nz_data[2], 1);
  EXPECT_EQ(nz_data[3], 1);

  // Column indices (second dimension)
  EXPECT_EQ(nz_data[4], 1);
  EXPECT_EQ(nz_data[5], 3);
  EXPECT_EQ(nz_data[6], 1);
  EXPECT_EQ(nz_data[7], 3);
}

TEST_F(IndexingTest, Flatnonzero) {
  std::vector<float> data = {0, 1, 0, 2, 0, 3, 0, 4};
  Array x = to_array(data).reshape(Shape({2, 4}));
  Array fnz = flatnonzero(x);

  EXPECT_EQ(fnz.numel(), 4);
  const int64_t *data_fnz = fnz.data<int64_t>();
  EXPECT_EQ(data_fnz[0], 1);
  EXPECT_EQ(data_fnz[1], 3);
  EXPECT_EQ(data_fnz[2], 5);
  EXPECT_EQ(data_fnz[3], 7);
}

// ========== argsort / sort tests ==========

TEST_F(IndexingTest, Argsort) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).reshape(Shape({2, 4}));
  Array idx = argsort(x, 1);

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

TEST_F(IndexingTest, Sort) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).reshape(Shape({2, 4}));
  Array s = sort(x, 1);

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

TEST_F(IndexingTest, TopkLargest) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data).reshape(Shape({2, 4}));
  auto [vals, idxs] = topk(x, 3, 1, true, true);

  const float *v_data = vals.data<float>();
  const int64_t *i_data = idxs.data<int64_t>();

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

TEST_F(IndexingTest, SearchsortedLeft) {
  Array x = to_array(std::vector<float>{1, 2, 2, 3, 3, 3, 4, 5});
  Array v = to_array(std::vector<float>{2, 3, 4, 6});
  Array left = searchsorted(x, v, "left");

  const int64_t *data = left.data<int64_t>();
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 3);
  EXPECT_EQ(data[2], 6);
  EXPECT_EQ(data[3], 8);
}

TEST_F(IndexingTest, SearchsortedRight) {
  Array x = to_array(std::vector<float>{1, 2, 2, 3, 3, 3, 4, 5});
  Array v = to_array(std::vector<float>{2, 3, 4, 6});
  Array right = searchsorted(x, v, "right");

  const int64_t *data = right.data<int64_t>();
  EXPECT_EQ(data[0], 3);
  EXPECT_EQ(data[1], 6);
  EXPECT_EQ(data[2], 7);
  EXPECT_EQ(data[3], 8);
}

// ========== unique tests ==========

TEST_F(IndexingTest, UniqueBasic) {
  std::vector<float> data = {3, 1, 2, 2, 3, 1, 4, 2};
  Array x = to_array(data);
  UniqueResult u = unique(x);

  EXPECT_EQ(u.unique.numel(), 4);
  const float *u_data = u.unique.data<float>();
  std::vector<float> expected = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(u_data[i], expected[i], 1e-5);
  }
}

TEST_F(IndexingTest, UniqueWithCounts) {
  std::vector<float> data = {3, 1, 2, 2, 3, 1, 4, 2};
  Array x = to_array(data);
  UniqueResult u = unique(x, true, true, true);

  const int64_t *cnt = u.counts.data<int64_t>();
  EXPECT_EQ(cnt[0], 2);
  EXPECT_EQ(cnt[1], 3);
  EXPECT_EQ(cnt[2], 2);
  EXPECT_EQ(cnt[3], 1);
}

// ========== lexsort tests ==========

TEST_F(IndexingTest, Lexsort) {
  std::vector<int> key0 = {1, 1, 0, 0};
  std::vector<int> key1 = {0, 1, 0, 1};
  std::vector<int> key2 = {25, 30, 20, 35};

  std::vector<int> stacked(3 * 4);
  for (int i = 0; i < 4; ++i) {
    stacked[0 * 4 + i] = key0[i];
    stacked[1 * 4 + i] = key1[i];
    stacked[2 * 4 + i] = key2[i];
  }
  Array keys = to_array(stacked, Shape({3, 4}));
  Array idx = lexsort(keys, 1);

  const int64_t *idx_data = idx.data<int64_t>();
  EXPECT_EQ(idx_data[0], 2);
  EXPECT_EQ(idx_data[1], 3);
  EXPECT_EQ(idx_data[2], 0);
  EXPECT_EQ(idx_data[3], 1);
}

// ========== indices tests ==========

TEST_F(IndexingTest, Indices2D) {
  Array idx = indices({2, 3});

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

TEST_F(IndexingTest, Ix_) {
  std::vector<Array> grids = ix_({to_array(std::vector<int64_t>{0, 2}),
                                  to_array(std::vector<int64_t>{1, 3, 5})});

  EXPECT_EQ(grids.size(), 2);
  EXPECT_EQ(grids[0].shape(), Shape({2, 1}));
  EXPECT_EQ(grids[1].shape(), Shape({1, 3}));

  const int64_t *d0 = grids[0].data<int64_t>();
  EXPECT_EQ(d0[0], 0);
  EXPECT_EQ(d0[1], 2);

  const int64_t *d1 = grids[1].data<int64_t>();
  EXPECT_EQ(d1[0], 1);
  EXPECT_EQ(d1[1], 3);
  EXPECT_EQ(d1[2], 5);
}

// ========== partition tests ==========

TEST_F(IndexingTest, Partition1D) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data);
  Array p = partition(x, 3);

  const float *p_data = p.data<float>();
  float kth_val = p_data[3];

  int less_count = 0;
  for (int i = 0; i < 8; ++i) {
    if (p_data[i] < kth_val)
      less_count++;
  }
  EXPECT_EQ(less_count, 3);
}

TEST_F(IndexingTest, Partition2D) {
  std::vector<float> data = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5};
  Array x = to_array(data, Shape({3, 4}));
  Array p = partition(x, 2, 1);

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

TEST_F(IndexingTest, Argpartition1D) {
  std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
  Array x = to_array(data);
  Array ap = argpartition(x, 3);

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

  Array values = take(x, ap);
  const float *val_data = values.data<float>();
  float kth_val = val_data[3];

  int less_count = 0;
  for (int i = 0; i < 8; ++i) {
    if (val_data[i] < kth_val)
      less_count++;
  }
  EXPECT_EQ(less_count, 3);
}

TEST_F(IndexingTest, Argpartition2D) {
  std::vector<float> data = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5};
  Array x = to_array(data, Shape({3, 4}));
  Array ap = argpartition(x, 2, 1);

  Array values = take_along_axis(x, ap, 1);
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

TEST_F(IndexingTest, InterpBasic) {
  Array xp = to_array({1.0, 2.0, 3.0, 4.0});
  Array fp = to_array({2.0, 4.0, 6.0, 8.0});
  Array x = to_array({1.5, 2.5, 3.5});
  Array y = interp(x, xp, fp);
  EXPECT_EQ(y.shape(), Shape({3}));
  const double *data = y.data<double>();
  EXPECT_NEAR(data[0], 3.0, 1e-6);
  EXPECT_NEAR(data[1], 5.0, 1e-6);
  EXPECT_NEAR(data[2], 7.0, 1e-6);
}

TEST_F(IndexingTest, InterpNonUniform) {
  Array xp = to_array({1.0, 2.0, 4.0, 8.0});
  Array fp = to_array({10.0, 20.0, 40.0, 80.0});
  Array x = to_array({3.0, 5.0, 6.0});
  Array y = interp(x, xp, fp);
  const double *data = y.data<double>();
  EXPECT_NEAR(data[0], 30.0, 1e-6);
  EXPECT_NEAR(data[1], 50.0, 1e-6);
  EXPECT_NEAR(data[2], 60.0, 1e-6);
}

TEST_F(IndexingTest, InterpBoundaryDefault) {
  Array xp = to_array({10.0, 20.0, 30.0, 40.0});
  Array fp = to_array({100.0, 200.0, 300.0, 400.0});
  Array x = to_array({5.0, 15.0, 25.0, 35.0, 45.0});
  Array y = interp(x, xp, fp);
  const double *data = y.data<double>();
  EXPECT_NEAR(data[0], 100.0, 1e-6);
  EXPECT_NEAR(data[1], 150.0, 1e-6);
  EXPECT_NEAR(data[2], 250.0, 1e-6);
  EXPECT_NEAR(data[3], 350.0, 1e-6);
  EXPECT_NEAR(data[4], 400.0, 1e-6);
}

TEST_F(IndexingTest, InterpBoundaryCustom) {
  Array xp = to_array({10.0, 20.0, 30.0, 40.0});
  Array fp = to_array({100.0, 200.0, 300.0, 400.0});
  Array x = to_array({5.0, 15.0, 25.0, 35.0, 45.0});
  Array y = interp(x, xp, fp, 0.0, 999.0);
  const double *data = y.data<double>();
  EXPECT_NEAR(data[0], 0.0, 1e-6);
  EXPECT_NEAR(data[1], 150.0, 1e-6);
  EXPECT_NEAR(data[2], 250.0, 1e-6);
  EXPECT_NEAR(data[3], 350.0, 1e-6);
  EXPECT_NEAR(data[4], 999.0, 1e-6);
}

TEST_F(IndexingTest, InterpScalar) {
  Array xp = to_array({0.0, 1.0, 2.0, 3.0, 4.0});
  Array fp = to_array({0.0, 1.0, 4.0, 9.0, 16.0});
  Array x = to_array({2.5});
  Array y = interp(x, xp, fp);
  EXPECT_NEAR(y.item<double>(), 6.5, 1e-6);
}

TEST_F(IndexingTest, InterpUnsortedXp) {
  Array xp = to_array({4.0, 1.0, 3.0, 2.0});
  Array fp = to_array({40.0, 10.0, 30.0, 20.0});
  Array x = to_array({1.5, 2.5, 3.5});
  Array y = interp(x, xp, fp);
  const double *data = y.data<double>();
  EXPECT_NEAR(data[0], 15.0, 1e-6);
  EXPECT_NEAR(data[1], 25.0, 1e-6);
  EXPECT_NEAR(data[2], 35.0, 1e-6);
}

// ========== String-based slicing (operator[]) ==========

TEST_F(IndexingTest, StringSlice1D) {
  Array a = arange(10, DType::F32, CPUPlace()); // [0,1,2,...,9]
  Array b = a[":5"];
  EXPECT_EQ(b.numel(), 5);
  EXPECT_FLOAT_EQ(b.at({0}).item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(b.at({4}).item<float>(), 4.0f);

  Array c = a["2:"];
  EXPECT_EQ(c.numel(), 8);
  EXPECT_FLOAT_EQ(c.at({0}).item<float>(), 2.0f);

  Array d = a["1:8:2"];
  EXPECT_EQ(d.numel(), 4);
  EXPECT_FLOAT_EQ(d.at({0}).item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(d.at({1}).item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(d.at({2}).item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(d.at({3}).item<float>(), 7.0f);
}

TEST_F(IndexingTest, StringSliceInteger) {
  Array a = arange(10, DType::F32, CPUPlace());
  Array b = a["3"];
  EXPECT_EQ(b.numel(), 1);
  EXPECT_FLOAT_EQ(b.item<float>(), 3.0f);

  Array c = a["-1"];
  EXPECT_FLOAT_EQ(c.item<float>(), 9.0f);
}

TEST_F(IndexingTest, StringSliceMultiDim) {
  // 4x5 array
  Array a = arange(20, DType::F32, CPUPlace()).reshape({4, 5});

  // arr[:, 1:-1] → 4x3
  Array b = a[":,1:-1"];
  EXPECT_EQ(b.shape().dim(0), 4);
  EXPECT_EQ(b.shape().dim(1), 3);
  EXPECT_FLOAT_EQ(b.data<float>()[0], 1.0f); // a[0,1]
  EXPECT_FLOAT_EQ(b.data<float>()[1], 2.0f); // a[0,2]
  EXPECT_FLOAT_EQ(b.data<float>()[2], 3.0f); // a[0,3]

  // arr[1:3, :] → 2x5
  Array c = a["1:3,:"];
  EXPECT_EQ(c.shape().dim(0), 2);
  EXPECT_EQ(c.shape().dim(1), 5);
  EXPECT_FLOAT_EQ(c.data<float>()[0], 5.0f); // a[1,0]
}

TEST_F(IndexingTest, StringSliceMultiDimIndexAndSlice) {
  // 4x5 array
  Array a = arange(20, DType::F32, CPUPlace()).reshape({4, 5});

  // arr[2, 1:-1] → 1D of size 3 (dim 0 squeezed)
  Array b = a["2,1:-1"];
  EXPECT_EQ(b.shape().ndim(), 1);
  EXPECT_EQ(b.numel(), 3);
  EXPECT_FLOAT_EQ(b.data<float>()[0], 11.0f); // a[2,1]
  EXPECT_FLOAT_EQ(b.data<float>()[1], 12.0f); // a[2,2]
  EXPECT_FLOAT_EQ(b.data<float>()[2], 13.0f); // a[2,3]

  // arr[1:3, 2] → 1D of size 2 (dim 1 squeezed)
  Array c = a["1:3,2"];
  EXPECT_EQ(c.shape().ndim(), 1);
  EXPECT_EQ(c.numel(), 2);
  EXPECT_FLOAT_EQ(c.data<float>()[0], 7.0f);  // a[1,2]
  EXPECT_FLOAT_EQ(c.data<float>()[1], 12.0f); // a[2,2]
}

TEST_F(IndexingTest, StringSliceMultiDimAllInt) {
  // 4x5 array
  Array a = arange(20, DType::F32, CPUPlace()).reshape({4, 5});

  // arr[1, 3] → scalar
  Array b = a["1,3"];
  EXPECT_EQ(b.numel(), 1);
  EXPECT_FLOAT_EQ(b.item<float>(), 8.0f); // a[1,3]
}

TEST_F(IndexingTest, StringSliceReverse) {
  Array a = arange(5, DType::F32, CPUPlace());
  Array b = a["::-1"];
  EXPECT_EQ(b.numel(), 5);
  EXPECT_FLOAT_EQ(b.at({0}).item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(b.at({4}).item<float>(), 0.0f);
}

TEST_F(IndexingTest, StringSlice3D) {
  // 2x5x4 array
  Array a = arange(40, DType::F32, CPUPlace()).reshape({2, 5, 4});

  // arr[:, 1:-1, ::2] → 2x3x2
  Array b = a[":,1:-1,::2"];
  EXPECT_EQ(b.shape().dim(0), 2);
  EXPECT_EQ(b.shape().dim(1), 3);
  EXPECT_EQ(b.shape().dim(2), 2);
  // a[0,1,0]=4, a[0,1,2]=6, a[0,2,0]=8, a[0,2,2]=10, a[0,3,0]=12, a[0,3,2]=14
  EXPECT_FLOAT_EQ(b.at({0, 0, 0}).item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(b.at({0, 0, 1}).item<float>(), 6.0f);
  EXPECT_FLOAT_EQ(b.at({0, 1, 0}).item<float>(), 8.0f);

  // arr[1, :, 2] → 1D of size 5 (dim0 and dim2 squeezed)
  Array c = a["1,:,2"];
  EXPECT_EQ(c.shape().ndim(), 1);
  EXPECT_EQ(c.numel(), 5);
  // a[1,0,2]=22, a[1,1,2]=26, a[1,2,2]=30
  EXPECT_FLOAT_EQ(c.at({0}).item<float>(), 22.0f);
  EXPECT_FLOAT_EQ(c.at({1}).item<float>(), 26.0f);
  EXPECT_FLOAT_EQ(c.at({2}).item<float>(), 30.0f);

  // arr[0, 1, 3] → scalar
  Array d = a["0,1,3"];
  EXPECT_EQ(d.numel(), 1);
  // a[0,1,3] = 0*20 + 1*4 + 3 = 7
  EXPECT_FLOAT_EQ(d.item<float>(), 7.0f);
}

TEST_F(IndexingTest, StringSliceInvalidThrows) {
  Array a = arange(6, DType::F32, CPUPlace()).reshape({2, 3});
  EXPECT_THROW(a["abc"], std::exception);
}

// ========== Partial indexing (NumPy-style a[1] on multi-dim) ==========

TEST_F(IndexingTest, PartialIndex2DRow) {
  // 4x5 array: a[1] should return row 1, shape (5,)
  Array a = arange(20, DType::F32, CPUPlace()).reshape({4, 5});

  // operator[](int64_t)
  Array row = a[1];
  EXPECT_EQ(row.shape().ndim(), 1);
  EXPECT_EQ(row.numel(), 5);
  EXPECT_FLOAT_EQ(row.at({0}).item<float>(), 5.0f); // a[1,0]
  EXPECT_FLOAT_EQ(row.at({1}).item<float>(), 6.0f); // a[1,1]
  EXPECT_FLOAT_EQ(row.at({4}).item<float>(), 9.0f); // a[1,4]

  // Negative index
  Array last = a[-1];
  EXPECT_EQ(last.numel(), 5);
  EXPECT_FLOAT_EQ(last.at({0}).item<float>(), 15.0f); // a[3,0]
}

TEST_F(IndexingTest, PartialIndex2DString) {
  // a["1"] on 2D should also return row
  Array a = arange(20, DType::F32, CPUPlace()).reshape({4, 5});

  Array row = a["1"];
  EXPECT_EQ(row.shape().ndim(), 1);
  EXPECT_EQ(row.numel(), 5);
  EXPECT_FLOAT_EQ(row.at({0}).item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(row.at({4}).item<float>(), 9.0f);
}

TEST_F(IndexingTest, PartialIndex3D) {
  // 2x3x4 array: a[1] should return shape (3,4)
  Array a = arange(24, DType::F32, CPUPlace()).reshape({2, 3, 4});

  Array slab = a[1];
  EXPECT_EQ(slab.shape().ndim(), 2);
  EXPECT_EQ(slab.shape().dim(0), 3);
  EXPECT_EQ(slab.shape().dim(1), 4);
  // a[1,0,0] = 12
  EXPECT_FLOAT_EQ(slab.at({0, 0}).item<float>(), 12.0f);
  // a[1,2,3] = 12 + 2*4 + 3 = 23
  EXPECT_FLOAT_EQ(slab.at({2, 3}).item<float>(), 23.0f);

  // a["0"] on 3D
  Array slab0 = a["0"];
  EXPECT_EQ(slab0.shape().ndim(), 2);
  EXPECT_FLOAT_EQ(slab0.at({0, 0}).item<float>(), 0.0f);
}

TEST_F(IndexingTest, PartialIndex3DTwoIndices) {
  // 2x3x4: a.at({1, 2}) should return shape (4,) — first two dims indexed
  Array a = arange(24, DType::F32, CPUPlace()).reshape({2, 3, 4});

  Array row = a.at({1, 2});
  EXPECT_EQ(row.shape().ndim(), 1);
  EXPECT_EQ(row.numel(), 4);
  // a[1,2,0] = 12 + 2*4 + 0 = 20
  EXPECT_FLOAT_EQ(row.at({0}).item<float>(), 20.0f);
  EXPECT_FLOAT_EQ(row.at({3}).item<float>(), 23.0f);
}

TEST_F(IndexingTest, PartialIndex1DStillScalar) {
  // 1D: a["3"] should still return scalar (not slice)
  Array a = arange(10, DType::F32, CPUPlace());
  Array val = a["3"];
  EXPECT_EQ(val.numel(), 1);
  EXPECT_FLOAT_EQ(val.item<float>(), 3.0f);

  Array val2 = a[5];
  EXPECT_EQ(val2.numel(), 1);
  EXPECT_FLOAT_EQ(val2.item<float>(), 5.0f);
}

TEST_F(IndexingTest, PartialIndexTooManyThrows) {
  Array a = arange(6, DType::F32, CPUPlace()).reshape({2, 3});
  // 3 indices on 2D array should throw
  EXPECT_THROW(a.at({0, 1, 2}), std::exception);
}
TEST_F(IndexingTest, StringIndexNegativeOneWithSlice) {
  Array a = arange_2d(4, 5);
  Array last_col = a[":,-1"];
  ASSERT_EQ(last_col.shape().ndim(), 1);
  ASSERT_EQ(last_col.numel(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(last_col.at({i}).item<float>(),
                    static_cast<float>(i * 5 + 4));
  }

  Array second_last = a[":,-2"];
  ASSERT_EQ(second_last.numel(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(second_last.at({i}).item<float>(),
                    static_cast<float>(i * 5 + 3));
  }
}

TEST_F(IndexingTest, SliceStartBeyondDimIsEmpty) {
  Array a = arange_1d(5);
  EXPECT_EQ(a["10:"].numel(), 0);
  EXPECT_EQ(a["5:"].numel(), 0);
  EXPECT_EQ(a["4:2"].numel(), 0);
  EXPECT_EQ(a.slice(0, 7, 9).numel(), 0);
}

TEST_F(IndexingTest, SliceOperatorPreservesReverseDefaults) {
  Array a = arange_1d(5);
  Array rev = a[Slice(std::nullopt, std::nullopt, -1)];
  ASSERT_EQ(rev.numel(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(rev.at({i}).item<float>(), static_cast<float>(4 - i));
  }

  Array rev_str = a["::-1"];
  ASSERT_EQ(rev_str.numel(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(rev_str.at({i}).item<float>(), static_cast<float>(4 - i));
  }
}

TEST_F(IndexingTest, ViewReshapeUpdatesKernelVisibleLayout) {
  // view() must refresh the C-ABI layout, otherwise kernels still see the
  // original dims and reduce along the wrong extents.
  Array a = arange_1d(6);
  Array v = a.view(Shape({3, 2}));
  ASSERT_EQ(v.shape().ndim(), 2);
  EXPECT_EQ(v.layout_ptr()->ndim, 2);
  EXPECT_EQ(v.layout_ptr()->dims[0], 3);
  EXPECT_EQ(v.layout_ptr()->dims[1], 2);
  EXPECT_EQ(v.layout_ptr()->strides[0], 2);
  EXPECT_EQ(v.layout_ptr()->strides[1], 1);
}
