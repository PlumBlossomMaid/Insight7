// tests/cuda/test_manipulation.cpp
#include "insight/insight.h"
#include "insight/ops/manipulation.h"
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

class ManipulationTestGPU : public ::testing::Test {
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

template <typename T> void fill_sequential_gpu(Array &gpu_arr, T start = 0) {
  Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
  T *data = cpu_arr.data<T>();
  int64_t n = cpu_arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = static_cast<T>(start + i);
  }
  gpu_arr = cpu_arr.to(GPUPlace(0));
}

template <typename T>
void expect_equal_gpu(const Array &gpu_arr, const std::vector<T> &expected) {
  Array cpu_arr = gpu_arr.to(CPUPlace());
  ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = cpu_arr.data<T>();
  for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
  }
}

template <typename T>
void expect_float_equal_gpu(const Array &gpu_arr,
                            const std::vector<T> &expected, T tol = 1e-5) {
  Array cpu_arr = gpu_arr.to(CPUPlace());
  ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = cpu_arr.data<T>();
  for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol);
  }
}

// ========== reshape ==========

TEST_F(ManipulationTestGPU, Reshape2DTo3D) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = reshape(a, {3, 2});
  EXPECT_EQ(b.shape(), Shape({3, 2}));
  expect_float_equal_gpu<float>(b, {0, 1, 2, 3, 4, 5});
}

TEST_F(ManipulationTestGPU, ReshapeTo1D) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = reshape(a, {6});
  EXPECT_EQ(b.shape(), Shape({6}));
  expect_float_equal_gpu<float>(b, {0, 1, 2, 3, 4, 5});
}

// ========== flatten / ravel ==========

TEST_F(ManipulationTestGPU, Flatten) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = flatten(a);
  EXPECT_EQ(b.shape(), Shape({6}));
  expect_float_equal_gpu<float>(b, {0, 1, 2, 3, 4, 5});
}

TEST_F(ManipulationTestGPU, Ravel) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = ravel(a);
  EXPECT_EQ(b.shape(), Shape({6}));
  expect_float_equal_gpu<float>(b, {0, 1, 2, 3, 4, 5});
}

// ========== squeeze / unsqueeze ==========

TEST_F(ManipulationTestGPU, Squeeze) {
  Array a({1, 3, 1, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = squeeze(a);
  EXPECT_EQ(b.shape(), Shape({3, 4}));

  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0);
  EXPECT_FLOAT_EQ(data[3], 3);
  EXPECT_FLOAT_EQ(data[4], 4);
  EXPECT_FLOAT_EQ(data[11], 11);
}

TEST_F(ManipulationTestGPU, SqueezeAxis) {
  Array a({1, 3, 1, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = squeeze(a, 2); // squeeze axis 2 (size 1)
  EXPECT_EQ(b.shape(), Shape({1, 3, 4}));
}

TEST_F(ManipulationTestGPU, Unsqueeze) {
  Array a({3, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = unsqueeze(a, 0);
  EXPECT_EQ(b.shape(), Shape({1, 3, 4}));

  Array c = unsqueeze(a, -1);
  EXPECT_EQ(c.shape(), Shape({3, 4, 1}));
}

// ========== transpose / permute / swapaxes / moveaxis ==========

TEST_F(ManipulationTestGPU, Transpose) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = transpose(a).contiguous();
  EXPECT_EQ(b.shape(), Shape({3, 2}));
  expect_float_equal_gpu<float>(b, {0, 3, 1, 4, 2, 5});
}

TEST_F(ManipulationTestGPU, Permute) {
  Array a({2, 3, 4}, DType::F32, GPUPlace(0));

  Array b = permute(a, {2, 0, 1});
  EXPECT_EQ(b.shape(), Shape({4, 2, 3}));
}

TEST_F(ManipulationTestGPU, Swapaxes) {
  Array a({2, 3, 4}, DType::F32, GPUPlace(0));

  Array b = swapaxes(a, 0, 2);
  EXPECT_EQ(b.shape(), Shape({4, 3, 2}));
}

TEST_F(ManipulationTestGPU, Moveaxis) {
  Array a({2, 3, 4}, DType::F32, GPUPlace(0));

  Array b = moveaxis(a, 0, 2);
  EXPECT_EQ(b.shape(), Shape({3, 4, 2}));
}

// ========== flip / fliplr / flipud ==========

TEST_F(ManipulationTestGPU, Flip) {
  Array a({2, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = flip(a, 0);
  expect_float_equal_gpu<float>(b, {4, 5, 6, 7, 0, 1, 2, 3});

  Array c = flip(a, 1);
  expect_float_equal_gpu<float>(c, {3, 2, 1, 0, 7, 6, 5, 4});
}

TEST_F(ManipulationTestGPU, Fliplr) {
  Array a({2, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = fliplr(a);
  expect_float_equal_gpu<float>(b, {3, 2, 1, 0, 7, 6, 5, 4});
}

TEST_F(ManipulationTestGPU, Flipud) {
  Array a({2, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = flipud(a);
  expect_float_equal_gpu<float>(b, {4, 5, 6, 7, 0, 1, 2, 3});
}

// ========== rot90 ==========

TEST_F(ManipulationTestGPU, Rot90) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = rot90(a);
  EXPECT_EQ(b.shape(), Shape({3, 2}));
  expect_float_equal_gpu<float>(b, {2, 5, 1, 4, 0, 3});

  Array c = rot90(a, 2);
  EXPECT_EQ(c.shape(), Shape({2, 3}));
  expect_float_equal_gpu<float>(c, {5, 4, 3, 2, 1, 0});
}

// ========== concat ==========

TEST_F(ManipulationTestGPU, ConcatAxis0) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  Array b({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);
  fill_sequential_gpu<float>(b, 4);

  Array c = concat({a, b}, 0);
  EXPECT_EQ(c.shape(), Shape({4, 2}));
  expect_float_equal_gpu<float>(c, {0, 1, 2, 3, 4, 5, 6, 7});
}

TEST_F(ManipulationTestGPU, ConcatAxis1) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  Array b({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);
  fill_sequential_gpu<float>(b, 4);

  Array c = concat({a, b}, 1);
  EXPECT_EQ(c.shape(), Shape({2, 4}));
  expect_float_equal_gpu<float>(c, {0, 1, 4, 5, 2, 3, 6, 7});
}

TEST_F(ManipulationTestGPU, ConcatSingle) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array c = concat({a}, 0);
  EXPECT_EQ(c.shape(), Shape({2, 3}));
  expect_float_equal_gpu<float>(c, {0, 1, 2, 3, 4, 5});
}

// ========== stack ==========

TEST_F(ManipulationTestGPU, StackAxis0) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  Array b({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);
  fill_sequential_gpu<float>(b, 4);

  Array c = stack({a, b}, 0);
  EXPECT_EQ(c.shape(), Shape({2, 2, 2}));
}

TEST_F(ManipulationTestGPU, StackAxis1) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  Array b({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);
  fill_sequential_gpu<float>(b, 4);

  Array c = stack({a, b}, 1);
  EXPECT_EQ(c.shape(), Shape({2, 2, 2}));
}

// ========== split ==========

TEST_F(ManipulationTestGPU, SplitEqualParts) {
  Array a({4, 6}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  auto parts = split(a, 2, 0);
  ASSERT_EQ(parts.size(), 2);
  EXPECT_EQ(parts[0].shape(), Shape({2, 6}));
  EXPECT_EQ(parts[1].shape(), Shape({2, 6}));
}

TEST_F(ManipulationTestGPU, SplitIndices) {
  Array a({4, 6}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  auto parts = split(a, {2, 4}, 1);
  ASSERT_EQ(parts.size(), 3);
  EXPECT_EQ(parts[0].shape(), Shape({4, 2}));
  EXPECT_EQ(parts[1].shape(), Shape({4, 2}));
  EXPECT_EQ(parts[2].shape(), Shape({4, 2}));
}

// ========== repeat ==========

TEST_F(ManipulationTestGPU, RepeatAxis0) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = repeat(a, 2, 0);
  EXPECT_EQ(b.shape(), Shape({4, 3}));
  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0);
  EXPECT_FLOAT_EQ(data[3], 0);
  EXPECT_FLOAT_EQ(data[6], 3);
}

TEST_F(ManipulationTestGPU, RepeatAxis1) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = repeat(a, 2, 1);
  EXPECT_EQ(b.shape(), Shape({2, 6}));
}

// ========== tile ==========

TEST_F(ManipulationTestGPU, Tile2D) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = tile(a, {2, 2});
  EXPECT_EQ(b.shape(), Shape({4, 4}));
}

TEST_F(ManipulationTestGPU, TileWithLeadingOnes) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = tile(a, {1, 3});
  EXPECT_EQ(b.shape(), Shape({2, 6}));
}

// ========== pad ==========

TEST_F(ManipulationTestGPU, PadConstant) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = pad(a, {1, 1, 1, 1}, 0);
  EXPECT_EQ(b.shape(), Shape({4, 5}));

  std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                                 0, 3, 4, 5, 0, 0, 0, 0, 0, 0};
  expect_float_equal_gpu<float>(b, expected);
}

TEST_F(ManipulationTestGPU, PadC64) {
  std::vector<std::complex<double>> data = {{1, 2}, {3, 4}, {5, 6}};
  Array a = to_array(data, Shape({3}), DType::C64, CPUPlace()).to(GPUPlace(0));

  Array b = pad(a, {2, 2}, 0);
  EXPECT_EQ(b.shape(), Shape({7}));
  EXPECT_EQ(b.dtype(), DType::C64);

  Array b_cpu = b.to(CPUPlace());
  const std::complex<double> *bd =
      reinterpret_cast<const std::complex<double> *>(b_cpu.data<char>());
  EXPECT_EQ(bd[0], std::complex<double>(0, 0));
  EXPECT_EQ(bd[2], std::complex<double>(1, 2));
  EXPECT_EQ(bd[3], std::complex<double>(3, 4));
  EXPECT_EQ(bd[4], std::complex<double>(5, 6));
  EXPECT_EQ(bd[6], std::complex<double>(0, 0));
}

TEST_F(ManipulationTestGPU, PadC32) {
  std::vector<std::complex<float>> data = {{1, 2}, {3, 4}};
  Array a = to_array(data, Shape({2}), DType::C32, CPUPlace()).to(GPUPlace(0));

  Array b = pad(a, {1, 1}, 0);
  EXPECT_EQ(b.shape(), Shape({4}));
  EXPECT_EQ(b.dtype(), DType::C32);

  Array b_cpu = b.to(CPUPlace());
  const std::complex<float> *bd =
      reinterpret_cast<const std::complex<float> *>(b_cpu.data<char>());
  EXPECT_EQ(bd[0], std::complex<float>(0, 0));
  EXPECT_EQ(bd[1], std::complex<float>(1, 2));
  EXPECT_EQ(bd[2], std::complex<float>(3, 4));
  EXPECT_EQ(bd[3], std::complex<float>(0, 0));
}

// ========== roll ==========

TEST_F(ManipulationTestGPU, RollAlongAxis) {
  Array a({2, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0); // [[0,1,2,3], [4,5,6,7]]

  Array b = roll(a, 1, 0);
  std::vector<float> expected = {4, 5, 6, 7, 0, 1, 2, 3};
  expect_float_equal_gpu<float>(b, expected);

  Array c = roll(a, 2, 0);
  expect_float_equal_gpu<float>(c, {0, 1, 2, 3, 4, 5, 6, 7});
}

TEST_F(ManipulationTestGPU, RollFlatten) {
  Array a({2, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = roll(a, 3);
  EXPECT_EQ(b.shape(), Shape({2, 4}));
}

// ========== diag ==========

TEST_F(ManipulationTestGPU, DiagExtract) {
  Array a({3, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = diag(a);
  EXPECT_EQ(b.shape(), Shape({3}));
  expect_float_equal_gpu<float>(b, {0, 4, 8});

  Array c = diag(a, 1);
  EXPECT_EQ(c.shape(), Shape({2}));
  expect_float_equal_gpu<float>(c, {1, 5});
}

TEST_F(ManipulationTestGPU, DiagConstruct) {
  Array a({3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = diag(a);
  EXPECT_EQ(b.shape(), Shape({3, 3}));
  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0);
  EXPECT_FLOAT_EQ(data[4], 1);
  EXPECT_FLOAT_EQ(data[8], 2);
}

// ========== tril / triu ==========

TEST_F(ManipulationTestGPU, Tril) {
  Array a({3, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = tril(a);
  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0);
  EXPECT_FLOAT_EQ(data[1], 0);
  EXPECT_FLOAT_EQ(data[3], 3);
  EXPECT_FLOAT_EQ(data[4], 4);
  EXPECT_FLOAT_EQ(data[8], 8);
}

TEST_F(ManipulationTestGPU, TrilWithK) {
  Array a({3, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = tril(a, 1);
  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_FLOAT_EQ(data[1], 1);
  EXPECT_FLOAT_EQ(data[5], 5);
}

TEST_F(ManipulationTestGPU, Triu) {
  Array a({3, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  Array b = triu(a);
  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0);
  EXPECT_FLOAT_EQ(data[1], 1);
  EXPECT_FLOAT_EQ(data[3], 0);
  EXPECT_FLOAT_EQ(data[4], 4);
  EXPECT_FLOAT_EQ(data[8], 8);
}

// ========== vstack / hstack ==========

TEST_F(ManipulationTestGPU, Vstack) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  Array b({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);
  fill_sequential_gpu<float>(b, 4);

  Array c = vstack({a, b});
  EXPECT_EQ(c.shape(), Shape({4, 2}));
  expect_float_equal_gpu<float>(c, {0, 1, 2, 3, 4, 5, 6, 7});
}

TEST_F(ManipulationTestGPU, Hstack) {
  Array a({2, 2}, DType::F32, GPUPlace(0));
  Array b({2, 2}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);
  fill_sequential_gpu<float>(b, 4);

  Array c = hstack({a, b});
  EXPECT_EQ(c.shape(), Shape({2, 4}));
  expect_float_equal_gpu<float>(c, {0, 1, 4, 5, 2, 3, 6, 7});
}

// ========== diff tests ==========

TEST_F(ManipulationTestGPU, Diff1DBasic) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(ManipulationTestGPU, Diff2ndOrder) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(ManipulationTestGPU, DiffInt) {
  Array a = to_array({1, 3, 5, 7, 9}).to(GPUPlace(0));
  Array d = diff(a).to(CPUPlace());
  const int32_t *data = d.data<int32_t>();
  EXPECT_EQ(data[0], 2);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 2);
  EXPECT_EQ(data[3], 2);
}

TEST_F(ManipulationTestGPU, DiffWithNegativeAxis) {
  GTEST_SKIP() << "Iluvatar: no native FP64";
}

TEST_F(ManipulationTestGPU, Diff2DAxis0) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  Array m = to_array(data, Shape({3, 4})).to(GPUPlace(0));
  Array d = diff(m, 1, 0).to(CPUPlace());
  EXPECT_EQ(d.shape(), Shape({2, 4}));
  const float *d_data = d.data<float>();
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(d_data[i], 4.0f);
  }
}

TEST_F(ManipulationTestGPU, Diff2DAxis1) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  Array m = to_array(data, Shape({3, 4})).to(GPUPlace(0));
  Array d = diff(m, 1, 1).to(CPUPlace());
  EXPECT_EQ(d.shape(), Shape({3, 3}));
  const float *d_data = d.data<float>();
  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(d_data[i], 1.0f);
  }
}

// ========== In-place assignment: fill_, copy_from_, operator=, operator[]
// ==========

TEST_F(ManipulationTestGPU, FillScalar) {
  Array a({3, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  a.fill_(5.0);

  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  for (int64_t i = 0; i < cpu_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(data[i], 5.0f);
  }
}

TEST_F(ManipulationTestGPU, FillViaOperator) {
  Array a({3, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  a = 7.0;

  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  for (int64_t i = 0; i < cpu_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(data[i], 7.0f);
  }
}

TEST_F(ManipulationTestGPU, CopyFromArray) {
  Array src({3, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(src, 10);

  Array dst({3, 4}, DType::F32, GPUPlace(0));
  dst.fill_(0.0);

  dst.copy_from_(src);

  Array cpu_dst = dst.to(CPUPlace());
  const float *data = cpu_dst.data<float>();
  for (int64_t i = 0; i < cpu_dst.numel(); ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(10 + i));
  }
  // src unchanged
  Array cpu_src = src.to(CPUPlace());
  const float *src_data = cpu_src.data<float>();
  for (int64_t i = 0; i < cpu_src.numel(); ++i) {
    EXPECT_FLOAT_EQ(src_data[i], static_cast<float>(10 + i));
  }
}

TEST_F(ManipulationTestGPU, SliceAssignScalar) {
  Array a({6}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0); // [0, 1, 2, 3, 4, 5]

  a[":"] = 3.0;

  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  for (int64_t i = 0; i < cpu_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(data[i], 3.0f);
  }
}

TEST_F(ManipulationTestGPU, SliceAssignArray) {
  Array a({6}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0); // [0, 1, 2, 3, 4, 5]

  Array src({6}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(src, 100); // [100, 101, 102, 103, 104, 105]

  a[":"] = src;

  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  for (int64_t i = 0; i < cpu_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(100 + i));
  }
}

TEST_F(ManipulationTestGPU, SliceAssign2D) {
  // 4x4 array filled sequentially on GPU
  Array a({4, 4}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  // 2x2 source filled with 99 on GPU
  Array src({2, 2}, DType::F32, GPUPlace(0));
  src.fill_(99.0);

  // Assign to rows 0..1, cols 1..2
  a["0:2,1:3"] = src;

  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  // Row 0: [0, 99, 99, 3]
  EXPECT_FLOAT_EQ(data[0], 0.0f);
  EXPECT_FLOAT_EQ(data[1], 99.0f);
  EXPECT_FLOAT_EQ(data[2], 99.0f);
  EXPECT_FLOAT_EQ(data[3], 3.0f);
  // Row 1: [4, 99, 99, 7]
  EXPECT_FLOAT_EQ(data[4], 4.0f);
  EXPECT_FLOAT_EQ(data[5], 99.0f);
  EXPECT_FLOAT_EQ(data[6], 99.0f);
  EXPECT_FLOAT_EQ(data[7], 7.0f);
  // Row 2 unchanged: [8, 9, 10, 11]
  EXPECT_FLOAT_EQ(data[8], 8.0f);
  EXPECT_FLOAT_EQ(data[9], 9.0f);
  EXPECT_FLOAT_EQ(data[10], 10.0f);
  EXPECT_FLOAT_EQ(data[11], 11.0f);
  // Row 3 unchanged: [12, 13, 14, 15]
  EXPECT_FLOAT_EQ(data[12], 12.0f);
  EXPECT_FLOAT_EQ(data[13], 13.0f);
  EXPECT_FLOAT_EQ(data[14], 14.0f);
  EXPECT_FLOAT_EQ(data[15], 15.0f);
}

TEST_F(ManipulationTestGPU, ScalarIndexAssign) {
  // 3x3 array filled sequentially on GPU
  Array a({3, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a, 0);

  // Get a scalar view at position (1,1) — value is 4
  Array view = a.at({1, 1});
  EXPECT_EQ(view.numel(), 1);

  // Assign through the view — should modify the parent
  view = 99.0;

  // Parent array should have 99 at (1,1)
  Array cpu_a = a.to(CPUPlace());
  const float *data = cpu_a.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0.0f);
  EXPECT_FLOAT_EQ(data[1], 1.0f);
  EXPECT_FLOAT_EQ(data[4], 99.0f); // (1,1) = row*3 + col = 1*3+1 = 4
  EXPECT_FLOAT_EQ(data[8], 8.0f);
}
