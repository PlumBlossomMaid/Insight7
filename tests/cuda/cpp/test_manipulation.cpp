// tests/cuda/test_manipulation.cpp
#include "insight/insight.h"
#include "insight/ops/manipulation.h"
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

class ManipulationTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
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
  if (active_gpu_backend_name() == "ixuca") GTEST_SKIP() << "IXUCA: no native FP64";
}

TEST_F(ManipulationTestGPU, Diff2ndOrder) {
  if (active_gpu_backend_name() == "ixuca") GTEST_SKIP() << "IXUCA: no native FP64";
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
  if (active_gpu_backend_name() == "ixuca") GTEST_SKIP() << "IXUCA: no native FP64";
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

// ========== ediff1d / meshgrid / trapz / gradient (composite, GPU) ==========

TEST_F(ManipulationTestGPU, Ediff1dBasic) {
  Array a = to_array({1.0, 2.0, 4.0, 7.0, 11.0}).to(GPUPlace(0));
  Array d = ediff1d(a).to(CPUPlace());
  EXPECT_EQ(d.shape(), Shape({4}));
  const double *data = d.data<double>();
  EXPECT_NEAR(data[0], 1.0, 1e-6);
  EXPECT_NEAR(data[1], 2.0, 1e-6);
  EXPECT_NEAR(data[2], 3.0, 1e-6);
  EXPECT_NEAR(data[3], 4.0, 1e-6);
}

TEST_F(ManipulationTestGPU, Ediff1dToBeginToEnd) {
  Array a = to_array({1.0, 2.0, 4.0, 7.0}).to(GPUPlace(0));
  Array begin = to_array({0.0}).to(GPUPlace(0));
  Array end = to_array({99.0}).to(GPUPlace(0));
  Array d = ediff1d(a, end, begin).to(CPUPlace());
  EXPECT_EQ(d.shape(), Shape({5}));
  const double *data = d.data<double>();
  EXPECT_NEAR(data[0], 0.0, 1e-6);   // to_begin
  EXPECT_NEAR(data[1], 1.0, 1e-6);   // 2-1
  EXPECT_NEAR(data[2], 2.0, 1e-6);   // 4-2
  EXPECT_NEAR(data[3], 3.0, 1e-6);   // 7-4
  EXPECT_NEAR(data[4], 99.0, 1e-6);  // to_end
}

TEST_F(ManipulationTestGPU, Ediff1dIntKeepsDtype) {
  Array a = to_array({1, 3, 5, 7, 9}).to(GPUPlace(0));
  Array d = ediff1d(a).to(CPUPlace());
  EXPECT_EQ(d.dtype(), DType::I32);
  const int32_t *data = d.data<int32_t>();
  EXPECT_EQ(data[0], 2);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 2);
  EXPECT_EQ(data[3], 2);
}

TEST_F(ManipulationTestGPU, Meshgrid2DXY) {
  Array x = to_array({1.0, 2.0, 3.0}).to(GPUPlace(0));
  Array y = to_array({4.0, 5.0}).to(GPUPlace(0));
  std::vector<Array> g = meshgrid({x, y}, "xy");

  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[0].shape(), Shape({2, 3}));
  EXPECT_EQ(g[1].shape(), Shape({2, 3}));

  Array X = g[0].to(CPUPlace()).contiguous();
  Array Y = g[1].to(CPUPlace()).contiguous();
  const double *xd = X.data<double>();
  const double *yd = Y.data<double>();
  EXPECT_NEAR(xd[0], 1.0, 1e-6); // X[0][0] = x[0]
  EXPECT_NEAR(xd[1], 2.0, 1e-6); // X[0][1] = x[1]
  EXPECT_NEAR(xd[3], 1.0, 1e-6); // X[1][0] = x[0]
  EXPECT_NEAR(yd[0], 4.0, 1e-6); // Y[0][0] = y[0]
  EXPECT_NEAR(yd[3], 5.0, 1e-6); // Y[1][0] = y[1]
}

TEST_F(ManipulationTestGPU, Meshgrid2DIJ) {
  Array x = to_array({1.0, 2.0, 3.0}).to(GPUPlace(0));
  Array y = to_array({4.0, 5.0}).to(GPUPlace(0));
  std::vector<Array> g = meshgrid({x, y}, "ij");

  EXPECT_EQ(g[0].shape(), Shape({3, 2}));
  EXPECT_EQ(g[1].shape(), Shape({3, 2}));

  Array X = g[0].to(CPUPlace()).contiguous();
  const double *xd = X.data<double>();
  EXPECT_NEAR(xd[0], 1.0, 1e-6);  // X[0][0] = x[0]
  EXPECT_NEAR(xd[2], 2.0, 1e-6);  // X[1][0] = x[1]
  EXPECT_NEAR(xd[4], 3.0, 1e-6);  // X[2][0] = x[2]
}

TEST_F(ManipulationTestGPU, Trapz1DUniform) {
  Array y = to_array({1.0, 2.0, 3.0}).to(GPUPlace(0));
  Array r = trapz(y).to(CPUPlace());
  EXPECT_NEAR(r.item<double>(), 4.0, 1e-6);
}

TEST_F(ManipulationTestGPU, Trapz2DAxis1) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Array m = to_array(data, Shape({2, 3})).to(GPUPlace(0));
  Array r = trapz(m, std::nullopt, 1.0f, 1).to(CPUPlace());
  EXPECT_EQ(r.shape(), Shape({2}));
  const float *rd = r.data<float>();
  EXPECT_NEAR(rd[0], 4.0f, 1e-5);
  EXPECT_NEAR(rd[1], 10.0f, 1e-5);
}

TEST_F(ManipulationTestGPU, TrapzWithX) {
  Array y = to_array({1.0, 2.0, 4.0}).to(GPUPlace(0));
  Array x = to_array({0.0, 0.5, 1.0}).to(GPUPlace(0));
  Array r = trapz(y, x).to(CPUPlace());
  EXPECT_NEAR(r.item<double>(), 2.25, 1e-6);
}

TEST_F(ManipulationTestGPU, Gradient1D) {
  Array f = to_array({1.0, 2.0, 4.0, 7.0, 11.0}).to(GPUPlace(0));
  Array g = gradient(f, 1.0, -1).to(CPUPlace());
  EXPECT_EQ(g.shape(), Shape({5}));
  const double *gd = g.data<double>();
  EXPECT_NEAR(gd[0], 1.0, 1e-6);  // (2-1)/1
  EXPECT_NEAR(gd[1], 1.5, 1e-6);  // (4-1)/2
  EXPECT_NEAR(gd[2], 2.5, 1e-6);  // (7-2)/2
  EXPECT_NEAR(gd[3], 3.5, 1e-6);  // (11-4)/2
  EXPECT_NEAR(gd[4], 4.0, 1e-6);  // (11-7)/1
}

TEST_F(ManipulationTestGPU, Gradient2DAxis) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Array m = to_array(data, Shape({2, 3})).to(GPUPlace(0));
  Array g0 = gradient(m, 1.0f, 0).to(CPUPlace());
  const float *g0d = g0.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(g0d[i], 3.0f, 1e-5);
  }
  Array g1 = gradient(m, 1.0f, 1).to(CPUPlace());
  const float *g1d = g1.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(g1d[i], 1.0f, 1e-5);
  }
}

TEST_F(ManipulationTestGPU, GradientAllAxes) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Array m = to_array(data, Shape({2, 3})).to(GPUPlace(0));
  std::vector<Array> gs = gradient(m, 1.0f);
  ASSERT_EQ(gs.size(), 2u);
  EXPECT_EQ(gs[0].shape(), Shape({2, 3}));
  EXPECT_EQ(gs[1].shape(), Shape({2, 3}));
}

// ========== repeat_interleave / one_hot / histogram (composite, GPU) ==========

TEST_F(ManipulationTestGPU, RepeatInterleaveFlatten) {
  Array a = to_array({1.0, 2.0, 3.0}).to(GPUPlace(0));
  Array r = repeat_interleave(a, 2).to(CPUPlace());
  EXPECT_EQ(r.shape(), Shape({6}));
  const double *data = r.data<double>();
  EXPECT_NEAR(data[0], 1.0, 1e-6);
  EXPECT_NEAR(data[1], 1.0, 1e-6);
  EXPECT_NEAR(data[2], 2.0, 1e-6);
  EXPECT_NEAR(data[3], 2.0, 1e-6);
  EXPECT_NEAR(data[4], 3.0, 1e-6);
  EXPECT_NEAR(data[5], 3.0, 1e-6);
}

TEST_F(ManipulationTestGPU, RepeatInterleaveAxis1) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Array m = to_array(data, Shape({2, 3})).to(GPUPlace(0));
  Array r = repeat_interleave(m, 2, 1).to(CPUPlace());
  EXPECT_EQ(r.shape(), Shape({2, 6}));
  const float *data_r = r.data<float>();
  EXPECT_FLOAT_EQ(data_r[0], 1.0f);
  EXPECT_FLOAT_EQ(data_r[1], 1.0f);
  EXPECT_FLOAT_EQ(data_r[2], 2.0f);
  EXPECT_FLOAT_EQ(data_r[3], 2.0f);
  EXPECT_FLOAT_EQ(data_r[4], 3.0f);
  EXPECT_FLOAT_EQ(data_r[5], 3.0f);
  EXPECT_FLOAT_EQ(data_r[6], 4.0f);
  EXPECT_FLOAT_EQ(data_r[7], 4.0f);
  EXPECT_FLOAT_EQ(data_r[8], 5.0f);
  EXPECT_FLOAT_EQ(data_r[9], 5.0f);
  EXPECT_FLOAT_EQ(data_r[10], 6.0f);
  EXPECT_FLOAT_EQ(data_r[11], 6.0f);
}

TEST_F(ManipulationTestGPU, OneHot1D) {
  Array idx = to_array({0, 1, 2}).to(GPUPlace(0));
  Array r = one_hot(idx, 3).to(CPUPlace());
  EXPECT_EQ(r.shape(), Shape({3, 3}));
  EXPECT_EQ(r.dtype(), DType::F32);
  const float *data = r.data<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[4], 1.0f);
  EXPECT_FLOAT_EQ(data[8], 1.0f);
  EXPECT_FLOAT_EQ(data[1] + data[2] + data[3] + data[5] + data[6] + data[7],
                  0.0f);
}

TEST_F(ManipulationTestGPU, OneHotInferClasses) {
  Array idx = to_array({0, 2}).to(GPUPlace(0));
  Array r = one_hot(idx).to(CPUPlace()); // num_classes = 3
  EXPECT_EQ(r.shape(), Shape({2, 3}));
}

TEST_F(ManipulationTestGPU, OneHotIntDtype) {
  Array idx = to_array({1}).to(GPUPlace(0));
  Array r = one_hot(idx, 2, DType::I64).to(CPUPlace());
  EXPECT_EQ(r.dtype(), DType::I64);
  const int64_t *data = r.data<int64_t>();
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 1);
}

TEST_F(ManipulationTestGPU, HistogramBasic) {
  Array x = to_array({0.0, 1.0, 2.0, 3.0, 4.0}).to(GPUPlace(0));
  std::vector<Array> res = histogram(x, 5);
  ASSERT_EQ(res.size(), 2u);
  Array hist = res[0].to(CPUPlace());
  Array edges = res[1].to(CPUPlace());
  EXPECT_EQ(hist.shape(), Shape({5}));
  EXPECT_EQ(edges.shape(), Shape({6}));
  const int64_t *hist_data = hist.data<int64_t>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(hist_data[i], 1);
  }
  const double *edges_data = edges.data<double>();
  // Default range is (min, max) = (0, 4); width = 0.8.
  EXPECT_NEAR(edges_data[0], 0.0, 1e-6);
  EXPECT_NEAR(edges_data[1], 0.8, 1e-6);
  EXPECT_NEAR(edges_data[5], 4.0, 1e-6);
}

TEST_F(ManipulationTestGPU, HistogramRange) {
  Array x = to_array({0.1, 0.3, 0.5, 0.7, 0.9}).to(GPUPlace(0));
  std::vector<Array> res = histogram(x, 5, std::make_pair(0.0, 1.0));
  ASSERT_EQ(res.size(), 2u);
  Array hist = res[0].to(CPUPlace());
  const int64_t *hist_data = hist.data<int64_t>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(hist_data[i], 1);
  }
}

TEST_F(ManipulationTestGPU, HistogramUniform) {
  Array x = to_array({2.0, 2.0, 2.0}).to(GPUPlace(0));
  std::vector<Array> res = histogram(x, 4, std::make_pair(0.0, 4.0));
  Array hist = res[0].to(CPUPlace());
  const int64_t *hist_data = hist.data<int64_t>();
  EXPECT_EQ(hist_data[2], 3);
  EXPECT_EQ(hist_data[0] + hist_data[1] + hist_data[3], 0);
}
