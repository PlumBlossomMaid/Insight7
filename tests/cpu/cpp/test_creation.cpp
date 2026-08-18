// tests/cpu/test_creation.cpp
#include "insight/insight.h"
#include "insight/ops/creation.h"
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

class CreationTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

// ========== zeros / ones / full ==========

TEST_F(CreationTest, Zeros) {
  Array a = zeros({2, 3}, DType::F32);
  EXPECT_EQ(a.shape(), Shape({2, 3}));
  EXPECT_EQ(a.dtype(), DType::F32);

  const float *data = a.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0.0f);
  }
}

TEST_F(CreationTest, Ones) {
  Array a = ones({2, 3}, DType::F32);
  const float *data = a.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.0f);
  }
}

TEST_F(CreationTest, Full) {
  Array a = full({2, 3}, 3.14, DType::F32);
  const float *data = a.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 3.14f);
  }
}

TEST_F(CreationTest, FullInt) {
  Array a = full({2, 3}, 42, DType::I32);
  const int32_t *data = a.data<int32_t>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], 42);
  }
}

TEST_F(CreationTest, FullFloat64) {
  Array a = full({2, 2}, 2.718281828, DType::F64);
  EXPECT_EQ(a.dtype(), DType::F64);
  const double *data = a.data<double>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(data[i], 2.718281828, 1e-9);
  }
}

TEST_F(CreationTest, FullScalar) {
  Array a = full({}, 7.0, DType::F32);
  EXPECT_EQ(a.shape(), Shape({}));
  EXPECT_FLOAT_EQ(a.item<float>(), 7.0f);
}

// ========== eye ==========

TEST_F(CreationTest, Eye) {
  Array a = eye(3);
  EXPECT_EQ(a.shape(), Shape({3, 3}));
  const float *data = a.data<float>();

  // Expected: [[1,0,0],[0,1,0],[0,0,1]]
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_FLOAT_EQ(data[i * 3 + j], 1.0f);
      } else {
        EXPECT_FLOAT_EQ(data[i * 3 + j], 0.0f);
      }
    }
  }
}

TEST_F(CreationTest, EyeWithOffset) {
  Array a = eye(3, 3, 1);
  const float *data = a.data<float>();
  // Expected: [[0,1,0],[0,0,1],[0,0,0]]
  EXPECT_FLOAT_EQ(data[1], 1.0f); // (0,1)
  EXPECT_FLOAT_EQ(data[5], 1.0f); // (1,2)
  EXPECT_FLOAT_EQ(data[0], 0.0f);
  EXPECT_FLOAT_EQ(data[4], 0.0f);
  EXPECT_FLOAT_EQ(data[8], 0.0f);
}

TEST_F(CreationTest, EyeRectangular) {
  Array a = eye(2, 3);
  const float *data = a.data<float>();
  // Expected: [[1,0,0],[0,1,0]]
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[4], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 0.0f);
  EXPECT_FLOAT_EQ(data[2], 0.0f);
  EXPECT_FLOAT_EQ(data[3], 0.0f);
  EXPECT_FLOAT_EQ(data[5], 0.0f);
}

TEST_F(CreationTest, EyeNegativeOffset) {
  Array a = eye(3, 3, -1);
  const float *data = a.data<float>();
  // Expected: [[0,0,0],[1,0,0],[0,1,0]]
  EXPECT_FLOAT_EQ(data[3], 1.0f); // (1,0)
  EXPECT_FLOAT_EQ(data[7], 1.0f); // (2,1)
  EXPECT_FLOAT_EQ(data[0], 0.0f);
  EXPECT_FLOAT_EQ(data[4], 0.0f);
  EXPECT_FLOAT_EQ(data[8], 0.0f);
}

// ========== arange ==========

TEST_F(CreationTest, ArangeSingle) {
  Array a = arange(5);
  EXPECT_EQ(a.shape(), Shape({5}));
  const int64_t *data = a.data<int64_t>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(data[i], i);
  }
}

TEST_F(CreationTest, ArangeStartStop) {
  Array a = arange(2, 7);
  const int64_t *data = a.data<int64_t>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(data[i], 2 + i);
  }
}

TEST_F(CreationTest, ArangeWithStep) {
  Array a = arange(1, 10, 2);
  const int64_t *data = a.data<int64_t>();
  EXPECT_EQ(a.numel(), 5);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 3);
  EXPECT_EQ(data[2], 5);
  EXPECT_EQ(data[3], 7);
  EXPECT_EQ(data[4], 9);
}

TEST_F(CreationTest, ArangeFloat) {
  Array a = arange(0.0, 1.0, 0.2, DType::F32);
  const float *data = a.data<float>();
  EXPECT_EQ(a.numel(), 5);
  EXPECT_NEAR(data[0], 0.0f, 1e-6);
  EXPECT_NEAR(data[1], 0.2f, 1e-6);
  EXPECT_NEAR(data[2], 0.4f, 1e-6);
  EXPECT_NEAR(data[3], 0.6f, 1e-6);
  EXPECT_NEAR(data[4], 0.8f, 1e-6);
}

TEST_F(CreationTest, ArangeFloat64) {
  Array a = arange(0.0, 1.0, 0.25, DType::F64);
  EXPECT_EQ(a.dtype(), DType::F64);
  const double *data = a.data<double>();
  EXPECT_EQ(a.numel(), 4);
  EXPECT_NEAR(data[0], 0.0, 1e-12);
  EXPECT_NEAR(data[1], 0.25, 1e-12);
  EXPECT_NEAR(data[2], 0.5, 1e-12);
  EXPECT_NEAR(data[3], 0.75, 1e-12);
}

// ========== linspace ==========

TEST_F(CreationTest, Linspace) {
  Array a = linspace(0, 1, 5);
  const float *data = a.data<float>();
  EXPECT_EQ(a.numel(), 5);
  EXPECT_NEAR(data[0], 0.0f, 1e-6);
  EXPECT_NEAR(data[1], 0.25f, 1e-6);
  EXPECT_NEAR(data[2], 0.5f, 1e-6);
  EXPECT_NEAR(data[3], 0.75f, 1e-6);
  EXPECT_NEAR(data[4], 1.0f, 1e-6);
}

TEST_F(CreationTest, LinspaceSingle) {
  Array a = linspace(0, 1, 1);
  const float *data = a.data<float>();
  EXPECT_EQ(a.numel(), 1);
  EXPECT_NEAR(data[0], 0.0f, 1e-6);
}

TEST_F(CreationTest, LinspaceFloat64) {
  Array a = linspace(0, 1, 5, DType::F64);
  const double *data = a.data<double>();
  EXPECT_NEAR(data[0], 0.0, 1e-12);
  EXPECT_NEAR(data[2], 0.5, 1e-12);
  EXPECT_NEAR(data[4], 1.0, 1e-12);
}

TEST_F(CreationTest, LinspaceNegativeRange) {
  Array a = linspace(-1, 1, 5);
  const float *data = a.data<float>();
  EXPECT_EQ(a.numel(), 5);
  EXPECT_NEAR(data[0], -1.0f, 1e-6);
  EXPECT_NEAR(data[2], 0.0f, 1e-6);
  EXPECT_NEAR(data[4], 1.0f, 1e-6);
}

// ========== logspace ==========

TEST_F(CreationTest, Logspace) {
  Array a = logspace(0, 2, 3);
  const float *data = a.data<float>();
  EXPECT_EQ(a.numel(), 3);
  // base=10, start=0, stop=2: [10^0, 10^1, 10^2] = [1, 10, 100]
  EXPECT_NEAR(data[0], 1.0f, 1e-6);
  EXPECT_NEAR(data[1], 10.0f, 1e-6);
  EXPECT_NEAR(data[2], 100.0f, 1e-6);
}

TEST_F(CreationTest, LogspaceBase2) {
  Array a = logspace(0, 3, 4, 2.0);
  const float *data = a.data<float>();
  // base=2, start=0, stop=3: [2^0, 2^1, 2^2, 2^3] = [1, 2, 4, 8]
  EXPECT_NEAR(data[0], 1.0f, 1e-6);
  EXPECT_NEAR(data[1], 2.0f, 1e-6);
  EXPECT_NEAR(data[2], 4.0f, 1e-6);
  EXPECT_NEAR(data[3], 8.0f, 1e-6);
}

TEST_F(CreationTest, LogspaceFloat64) {
  Array a = logspace(0, 1, 3, 10.0, DType::F64);
  EXPECT_EQ(a.dtype(), DType::F64);
  const double *data = a.data<double>();
  EXPECT_EQ(a.numel(), 3);
  // base=10, start=0, stop=1, n=3: step=0.5
  // [10^0, 10^0.5, 10^1] = [1, sqrt(10), 10]
  EXPECT_NEAR(data[0], 1.0, 1e-12);
  EXPECT_NEAR(data[1], std::sqrt(10.0), 1e-12);
  EXPECT_NEAR(data[2], 10.0, 1e-12);
}

TEST_F(CreationTest, LogspaceSingle) {
  Array a = logspace(2, 2, 1, 10.0);
  const float *data = a.data<float>();
  EXPECT_EQ(a.numel(), 1);
  EXPECT_NEAR(data[0], 100.0f, 1e-6); // 10^2
}

// ========== to_array ==========

TEST_F(CreationTest, ToArrayFromVector) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Array a = to_array(data);
  EXPECT_EQ(a.shape(), Shape({4}));
  const float *ptr = a.data<float>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(ptr[i], data[i]);
  }
}

TEST_F(CreationTest, ToArrayFromInitList) {
  Array a = to_array({1.0f, 2.0f, 3.0f});
  EXPECT_EQ(a.shape(), Shape({3}));
  const float *ptr = a.data<float>();
  EXPECT_FLOAT_EQ(ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(ptr[2], 3.0f);
}

TEST_F(CreationTest, ToArrayIntToFloat) {
  std::vector<int> data = {1, 2, 3, 4};
  Array a = to_array(data, DType::F32);
  EXPECT_EQ(a.dtype(), DType::F32);
  const float *ptr = a.data<float>();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(data[i]));
  }
}

// ========== zeros_like / ones_like / rand_like / randn_like ==========

TEST_F(CreationTest, ZerosLike) {
  Array original({2, 3}, DType::F32);
  Array copy = zeros_like(original);
  EXPECT_EQ(copy.shape(), original.shape());
  EXPECT_EQ(copy.dtype(), original.dtype());
  EXPECT_EQ(copy.place(), original.place());

  const float *data = copy.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0.0f);
  }
}

TEST_F(CreationTest, OnesLike) {
  Array original({2, 3}, DType::F32);
  Array copy = ones_like(original);
  const float *data = copy.data<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.0f);
  }
}

// ========== GPU not available: must throw ==========

TEST_F(CreationTest, GpuThrowsWithoutBackend) {
  // Only CPU backend is loaded (init() in SetUpTestSuite).
  // Attempting to use GPU must throw an exception.
  Array a = ones({3}, DType::F32, CPUPlace());
  EXPECT_THROW(a.to(GPUPlace(0)), std::exception);
}

TEST_F(CreationTest, SetDeviceGpuThrowsWithoutBackend) {
  Array a = ones({3}, DType::F32, CPUPlace());
  EXPECT_THROW(a.to(GPUPlace(0)), std::exception);
}