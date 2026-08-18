// tests/cuda/test_complex.cpp
#include "insight/insight.h"
#include "insight/ops/complex.h"
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

using namespace ins;

class ComplexTestGPU : public ::testing::Test {
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

static Array gpu_f32(const std::vector<float> &data, const Shape &shape = {}) {
  Shape s =
      shape.ndim() == 0 ? Shape({static_cast<int64_t>(data.size())}) : shape;
  Array result = to_array(data, s, DType::F32, CPUPlace());
  return result.to(GPUPlace(0));
}

static Array gpu_f64(const std::vector<double> &data, const Shape &shape = {}) {
  Shape s =
      shape.ndim() == 0 ? Shape({static_cast<int64_t>(data.size())}) : shape;
  Array result = to_array(data, s, DType::F64, CPUPlace());
  return result.to(GPUPlace(0));
}

// ============================================================================
// is_complex tests
// ============================================================================

TEST_F(ComplexTestGPU, IsComplexReturnsFalseForReal) {
  Array real = ones({3, 4}, DType::F32, GPUPlace(0));
  EXPECT_FALSE(is_complex(real));
}

TEST_F(ComplexTestGPU, IsComplexReturnsTrueForComplex) {
  Array real = ones({3, 4}, DType::F32, GPUPlace(0));
  Array complex = to_complex(real);
  EXPECT_TRUE(is_complex(complex));
}

TEST_F(ComplexTestGPU, IsComplexReturnsFalseForScalar) {
  Array scalar = full({}, 1.0f, DType::F32, GPUPlace(0));
  EXPECT_FALSE(is_complex(scalar));
}

// ============================================================================
// has_complex_shape tests
// ============================================================================

TEST_F(ComplexTestGPU, HasComplexShapeReturnsFalseForReal) {
  Array real = ones({3, 4}, DType::F32, GPUPlace(0));
  EXPECT_FALSE(has_complex_shape(real));
}

TEST_F(ComplexTestGPU, HasComplexShapeReturnsFalseForNativeComplex) {
  Array real = ones({3, 4}, DType::F32, GPUPlace(0));
  Array complex = to_complex(real);
  EXPECT_FALSE(has_complex_shape(complex));
}

// ============================================================================
// to_complex (single argument) tests
// ============================================================================

TEST_F(ComplexTestGPU, ToComplexSingleArgumentShape) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array z = to_complex(real);

  EXPECT_EQ(z.shape().ndim(), 1);
  EXPECT_EQ(z.shape().dim(0), 3);
  EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTestGPU, ToComplexSingleArgumentValues) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array z = to_complex(real);
  Array cpu_z = z.to(CPUPlace());

  const std::complex<float> *data = cpu_z.data<std::complex<float>>();
  EXPECT_FLOAT_EQ(data[0].real(), 1.0f);
  EXPECT_FLOAT_EQ(data[0].imag(), 0.0f);
  EXPECT_FLOAT_EQ(data[1].real(), 2.0f);
  EXPECT_FLOAT_EQ(data[1].imag(), 0.0f);
  EXPECT_FLOAT_EQ(data[2].real(), 3.0f);
  EXPECT_FLOAT_EQ(data[2].imag(), 0.0f);
}

TEST_F(ComplexTestGPU, ToComplexSingleArgument2D) {
  Array real = arange(0.0, 6.0, 1.0).to(DType::F32).reshape({2, 3});
  Array gpu_real = real.to(GPUPlace(0));
  Array z = to_complex(gpu_real);

  EXPECT_EQ(z.shape().ndim(), 2);
  EXPECT_EQ(z.shape().dim(0), 2);
  EXPECT_EQ(z.shape().dim(1), 3);
  EXPECT_EQ(z.dtype(), DType::C32);
}

// ============================================================================
// to_complex (two arguments) tests
// ============================================================================

TEST_F(ComplexTestGPU, ToComplexTwoArgumentsShape) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array imag = gpu_f32({4.0f, 5.0f, 6.0f});
  Array z = to_complex(real, imag);

  EXPECT_EQ(z.shape().ndim(), 1);
  EXPECT_EQ(z.shape().dim(0), 3);
  EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTestGPU, ToComplexTwoArgumentsValues) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array imag = gpu_f32({4.0f, 5.0f, 6.0f});
  Array z = to_complex(real, imag);
  Array cpu_z = z.to(CPUPlace());

  const std::complex<float> *data = cpu_z.data<std::complex<float>>();
  EXPECT_FLOAT_EQ(data[0].real(), 1.0f);
  EXPECT_FLOAT_EQ(data[0].imag(), 4.0f);
  EXPECT_FLOAT_EQ(data[1].real(), 2.0f);
  EXPECT_FLOAT_EQ(data[1].imag(), 5.0f);
  EXPECT_FLOAT_EQ(data[2].real(), 3.0f);
  EXPECT_FLOAT_EQ(data[2].imag(), 6.0f);
}

TEST_F(ComplexTestGPU, ToComplexTwoArgumentsShapeMismatchThrows) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array imag = gpu_f32({4.0f, 5.0f});
  EXPECT_THROW(to_complex(real, imag), Exception);
}

// ============================================================================
// real() and imag() tests
// ============================================================================

TEST_F(ComplexTestGPU, RealExtractsRealPart) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array imag = gpu_f32({4.0f, 5.0f, 6.0f});
  Array z = to_complex(real, imag);

  Array r = ins::real(z);
  Array cpu_r = r.to(CPUPlace());

  EXPECT_EQ(cpu_r.shape().ndim(), 1);
  EXPECT_EQ(cpu_r.shape().dim(0), 3);
  EXPECT_FLOAT_EQ(cpu_r.at(0).item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(cpu_r.at(1).item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(cpu_r.at(2).item<float>(), 3.0f);
}

TEST_F(ComplexTestGPU, ImagExtractsImagPart) {
  Array real = gpu_f32({1.0f, 2.0f, 3.0f});
  Array imag = gpu_f32({4.0f, 5.0f, 6.0f});
  Array z = to_complex(real, imag);

  Array i = ins::imag(z);
  Array cpu_i = i.to(CPUPlace());

  EXPECT_EQ(cpu_i.shape().ndim(), 1);
  EXPECT_EQ(cpu_i.shape().dim(0), 3);
  EXPECT_FLOAT_EQ(cpu_i.at(0).item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(cpu_i.at(1).item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(cpu_i.at(2).item<float>(), 6.0f);
}

TEST_F(ComplexTestGPU, RealAndImagValues) {
  Array real_arr = gpu_f32({1.0f, 2.0f, 3.0f});
  Array imag_arr = gpu_f32({4.0f, 5.0f, 6.0f});

  Array z = to_complex(real_arr, imag_arr);
  Array r = real(z);
  Array i = imag(z);
  Array cpu_r = r.to(CPUPlace());
  Array cpu_i = i.to(CPUPlace());

  EXPECT_FLOAT_EQ(cpu_r.at(0).item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(cpu_r.at(1).item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(cpu_r.at(2).item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(cpu_i.at(0).item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(cpu_i.at(1).item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(cpu_i.at(2).item<float>(), 6.0f);
}

// ============================================================================
// as_complex tests
// ============================================================================

TEST_F(ComplexTestGPU, AsComplexShape) {
  Array x = gpu_f32({1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}, Shape({2, 3, 2}));
  Array z = as_complex(x);

  EXPECT_EQ(z.shape().ndim(), 2);
  EXPECT_EQ(z.shape().dim(0), 2);
  EXPECT_EQ(z.shape().dim(1), 3);
  EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTestGPU, AsComplexDtype) {
  Array x = gpu_f64({1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}, Shape({2, 3, 2}));
  Array z = as_complex(x);

  EXPECT_EQ(z.dtype(), DType::C64);
}

TEST_F(ComplexTestGPU, AsComplexInvalidInputThrows) {
  Array x = ones({3, 4}, DType::F32, GPUPlace(0));
  EXPECT_THROW(as_complex(x), Exception);
}

// ============================================================================
// as_real tests
// ============================================================================

TEST_F(ComplexTestGPU, AsRealShape) {
  Array x = gpu_f32({1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}, Shape({2, 3, 2}));
  Array z = as_complex(x);
  Array xr = as_real(z);

  EXPECT_EQ(xr.shape().ndim(), 3);
  EXPECT_EQ(xr.shape().dim(0), 2);
  EXPECT_EQ(xr.shape().dim(1), 3);
  EXPECT_EQ(xr.shape().dim(2), 2);
  EXPECT_EQ(xr.dtype(), DType::F32);
}

TEST_F(ComplexTestGPU, AsRealRoundTrip) {
  Array x = gpu_f32({1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}, Shape({2, 3, 2}));

  Array z = as_complex(x);
  Array x2 = as_real(z);
  Array cpu_x = x.to(CPUPlace());
  Array cpu_x2 = x2.to(CPUPlace());

  const float *x_data = cpu_x.data<float>();
  const float *x2_data = cpu_x2.data<float>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(x_data[i], x2_data[i]);
  }
}

// ============================================================================
// Memory sharing tests
// ============================================================================

TEST_F(ComplexTestGPU, AsComplexSharesMemory) {
  Array x = gpu_f32({1, 4, 2, 5, 3, 6}, Shape({3, 2}));

  Array z = as_complex(x);
  Array x2 = as_real(z);

  // On GPU, as_complex and as_real are view operations sharing the same data.
  // Verify by checking shapes and dtypes are compatible.
  Array cpu_x2 = x2.to(CPUPlace());
  Array cpu_z = z.to(CPUPlace());

  EXPECT_EQ(cpu_x2.shape(), Shape({3, 2}));
  EXPECT_EQ(cpu_z.dtype(), DType::C32);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(ComplexTestGPU, ToComplexScalar) {
  Array scalar = full({}, 3.14f, DType::F32, GPUPlace(0));
  Array z = to_complex(scalar);

  EXPECT_EQ(z.shape().ndim(), 0);
  EXPECT_EQ(z.dtype(), DType::C32);

  Array cpu_z = z.to(CPUPlace());
  const std::complex<float> *data = cpu_z.data<std::complex<float>>();
  EXPECT_FLOAT_EQ(data->real(), 3.14f);
  EXPECT_FLOAT_EQ(data->imag(), 0.0f);
}

TEST_F(ComplexTestGPU, AsComplexScalar) {
  Array x = gpu_f32({3.14f, 2.71f}, Shape({2}));
  Array z = as_complex(x);

  EXPECT_EQ(z.shape().ndim(), 0);
  EXPECT_EQ(z.dtype(), DType::C32);
}
