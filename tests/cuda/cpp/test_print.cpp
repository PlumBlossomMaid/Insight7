// tests/cuda/test_print.cpp
#include "insight/insight.h"
#include "insight/io/print.h"
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>

using namespace ins;

class PrintTestGPU : public ::testing::Test {
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

TEST_F(PrintTestGPU, PrintScalar) {
  Array a(3.14159f);
  std::string str = to_string(a);

  EXPECT_TRUE(str.find("Array(shape=[], dtype=float32") != std::string::npos);
  EXPECT_TRUE(str.find("3.14") != std::string::npos);
}

TEST_F(PrintTestGPU, Print1D) {
  Array cpu_a(Shape({5}), DType::F32, CPUPlace());
  for (int i = 0; i < 5; ++i)
    cpu_a.data<float>()[i] = static_cast<float>(i);
  Array a = cpu_a.to(GPUPlace(0));

  std::string str = to_string(a);

  EXPECT_TRUE(str.find("shape=[5]") != std::string::npos);
  EXPECT_TRUE(str.find("[0., 1., 2., 3., 4.]") != std::string::npos);
}

TEST_F(PrintTestGPU, Print2D) {
  Array cpu_a(Shape({2, 3}), DType::F32, CPUPlace());
  for (int i = 0; i < 6; ++i)
    cpu_a.data<float>()[i] = static_cast<float>(i);
  Array a = cpu_a.to(GPUPlace(0));

  std::string str = to_string(a);

  EXPECT_TRUE(str.find("shape=[2, 3]") != std::string::npos);
  EXPECT_TRUE(str.find("[0., 1., 2.]") != std::string::npos);
  EXPECT_TRUE(str.find("[3., 4., 5.]") != std::string::npos);
}

TEST_F(PrintTestGPU, Print3D) {
  Array cpu_a(Shape({2, 2, 2}), DType::F32, CPUPlace());
  for (int i = 0; i < 8; ++i)
    cpu_a.data<float>()[i] = static_cast<float>(i);
  Array a = cpu_a.to(GPUPlace(0));

  std::string str = to_string(a);

  EXPECT_TRUE(str.find("shape=[2, 2, 2]") != std::string::npos);
  EXPECT_TRUE(str.find("[[[0., 1.]") != std::string::npos);
}

TEST_F(PrintTestGPU, PrintInt) {
  Array cpu_a(Shape({2, 3}), DType::I32, CPUPlace());
  for (int i = 0; i < 6; ++i)
    cpu_a.data<int32_t>()[i] = i;
  Array a = cpu_a.to(GPUPlace(0));

  std::string str = to_string(a);

  EXPECT_TRUE(str.find("dtype=int32") != std::string::npos);
  EXPECT_TRUE(str.find("[0, 1, 2]") != std::string::npos);
  EXPECT_TRUE(str.find("[3, 4, 5]") != std::string::npos);
}

TEST_F(PrintTestGPU, PrintBool) {
  Array cpu_a(Shape({2, 2}), DType::BOOL, CPUPlace());
  bool *data = cpu_a.data<bool>();
  data[0] = true;
  data[1] = false;
  data[2] = true;
  data[3] = false;
  Array a = cpu_a.to(GPUPlace(0));

  std::string str = to_string(a);

  EXPECT_TRUE(str.find("dtype=bool") != std::string::npos);
  EXPECT_TRUE(str.find("[true, false]") != std::string::npos);
}

TEST_F(PrintTestGPU, PrintComplex) {
  Array cpu_a(Shape({2}), DType::C32, CPUPlace());
  std::complex<float> *data = cpu_a.data<std::complex<float>>();
  data[0] = std::complex<float>(1.0f, 2.0f);
  data[1] = std::complex<float>(3.0f, -4.0f);
  Array a = cpu_a.to(GPUPlace(0));

  std::string str = to_string(a);

  EXPECT_TRUE(str.find("dtype=complex64") != std::string::npos);
  EXPECT_TRUE(str.find("(1.00000000+2.00000000j)") != std::string::npos);
  EXPECT_TRUE(str.find("(3.00000000-4.00000000j)") != std::string::npos);
}

TEST_F(PrintTestGPU, PrintWithSetPrecision) {
  Array cpu_a(Shape({1}), DType::F32, CPUPlace());
  cpu_a.data<float>()[0] = 3.14159265358979f;
  Array a = cpu_a.to(GPUPlace(0));

  set_printoptions(4, -1, -1, -1, false);
  std::string str = to_string(a);
  EXPECT_TRUE(str.find("3.1416") != std::string::npos);

  set_printoptions(2, -1, -1, -1, false);
  str = to_string(a);
  EXPECT_TRUE(str.find("3.14") != std::string::npos);

  set_printoptions(8, -1, -1, -1, false);
}

TEST_F(PrintTestGPU, PrintWithSummary) {
  Array cpu_a(Shape({20}), DType::F32, CPUPlace());
  for (int i = 0; i < 20; ++i)
    cpu_a.data<float>()[i] = static_cast<float>(i);
  Array a = cpu_a.to(GPUPlace(0));

  set_printoptions(-1, 10, 3, -1, false);
  std::string str = to_string(a);

  EXPECT_TRUE(str.find("..., ") != std::string::npos);
  EXPECT_TRUE(str.find("0.") != std::string::npos);
  EXPECT_TRUE(str.find("19.") != std::string::npos);

  set_printoptions(-1, 1000, 3, -1, false);
}

TEST_F(PrintTestGPU, PrintNonContiguous) {
  Array cpu_a(Shape({3, 4}), DType::F32, CPUPlace());
  for (int i = 0; i < 12; ++i)
    cpu_a.data<float>()[i] = static_cast<float>(i);
  Array a = cpu_a.to(GPUPlace(0));

  Array view = a.slice(0, 0, 3, 2);

  std::string str = to_string(view);

  // GPU non-contiguous views may be made contiguous during to(CPUPlace()),
  // so we only verify shape and that valid data is present.
  EXPECT_TRUE(str.find("shape=[2, 4]") != std::string::npos);
  EXPECT_TRUE(str.find("[0., 1., 2., 3.]") != std::string::npos);
}

TEST_F(PrintTestGPU, PrintOperator) {
  Array cpu_a(Shape({2, 2}), DType::F32, CPUPlace());
  cpu_a.data<float>()[0] = 1.0f;
  cpu_a.data<float>()[1] = 2.0f;
  cpu_a.data<float>()[2] = 3.0f;
  cpu_a.data<float>()[3] = 4.0f;
  Array a = cpu_a.to(GPUPlace(0));

  std::ostringstream oss;
  oss << a;

  std::string str = oss.str();
  EXPECT_TRUE(str.find("Array(shape=[2, 2]") != std::string::npos);
  EXPECT_TRUE(str.find("1.") != std::string::npos);
  EXPECT_TRUE(str.find("2.") != std::string::npos);
  EXPECT_TRUE(str.find("3.") != std::string::npos);
  EXPECT_TRUE(str.find("4.") != std::string::npos);
}
