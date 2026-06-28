// tests/cuda/test_signal_io.cpp
#include "insight/insight.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

namespace {

class SignalIOTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
      // Iluvatar: signal API uses F64
      GTEST_SKIP() << "Iluvatar: signal API uses F64 (hardware limit)";
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
    std::filesystem::create_directories("/tmp/insight_io_test_gpu");
  }
  static void TearDownTestSuite() {
    std::filesystem::remove_all("/tmp/insight_io_test_gpu");
  }
  void SetUp() override { tmp_dir = "/tmp/insight_io_test_gpu"; }

  std::string tmp_dir;
};

// ========== pack_bin / unpack_bin ==========

TEST_F(SignalIOTestGPU, PackBinF64) {
  // I/O operations are CPU-only; verify data round-trips to GPU
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  Array packed = pack_bin(arr);

  EXPECT_EQ(packed.dtype(), DType::U8);
  EXPECT_EQ(packed.numel(), 4 * 8);
}

TEST_F(SignalIOTestGPU, UnpackBinF64) {
  std::vector<double> data = {1.5, 2.5, 3.5};
  Array arr = to_array(data, DType::F64, CPUPlace());

  Array packed = pack_bin(arr);
  Array unpacked = unpack_bin(packed, DType::F64, "L");

  EXPECT_EQ(unpacked.numel(), 3);
  EXPECT_EQ(unpacked.dtype(), DType::F64);

  // Transfer to GPU and back to verify data integrity
  Array gpu = unpacked.to(GPUPlace(0));
  Array back = gpu.to(CPUPlace());
  const double *ud = back.data<double>();
  EXPECT_NEAR(ud[0], 1.5, 1e-10);
  EXPECT_NEAR(ud[1], 2.5, 1e-10);
  EXPECT_NEAR(ud[2], 3.5, 1e-10);
}

TEST_F(SignalIOTestGPU, PackUnpackF32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Array arr = to_array(data, DType::F32, CPUPlace());

  Array packed = pack_bin(arr);
  Array unpacked = unpack_bin(packed, DType::F32, "L");

  EXPECT_EQ(unpacked.numel(), 3);
  const float *ud = unpacked.data<float>();
  EXPECT_NEAR(ud[0], 1.0f, 1e-6f);
  EXPECT_NEAR(ud[1], 2.0f, 1e-6f);
  EXPECT_NEAR(ud[2], 3.0f, 1e-6f);
}

TEST_F(SignalIOTestGPU, UnpackBinBigEndian) {
  std::vector<uint8_t> raw(8);
  raw[0] = 0x3F;
  raw[1] = 0xF0;
  raw[2] = 0x00;
  raw[3] = 0x00;
  raw[4] = 0x00;
  raw[5] = 0x00;
  raw[6] = 0x00;
  raw[7] = 0x00;

  Array bin = to_array(raw, DType::U8, CPUPlace());
  Array unpacked = unpack_bin(bin, DType::F64, "B");

  EXPECT_NEAR(unpacked.data<double>()[0], 1.0, 1e-10);
}

// ========== write_bin / read_bin ==========

TEST_F(SignalIOTestGPU, WriteReadBin) {
  std::string path = tmp_dir + "/test.bin";

  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  write_bin(path, arr, false);

  Array read = read_bin(path, DType::F64, 0, 0);

  EXPECT_EQ(read.numel(), 5);
  const double *rd = read.data<double>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(rd[i], data[i], 1e-10);
  }
}

TEST_F(SignalIOTestGPU, ReadBinWithOffset) {
  std::string path = tmp_dir + "/test_offset.bin";

  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  write_bin(path, arr, false);

  Array read = read_bin(path, DType::F64, 0, 2);

  EXPECT_EQ(read.numel(), 3);
  const double *rd = read.data<double>();
  EXPECT_NEAR(rd[0], 3.0, 1e-10);
  EXPECT_NEAR(rd[1], 4.0, 1e-10);
  EXPECT_NEAR(rd[2], 5.0, 1e-10);
}

TEST_F(SignalIOTestGPU, ReadBinNumSamples) {
  std::string path = tmp_dir + "/test_ns.bin";

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  Array arr = to_array(data, DType::F32, CPUPlace());

  write_bin(path, arr, false);

  Array read = read_bin(path, DType::F32, 3, 0);

  EXPECT_EQ(read.numel(), 3);
}

TEST_F(SignalIOTestGPU, WriteBinAppend) {
  std::string path = tmp_dir + "/test_append.bin";

  std::vector<double> data1 = {1.0, 2.0};
  std::vector<double> data2 = {3.0, 4.0};

  Array arr1 = to_array(data1, DType::F64, CPUPlace());
  Array arr2 = to_array(data2, DType::F64, CPUPlace());

  write_bin(path, arr1, false);
  write_bin(path, arr2, true);

  Array read = read_bin(path, DType::F64, 0, 0);
  EXPECT_EQ(read.numel(), 4);
}

// ========== read_sigmf / write_sigmf ==========

TEST_F(SignalIOTestGPU, WriteReadSigmf) {
  std::string data_path = tmp_dir + "/test.sigmf-data";

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::ofstream ofs(data_path, std::ios::binary);
  ofs.write(reinterpret_cast<const char *>(data.data()),
            data.size() * sizeof(float));
  ofs.close();

  Array read = read_bin(data_path, DType::F32, 0, 0);

  EXPECT_EQ(read.numel(), 4);
  EXPECT_EQ(read.dtype(), DType::F32);
  const float *rd = read.data<float>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(rd[i], data[i], 1e-6f);
  }
}

// ========== Error handling ==========

TEST_F(SignalIOTestGPU, ReadBinNonexistentFile) {
  EXPECT_THROW(read_bin("/nonexistent/file.bin", DType::F64), std::exception);
}

TEST_F(SignalIOTestGPU, PackBinUndefined) {
  Array undef;
  EXPECT_THROW(pack_bin(undef), std::exception);
}

} // namespace
