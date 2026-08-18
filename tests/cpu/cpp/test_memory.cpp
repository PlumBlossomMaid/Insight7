// tests/cpu/cpp/test_memory.cpp
#include "insight/c_api/memory.h"
#include "insight/core/place.h"
#include "insight/insight.h"
#include "gtest/gtest.h"

class MemoryTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

// ========== C++ API ==========

TEST_F(MemoryTestCPU, DeviceMemoryInfoCPU) {
  auto info = ins::device_memory_info(ins::DeviceKind::CPU);
  // CPU should report some memory (at least > 0)
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
}

TEST_F(MemoryTestCPU, DeviceMemoryInfoCPUSpecific) {
  auto info = ins::device_memory_info(ins::DeviceKind::CPU, 0);
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
}

TEST_F(MemoryTestCPU, CArrayScalarIsContiguous) {
  ins::Array scalar(3.0);
  EXPECT_EQ(insight_array_is_contiguous(scalar.layout_ptr()), 1);
}

TEST_F(MemoryTestCPU, CArraySingleElementViewIsContiguous) {
  ins::Array a({4}, ins::DType::F64);
  double *data = a.data<double>();
  data[0] = 1.0;
  data[1] = 2.0;
  data[2] = 3.0;
  data[3] = 4.0;

  ins::Array view = a.slice(0, 1, 2);
  EXPECT_EQ(view.numel(), 1);
  EXPECT_EQ(insight_array_is_contiguous(view.layout_ptr()), 1);
  EXPECT_TRUE(view.is_contiguous());
}

TEST_F(MemoryTestCPU, ArrayStorageMetadataRoot) {
  ins::Array a({2, 3}, ins::DType::F32);
  auto storage = a.storage_metadata();

  EXPECT_EQ(storage.data, a.storage_data());
  EXPECT_EQ(storage.nbytes, 6 * sizeof(float));
  EXPECT_EQ(storage.place, a.place());
  EXPECT_EQ(storage.ref_count, 1);
  EXPECT_FALSE(storage.is_view);
}

TEST_F(MemoryTestCPU, ArrayStorageMetadataViewSharesRoot) {
  ins::Array a({3, 4}, ins::DType::F32);
  ins::Array view = a.slice(0, 1, 3);

  auto root_storage = a.storage_metadata();
  auto view_storage = view.storage_metadata();

  EXPECT_EQ(view_storage.data, root_storage.data);
  EXPECT_EQ(view_storage.nbytes, root_storage.nbytes);
  EXPECT_EQ(view_storage.place, root_storage.place);
  EXPECT_EQ(root_storage.ref_count, 2);
  EXPECT_EQ(view_storage.ref_count, 2);
  EXPECT_FALSE(root_storage.is_view);
  EXPECT_TRUE(view_storage.is_view);
  EXPECT_EQ(view.data(), static_cast<const char *>(view_storage.data) +
                            view.offset() * sizeof(float));
}

TEST_F(MemoryTestCPU, ViewKeepsStorageAliveAfterRootDestruction) {
  ins::Array view;
  {
    ins::Array root({4}, ins::DType::F64);
    root.data<double>()[0] = 1.0;
    root.data<double>()[1] = 2.0;
    root.data<double>()[2] = 3.0;
    root.data<double>()[3] = 4.0;
    view = root.slice(0, 1, 4);
  }

  ASSERT_TRUE(view.defined());
  EXPECT_EQ(view.storage_metadata().ref_count, 1);
  EXPECT_DOUBLE_EQ(view.data<double>()[0], 2.0);
  EXPECT_DOUBLE_EQ(view.data<double>()[2], 4.0);
}

TEST_F(MemoryTestCPU, ViewChainSharesOneStorageRecord) {
  ins::Array root({4, 4}, ins::DType::F32);
  ins::Array row_view = root.slice(0, 1, 4);
  ins::Array column_view = row_view.slice(1, 1, 4, 2);

  EXPECT_EQ(root.storage_data(), row_view.storage_data());
  EXPECT_EQ(root.storage_data(), column_view.storage_data());
  EXPECT_EQ(root.storage_metadata().ref_count, 3);
  EXPECT_EQ(row_view.storage_metadata().ref_count, 3);
  EXPECT_EQ(column_view.storage_metadata().ref_count, 3);
}

// ========== C API ==========

TEST_F(MemoryTestCPU, CDeviceMemoryInfoCPU) {
  size_t total = 0, free_mem = 0;
  C_Status status = insight_device_memory_info(0, 0, &total, &free_mem);
  EXPECT_EQ(status, C_SUCCESS);
  EXPECT_GT(total, 0);
  EXPECT_GT(free_mem, 0);
}

TEST_F(MemoryTestCPU, CDeviceMemoryInfoNullPtrs) {
  EXPECT_NE(insight_device_memory_info(0, 0, nullptr, nullptr), C_SUCCESS);
}
