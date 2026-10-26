// tests/cpu/cpp/test_dlpack.cpp
#include "insight/c_api/dlpack.h"
#include "insight/insight.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace ins;

class DLPackTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

// ========== dtype mapping ==========

TEST_F(DLPackTest, DTypeRoundTrip) {
  DLDataType dl;
  ASSERT_EQ(insight_dtype_to_dlpack(INSIGHT_DTYPE_F32, &dl), C_SUCCESS);
  EXPECT_EQ(dl.code, kDLFloat);
  EXPECT_EQ(dl.bits, 32);
  EXPECT_EQ(dl.lanes, 1);

  int32_t back;
  ASSERT_EQ(insight_dtype_from_dlpack(&dl, &back), C_SUCCESS);
  EXPECT_EQ(back, INSIGHT_DTYPE_F32);
}

TEST_F(DLPackTest, DTypeAllMapped) {
  const int32_t all[] = {
      INSIGHT_DTYPE_BOOL, INSIGHT_DTYPE_U8,   INSIGHT_DTYPE_I8,
      INSIGHT_DTYPE_I16,  INSIGHT_DTYPE_I32,  INSIGHT_DTYPE_I64,
      INSIGHT_DTYPE_F16,  INSIGHT_DTYPE_BF16, INSIGHT_DTYPE_F32,
      INSIGHT_DTYPE_F64,  INSIGHT_DTYPE_C32,  INSIGHT_DTYPE_C64,
      INSIGHT_DTYPE_U16,  INSIGHT_DTYPE_U32,  INSIGHT_DTYPE_U64};
  for (int32_t dt : all) {
    DLDataType dl;
    ASSERT_EQ(insight_dtype_to_dlpack(dt, &dl), C_SUCCESS);
    int32_t back;
    ASSERT_EQ(insight_dtype_from_dlpack(&dl, &back), C_SUCCESS);
    EXPECT_EQ(back, dt);
  }
}

// ========== to_dlpack ==========

TEST_F(DLPackTest, ToDLPackBasic) {
  Array a = full({2, 3}, 3.5f, DType::F32);
  DLManagedTensorVersioned *m = nullptr;
  ASSERT_EQ(insight_array_to_dlpack(a.layout_ptr(), &m), C_SUCCESS);
  ASSERT_NE(m, nullptr);
  ASSERT_EQ(m->version.major, DLPACK_MAJOR_VERSION);

  EXPECT_EQ(m->dl_tensor.ndim, 2);
  EXPECT_EQ(m->dl_tensor.shape[0], 2);
  EXPECT_EQ(m->dl_tensor.shape[1], 3);
  EXPECT_EQ(m->dl_tensor.strides[0], 3);
  EXPECT_EQ(m->dl_tensor.strides[1], 1);
  EXPECT_EQ(m->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(m->dl_tensor.dtype.bits, 32);
  EXPECT_EQ(m->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(m->dl_tensor.byte_offset, 0);

  // data pointer equals the array's logical first element
  EXPECT_EQ(m->dl_tensor.data, static_cast<void *>(a.data<float>()));

  m->deleter(m);
}

TEST_F(DLPackTest, ToDLPackScalar) {
  Array s = full({}, 2.0, DType::F64); // 0-d
  DLManagedTensorVersioned *m = nullptr;
  ASSERT_EQ(insight_array_to_dlpack(s.layout_ptr(), &m), C_SUCCESS);
  EXPECT_EQ(m->dl_tensor.ndim, 0);
  EXPECT_EQ(m->dl_tensor.shape, nullptr);
  EXPECT_EQ(m->dl_tensor.strides, nullptr);
  EXPECT_EQ(m->dl_tensor.dtype.bits, 64);
  m->deleter(m);
}

TEST_F(DLPackTest, ToDLPackViewOffset) {
  // Create a view starting partway into the storage
  Array base = arange(12, DType::F32); // 12 elements
  Array view = base.slice(0, 3, 9);    // elements 3..8 (dim 0)
  ASSERT_NE(view.layout_ptr()->offset, 0);
  DLManagedTensorVersioned *m = nullptr;
  ASSERT_EQ(insight_array_to_dlpack(view.layout_ptr(), &m), C_SUCCESS);
  // logical data pointer must include the view offset
  EXPECT_EQ(m->dl_tensor.data, static_cast<void *>(view.data<float>()));
  m->deleter(m);
}

// ========== from_dlpack ==========

TEST_F(DLPackTest, FromDLPackBasic) {
  // Build a synthetic DLPack tensor describing 2x3 float32 data
  float data[6] = {1, 2, 3, 4, 5, 6};
  int64_t shape[2] = {2, 3};
  int64_t strides[2] = {3, 1};

  DLManagedTensorVersioned m;
  std::memset(&m, 0, sizeof(m));
  m.version.major = DLPACK_MAJOR_VERSION;
  m.version.minor = DLPACK_MINOR_VERSION;
  m.dl_tensor.data = data;
  m.dl_tensor.device.device_type = kDLCPU;
  m.dl_tensor.device.device_id = 0;
  m.dl_tensor.ndim = 2;
  m.dl_tensor.dtype.code = kDLFloat;
  m.dl_tensor.dtype.bits = 32;
  m.dl_tensor.dtype.lanes = 1;
  m.dl_tensor.shape = shape;
  m.dl_tensor.strides = strides;
  m.dl_tensor.byte_offset = 0;

  InsightArray out;
  ASSERT_EQ(insight_array_from_dlpack(&m, &out), C_SUCCESS);
  Array arr(&out);             // adopts ref → ref_count 2
  insight_array_destroy(&out); // drop the C-side ref → arr owns the last one
  EXPECT_EQ(arr.shape(), Shape({2, 3}));
  EXPECT_EQ(arr.dtype(), DType::F32);
  const float *d = arr.data<float>();
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(d[i], data[i]);
}

TEST_F(DLPackTest, FromDLPackNonContiguous) {
  // describe a strided view: rows 1..2 of a 3x3 matrix (3 elements wide,
  // stride 3) -> shape {2,3} with row stride 3
  float storage[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int64_t shape[2] = {2, 3};
  int64_t strides[2] = {3, 1};
  // data pointer at storage, byte_offset of 3 floats (row 1)
  DLManagedTensorVersioned m;
  std::memset(&m, 0, sizeof(m));
  m.version.major = DLPACK_MAJOR_VERSION;
  m.version.minor = DLPACK_MINOR_VERSION;
  m.dl_tensor.data = storage;
  m.dl_tensor.device.device_type = kDLCPU;
  m.dl_tensor.ndim = 2;
  m.dl_tensor.dtype.code = kDLFloat;
  m.dl_tensor.dtype.bits = 32;
  m.dl_tensor.dtype.lanes = 1;
  m.dl_tensor.shape = shape;
  m.dl_tensor.strides = strides;
  m.dl_tensor.byte_offset = 3 * sizeof(float);

  InsightArray out;
  ASSERT_EQ(insight_array_from_dlpack(&m, &out), C_SUCCESS);
  Array arr(&out);
  insight_array_destroy(&out);
  const float *d = arr.data<float>();
  // expected rows: 4,5,6 / 7,8,9
  float expected[6] = {4, 5, 6, 7, 8, 9};
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(d[i], expected[i]);
}

// ========== round trip ==========

TEST_F(DLPackTest, RoundTrip) {
  Array a = arange(12, DType::F32).reshape({3, 4}) + 1.0f;
  DLManagedTensorVersioned *m = nullptr;
  ASSERT_EQ(insight_array_to_dlpack(a.layout_ptr(), &m), C_SUCCESS);

  InsightArray out;
  ASSERT_EQ(insight_array_from_dlpack(m, &out), C_SUCCESS);
  Array b(&out);
  insight_array_destroy(&out);
  EXPECT_EQ(b.shape(), a.shape());
  EXPECT_EQ(b.dtype(), a.dtype());
  const float *da = a.data<float>();
  const float *db = b.data<float>();
  for (int i = 0; i < 12; ++i)
    EXPECT_EQ(db[i], da[i]);

  m->deleter(m);
}
