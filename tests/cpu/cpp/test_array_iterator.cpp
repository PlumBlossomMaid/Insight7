// tests/cpu/cpp/test_array_iterator.cpp
#include "insight/insight.h"
#include <gtest/gtest.h>

using namespace ins;

class ArrayIteratorTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

TEST_F(ArrayIteratorTestCPU, ElementwiseBroadcastMetadata) {
  Array a({2, 1}, DType::F32);
  Array b({1, 3}, DType::F32);

  ArrayIterator iter = ArrayIterator::elementwise({a, b});

  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  EXPECT_EQ(iter.ndim(), 2);
  EXPECT_EQ(iter.numel(), 6);
  ASSERT_EQ(iter.operands().size(), 2u);
  ASSERT_EQ(iter.arrays().size(), 2u);
  EXPECT_EQ(iter.array(0).shape(), Shape({2, 3}));
  EXPECT_EQ(iter.array(1).shape(), Shape({2, 3}));

  const auto &a_meta = iter.operands()[0];
  const auto &b_meta = iter.operands()[1];
  EXPECT_EQ(a_meta.access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(b_meta.access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(a_meta.dtype, DType::F32);
  EXPECT_EQ(b_meta.dtype, DType::F32);
  EXPECT_EQ(a_meta.shape, Shape({2, 3}));
  EXPECT_EQ(b_meta.shape, Shape({2, 3}));
  EXPECT_EQ(a_meta.strides, Strides(std::vector<int64_t>{1, 0}));
  EXPECT_EQ(b_meta.strides, Strides(std::vector<int64_t>{0, 1}));
  EXPECT_TRUE(a_meta.is_broadcast);
  EXPECT_TRUE(b_meta.is_broadcast);
  EXPECT_FALSE(iter.all_contiguous());
  EXPECT_TRUE(iter.has_cpu_operands());
  EXPECT_FALSE(iter.has_gpu_operands());
}

TEST_F(ArrayIteratorTestCPU, ElementwiseWithOutputMetadata) {
  Array a({2, 3}, DType::F64);
  Array b({2, 3}, DType::F64);
  Array out({2, 3}, DType::F64);

  ArrayIterator iter = ArrayIterator::elementwise({a, b}, {out});

  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  ASSERT_EQ(iter.operands().size(), 3u);
  ASSERT_EQ(iter.arrays().size(), 3u);
  EXPECT_EQ(iter.arrays()[2].shape(), Shape({2, 3}));
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[2].access, ArrayIteratorOperandAccess::WriteOnly);
  EXPECT_TRUE(iter.all_contiguous());
  EXPECT_FALSE(iter.operands()[2].is_broadcast);
}

TEST_F(ArrayIteratorTestCPU, RejectsMismatchedOutputShape) {
  Array a({2, 3}, DType::F32);
  Array out({3, 2}, DType::F32);

  EXPECT_THROW(ArrayIterator::elementwise({a}, {out}), ins::Exception);
}

TEST_F(ArrayIteratorTestCPU, PreservesViewOffsetAndStrides) {
  Array a({4, 3}, DType::F32);
  Array view = a.slice(0, 1, 4, 2);

  ArrayIterator iter = ArrayIterator::elementwise({view});

  ASSERT_EQ(iter.operands().size(), 1u);
  const auto &meta = iter.operands()[0];
  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  EXPECT_EQ(meta.offset, 3);
  EXPECT_EQ(meta.strides, Strides(std::vector<int64_t>{6, 1}));
  EXPECT_FALSE(meta.is_contiguous);
  EXPECT_FALSE(meta.is_broadcast);
}

TEST_F(ArrayIteratorTestCPU, ReductionMetadata) {
  Array input({2, 3}, DType::F32);
  Array output({2}, DType::F32);

  ArrayIterator iter = ArrayIterator::reduction(input, output, 2, 3);

  EXPECT_TRUE(iter.is_reduction());
  EXPECT_EQ(iter.shape(), Shape({2}));
  EXPECT_EQ(iter.reduction_batch_size(), 2);
  EXPECT_EQ(iter.reduction_reduce_size(), 3);
  ASSERT_EQ(iter.arrays().size(), 2u);
  ASSERT_EQ(iter.operands().size(), 2u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::WriteOnly);
  EXPECT_TRUE(iter.all_contiguous());
  EXPECT_TRUE(iter.has_cpu_operands());
  EXPECT_FALSE(iter.has_gpu_operands());
}

TEST_F(ArrayIteratorTestCPU, MultiInputReductionMetadata) {
  Array input({5}, DType::I32);
  Array weights({5}, DType::F32);
  Array output({3}, DType::F32);

  ArrayIterator iter = ArrayIterator::reduction({input, weights}, {output}, 5, 1);

  EXPECT_TRUE(iter.is_reduction());
  EXPECT_EQ(iter.shape(), Shape({3}));
  EXPECT_EQ(iter.reduction_batch_size(), 5);
  EXPECT_EQ(iter.reduction_reduce_size(), 1);
  ASSERT_EQ(iter.arrays().size(), 3u);
  ASSERT_EQ(iter.operands().size(), 3u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[2].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(ArrayIteratorTestCPU, CreationMetadata) {
  Array output({2, 3}, DType::F32);

  ArrayIterator iter = ArrayIterator::creation({output});

  EXPECT_FALSE(iter.is_reduction());
  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  ASSERT_EQ(iter.arrays().size(), 1u);
  ASSERT_EQ(iter.operands().size(), 1u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::WriteOnly);
  EXPECT_TRUE(iter.all_contiguous());
  EXPECT_TRUE(iter.has_cpu_operands());
  EXPECT_FALSE(iter.has_gpu_operands());
}

TEST_F(ArrayIteratorTestCPU, TransformMetadata) {
  Array input({2, 3}, DType::F32);
  Array output({3, 2}, DType::F32);

  ArrayIterator iter = ArrayIterator::transform({input}, {output});

  EXPECT_FALSE(iter.is_reduction());
  EXPECT_EQ(iter.shape(), Shape({3, 2}));
  ASSERT_EQ(iter.arrays().size(), 2u);
  ASSERT_EQ(iter.operands().size(), 2u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(ArrayIteratorTestCPU, TransformSupportsMultipleOutputShapes) {
  Array input({3, 2}, DType::F32);
  Array first({3, 3}, DType::F32);
  Array second({2}, DType::F32);

  ArrayIterator iter = ArrayIterator::transform({input}, {first, second});

  EXPECT_EQ(iter.shape(), Shape({3, 3}));
  ASSERT_EQ(iter.arrays().size(), 3u);
  ASSERT_EQ(iter.operands().size(), 3u);
  EXPECT_EQ(iter.arrays()[1].shape(), Shape({3, 3}));
  EXPECT_EQ(iter.arrays()[2].shape(), Shape({2}));
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::WriteOnly);
  EXPECT_EQ(iter.operands()[2].access, ArrayIteratorOperandAccess::WriteOnly);
}
