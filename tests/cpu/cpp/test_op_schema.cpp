// tests/cpu/cpp/test_op_schema.cpp
#include "insight/insight.h"
#include "insight/c_api/exception.h"
#include "insight/c_api/kernel.h"
#include "insight/c_api/place.h"
#include <gtest/gtest.h>
#include <string>

using namespace ins;

class OpSchemaTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
  void TearDown() override { OpSchemaRegistry::clear(); }
};

static OpSchema binary_add_schema() {
  return OpSchema(
             "add", OpKind::BinaryElementwise,
             {{"a", OpArgumentKind::Array, OpArgumentAccess::ReadOnly},
              {"b", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
             {{"out", OpArgumentKind::Array, OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Numeric)
      .broadcast(OpBroadcastRule::Inputs)
      .fallback(OpFallbackRule::StructuredCpu)
      .dispatch_dtypes({DType::F32, DType::F64});
}

static OpSchema reduction_sum_schema() {
  return OpSchema("sum", OpKind::Reduction,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu)
      .dispatch_dtypes({DType::F32, DType::F64});
}

static OpSchema weighted_bincount_schema() {
  return OpSchema("bincount_weighted", OpKind::Reduction,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly},
                   {"weights", OpArgumentKind::Array,
                    OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu)
      .dispatch_dtypes({DType::I32, DType::I64});
}

static OpSchema full_schema() {
  return OpSchema("full", OpKind::Creation, {},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static OpSchema flip_schema() {
  return OpSchema("flip", OpKind::Manipulation,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static OpSchema fft_schema() {
  return OpSchema("fft", OpKind::Fft,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static C_Status fallback_test_kernel(void **, void **) { return C_FALLBACK; }

TEST_F(OpSchemaTestCPU, RawLaunchRejectsFallbackWithoutSchema) {
  ASSERT_EQ(insight_register_kernel("schema_required_fallback", INSIGHT_DEVICE_GPU,
                                    INSIGHT_DTYPE_F32, fallback_test_kernel),
            C_SUCCESS);

  void *inputs[] = {nullptr};
  void *outputs[] = {nullptr};
  C_Status status = insight_kernel_launch("schema_required_fallback",
                                          INSIGHT_DEVICE_GPU,
                                          INSIGHT_DTYPE_F32, inputs, outputs);

  EXPECT_EQ(status, C_FAILED);
  std::string error = insight_get_last_error();
  EXPECT_NE(error.find("requires explicit argument schema"), std::string::npos);
}

TEST_F(OpSchemaTestCPU, SchemaLaunchRejectsFallbackWhenOutputKindMissing) {
  ASSERT_EQ(insight_register_kernel("schema_kind_required_fallback",
                                    INSIGHT_DEVICE_GPU, INSIGHT_DTYPE_F32,
                                    fallback_test_kernel),
            C_SUCCESS);

  InsightKernelArgKind input_kinds[] = {INSIGHT_KERNEL_ARG_HOST_SCALAR};
  int dummy_output = 0;
  void *inputs[] = {nullptr};
  void *outputs[] = {&dummy_output, nullptr};
  C_Status status = insight_kernel_launch_schema(
      "schema_kind_required_fallback", INSIGHT_DEVICE_GPU, INSIGHT_DTYPE_F32,
      inputs, input_kinds, 0, outputs, nullptr, 1);

  EXPECT_EQ(status, C_FAILED);
  std::string error = insight_get_last_error();
  EXPECT_NE(error.find("requires explicit argument schema"), std::string::npos);
}

TEST_F(OpSchemaTestCPU, RegistryStoresAndListsSchemas) {
  OpSchemaRegistry::register_schema(binary_add_schema());

  EXPECT_TRUE(OpSchemaRegistry::has("add"));
  EXPECT_FALSE(OpSchemaRegistry::has("missing"));

  const OpSchema &schema = OpSchemaRegistry::get("add");
  EXPECT_EQ(schema.name(), "add");
  EXPECT_EQ(schema.kind(), OpKind::BinaryElementwise);
  EXPECT_EQ(schema.promotion_rule(), OpPromotionRule::Numeric);
  EXPECT_EQ(schema.broadcast_rule(), OpBroadcastRule::Inputs);
  EXPECT_EQ(schema.fallback_rule(), OpFallbackRule::StructuredCpu);
  ASSERT_EQ(schema.inputs().size(), 2u);
  ASSERT_EQ(schema.outputs().size(), 1u);
  EXPECT_EQ(schema.dispatch_dtypes(), std::vector<DType>({DType::F32, DType::F64}));

  auto names = OpSchemaRegistry::names();
  ASSERT_EQ(names.size(), 1u);
  EXPECT_EQ(names[0], "add");
}

TEST_F(OpSchemaTestCPU, InfersBroadcastShapeAndDType) {
  OpSchema schema = binary_add_schema();
  Array a({2, 1}, DType::F32);
  Array b({1, 3}, DType::F64);

  EXPECT_EQ(schema.infer_elementwise_shape({a, b}), Shape({2, 3}));
  EXPECT_EQ(schema.infer_binary_dtype(a.dtype(), b.dtype()), DType::F64);
}

TEST_F(OpSchemaTestCPU, ComparisonPromotionReturnsBool) {
  OpSchema schema("equal", OpKind::BinaryElementwise,
                  {{"a", OpArgumentKind::Array, OpArgumentAccess::ReadOnly},
                   {"b", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array, OpArgumentAccess::WriteOnly}});
  schema.promotion(OpPromotionRule::Comparison);

  EXPECT_EQ(schema.infer_binary_dtype(DType::F32, DType::F64), DType::BOOL);
}

TEST_F(OpSchemaTestCPU, BuildsArrayIteratorFromSchema) {
  OpSchema schema = binary_add_schema();
  Array a({2, 1}, DType::F32);
  Array b({1, 3}, DType::F32);
  Array out({2, 3}, DType::F32);

  ArrayIterator iter = schema.make_array_iterator({a, b}, {out});

  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  ASSERT_EQ(iter.operands().size(), 3u);
  EXPECT_TRUE(iter.operands()[0].is_broadcast);
  EXPECT_TRUE(iter.operands()[1].is_broadcast);
  EXPECT_EQ(iter.operands()[2].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(OpSchemaTestCPU, RejectsSchemaArityMismatch) {
  OpSchema schema = binary_add_schema();
  Array a({2, 3}, DType::F32);
  Array out({2, 3}, DType::F32);

  EXPECT_THROW(schema.make_array_iterator({a}, {out}), ins::Exception);
}

TEST_F(OpSchemaTestCPU, BuildsReductionIteratorFromSchema) {
  OpSchema schema = reduction_sum_schema();
  Array input({2, 3}, DType::F32);
  Array output({2}, DType::F32);

  ArrayIterator iter = schema.make_reduction_iterator(input, output, 2, 3);

  EXPECT_TRUE(iter.is_reduction());
  EXPECT_EQ(iter.reduction_batch_size(), 2);
  EXPECT_EQ(iter.reduction_reduce_size(), 3);
  ASSERT_EQ(iter.operands().size(), 2u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(OpSchemaTestCPU, RejectsReductionIteratorFromElementwiseSchema) {
  OpSchema schema = binary_add_schema();
  Array input({2, 3}, DType::F32);
  Array output({2}, DType::F32);

  EXPECT_THROW(schema.make_reduction_iterator(input, output, 2, 3),
               ins::Exception);
}

TEST_F(OpSchemaTestCPU, BuildsMultiInputReductionIteratorFromSchema) {
  OpSchema schema = weighted_bincount_schema();
  Array input({5}, DType::I32);
  Array weights({5}, DType::F32);
  Array output({3}, DType::F32);

  ArrayIterator iter = schema.make_reduction_iterator({input, weights}, {output}, 5, 1);

  EXPECT_TRUE(iter.is_reduction());
  EXPECT_EQ(iter.reduction_batch_size(), 5);
  EXPECT_EQ(iter.reduction_reduce_size(), 1);
  ASSERT_EQ(iter.operands().size(), 3u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[2].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(OpSchemaTestCPU, BuildsCreationIteratorFromSchema) {
  OpSchema schema = full_schema();
  Array output({2, 3}, DType::F32);

  ArrayIterator iter = schema.make_creation_iterator({output});

  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  ASSERT_EQ(iter.operands().size(), 1u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(OpSchemaTestCPU, BuildsTransformIteratorFromSchema) {
  OpSchema schema = flip_schema();
  Array input({2, 3}, DType::F32);
  Array output({2, 3}, DType::F32);

  ArrayIterator iter = schema.make_transform_iterator({input}, {output});

  EXPECT_EQ(iter.shape(), Shape({2, 3}));
  ASSERT_EQ(iter.operands().size(), 2u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::WriteOnly);
}

TEST_F(OpSchemaTestCPU, BuildsFftIteratorFromSchema) {
  OpSchema schema = fft_schema();
  Array input({2, 4}, DType::C64);
  Array output({2, 4}, DType::C64);

  ArrayIterator iter = schema.make_transform_iterator({input}, {output});

  EXPECT_EQ(iter.shape(), Shape({2, 4}));
  ASSERT_EQ(iter.operands().size(), 2u);
  EXPECT_EQ(iter.operands()[0].access, ArrayIteratorOperandAccess::ReadOnly);
  EXPECT_EQ(iter.operands()[1].access, ArrayIteratorOperandAccess::WriteOnly);
}
