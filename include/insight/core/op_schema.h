// insight/core/op_schema.h
#pragma once
#include "insight/core/array_iterator.h"
#include "insight/core/dtype.h"
#include "insight/core/shape.h"
#include <optional>
#include <string>
#include <vector>

namespace ins {

enum class OpKind {
  UnaryElementwise,
  BinaryElementwise,
  Reduction,
  Creation,
  Manipulation,
  Indexing,
  Linalg,
  Fft,
  Signal,
  Custom,
};

enum class OpArgumentKind {
  Array,
  HostScalar,
  DType,
  Shape,
  Int,
  Float,
  String,
};

enum class OpArgumentAccess {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

enum class OpPromotionRule {
  None,
  Identity,
  Numeric,
  Comparison,
  Floating,
};

enum class OpBroadcastRule {
  None,
  Inputs,
};

enum class OpFallbackRule {
  None,
  StructuredCpu,
};

struct OpArgumentSchema {
  std::string name;
  OpArgumentKind kind = OpArgumentKind::Array;
  OpArgumentAccess access = OpArgumentAccess::ReadOnly;
};

class OpSchema {
public:
  OpSchema(std::string name, OpKind kind,
           std::vector<OpArgumentSchema> inputs,
           std::vector<OpArgumentSchema> outputs);

  const std::string &name() const { return name_; }
  OpKind kind() const { return kind_; }
  const std::vector<OpArgumentSchema> &inputs() const { return inputs_; }
  const std::vector<OpArgumentSchema> &outputs() const { return outputs_; }
  OpPromotionRule promotion_rule() const { return promotion_rule_; }
  OpBroadcastRule broadcast_rule() const { return broadcast_rule_; }
  OpFallbackRule fallback_rule() const { return fallback_rule_; }
  const std::vector<DType> &dispatch_dtypes() const { return dispatch_dtypes_; }

  OpSchema &promotion(OpPromotionRule rule);
  OpSchema &broadcast(OpBroadcastRule rule);
  OpSchema &fallback(OpFallbackRule rule);
  OpSchema &dispatch_dtypes(std::vector<DType> dtypes);

  Shape infer_elementwise_shape(const std::vector<Array> &inputs) const;
  DType infer_binary_dtype(DType a, DType b) const;
  ArrayIterator make_array_iterator(const std::vector<Array> &inputs,
                                    const std::vector<Array> &outputs = {}) const;
  ArrayIterator make_creation_iterator(const std::vector<Array> &outputs) const;
  ArrayIterator make_transform_iterator(const std::vector<Array> &inputs,
                                        const std::vector<Array> &outputs) const;
  ArrayIterator make_reduction_iterator(const Array &input, const Array &output,
                                        int64_t batch_size,
                                        int64_t reduce_size) const;
  ArrayIterator make_reduction_iterator(const Array &input,
                                        const std::vector<Array> &outputs,
                                        int64_t batch_size,
                                        int64_t reduce_size) const;
  ArrayIterator make_reduction_iterator(const std::vector<Array> &inputs,
                                        const std::vector<Array> &outputs,
                                        int64_t batch_size,
                                        int64_t reduce_size) const;

private:
  void validate() const;

  std::string name_;
  OpKind kind_ = OpKind::Custom;
  std::vector<OpArgumentSchema> inputs_;
  std::vector<OpArgumentSchema> outputs_;
  OpPromotionRule promotion_rule_ = OpPromotionRule::None;
  OpBroadcastRule broadcast_rule_ = OpBroadcastRule::None;
  OpFallbackRule fallback_rule_ = OpFallbackRule::None;
  std::vector<DType> dispatch_dtypes_;
};

class OpSchemaRegistry {
public:
  static void register_schema(const OpSchema &schema);
  static bool has(const std::string &name);
  static const OpSchema &get(const std::string &name);
  static std::vector<std::string> names();
  static void clear();
};

} // namespace ins
