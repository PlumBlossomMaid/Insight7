// src/core/op_schema.cpp
#include "insight/core/op_schema.h"
#include "insight/core/exception.h"
#include "insight/ops/broadcast.h"
#include "insight/utils/promotion.h"
#include <algorithm>
#include <unordered_map>
#include <utility>

namespace ins {

namespace {

std::unordered_map<std::string, OpSchema> &schema_map() {
  static std::unordered_map<std::string, OpSchema> schemas;
  return schemas;
}

bool is_array_argument(const OpArgumentSchema &arg) {
  return arg.kind == OpArgumentKind::Array;
}

} // namespace

OpSchema::OpSchema(std::string name, OpKind kind,
                   std::vector<OpArgumentSchema> inputs,
                   std::vector<OpArgumentSchema> outputs)
    : name_(std::move(name)), kind_(kind), inputs_(std::move(inputs)),
      outputs_(std::move(outputs)) {
  validate();
}

void OpSchema::validate() const {
  INS_CHECK(!name_.empty(), "OpSchema: name cannot be empty");
  INS_CHECK(!outputs_.empty(), "OpSchema '", name_,
            "': at least one output is required");
  for (const auto &input : inputs_) {
    INS_CHECK(!input.name.empty(), "OpSchema '", name_,
              "': input name cannot be empty");
  }
  for (const auto &output : outputs_) {
    INS_CHECK(!output.name.empty(), "OpSchema '", name_,
              "': output name cannot be empty");
    INS_CHECK(output.access != OpArgumentAccess::ReadOnly, "OpSchema '", name_,
              "': output '", output.name, "' cannot be read-only");
  }
}

OpSchema &OpSchema::promotion(OpPromotionRule rule) {
  promotion_rule_ = rule;
  return *this;
}

OpSchema &OpSchema::broadcast(OpBroadcastRule rule) {
  broadcast_rule_ = rule;
  return *this;
}

OpSchema &OpSchema::fallback(OpFallbackRule rule) {
  fallback_rule_ = rule;
  return *this;
}

OpSchema &OpSchema::dispatch_dtypes(std::vector<DType> dtypes) {
  dispatch_dtypes_ = std::move(dtypes);
  return *this;
}

Shape OpSchema::infer_elementwise_shape(const std::vector<Array> &inputs) const {
  INS_CHECK(!inputs.empty(), "OpSchema '", name_,
            "': at least one input is required");
  Shape shape = inputs[0].shape();
  if (broadcast_rule_ == OpBroadcastRule::Inputs) {
    for (size_t i = 1; i < inputs.size(); ++i) {
      shape = broadcast_shape(shape, inputs[i].shape());
    }
  } else {
    for (size_t i = 1; i < inputs.size(); ++i) {
      INS_CHECK(inputs[i].shape() == shape, "OpSchema '", name_,
                "': input shape ", inputs[i].shape(),
                " does not match ", shape);
    }
  }
  return shape;
}

DType OpSchema::infer_binary_dtype(DType a, DType b) const {
  switch (promotion_rule_) {
  case OpPromotionRule::None:
  case OpPromotionRule::Identity:
    INS_CHECK(a == b, "OpSchema '", name_, "': dtype mismatch ", dtype_name(a),
              " vs ", dtype_name(b));
    return a;
  case OpPromotionRule::Numeric:
  case OpPromotionRule::Floating:
    return promote_types(a, b);
  case OpPromotionRule::Comparison:
    return DType::BOOL;
  }
  INS_THROW("OpSchema '", name_, "': invalid promotion rule");
}

ArrayIterator OpSchema::make_array_iterator(
    const std::vector<Array> &inputs, const std::vector<Array> &outputs) const {
  INS_CHECK(std::all_of(inputs_.begin(), inputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': array iterator only supports array input schemas");
  INS_CHECK(std::all_of(outputs_.begin(), outputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': array iterator only supports array output schemas");
  INS_CHECK(inputs.size() == inputs_.size(), "OpSchema '", name_,
            "': expected ", inputs_.size(), " inputs, got ", inputs.size());
  INS_CHECK(outputs.size() == outputs_.size(), "OpSchema '", name_,
            "': expected ", outputs_.size(), " outputs, got ", outputs.size());
  return ArrayIterator::elementwise(inputs, outputs);
}

ArrayIterator OpSchema::make_creation_iterator(
    const std::vector<Array> &outputs) const {
  INS_CHECK(kind_ == OpKind::Creation, "OpSchema '", name_,
            "': schema is not a creation op");
  INS_CHECK(inputs_.empty(), "OpSchema '", name_,
            "': creation iterator expects no array input schemas");
  INS_CHECK(outputs.size() == outputs_.size(), "OpSchema '", name_,
            "': expected ", outputs_.size(), " outputs, got ", outputs.size());
  INS_CHECK(std::all_of(outputs_.begin(), outputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': creation iterator only supports array output schemas");
  return ArrayIterator::creation(outputs);
}

ArrayIterator OpSchema::make_transform_iterator(
    const std::vector<Array> &inputs, const std::vector<Array> &outputs) const {
  INS_CHECK(kind_ == OpKind::Manipulation || kind_ == OpKind::Indexing ||
                kind_ == OpKind::Linalg || kind_ == OpKind::Fft ||
                kind_ == OpKind::Signal,
            "OpSchema '", name_,
            "': schema is not a transform op");
  INS_CHECK(inputs.size() == inputs_.size(), "OpSchema '", name_,
            "': expected ", inputs_.size(), " inputs, got ", inputs.size());
  INS_CHECK(outputs.size() == outputs_.size(), "OpSchema '", name_,
            "': expected ", outputs_.size(), " outputs, got ", outputs.size());
  INS_CHECK(std::all_of(inputs_.begin(), inputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': transform iterator only supports array input schemas");
  INS_CHECK(std::all_of(outputs_.begin(), outputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': transform iterator only supports array output schemas");
  return ArrayIterator::transform(inputs, outputs);
}

ArrayIterator OpSchema::make_reduction_iterator(const Array &input,
                                                const Array &output,
                                                int64_t batch_size,
                                                int64_t reduce_size) const {
  return make_reduction_iterator(input, std::vector<Array>{output}, batch_size,
                                 reduce_size);
}

ArrayIterator OpSchema::make_reduction_iterator(
    const Array &input, const std::vector<Array> &outputs, int64_t batch_size,
    int64_t reduce_size) const {
  return make_reduction_iterator(std::vector<Array>{input}, outputs, batch_size,
                                 reduce_size);
}

ArrayIterator OpSchema::make_reduction_iterator(
    const std::vector<Array> &inputs, const std::vector<Array> &outputs,
    int64_t batch_size, int64_t reduce_size) const {
  INS_CHECK(kind_ == OpKind::Reduction, "OpSchema '", name_,
            "': schema is not a reduction");
  INS_CHECK(inputs.size() == inputs_.size(), "OpSchema '", name_,
            "': expected ", inputs_.size(), " inputs, got ", inputs.size());
  INS_CHECK(outputs.size() == outputs_.size(), "OpSchema '", name_,
            "': expected ", outputs_.size(), " outputs, got ", outputs.size());
  INS_CHECK(std::all_of(inputs_.begin(), inputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': reduction iterator only supports array input schemas");
  INS_CHECK(std::all_of(outputs_.begin(), outputs_.end(), is_array_argument),
            "OpSchema '", name_,
            "': reduction iterator only supports array output schemas");
  return ArrayIterator::reduction(inputs, outputs, batch_size, reduce_size);
}

void OpSchemaRegistry::register_schema(const OpSchema &schema) {
  schema_map().insert_or_assign(schema.name(), schema);
}

bool OpSchemaRegistry::has(const std::string &name) {
  return schema_map().find(name) != schema_map().end();
}

const OpSchema &OpSchemaRegistry::get(const std::string &name) {
  auto it = schema_map().find(name);
  INS_CHECK(it != schema_map().end(), "OpSchemaRegistry: schema not found: ",
            name);
  return it->second;
}

std::vector<std::string> OpSchemaRegistry::names() {
  std::vector<std::string> result;
  result.reserve(schema_map().size());
  for (const auto &entry : schema_map()) {
    result.push_back(entry.first);
  }
  std::sort(result.begin(), result.end());
  return result;
}

void OpSchemaRegistry::clear() { schema_map().clear(); }

} // namespace ins
