// src/core/array_iterator.cpp
#include "insight/core/array_iterator.h"
#include "insight/core/exception.h"
#include "insight/ops/broadcast.h"

namespace ins {

namespace {

Shape common_shape_for(const std::vector<Array> &arrays) {
  INS_CHECK(!arrays.empty(), "ArrayIterator: at least one operand required");

  Shape shape = arrays[0].shape();
  for (size_t i = 1; i < arrays.size(); ++i) {
    shape = broadcast_shape(shape, arrays[i].shape());
  }
  return shape;
}

ArrayIteratorOperand make_operand(const Array &view, const Array &source,
                                   const Shape &iterator_shape,
                                   ArrayIteratorOperandAccess access) {
  ArrayIteratorOperand operand;
  operand.access = access;
  operand.dtype = view.dtype();
  operand.place = view.place();
  operand.shape = view.shape();
  operand.strides = view.strides();
  operand.offset = view.offset();
  operand.is_contiguous = view.is_contiguous();
  operand.is_broadcast = view.shape() != source.shape();
  if (!operand.is_broadcast) {
    for (int i = 0; i < operand.strides.ndim(); ++i) {
      if (operand.strides[i] == 0 && iterator_shape.dim(i) > 1) {
        operand.is_broadcast = true;
        break;
      }
    }
  }
  return operand;
}

} // namespace

ArrayIterator ArrayIterator::elementwise(const std::vector<Array> &inputs) {
  return elementwise(inputs, {});
}

ArrayIterator ArrayIterator::elementwise(const std::vector<Array> &inputs,
                                           const std::vector<Array> &outputs) {
  INS_CHECK(!inputs.empty(), "ArrayIterator: at least one input required");
  for (size_t i = 0; i < inputs.size(); ++i) {
    INS_CHECK(inputs[i].defined(), "ArrayIterator: input ", i,
              " is undefined");
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    INS_CHECK(outputs[i].defined(), "ArrayIterator: output ", i,
              " is undefined");
  }

  ArrayIterator iter;
  iter.shape_ = common_shape_for(inputs);

  for (const auto &input : inputs) {
    Array view = broadcast_to(input, iter.shape_);
    iter.arrays_.push_back(view);
    iter.operands_.push_back(make_operand(
        view, input, iter.shape_, ArrayIteratorOperandAccess::ReadOnly));
  }
  for (const auto &output : outputs) {
    INS_CHECK(output.shape() == iter.shape_, "ArrayIterator: output shape ",
              output.shape(), " does not match iterator shape ", iter.shape_);
    iter.arrays_.push_back(output);
    iter.operands_.push_back(make_operand(
        output, output, iter.shape_, ArrayIteratorOperandAccess::WriteOnly));
  }

  iter.all_contiguous_ = true;
  for (const auto &operand : iter.operands_) {
    iter.all_contiguous_ = iter.all_contiguous_ && operand.is_contiguous;
    iter.has_cpu_operands_ = iter.has_cpu_operands_ || operand.place.is_cpu();
    iter.has_gpu_operands_ = iter.has_gpu_operands_ || operand.place.is_gpu();
  }

  return iter;
}

ArrayIterator ArrayIterator::creation(const std::vector<Array> &outputs) {
  INS_CHECK(!outputs.empty(), "ArrayIterator: at least one creation output required");

  ArrayIterator iter;
  iter.shape_ = outputs[0].shape();
  for (const auto &output : outputs) {
    INS_CHECK(output.defined(), "ArrayIterator: creation output is undefined");
    INS_CHECK(output.shape() == iter.shape_, "ArrayIterator: creation output shape ",
              output.shape(), " does not match first output shape ", iter.shape_);
    iter.arrays_.push_back(output);
    iter.operands_.push_back(make_operand(
        output, output, output.shape(), ArrayIteratorOperandAccess::WriteOnly));
  }

  iter.all_contiguous_ = true;
  for (const auto &operand : iter.operands_) {
    iter.all_contiguous_ = iter.all_contiguous_ && operand.is_contiguous;
    iter.has_cpu_operands_ = iter.has_cpu_operands_ || operand.place.is_cpu();
    iter.has_gpu_operands_ = iter.has_gpu_operands_ || operand.place.is_gpu();
  }

  return iter;
}

ArrayIterator ArrayIterator::transform(const std::vector<Array> &inputs,
                                       const std::vector<Array> &outputs) {
  INS_CHECK(!inputs.empty(), "ArrayIterator: at least one transform input required");
  INS_CHECK(!outputs.empty(), "ArrayIterator: at least one transform output required");

  ArrayIterator iter;
  iter.shape_ = outputs[0].shape();

  for (const auto &input : inputs) {
    INS_CHECK(input.defined(), "ArrayIterator: transform input is undefined");
    iter.arrays_.push_back(input);
    iter.operands_.push_back(make_operand(
        input, input, input.shape(), ArrayIteratorOperandAccess::ReadOnly));
  }

  for (const auto &output : outputs) {
    INS_CHECK(output.defined(), "ArrayIterator: transform output is undefined");
    iter.arrays_.push_back(output);
    iter.operands_.push_back(make_operand(
        output, output, output.shape(), ArrayIteratorOperandAccess::WriteOnly));
  }

  iter.all_contiguous_ = true;
  for (const auto &operand : iter.operands_) {
    iter.all_contiguous_ = iter.all_contiguous_ && operand.is_contiguous;
    iter.has_cpu_operands_ = iter.has_cpu_operands_ || operand.place.is_cpu();
    iter.has_gpu_operands_ = iter.has_gpu_operands_ || operand.place.is_gpu();
  }

  return iter;
}

ArrayIterator ArrayIterator::reduction(const Array &input, const Array &output,
                                       int64_t batch_size,
                                       int64_t reduce_size) {
  return reduction(input, std::vector<Array>{output}, batch_size, reduce_size);
}

ArrayIterator ArrayIterator::reduction(const Array &input,
                                       const std::vector<Array> &outputs,
                                       int64_t batch_size,
                                       int64_t reduce_size) {
  return reduction(std::vector<Array>{input}, outputs, batch_size, reduce_size);
}

ArrayIterator ArrayIterator::reduction(const std::vector<Array> &inputs,
                                       const std::vector<Array> &outputs,
                                       int64_t batch_size,
                                       int64_t reduce_size) {
  INS_CHECK(!inputs.empty(), "ArrayIterator: at least one reduction input required");
  INS_CHECK(!outputs.empty(), "ArrayIterator: at least one reduction output required");
  INS_CHECK(batch_size >= 0, "ArrayIterator: reduction batch size must be >= 0");
  INS_CHECK(reduce_size >= 0, "ArrayIterator: reduction size must be >= 0");

  ArrayIterator iter;
  iter.shape_ = outputs[0].shape();

  for (const auto &input : inputs) {
    INS_CHECK(input.defined(), "ArrayIterator: reduction input is undefined");
    iter.arrays_.push_back(input);
    iter.operands_.push_back(make_operand(
        input, input, input.shape(), ArrayIteratorOperandAccess::ReadOnly));
  }

  for (const auto &output : outputs) {
    INS_CHECK(output.defined(), "ArrayIterator: reduction output is undefined");
    INS_CHECK(output.shape() == iter.shape_,
              "ArrayIterator: reduction output shape ", output.shape(),
              " does not match first output shape ", iter.shape_);
    iter.arrays_.push_back(output);
    iter.operands_.push_back(make_operand(
        output, output, output.shape(), ArrayIteratorOperandAccess::WriteOnly));
  }

  iter.is_reduction_ = true;
  iter.reduction_batch_size_ = batch_size;
  iter.reduction_reduce_size_ = reduce_size;
  iter.all_contiguous_ = true;
  for (const auto &operand : iter.operands_) {
    iter.all_contiguous_ = iter.all_contiguous_ && operand.is_contiguous;
    iter.has_cpu_operands_ = iter.has_cpu_operands_ || operand.place.is_cpu();
    iter.has_gpu_operands_ = iter.has_gpu_operands_ || operand.place.is_gpu();
  }

  return iter;
}

} // namespace ins
