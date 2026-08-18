// insight/core/array_iterator.h
#pragma once
#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/strides.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ins {

enum class ArrayIteratorOperandAccess {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

struct ArrayIteratorOperand {
  ArrayIteratorOperandAccess access = ArrayIteratorOperandAccess::ReadOnly;
  DType dtype = DType::UNKNOWN;
  Place place = CPUPlace();
  Shape shape;
  Strides strides;
  int64_t offset = 0;
  bool is_contiguous = false;
  bool is_broadcast = false;
};

class ArrayIterator {
public:
  static ArrayIterator elementwise(const std::vector<Array> &inputs);
  static ArrayIterator elementwise(const std::vector<Array> &inputs,
                                    const std::vector<Array> &outputs);
  static ArrayIterator creation(const std::vector<Array> &outputs);
  static ArrayIterator transform(const std::vector<Array> &inputs,
                                  const std::vector<Array> &outputs);
  static ArrayIterator reduction(const Array &input, const Array &output,
                                  int64_t batch_size, int64_t reduce_size);
  static ArrayIterator reduction(const Array &input,
                                  const std::vector<Array> &outputs,
                                  int64_t batch_size, int64_t reduce_size);
  static ArrayIterator reduction(const std::vector<Array> &inputs,
                                  const std::vector<Array> &outputs,
                                  int64_t batch_size, int64_t reduce_size);

  const Shape &shape() const { return shape_; }
  int ndim() const { return shape_.ndim(); }
  int64_t numel() const { return shape_.numel(); }
  const std::vector<ArrayIteratorOperand> &operands() const {
    return operands_;
  }
  const std::vector<Array> &arrays() const { return arrays_; }
  const Array &array(size_t index) const { return arrays_.at(index); }

  bool all_contiguous() const { return all_contiguous_; }
  bool has_cpu_operands() const { return has_cpu_operands_; }
  bool has_gpu_operands() const { return has_gpu_operands_; }
  bool is_reduction() const { return is_reduction_; }
  int64_t reduction_batch_size() const { return reduction_batch_size_; }
  int64_t reduction_reduce_size() const { return reduction_reduce_size_; }

private:
  Shape shape_;
  std::vector<ArrayIteratorOperand> operands_;
  std::vector<Array> arrays_;
  bool all_contiguous_ = false;
  bool has_cpu_operands_ = false;
  bool has_gpu_operands_ = false;
  bool is_reduction_ = false;
  int64_t reduction_batch_size_ = 1;
  int64_t reduction_reduce_size_ = 1;
};

} // namespace ins
