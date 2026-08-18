// insight/core/shape.h
#pragma once
#include "insight/c_api/array.h" // for INSIGHT_MAX_NDIM
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace ins {

/**
 * @brief Multi-dimensional shape descriptor.
 *
 * Shape represents the size of each dimension of an array.
 * Examples:
 *   Shape()          -> 0-dimensional (scalar)
 *   Shape({2, 3})    -> 2x3 matrix (6 elements)
 *   Shape({2, 3, 4}) -> 3D array (24 elements)
 */
class Shape {
public:
  // Constructors
  Shape() = default;
  Shape(std::initializer_list<int64_t> dims);
  explicit Shape(const std::vector<int64_t> &dims);

  // Basic accessors
  int ndim() const { return ndim_; }
  int64_t dim(int i) const; // supports negative indexing
  int64_t numel() const { return numel_; }
  bool empty() const { return ndim_ == 0; }
  std::vector<int64_t> dims() const {
    return std::vector<int64_t>(dims_, dims_ + ndim_);
  }

  // Shape manipulation (return new Shape, immutable)
  Shape squeeze() const;
  Shape squeeze(int dim) const;
  Shape unsqueeze(int dim) const;

  // Comparison
  bool operator==(const Shape &other) const;
  bool operator!=(const Shape &other) const;

  // String conversion
  std::string to_string() const;

private:
  int64_t dims_[INSIGHT_MAX_NDIM] = {0};
  int ndim_ = 0;
  int64_t numel_ = 1;
};

// Stream output
std::ostream &operator<<(std::ostream &os, const Shape &shape);

} // namespace ins