// insight/core/array.h
#pragma once
#include "insight/c_api/array.h"
#include "insight/core/dtype.h"
#include "insight/core/exception.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/slice.h"
#include "insight/core/strides.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace ins {

struct ArrayStorageMetadata {
  const void *data = nullptr;
  size_t nbytes = 0;
  Place place;
  int32_t ref_count = 0;
  bool is_view = false;
};

/**
 * @brief Multi-dimensional array container.
 *
 * Array is the core data structure of Insight. It represents a dense
 * multi-dimensional array with a given shape, data type, and device placement.
 *
 * Examples:
 *   Array a({2, 3});                    // 2x3 float32 array on CPU
 *   Array b({2, 3}, DType::F64);        // 2x3 float64 array on CPU
 *   Array c({2, 3}, DType::F32, Place::GPU());  // 2x3 on GPU
 */
class Array {
public:
  // ========== Constructors ==========

  /// Create an empty (undefined) array
  Array();

  /// Create a new array with given shape, dtype, and place
  Array(const Shape &shape, DType dtype = DType::F32,
        const Place &place = ins::get_device());

  /**
   * @brief Construct an Array from an existing InsightArray layout.
   *
   * Takes ownership of the layout. The layout's ref_count is incremented.
   *
   * @param layout Pointer to an initialized InsightArray
   */
  explicit Array(InsightArray *layout);

  /// Create a scalar array from a value
  // Boolean
  explicit Array(bool value);

  // Unsigned integers
  explicit Array(uint8_t value);
  explicit Array(uint16_t value);
  explicit Array(uint32_t value);
  explicit Array(uint64_t value);

  // Signed integers
  explicit Array(int8_t value);
  explicit Array(int16_t value);
  explicit Array(int32_t value);
  explicit Array(int64_t value);

  // Floating point
  explicit Array(float value);
  explicit Array(double value);

  // Complex
  explicit Array(std::complex<float> value);
  explicit Array(std::complex<double> value);

  /// Copy constructor
  Array(const Array &other);

  /// Move constructor
  Array(Array &&other) noexcept;

  /// View constructor
  Array(const Array &parent, const Shape &view_shape,
        const Strides &view_strides, int64_t offset);

  /// Destructor
  ~Array();

  // ========== Assignment ==========

  /// Deep copy assignment (view: writes through shared memory)
  Array &operator=(const Array &other);

  /// Scalar fill assignment (in-place, works on views)
  Array &operator=(double value);

  /// Move assignment
  Array &operator=(Array &&other) noexcept;

  // ========== State ==========

  /// Check if array is initialized
  bool defined() const;

  // ========== Metadata ==========

  /// Return the shape of the array
  Shape shape() const;

  /// Return the data type
  DType dtype() const;

  /// Return the device placement
  Place place() const;

  /// Return the total number of elements
  int64_t numel() const;

  /// Return the total number of bytes (logical numel * dtype size)
  size_t nbytes() const;

  /// Return the byte size of the shared base storage allocation
  size_t storage_nbytes() const;

  /// Return explicit shared storage metadata for layout/storage protocols
  ArrayStorageMetadata storage_metadata() const;

  // ========== Memory Layout ==========

  /// Check if the array is contiguous in memory
  bool is_contiguous() const;

  /// Ensure the array is contiguous (creates a copy if needed)
  Array contiguous() const;

  // ========== Data Access ==========

  /// Raw pointer to the first logical element, including view offset
  void *data();
  const void *data() const;

  /// Raw pointer to the shared base storage allocation
  void *storage_data();
  const void *storage_data() const;

  /// Typed raw pointer access
  template <typename T> T *data() { return static_cast<T *>(data()); }

  template <typename T> const T *data() const {
    return static_cast<const T *>(data());
  }

  InsightArray *layout_ptr();

  const InsightArray *layout_ptr() const;

  // ========== Element Access ==========

  /// Return a zero-dimensional array (scalar) at given indices
  /// Use .item<T>() to extract the value
  template <typename... Indices> Array at(Indices... indices) const {
    static_assert(sizeof...(Indices) > 0, "at() requires at least one index");
    std::vector<int64_t> idx = {static_cast<int64_t>(indices)...};
    return at(idx);
  }

  Array at(const std::vector<int64_t> &indices) const;

  template <typename T> T item() const {
    static_assert(std::is_arithmetic_v<T> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>,
                  "item<T>() requires arithmetic type or complex type");
    INS_CHECK(defined(), "Array is not initialized");
    INS_CHECK(numel() == 1, "item() only works for scalar arrays, got ",
              numel());

    // Check type compatibility
    DType expected = dtype_of<T>();
    INS_CHECK(dtype() == expected, "item<", dtype_name(expected),
              ">() called on array of dtype ", dtype_name(dtype()));

    return this->to(ins::CPUPlace()).data<T>()[0];
  }

  // ========== Slicing (Views) ==========

  /// Slice a single dimension
  Array slice(int dim, int64_t start, int64_t stop, int64_t step = 1) const;

  /// Multi-dimensional slice with Slice objects
  Array slice(const std::vector<Slice> &slices) const;

  // Python-style syntactic sugar (read)
  Array operator[](int64_t index) const;
  Array operator[](const std::string &spec) const;
  Array operator[](const Slice &slice) const;

  // Non-const indexing (returns view; assignment writes through shared memory)
  Array operator[](int64_t index);
  Array operator[](const std::string &spec);
  Array operator[](const Slice &slice);

  // ========== View Operations (Zero-Copy) ==========

  /// Reshape the array (view if possible, otherwise copy)
  Array reshape(const Shape &new_shape) const;

  /// Transpose the last two dimensions (view)
  Array transpose() const;

  /// Transpose with arbitrary permutation (advanced)
  Array transpose(const std::vector<int> &perm) const;

  /**
   * @brief Remove dimensions of size 1.
   * @param axis If provided, remove only the specified dimension (must be size
   * 1). If not provided, remove all dimensions of size 1.
   * @return Squeezed view (zero-copy if possible)
   */
  Array squeeze(std::optional<int> axis = std::nullopt) const;

  /// Add a dimension of size 1 (view)
  Array unsqueeze(int dim) const;

  /**
   * @brief Create a view with a different shape (zero-copy).
   *
   * @param new_shape New shape, must have same total number of elements.
   * @return View array sharing the same storage.
   */
  Array view(const Shape &new_shape) const;

  /**
   * @brief Create a view with a different dtype (zero-copy reinterpretation).
   *
   * The new dtype must have the same size as the original dtype.
   *
   * @param new_dtype Target dtype (must have same size)
   * @return View array with reinterpreted dtype.
   */
  Array view(DType new_dtype) const;

  /**
   * @brief Create a view with different shape and dtype (zero-copy).
   *
   * Total bytes: shape.numel() * dtype_size(new_dtype) must equal
   * total bytes of original array.
   *
   * @param new_shape New shape
   * @param new_dtype Target dtype
   * @return View array with new shape and reinterpreted dtype.
   */
  Array view(const Shape &new_shape, DType new_dtype) const;

  // ========== In-place Mutation ==========

  /**
   * @brief Fill all elements with a scalar value (in-place).
   *
   * Works on both contiguous and non-contiguous (view) arrays.
   * Writing through a view modifies the parent array's storage.
   */
  void fill_(double value);

  /**
   * @brief Copy data from src into this array (in-place).
   *
   * This array and src must have the same shape.
   * Dtype conversion is performed if needed.
   * Works on non-contiguous views (uses stride-aware copy).
   * Writing through a view modifies the parent array's storage.
   */
  void copy_from_(const Array &src);

  // ========== Device/Type Conversion ==========

  /// Convert to a different device
  Array to(const Place &target) const;

  /// Convert to a different data type
  Array to(DType target) const;

  /// Convert to a different device and data type
  Array to(const Place &target, DType target_dtype) const;

  /// Convert to same device and dtype as another array
  Array to(const Array &other) const;

  // ========== Copy ==========

  /// Create a deep copy of the array
  Array copy() const;

  // ========== Advanced Accessors ==========

  /**
   * @brief Get the stride information of the array.
   *
   * Strides represent the number of elements to skip to move to the next
   * element along each dimension. This is primarily used for implementing
   * zero-copy views and broadcasting.
   *
   * @return const reference to the strides object
   */
  const Strides &strides() const;

  /**
   * @brief Get the offset of the first element in the underlying storage.
   *
   * The offset is measured in number of elements from the start of the
   * shared storage to the first element of this array view.
   *
   * @return Offset in elements (0 for non-view arrays)
   */
  int64_t offset() const;

  // ========== Boolean conversion ==========

  /**
   * @brief Convert scalar array to bool.
   *
   * For arrays with more than one element, throws an exception to avoid
   * ambiguity. This matches NumPy/PyTorch/Paddle behavior.
   *
   * @throws Exception if numel() != 1
   * @return true if the scalar value is non-zero
   */
  explicit operator bool() const;

  /**
   * @brief Returns true if any element is non-zero.
   */
  bool any() const;

  /**
   * @brief Returns true if all elements are non-zero.
   */
  bool all() const;

private:
  InsightArray layout_;
  Shape shape_;
  Place place_;
  Strides strides_;

  void compute_strides();
  void allocate_storage();
  void deallocate_storage();

  friend class OpRegistry;
  friend Array broadcast_to(const Array &x, const Shape &target_shape);
  friend std::vector<Array> broadcast_arrays(const std::vector<Array> &arrays);

  template <typename T> void set_scalar(T value) {
    *data<T>() = value;
    // Move to default device
    Place default_dev = get_device();
    if (place_ != default_dev) {
      *this = this->to(default_dev);
    }
  }
};

} // namespace ins