// src/core/array.cpp
#include "insight/core/array.h"
#include "insight/c_api/array.h"
#include "insight/c_api/dtype.h"
#include "insight/c_api/exception.h"
#include "insight/c_api/place.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/slice.h"
#include "insight/core/strides.h"
#include "insight/io/print.h"
#include "insight/ops/cast.h"
#include "insight/ops/creation.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include <complex>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace {

int32_t ref_count_increment(int32_t *ref_count) {
#if defined(__GNUC__) || defined(__clang__)
  return __atomic_add_fetch(ref_count, 1, __ATOMIC_ACQ_REL);
#else
  return ++(*ref_count);
#endif
}

int32_t ref_count_decrement(int32_t *ref_count) {
#if defined(__GNUC__) || defined(__clang__)
  return __atomic_sub_fetch(ref_count, 1, __ATOMIC_ACQ_REL);
#else
  return --(*ref_count);
#endif
}

int32_t ref_count_value(const int32_t *ref_count) {
  if (!ref_count)
    return 0;
#if defined(__GNUC__) || defined(__clang__)
  return __atomic_load_n(ref_count, __ATOMIC_ACQUIRE);
#else
  return *ref_count;
#endif
}

} // namespace

// ========================================================================
// C API Implementations
// ========================================================================

extern "C" {

C_Status insight_array_create(InsightArray *array, void *data,
                              const int64_t *dims, int32_t ndim, int32_t dtype,
                              int32_t device_type, int32_t device_id) {

  // Validate array pointer
  if (!array) {
    insight_set_last_error("insight_array_create: array must be non-null");
    return C_FAILED;
  }

  // Validate ndim range (0 is valid for scalar)
  if (ndim < 0 || ndim > INSIGHT_MAX_NDIM) {
    insight_set_last_error(
        "insight_array_create: ndim out of range (0 to INSIGHT_MAX_NDIM)");
    return C_FAILED;
  }

  // Validate dims: must be non-null only when ndim > 0
  if (ndim > 0 && !dims) {
    insight_set_last_error(
        "insight_array_create: dims must be non-null when ndim > 0");
    return C_FAILED;
  }

  // Validate dtype
  if (dtype <= INSIGHT_DTYPE_UNKNOWN || dtype >= INSIGHT_DTYPE_COUNT) {
    insight_set_last_error("insight_array_create: invalid dtype");
    return C_FAILED;
  }

  // Initialize layout to zeros
  std::memset(array, 0, sizeof(InsightArray));

  // Set basic fields
  array->data = data;
  array->ndim = ndim;
  array->dtype = dtype;
  array->device_type = device_type;
  array->device_id = device_id;
  array->is_view = 0;
  array->offset = 0;

  // Calculate numel and copy dimensions
  array->numel = 1;
  for (int32_t i = 0; i < ndim; ++i) {
    array->dims[i] = dims[i];
    array->numel *= dims[i];
  }

  // Calculate strides (row-major / C-order)
  if (ndim == 0) {
    // Scalar: single element, stride is 1
    array->strides[0] = 1;
  } else {
    // Last dimension stride is 1
    array->strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
      array->strides[i] = array->strides[i + 1] * array->dims[i + 1];
    }
  }

  // Initialize reference count
  array->ref_count = new int32_t(1);
  array->storage_nbytes = insight_array_nbytes(array);

  return C_SUCCESS;
}

C_Status insight_array_destroy(InsightArray *array) {
  if (!array || !array->ref_count) {
    insight_set_last_error(
        "insight_array_destroy: array must be non-null with valid ref_count");
    return C_FAILED;
  }
  int32_t remaining = ref_count_decrement(array->ref_count);
  if (remaining == 0) {
    delete array->ref_count;
    array->ref_count = nullptr;
  }
  array->data = nullptr;
  array->ndim = 0;
  array->numel = 0;
  return C_SUCCESS;
}

C_Status insight_array_create_view(InsightArray *dst, const InsightArray *src,
                                   int64_t offset, const int64_t *dims,
                                   int32_t ndim, const int64_t *strides) {

  if (!dst || !src || !src->ref_count) {
    insight_set_last_error("insight_array_create_view: dst, src, and "
                           "src->ref_count must be non-null");
    return C_FAILED;
  }

  // Allow ndim == 0 (scalar view)
  if (ndim < 0 || ndim > INSIGHT_MAX_NDIM) {
    insight_set_last_error("insight_array_create_view: ndim out of range");
    return C_FAILED;
  }

  // dims and strides are only required if ndim > 0
  if (ndim > 0 && (!dims || !strides)) {
    insight_set_last_error("insight_array_create_view: dims and strides must "
                           "be non-null when ndim > 0");
    return C_FAILED;
  }

  std::memset(dst, 0, sizeof(InsightArray));
  dst->data = src->data;
  dst->ref_count = src->ref_count;
  ref_count_increment(dst->ref_count);
  dst->is_view = 1;
  dst->offset = offset;
  dst->ndim = ndim;
  dst->dtype = src->dtype;
  dst->device_type = src->device_type;
  dst->device_id = src->device_id;
  dst->storage_nbytes = src->storage_nbytes;

  dst->numel = 1;
  for (int32_t i = 0; i < ndim; ++i) {
    dst->dims[i] = dims[i];
    dst->strides[i] = strides[i];
    dst->numel *= dims[i];
  }

  // scalar when strides is set to 1
  if (ndim == 0) {
    dst->strides[0] = 1;
  }

  return C_SUCCESS;
}

int64_t insight_array_numel(const InsightArray *array) {
  return array ? array->numel : 0;
}

size_t insight_array_nbytes(const InsightArray *array) {
  if (!array)
    return 0;
  return static_cast<size_t>(array->numel) * insight_dtype_size(array->dtype);
}

const void *insight_array_data(const InsightArray *array) {
  if (!array || !array->data)
    return nullptr;
  size_t elem_size = insight_dtype_size(array->dtype);
  return static_cast<const char *>(array->data) + array->offset * elem_size;
}

const void *insight_array_storage_data(const InsightArray *array) {
  return array ? array->data : nullptr;
}

size_t insight_array_storage_nbytes(const InsightArray *array) {
  if (!array)
    return 0;
  return array->storage_nbytes ? array->storage_nbytes
                               : insight_array_nbytes(array);
}

int insight_array_is_contiguous(const InsightArray *array) {
  if (!array)
    return 0;
  if (array->ndim == 0 || array->numel <= 1)
    return 1;
  int64_t expected_stride = 1;
  for (int32_t i = array->ndim - 1; i >= 0; --i) {
    if (array->strides[i] != expected_stride)
      return 0;
    expected_stride *= array->dims[i];
  }
  return 1;
}

char *insight_array_tostring(const InsightArray *array) {
  if (!array || !array->data)
    return nullptr;
  try {
    // Build a non-const copy so to_string can copy-to-CPU
    InsightArray tmp = *array;
    ins::Array cpp_arr(&tmp);
    tmp.ref_count = nullptr;
    std::string s = ins::to_string(cpp_arr);
    char *result = static_cast<char *>(std::malloc(s.size() + 1));
    if (!result)
      return nullptr;
    std::memcpy(result, s.c_str(), s.size() + 1);
    return result;
  } catch (...) {
    return nullptr;
  }
}

} // extern "C"

// ========================================================================
// Array C++ Class Implementation
// ========================================================================

namespace ins {

namespace {

size_t storage_nbytes_or_logical(const InsightArray &layout) {
  return layout.storage_nbytes ? layout.storage_nbytes
                               : insight_array_nbytes(&layout);
}

void release_layout_storage(InsightArray &layout, Place &place) {
  if (!layout.ref_count)
    return;

  int32_t *ref_count = layout.ref_count;
  void *data = layout.data;
  size_t bytes = storage_nbytes_or_logical(layout);
  int32_t remaining = ref_count_decrement(ref_count);
  if (remaining == 0) {
    delete ref_count;
    if (data) {
      place.deallocate(data, bytes);
    }
  }
  std::memset(&layout, 0, sizeof(layout));
}

} // namespace

// ========== Constructors ==========

Array::Array() { std::memset(&layout_, 0, sizeof(layout_)); }

Array::Array(const Shape &shape, DType dtype, const Place &place)
    : shape_(shape), place_(place) {

  size_t bytes = shape.numel() * dtype_size(dtype);
  void *data = nullptr;
  if (bytes > 0) {
    data = place_.allocate(bytes);
    if (data && place_.is_cpu()) {
      std::memset(data, 0, bytes);
    } else if (data && place_.is_gpu()) {
      const C_DeviceInterface *iface = place_.device_interface();
      if (iface && iface->device_memory_set) {
        C_Device_st dev;
        dev.id = place_.device_id();
        iface->device_memory_set(&dev, data, 0, bytes);
      }
    }
  }

  std::vector<int64_t> dims_vec = shape.dims();
  insight_array_create(&layout_, data, dims_vec.data(), shape.ndim(),
                       static_cast<int32_t>(dtype),
                       place_.is_gpu() ? INSIGHT_DEVICE_GPU
                                       : INSIGHT_DEVICE_CPU,
                       place_.device_id());
  compute_strides();
}

Array::Array(const Array &other)
    : layout_(other.layout_), shape_(other.shape_), place_(other.place_),
      strides_(other.strides_) {
  if (layout_.ref_count) {
    ref_count_increment(layout_.ref_count);
  }
}

Array::Array(InsightArray *layout) {
  // Copy layout content
  layout_ = *layout;
  if (layout_.ref_count) {
    ref_count_increment(layout_.ref_count);
  }

  // Build shape_ from layout
  std::vector<int64_t> dims(layout_.ndim);
  for (int i = 0; i < layout_.ndim; ++i) {
    dims[i] = layout_.dims[i];
  }
  shape_ = Shape(dims);

  // Build strides_ from layout
  std::vector<int64_t> strides_vec(layout_.ndim);
  for (int i = 0; i < layout_.ndim; ++i) {
    strides_vec[i] = layout_.strides[i];
  }
  strides_ = Strides(strides_vec);

  // Build place_ from layout
  place_ = Place(layout_.device_type == INSIGHT_DEVICE_CPU ? DeviceKind::CPU
                                                           : DeviceKind::GPU,
                 layout_.device_id);
}

// Scalar constructors
Array::Array(bool value) : Array(Shape({}), DType::BOOL, CPUPlace()) {
  set_scalar(value);
}
Array::Array(int8_t value) : Array(Shape({}), DType::I8, CPUPlace()) {
  set_scalar(value);
}
Array::Array(int16_t value) : Array(Shape({}), DType::I16, CPUPlace()) {
  set_scalar(value);
}
Array::Array(int32_t value) : Array(Shape({}), DType::I32, CPUPlace()) {
  set_scalar(value);
}
Array::Array(int64_t value) : Array(Shape({}), DType::I64, CPUPlace()) {
  set_scalar(value);
}
Array::Array(uint8_t value) : Array(Shape({}), DType::U8, CPUPlace()) {
  set_scalar(value);
}
Array::Array(uint16_t value) : Array(Shape({}), DType::U16, CPUPlace()) {
  set_scalar(value);
}
Array::Array(uint32_t value) : Array(Shape({}), DType::U32, CPUPlace()) {
  set_scalar(value);
}
Array::Array(uint64_t value) : Array(Shape({}), DType::U64, CPUPlace()) {
  set_scalar(value);
}
Array::Array(float value) : Array(Shape({}), DType::F32, CPUPlace()) {
  set_scalar(value);
}
Array::Array(double value) : Array(Shape({}), DType::F64, CPUPlace()) {
  set_scalar(value);
}
Array::Array(std::complex<float> value)
    : Array(Shape({}), DType::C32, CPUPlace()) {
  set_scalar(value);
}
Array::Array(std::complex<double> value)
    : Array(Shape({}), DType::C64, CPUPlace()) {
  set_scalar(value);
}

// View constructor
Array::Array(const Array &parent, const Shape &view_shape,
             const Strides &view_strides, int64_t offset)
    : shape_(view_shape), place_(parent.place_), strides_(view_strides) {

  int64_t strides_arr[INSIGHT_MAX_NDIM];
  for (int i = 0; i < view_strides.ndim(); ++i) {
    strides_arr[i] = view_strides[i];
  }
  std::vector<int64_t> dims_vec = view_shape.dims();
  insight_array_create_view(&layout_, &parent.layout_, offset, dims_vec.data(),
                            view_shape.ndim(), strides_arr);
}

Array::~Array() { release_layout_storage(layout_, place_); }

// ========== Assignment ==========

Array &Array::operator=(const Array &other) {
  if (this == &other)
    return *this;

  // View with matching shape/dtype: write through shared memory
  if (layout_.is_view && defined() && other.defined() &&
      shape_ == other.shape_ && dtype() == other.dtype()) {
    copy_from_(other);
    return *this;
  }

  // Rebind (non-view, different shape/dtype, or uninitialized)
  if (other.layout_.ref_count) {
    ref_count_increment(other.layout_.ref_count);
  }
  release_layout_storage(layout_, place_);
  layout_ = other.layout_;
  shape_ = other.shape_;
  place_ = other.place_;
  strides_ = other.strides_;
  return *this;
}

Array &Array::operator=(double value) {
  fill_(value);
  return *this;
}

Array::Array(Array &&other) noexcept
    : layout_(other.layout_), shape_(std::move(other.shape_)),
      place_(other.place_), strides_(std::move(other.strides_)) {
  std::memset(&other.layout_, 0, sizeof(other.layout_));
}

Array &Array::operator=(Array &&other) noexcept {
  if (this != &other) {
    release_layout_storage(layout_, place_);

    layout_ = other.layout_;
    shape_ = std::move(other.shape_);
    place_ = other.place_;
    strides_ = std::move(other.strides_);
    std::memset(&other.layout_, 0, sizeof(other.layout_));
  }
  return *this;
}

// ========== State ==========

bool Array::defined() const { return layout_.ref_count != nullptr; }

// ========== Metadata ==========

Shape Array::shape() const {
  INS_CHECK(defined(), "Array is not initialized");
  return shape_;
}

DType Array::dtype() const {
  INS_CHECK(defined(), "Array is not initialized");
  return static_cast<DType>(layout_.dtype);
}

Place Array::place() const {
  INS_CHECK(defined(), "Array is not initialized");
  return place_;
}

int64_t Array::numel() const {
  INS_CHECK(defined(), "Array is not initialized");
  return layout_.numel;
}

size_t Array::nbytes() const {
  INS_CHECK(defined(), "Array is not initialized");
  return insight_array_nbytes(&layout_);
}

size_t Array::storage_nbytes() const {
  INS_CHECK(defined(), "Array is not initialized");
  return insight_array_storage_nbytes(&layout_);
}

ArrayStorageMetadata Array::storage_metadata() const {
  INS_CHECK(defined(), "Array is not initialized");
  return {insight_array_storage_data(&layout_),
          insight_array_storage_nbytes(&layout_), place_,
          ref_count_value(layout_.ref_count), layout_.is_view != 0};
}

// ========== Memory Layout ==========

bool Array::is_contiguous() const {
  INS_CHECK(defined(), "Array is not initialized");
  if (numel() <= 1)
    return true;
  return insight_array_is_contiguous(&layout_);
}

Array Array::contiguous() const { return ins::contiguous(*this); }

// ========== Data Access ==========

void *Array::data() {
  INS_CHECK(defined(), "Array is not initialized");
  return const_cast<void *>(insight_array_data(&layout_));
}

const void *Array::data() const {
  INS_CHECK(defined(), "Array is not initialized");
  return insight_array_data(&layout_);
}

void *Array::storage_data() {
  INS_CHECK(defined(), "Array is not initialized");
  return layout_.data;
}

const void *Array::storage_data() const {
  INS_CHECK(defined(), "Array is not initialized");
  return insight_array_storage_data(&layout_);
}

InsightArray *Array::layout_ptr() { return &layout_; }

const InsightArray *Array::layout_ptr() const { return &layout_; }

const Strides &Array::strides() const { return strides_; }

int64_t Array::offset() const { return layout_.offset; }

// ========== Element Access ==========

Array Array::at(const std::vector<int64_t> &indices) const {
  INS_CHECK(defined(), "Array is not initialized");
  int ndim = shape_.ndim();
  INS_CHECK(indices.size() <= static_cast<size_t>(ndim),
            "at(): too many indices (got ", indices.size(), ", array has ",
            ndim, " dimensions)");

  // Full indexing: return scalar view (original behavior)
  if (static_cast<int>(indices.size()) == ndim) {
    int64_t elem_offset = layout_.offset;
    for (int i = 0; i < ndim; ++i) {
      int64_t idx = indices[i];
      int64_t dim_size = shape_.dim(i);
      if (idx < 0)
        idx += dim_size;
      INS_CHECK(idx >= 0 && idx < dim_size, "at(): index out of range");
      elem_offset += idx * strides_[i];
    }
    return Array(*this, Shape({}), Strides(), elem_offset);
  }

  // Partial indexing (NumPy-style): indexed dims become scalars,
  // remaining dims are kept as full slices.
  // e.g. a[1] on (3,4) → shape (4,), a[1] on (2,3,4) → shape (3,4)
  std::vector<Slice> slices;
  int64_t new_offset = layout_.offset;
  for (int i = 0; i < ndim; ++i) {
    if (i < static_cast<int>(indices.size())) {
      int64_t idx = indices[i];
      int64_t dim_size = shape_.dim(i);
      if (idx < 0)
        idx += dim_size;
      INS_CHECK(idx >= 0 && idx < dim_size, "at(): index out of range");
      new_offset += idx * strides_[i];
    } else {
      slices.push_back(Slice::all());
    }
  }

  // Build result shape from remaining (non-indexed) dimensions
  std::vector<int64_t> new_dims;
  std::vector<int64_t> new_strides_vec;
  for (int i = static_cast<int>(indices.size()); i < ndim; ++i) {
    new_dims.push_back(shape_.dim(i));
    new_strides_vec.push_back(strides_[i]);
  }

  return Array(*this, Shape(new_dims), Strides(new_strides_vec), new_offset);
}

// ========== Slicing (Views) ==========

Array Array::slice(int dim, int64_t start, int64_t stop, int64_t step) const {
  INS_CHECK(defined(), "Cannot slice undefined array");
  INS_CHECK(dim >= 0 && dim < shape_.ndim(),
            "slice(): dimension index out of range");

  std::vector<Slice> slices(shape_.ndim(), Slice::all());
  slices[dim] = Slice(start, stop, step);
  return slice(slices);
}

Array Array::slice(const std::vector<Slice> &slices) const {
  INS_CHECK(defined(), "Cannot slice undefined array");
  INS_CHECK(slices.size() == static_cast<size_t>(shape_.ndim()),
            "slice(): number of slices must match number of dimensions");

  std::vector<int64_t> new_dims;
  int64_t new_offset = layout_.offset;

  for (size_t i = 0; i < slices.size(); ++i) {
    const Slice &s = slices[i];
    int64_t dim_size = shape_.dim(static_cast<int>(i));
    int64_t base_stride = strides_[i];

    int64_t start, stop, step;
    s.normalize(dim_size, start, stop, step);

    int64_t new_dim;
    if (step > 0) {
      new_dim = (stop - start + step - 1) / step;
    } else {
      new_dim = (start - stop + (-step) - 1) / (-step);
    }
    if (new_dim < 0)
      new_dim = 0;
    new_dims.push_back(new_dim);

    new_offset += start * base_stride;
  }

  // Build new strides from original strides and step values
  std::vector<int64_t> new_strides_vec(slices.size());
  for (size_t i = 0; i < slices.size(); ++i) {
    new_strides_vec[i] = strides_[i] * slices[i].step;
  }
  Strides new_strides(new_strides_vec);

  Shape new_shape(new_dims);
  return Array(*this, new_shape, new_strides, new_offset);
}

Array Array::operator[](int64_t index) const { return at({index}); }

Array Array::operator[](const Slice &slice) const {
  return this->slice(0, slice.start.value_or(0),
                     slice.stop.value_or(shape_.dim(0)), slice.step);
}

// Helper: parse a single dimension spec (":", "1:-1", "5", etc.)
// Returns either a Slice or an integer index stored as optional<int64_t>.
// If the spec is a pure integer, index_out is set and slice_out is nullopt.
// If the spec contains ':', slice_out is set and index_out is nullopt.
static void parse_dim_spec(const std::string &spec,
                           std::optional<Slice> &slice_out,
                           std::optional<int64_t> &index_out) {
  if (spec.empty()) {
    INS_THROW("Invalid index: empty dimension spec");
  }

  if (spec.find(':') != std::string::npos) {
    slice_out = parse_slice(spec);
    index_out = std::nullopt;
    return;
  }

  // Check if it's a valid integer
  bool is_number = true;
  for (size_t i = 0; i < spec.size(); ++i) {
    char c = spec[i];
    if (i == 0 && c == '-')
      continue;
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      is_number = false;
      break;
    }
  }

  if (is_number) {
    index_out = std::stoll(spec);
    slice_out = std::nullopt;
    return;
  }

  INS_THROW("Invalid index/slice syntax: ", spec);
}

Array Array::operator[](const std::string &spec) const {
  // Split on commas for multi-dimensional indexing
  std::vector<std::string> parts;
  std::string part;
  for (char c : spec) {
    if (c == ',') {
      parts.push_back(part);
      part.clear();
    } else {
      part += c;
    }
  }
  parts.push_back(part);

  // Parse each dimension
  std::vector<Slice> slices;
  bool any_slice = false;

  for (const auto &p : parts) {
    std::optional<Slice> s;
    std::optional<int64_t> idx;
    parse_dim_spec(p, s, idx);

    if (s.has_value()) {
      slices.push_back(s.value());
      any_slice = true;
    } else {
      // Integer index: convert to a slice that selects a single element
      // We'll apply at() for pure integer indexing below
      slices.push_back(Slice(idx.value(), idx.value() + 1, 1));
    }
  }

  // If all dimensions are integer indices, use at() with all indices at once
  if (!any_slice) {
    std::vector<int64_t> indices;
    for (const auto &p : parts) {
      indices.push_back(std::stoll(p));
    }
    return at(indices);
  }

  // Apply multi-dimensional slice
  int ndim = shape_.ndim();
  INS_CHECK(static_cast<int>(slices.size()) <= ndim,
            "Too many slice dimensions: got ", slices.size(), ", array has ",
            ndim, " dimensions");

  // Build full slice list (missing trailing dims = Slice::all())
  std::vector<Slice> full_slices(slices);
  while (static_cast<int>(full_slices.size()) < ndim) {
    full_slices.push_back(Slice::all());
  }

  // Apply multi-dimensional slice in one call
  // (Slice objects preserve optional nullopt for correct reverse-slice
  // defaults)
  Array result = this->slice(full_slices);

  // Squeeze dimensions that were integer-indexed (not slices)
  for (int i = static_cast<int>(parts.size()) - 1; i >= 0; --i) {
    const auto &p = parts[i];
    if (p.find(':') == std::string::npos) {
      // This was an integer index, squeeze dimension i
      result = result.squeeze(i);
    }
  }

  return result;
}

// Non-const operator[] (returns view for assignment-through)
Array Array::operator[](int64_t index) {
  return static_cast<const Array &>(*this)[index];
}
Array Array::operator[](const std::string &spec) {
  return static_cast<const Array &>(*this)[spec];
}
Array Array::operator[](const Slice &slice) {
  return static_cast<const Array &>(*this)[slice];
}

// ========== View Operations ==========

Array Array::reshape(const Shape &new_shape) const {
  INS_CHECK(defined(), "Cannot reshape undefined array");

  // Resolve -1 (infer one dimension from numel)
  auto dims = new_shape.dims();
  int neg1_count = 0;
  int64_t known_product = 1;
  for (int i = 0; i < new_shape.ndim(); i++) {
    if (dims[i] == -1) {
      neg1_count++;
    } else {
      INS_CHECK(dims[i] > 0,
                "reshape(): all dimensions must be > 0 or -1, got ", dims[i],
                " at dim ", i);
      known_product *= dims[i];
    }
  }
  INS_CHECK(neg1_count <= 1, "reshape(): can only infer one dimension, got ",
            neg1_count, " occurrences of -1");
  if (neg1_count == 1) {
    INS_CHECK(known_product > 0 && numel() % known_product == 0,
              "reshape(): cannot infer dimension: numel()=", numel(),
              " not divisible by product of known dims=", known_product);
    for (auto &d : dims) {
      if (d == -1)
        d = numel() / known_product;
    }
  }

  Shape resolved(dims);
  INS_CHECK(resolved.numel() == numel(),
            "reshape(): shape.numel() mismatch. Expected ", numel(), ", got ",
            resolved.numel());

  if (!is_contiguous()) {
    Array cont = contiguous();
    return cont.reshape(resolved);
  }

  Strides new_strides(resolved);
  return Array(*this, resolved, new_strides, layout_.offset);
}

Array Array::transpose() const {
  INS_CHECK(defined(), "Cannot transpose undefined array");
  INS_CHECK(shape_.ndim() >= 2,
            "transpose(): requires at least 2 dimensions, got ", shape_.ndim());

  int ndim = shape_.ndim();
  std::vector<int64_t> new_dims = shape_.dims();
  std::swap(new_dims[ndim - 1], new_dims[ndim - 2]);
  Shape new_shape(new_dims);

  std::vector<int64_t> new_strides_vec = strides_.data();
  std::swap(new_strides_vec[ndim - 1], new_strides_vec[ndim - 2]);
  Strides new_strides(new_strides_vec);

  return Array(*this, new_shape, new_strides, layout_.offset);
}

Array Array::transpose(const std::vector<int> &perm) const {
  INS_CHECK(defined(), "transpose: input is undefined");
  int ndim = shape_.ndim();
  INS_CHECK(perm.size() == static_cast<size_t>(ndim),
            "transpose(): perm size must match number of dimensions");

  std::vector<int64_t> new_dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    new_dims[i] = shape_.dim(perm[i]);
  }
  Shape new_shape(new_dims);

  std::vector<int64_t> new_strides_vec(ndim);
  for (int i = 0; i < ndim; ++i) {
    new_strides_vec[i] = strides_[perm[i]];
  }
  Strides new_strides(new_strides_vec);

  return Array(*this, new_shape, new_strides, layout_.offset);
}

Array Array::squeeze(std::optional<int> axis) const {
  INS_CHECK(defined(), "Cannot squeeze undefined array");

  if (axis.has_value()) {
    int ax = axis.value();
    int nd = shape_.ndim();
    if (ax < 0)
      ax += nd;
    INS_CHECK(ax >= 0 && ax < nd, "squeeze: axis out of range");
    INS_CHECK(shape_.dim(ax) == 1, "squeeze: axis ", ax, " has size ",
              shape_.dim(ax), ", cannot squeeze");

    std::vector<int64_t> new_dims;
    for (int i = 0; i < nd; ++i) {
      if (i != ax)
        new_dims.push_back(shape_.dim(i));
    }
    return reshape(Shape(new_dims));
  } else {
    std::vector<int64_t> new_dims;
    for (int i = 0; i < shape_.ndim(); ++i) {
      if (shape_.dim(i) != 1)
        new_dims.push_back(shape_.dim(i));
    }
    if (new_dims.size() == static_cast<size_t>(shape_.ndim()))
      return *this;
    return reshape(Shape(new_dims));
  }
}

Array Array::unsqueeze(int dim) const {
  INS_CHECK(defined(), "Cannot unsqueeze undefined array");
  int nd = shape_.ndim();
  if (dim < 0)
    dim += nd + 1;
  INS_CHECK(dim >= 0 && dim <= nd, "unsqueeze(): dim out of range, got ", dim);

  std::vector<int64_t> new_dims = shape_.dims();
  new_dims.insert(new_dims.begin() + dim, 1);
  return reshape(Shape(new_dims));
}

Array Array::view(const Shape &new_shape) const {
  INS_CHECK(defined(), "view: array is undefined");
  INS_CHECK(new_shape.numel() == numel(),
            "view: shape.numel() mismatch. Expected ", numel(), ", got ",
            new_shape.numel());

  if (!is_contiguous()) {
    INS_THROW(
        "view: only contiguous arrays can be viewed. Use .contiguous() first.");
  }

  Array result;

  result.layout_ = layout_;
  if (result.layout_.ref_count) {
    ref_count_increment(result.layout_.ref_count);
  }

  result.shape_ = new_shape;
  result.strides_ = Strides(new_shape);

  result.layout_.is_view = 1;

  result.place_ = place_;

  return result;
}

Array Array::view(DType new_dtype) const {
  INS_CHECK(defined(), "view: array is undefined");

  size_t old_size = dtype_size(dtype());
  size_t new_size = dtype_size(new_dtype);
  INS_CHECK(old_size == new_size,
            "view: dtype size mismatch. Original size=", old_size,
            ", target size=", new_size);

  if (!is_contiguous()) {
    INS_THROW(
        "view: only contiguous arrays can be viewed. Use .contiguous() first.");
  }

  Array result;

  result.layout_ = layout_;
  if (result.layout_.ref_count) {
    ref_count_increment(result.layout_.ref_count);
  }

  result.layout_.dtype = static_cast<int32_t>(new_dtype);
  result.shape_ = shape_;
  result.strides_ = strides_;
  result.layout_.is_view = 1;
  result.place_ = place_;

  return result;
}

Array Array::view(const Shape &new_shape, DType new_dtype) const {
  INS_CHECK(defined(), "view: array is undefined");

  size_t original_bytes = numel() * dtype_size(dtype());
  size_t new_bytes = new_shape.numel() * dtype_size(new_dtype);
  INS_CHECK(original_bytes == new_bytes,
            "view: total bytes mismatch. Original=", original_bytes,
            ", new=", new_bytes);

  Array source = *this;
  if (!source.is_contiguous()) {
    source = source.contiguous();
  }

  Array result;

  result.layout_ = source.layout_;
  if (result.layout_.ref_count) {
    ref_count_increment(result.layout_.ref_count);
  }

  result.layout_.ndim = new_shape.ndim();
  result.layout_.numel = new_shape.numel();
  for (int i = 0; i < new_shape.ndim(); ++i) {
    result.layout_.dims[i] = new_shape.dim(i);
  }
  result.layout_.dtype = static_cast<int32_t>(new_dtype);
  result.layout_.is_view = 1;

  result.shape_ = new_shape;
  result.compute_strides();
  result.place_ = place_;

  return result;
}

// ========== Device/Type Conversion ==========

Array Array::to(const Place &target) const {
  INS_CHECK(defined(), "Cannot convert undefined array");

  if (place_ == target)
    return *this;

  // For non-contiguous views (e.g. permuted arrays), make a contiguous copy
  // first so the raw memcpy below transfers the correct data layout.
  const Array &src = is_contiguous() ? *this : contiguous();

  Array result(shape_, dtype(), target);
  size_t bytes = numel() * dtype_size(dtype());

  if (src.place().is_cpu() && target.is_cpu()) {
    std::memcpy(result.data(), src.data(), bytes);
  } else if (src.place().is_cpu() && target.is_gpu()) {
    target.copy_from_host(result.data(), src.data(), bytes);
  } else if (src.place().is_gpu() && target.is_cpu()) {
    src.place().copy_to_host(result.data(), src.data(), bytes);
  } else if (src.place().is_gpu() && target.is_gpu()) {
    void *host_buf = std::malloc(bytes);
    src.place().copy_to_host(host_buf, src.data(), bytes);
    target.copy_from_host(result.data(), host_buf, bytes);
    std::free(host_buf);
  }

  return result;
}

Array Array::to(DType target) const {
  INS_CHECK(defined(), "Cannot convert undefined array");
  if (dtype() == target)
    return *this;
  return ins::cast(*this, target);
}

Array Array::to(const Place &target, DType target_dtype) const {
  return ins::cast(*this, target_dtype).to(target);
}

Array Array::to(const Array &other) const {
  return to(other.place(), other.dtype());
}

Array Array::copy() const {
  INS_CHECK(defined(), "Cannot copy undefined array");

  Array result(shape_, dtype(), place_);
  size_t bytes = numel() * dtype_size(dtype());

  place_.copy_on_device(result.data(), data(), bytes);

  return result;
}

// ========== Boolean conversion ==========

Array::operator bool() const {
  if (!defined())
    return false;
  INS_CHECK(
      numel() == 1,
      "The truth value of an array with more than one element is ambiguous.");

  switch (dtype()) {
  case DType::BOOL:
    return item<bool>();
  case DType::F32:
    return item<float>() != 0.0f;
  case DType::F64:
    return item<double>() != 0.0;
  case DType::I32:
    return item<int32_t>() != 0;
  case DType::I64:
    return item<int64_t>() != 0;
  case DType::U8:
    return item<uint8_t>() != 0;
  case DType::I8:
    return item<int8_t>() != 0;
  case DType::I16:
    return item<int16_t>() != 0;
  case DType::U16:
    return item<uint16_t>() != 0;
  case DType::U32:
    return item<uint32_t>() != 0;
  case DType::U64:
    return item<uint64_t>() != 0;
  default:
    INS_THROW("operator bool(): unsupported dtype");
  }
}

bool Array::any() const {
  Array result = ins::any(*this, std::nullopt, false);
  return result.item<bool>();
}

bool Array::all() const {
  Array result = ins::all(*this, std::nullopt, false);
  return result.item<bool>();
}

// ========== In-place Mutation ==========

void Array::fill_(double value) {
  INS_CHECK(defined(), "Array is not initialized");
  if (numel() == 0)
    return;
  if (is_contiguous()) {
    // Contiguous: use the "full" backend kernel directly
    double val = value;
    OpRegistry::launch_schema(
        "full", place(), dtype(),
        {OpRegistry::array_arg(layout_ptr()), OpRegistry::scalar_arg(&val)},
        {OpRegistry::array_arg(layout_ptr())});
  } else {
    // Non-contiguous view: create a contiguous filled array, then copy
    Array filled = full(shape(), value, dtype(), place());
    copy_from_(filled);
  }
}

void Array::copy_from_(const Array &src) {
  INS_CHECK(defined(), "Array is not initialized");
  INS_CHECK(src.defined(), "copy_from_(): source is not initialized");
  INS_CHECK(shape() == src.shape(), "copy_from_(): shape mismatch, got ",
            src.shape(), " expected ", shape());
  if (numel() == 0)
    return;

  if (dtype() == src.dtype()) {
    // Same dtype: use contiguous_copy backend kernel (CPU + GPU)
    Array src_dev = src.to(place());
    OpRegistry::launch_schema(
        "contiguous_copy", place(), dtype(),
        {OpRegistry::array_arg(src_dev.layout_ptr())},
        {OpRegistry::array_arg(layout_ptr())});
  } else {
    // Different dtype: cast first, then copy
    Array converted = src.to(dtype()).to(place());
    OpRegistry::launch_schema(
        "contiguous_copy", place(), dtype(),
        {OpRegistry::array_arg(converted.layout_ptr())},
        {OpRegistry::array_arg(layout_ptr())});
  }
}

// ========== Helper ==========

void Array::compute_strides() {
  strides_ = Strides(shape_);

  int ndim = shape_.ndim();
  for (int i = 0; i < ndim; ++i) {
    layout_.strides[i] = strides_[i];
  }
}

// Explicit template instantiations for set_scalar
template void Array::set_scalar<bool>(bool);
template void Array::set_scalar<int8_t>(int8_t);
template void Array::set_scalar<int16_t>(int16_t);
template void Array::set_scalar<int32_t>(int32_t);
template void Array::set_scalar<int64_t>(int64_t);
template void Array::set_scalar<uint8_t>(uint8_t);
template void Array::set_scalar<uint16_t>(uint16_t);
template void Array::set_scalar<uint32_t>(uint32_t);
template void Array::set_scalar<uint64_t>(uint64_t);
template void Array::set_scalar<float>(float);
template void Array::set_scalar<double>(double);
template void Array::set_scalar<std::complex<float>>(std::complex<float>);
template void Array::set_scalar<std::complex<double>>(std::complex<double>);

} // namespace ins