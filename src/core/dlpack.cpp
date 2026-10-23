// src/core/dlpack.cpp
//
// DLPack C ABI bridge implementation (see include/insight/c_api/dlpack.h).
//
// dtype / device mapping tables and the to/from DLPack converters.
// Lifecycle follows the header contract:
//   - to_dlpack: manager_ctx holds an ins::Array copy sharing the storage's
//     ref_count, so the underlying data stays alive until the consumer calls
//     the deleter.
//   - from_dlpack: bytes are copied into Insight-owned storage allocated with
//     CPUPlace().allocate(); the caller owns the returned InsightArray's data
//     (same convention as insight_array_create).

#include "insight/c_api/dlpack.h"
#include "insight/c_api/exception.h"
#include "insight/c_api/place.h"
#include "insight/core/array.h"
#include "insight/core/place.h"
#include <cstring>
#include <new>
#include <vector>

// ========================================================================
// dtype mapping (InsightDType <-> DLDataType)
// ========================================================================

namespace {

struct DLPackDTypeMap {
  int32_t insight;   // InsightDType enum value
  uint8_t code;      // DLDataTypeCode value
  uint8_t bits;      // number of bits
};

// Follows include/insight/core/dtype.h DType order. lanes is always 1.
const DLPackDTypeMap kDTypeMap[] = {
    {INSIGHT_DTYPE_BOOL, kDLBool, 8},
    {INSIGHT_DTYPE_U8, kDLUInt, 8},
    {INSIGHT_DTYPE_I8, kDLInt, 8},
    {INSIGHT_DTYPE_I16, kDLInt, 16},
    {INSIGHT_DTYPE_I32, kDLInt, 32},
    {INSIGHT_DTYPE_I64, kDLInt, 64},
    {INSIGHT_DTYPE_F16, kDLFloat, 16},
    {INSIGHT_DTYPE_BF16, kDLBfloat, 16},
    {INSIGHT_DTYPE_F32, kDLFloat, 32},
    {INSIGHT_DTYPE_F64, kDLFloat, 64},
    {INSIGHT_DTYPE_C32, kDLComplex, 64},
    {INSIGHT_DTYPE_C64, kDLComplex, 128},
    {INSIGHT_DTYPE_F8_E4M3, kDLFloat8_e4m3, 8},
    {INSIGHT_DTYPE_F8_E5M2, kDLFloat8_e5m2, 8},
    {INSIGHT_DTYPE_U16, kDLUInt, 16},
    {INSIGHT_DTYPE_U32, kDLUInt, 32},
    {INSIGHT_DTYPE_U64, kDLUInt, 64},
};

const DLPackDTypeMap *find_insight_to_dlpack(int32_t dtype) {
  for (const auto &m : kDTypeMap) {
    if (m.insight == dtype)
      return &m;
  }
  return nullptr;
}

const DLPackDTypeMap *find_dlpack_to_insight(const DLDataType &dt) {
  for (const auto &m : kDTypeMap) {
    if (m.code == dt.code && m.bits == dt.bits)
      return &m;
  }
  return nullptr;
}

// Copy data from a (possibly strided / byte-offset) DLTensor into a
// contiguous destination buffer of `itemsize`-sized elements.
// dst_elem_count == numel; strides are in element units (DLPack v1.2+).
void copy_strided(const char *src_data, uint64_t byte_offset,
                  const int64_t *strides, int32_t ndim, const int64_t *dims,
                  size_t itemsize, char *dst) {
  if (ndim == 0) {
    std::memcpy(dst, src_data + byte_offset, itemsize);
    return;
  }
  // Recursive strided copy: iterate the logical index space in row-major
  // order and gather from src_data + byte_offset + sum(i[k]*strides[k]).
  std::vector<int64_t> idx(ndim, 0);
  char *base = const_cast<char *>(src_data) + byte_offset;
  size_t total = 1;
  for (int32_t i = 0; i < ndim; ++i)
    total *= static_cast<size_t>(dims[i]);
  size_t elem = 0;
  while (elem < total) {
    int64_t offset = 0;
    for (int32_t k = 0; k < ndim; ++k)
      offset += idx[k] * strides[k];
    std::memcpy(dst + elem * itemsize, base + offset * itemsize, itemsize);
    ++elem;
    // increment row-major index
    for (int32_t k = ndim - 1; k >= 0; --k) {
      if (++idx[k] < dims[k])
        break;
      idx[k] = 0;
    }
  }
}

} // namespace

// ========================================================================
// public mapping helpers
// ========================================================================

extern "C" {

C_Status insight_dtype_to_dlpack(int32_t dtype, DLDataType *out) {
  if (!out)
    return C_FAILED;
  const DLPackDTypeMap *m = find_insight_to_dlpack(dtype);
  if (!m) {
    insight_set_last_error("insight_dtype_to_dlpack: unsupported InsightDType");
    return C_FAILED;
  }
  out->code = m->code;
  out->bits = m->bits;
  out->lanes = 1;
  return C_SUCCESS;
}

C_Status insight_dtype_from_dlpack(const DLDataType *src, int32_t *out) {
  if (!src || !out)
    return C_FAILED;
  if (src->lanes != 1) {
    insight_set_last_error(
        "insight_dtype_from_dlpack: vector lanes != 1 unsupported");
    return C_FAILED;
  }
  const DLPackDTypeMap *m = find_dlpack_to_insight(*src);
  if (!m) {
    insight_set_last_error("insight_dtype_from_dlpack: unsupported DLDataType");
    return C_FAILED;
  }
  *out = m->insight;
  return C_SUCCESS;
}

C_Status insight_device_to_dlpack(int32_t device_type, int32_t device_id,
                                  DLDevice *out) {
  if (!out)
    return C_FAILED;
  // Insight only exposes CPU / GPU abstractions; map GPU to kDLCUDA
  // (same convention as the Python binding's dlpack_device_type).
  out->device_type =
      device_type == INSIGHT_DEVICE_GPU ? (DLDeviceType)kDLCUDA : kDLCPU;
  out->device_id = device_type == INSIGHT_DEVICE_GPU ? device_id : 0;
  return C_SUCCESS;
}

// ========================================================================
// to_dlpack / from_dlpack
// ========================================================================

C_Status insight_array_to_dlpack(const InsightArray *array,
                                 DLManagedTensorVersioned **out) {
  if (!array || !out) {
    insight_set_last_error("insight_array_to_dlpack: null argument");
    return C_FAILED;
  }
  if (array->ndim < 0 || array->ndim > INSIGHT_MAX_NDIM) {
    insight_set_last_error("insight_array_to_dlpack: ndim out of range");
    return C_FAILED;
  }
  DLDataType dl_dtype;
  if (insight_dtype_to_dlpack(array->dtype, &dl_dtype) != C_SUCCESS)
    return C_FAILED;
  DLDevice dl_dev;
  if (insight_device_to_dlpack(array->device_type, array->device_id,
                               &dl_dev) != C_SUCCESS)
    return C_FAILED;

  try {
    DLManagedTensorVersioned *managed = new DLManagedTensorVersioned();
    std::memset(managed, 0, sizeof(*managed));
    managed->version.major = DLPACK_MAJOR_VERSION;
    managed->version.minor = DLPACK_MINOR_VERSION;
    managed->flags = 0;

    int32_t ndim = array->ndim;
    int64_t *shape = ndim > 0 ? new int64_t[ndim] : nullptr;
    int64_t *strides = ndim > 0 ? new int64_t[ndim] : nullptr;
    for (int32_t i = 0; i < ndim; ++i) {
      shape[i] = array->dims[i];
      strides[i] = array->strides[i];
    }

    managed->dl_tensor.data =
        const_cast<void *>(insight_array_data(array)); // logical first element
    managed->dl_tensor.device = dl_dev;
    managed->dl_tensor.ndim = ndim;
    managed->dl_tensor.dtype = dl_dtype;
    managed->dl_tensor.shape = shape;
    managed->dl_tensor.strides = strides;
    managed->dl_tensor.byte_offset = 0;

    // Keep the storage alive: manager_ctx holds an ins::Array copy sharing
    // the same ref_count. The deleter destroys it (and this struct).
    managed->manager_ctx =
        new ins::Array(const_cast<InsightArray *>(array));
    managed->deleter = [](DLManagedTensorVersioned *self) {
      if (!self)
        return;
      delete[] self->dl_tensor.shape;
      delete[] self->dl_tensor.strides;
      delete static_cast<ins::Array *>(self->manager_ctx);
      delete self;
    };

    *out = managed;
    return C_SUCCESS;
  } catch (const std::bad_alloc &) {
    insight_set_last_error("insight_array_to_dlpack: out of memory");
    return C_FAILED;
  } catch (const std::exception &e) {
    insight_set_last_error(e.what());
    return C_FAILED;
  }
}

C_Status insight_array_from_dlpack(const DLManagedTensorVersioned *src,
                                   InsightArray *out) {
  if (!src || !out) {
    insight_set_last_error("insight_array_from_dlpack: null argument");
    return C_FAILED;
  }
  if (src->version.major != DLPACK_MAJOR_VERSION) {
    insight_set_last_error(
        "insight_array_from_dlpack: DLPack major version mismatch");
    return C_FAILED;
  }
  const DLTensor &t = src->dl_tensor;
  if (t.ndim < 0 || t.ndim > INSIGHT_MAX_NDIM) {
    insight_set_last_error("insight_array_from_dlpack: ndim out of range");
    return C_FAILED;
  }
  int32_t dtype;
  if (insight_dtype_from_dlpack(&t.dtype, &dtype) != C_SUCCESS)
    return C_FAILED;

  // Only CPU import supported for now (device kind check).
  if (t.device.device_type != kDLCPU) {
    insight_set_last_error(
        "insight_array_from_dlpack: non-CPU import not supported yet");
    return C_FAILED;
  }

  int32_t ndim = t.ndim;
  std::vector<int64_t> dims(ndim);
  int64_t numel = 1;
  for (int32_t i = 0; i < ndim; ++i) {
    dims[i] = t.shape[i];
    numel *= t.shape[i];
  }
  size_t itemsize = (t.dtype.bits * t.dtype.lanes + 7) / 8;
  size_t bytes = static_cast<size_t>(numel) * itemsize;

  try {
    ins::Place place = ins::CPUPlace();
    void *data = bytes > 0 ? place.allocate(bytes) : nullptr;

    C_Status st = insight_array_create(
        out, data, dims.empty() ? nullptr : dims.data(), ndim, dtype,
        INSIGHT_DEVICE_CPU, 0);
    if (st != C_SUCCESS) {
      if (data)
        place.deallocate(data, bytes);
      return st;
    }

    if (bytes > 0 && t.data) {
      // Contiguous check: row-major standard strides (and byte_offset == 0).
      bool contiguous = (t.byte_offset == 0);
      if (contiguous && ndim > 0) {
        int64_t expected = 1;
        for (int32_t i = ndim - 1; i >= 0; --i) {
          if (t.strides[i] != expected) {
            contiguous = false;
            break;
          }
          expected *= t.shape[i];
        }
      }
      const char *src_data = static_cast<const char *>(t.data);
      if (contiguous) {
        std::memcpy(data, src_data, bytes);
      } else {
        copy_strided(src_data, t.byte_offset, t.strides, ndim, dims.data(),
                     itemsize, static_cast<char *>(data));
      }
    } else if (bytes > 0) {
      std::memset(data, 0, bytes);
    }
    return C_SUCCESS;
  } catch (const std::exception &e) {
    insight_set_last_error(e.what());
    return C_FAILED;
  }
}

} // extern "C"
