// src/core/dtype.cpp
#include "insight/core/dtype.h"
#include "insight/c_api/dtype.h"
#include <cstddef>

namespace ins {

namespace {

static const DTypeDescriptor dtype_descriptors[] = {
    {DType::UNKNOWN, "unknown", DTypeKind::Unknown, 0, 0,
     DTYPE_FLAG_NONE, 0, false},
    {DType::BOOL, "bool", DTypeKind::Bool, sizeof(bool), alignof(bool),
     DTYPE_FLAG_BUILTIN, 1, false},
    {DType::U8, "uint8", DTypeKind::UInt, sizeof(uint8_t), alignof(uint8_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 2, false},
    {DType::I8, "int8", DTypeKind::Int, sizeof(int8_t), alignof(int8_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 3, true},
    {DType::I16, "int16", DTypeKind::Int, sizeof(int16_t), alignof(int16_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 4, true},
    {DType::I32, "int32", DTypeKind::Int, sizeof(int32_t), alignof(int32_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 5, true},
    {DType::I64, "int64", DTypeKind::Int, sizeof(int64_t), alignof(int64_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 6, true},
    {DType::F16, "float16", DTypeKind::Float, sizeof(uint16_t),
     alignof(uint16_t), DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 7, true},
    {DType::BF16, "bfloat16", DTypeKind::Float, sizeof(uint16_t),
     alignof(uint16_t), DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 8, true},
    {DType::F32, "float32", DTypeKind::Float, sizeof(float), alignof(float),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 9, true},
    {DType::F64, "float64", DTypeKind::Float, sizeof(double), alignof(double),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 10, true},
    {DType::C32, "complex64", DTypeKind::Complex,
     sizeof(std::complex<float>), alignof(std::complex<float>),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 11, true},
    {DType::C64, "complex128", DTypeKind::Complex,
     sizeof(std::complex<double>), alignof(std::complex<double>),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 12, true},
    {DType::F8_E4M3, "float8_e4m3", DTypeKind::Float, sizeof(uint8_t),
     alignof(uint8_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC | DTYPE_FLAG_EXPERIMENTAL, 0,
     true},
    {DType::F8_E5M2, "float8_e5m2", DTypeKind::Float, sizeof(uint8_t),
     alignof(uint8_t),
     DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC | DTYPE_FLAG_EXPERIMENTAL, 0,
     true},
    {DType::U16, "uint16", DTypeKind::UInt, sizeof(uint16_t),
     alignof(uint16_t), DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 0, false},
    {DType::U32, "uint32", DTypeKind::UInt, sizeof(uint32_t),
     alignof(uint32_t), DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 0, false},
    {DType::U64, "uint64", DTypeKind::UInt, sizeof(uint64_t),
     alignof(uint64_t), DTYPE_FLAG_BUILTIN | DTYPE_FLAG_NUMERIC, 0, false},
};

static_assert(sizeof(dtype_descriptors) / sizeof(dtype_descriptors[0]) ==
                  static_cast<size_t>(DType::DTYPE_COUNT),
              "dtype_descriptors size mismatch");

const DTypeDescriptor &checked_descriptor(DType dtype) {
  int idx = static_cast<int>(dtype);
  if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
    return dtype_descriptors[0];
  return dtype_descriptors[idx];
}

} // namespace

const DTypeDescriptor &dtype_descriptor(DType dtype) {
  return checked_descriptor(dtype);
}

const char *dtype_name(DType dtype) { return dtype_descriptor(dtype).name; }

DType dtype_from_name(const std::string &name) {
  for (int i = 0; i < static_cast<int>(DType::DTYPE_COUNT); ++i) {
    if (name == dtype_descriptors[i].name) {
      return static_cast<DType>(i);
    }
  }
  return DType::UNKNOWN;
}

size_t dtype_size(DType dtype) { return dtype_descriptor(dtype).size; }

size_t dtype_alignment(DType dtype) {
  return dtype_descriptor(dtype).alignment;
}

DTypeKind dtype_kind(DType dtype) { return dtype_descriptor(dtype).kind; }

uint32_t dtype_flags(DType dtype) { return dtype_descriptor(dtype).flags; }

int dtype_promotion_rank(DType dtype) {
  return dtype_descriptor(dtype).promotion_rank;
}

bool is_floating_point(DType dtype) {
  return dtype_kind(dtype) == DTypeKind::Float;
}

bool is_integer(DType dtype) {
  DTypeKind kind = dtype_kind(dtype);
  return kind == DTypeKind::Int || kind == DTypeKind::UInt;
}

bool is_complex(DType dtype) { return dtype_kind(dtype) == DTypeKind::Complex; }

bool is_signed(DType dtype) { return dtype_descriptor(dtype).is_signed; }

std::ostream &operator<<(std::ostream &os, DType dtype) {
  os << dtype_name(dtype);
  return os;
}

} // namespace ins

// C API implementations

extern "C" {

const char *insight_dtype_name(int32_t dtype) {
  return ins::dtype_name(static_cast<ins::DType>(dtype));
}

int32_t insight_dtype_size(int32_t dtype) {
  return static_cast<int32_t>(ins::dtype_size(static_cast<ins::DType>(dtype)));
}

int32_t insight_dtype_alignment(int32_t dtype) {
  return static_cast<int32_t>(
      ins::dtype_alignment(static_cast<ins::DType>(dtype)));
}

int32_t insight_dtype_kind(int32_t dtype) {
  return static_cast<int32_t>(ins::dtype_kind(static_cast<ins::DType>(dtype)));
}

uint32_t insight_dtype_flags(int32_t dtype) {
  return ins::dtype_flags(static_cast<ins::DType>(dtype));
}

int32_t insight_dtype_promotion_rank(int32_t dtype) {
  return static_cast<int32_t>(
      ins::dtype_promotion_rank(static_cast<ins::DType>(dtype)));
}

int insight_dtype_is_float(int32_t dtype) {
  return ins::is_floating_point(static_cast<ins::DType>(dtype)) ? 1 : 0;
}

int insight_dtype_is_int(int32_t dtype) {
  return ins::is_integer(static_cast<ins::DType>(dtype)) ? 1 : 0;
}

int insight_dtype_is_complex(int32_t dtype) {
  return ins::is_complex(static_cast<ins::DType>(dtype)) ? 1 : 0;
}

int insight_dtype_is_signed(int32_t dtype) {
  return ins::is_signed(static_cast<ins::DType>(dtype)) ? 1 : 0;
}

} // extern "C"