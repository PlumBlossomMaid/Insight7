// include/insight/c_api/dtype.h
#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  C_SUCCESS = 0,
  C_WARNING,
  C_FAILED,
  C_ERROR,
  C_INTERNAL_ERROR,
  C_FALLBACK
} C_Status;

typedef enum {
  INSIGHT_DTYPE_UNKNOWN = 0,
  INSIGHT_DTYPE_BOOL,
  INSIGHT_DTYPE_U8,
  INSIGHT_DTYPE_I8,
  INSIGHT_DTYPE_I16,
  INSIGHT_DTYPE_I32,
  INSIGHT_DTYPE_I64,
  INSIGHT_DTYPE_F16,
  INSIGHT_DTYPE_BF16,
  INSIGHT_DTYPE_F32,
  INSIGHT_DTYPE_F64,
  INSIGHT_DTYPE_C32,
  INSIGHT_DTYPE_C64,
  INSIGHT_DTYPE_F8_E4M3,
  INSIGHT_DTYPE_F8_E5M2,
  INSIGHT_DTYPE_U16,
  INSIGHT_DTYPE_U32,
  INSIGHT_DTYPE_U64,
  INSIGHT_DTYPE_COUNT
} InsightDType;

typedef enum {
  INSIGHT_DTYPE_KIND_UNKNOWN = 0,
  INSIGHT_DTYPE_KIND_BOOL,
  INSIGHT_DTYPE_KIND_UINT,
  INSIGHT_DTYPE_KIND_INT,
  INSIGHT_DTYPE_KIND_FLOAT,
  INSIGHT_DTYPE_KIND_COMPLEX
} InsightDTypeKind;

typedef enum {
  INSIGHT_DTYPE_FLAG_NONE = 0,
  INSIGHT_DTYPE_FLAG_BUILTIN = 1u << 0,
  INSIGHT_DTYPE_FLAG_NUMERIC = 1u << 1,
  INSIGHT_DTYPE_FLAG_EXPERIMENTAL = 1u << 2
} InsightDTypeFlag;

const char *insight_dtype_name(int32_t dtype);
int32_t insight_dtype_size(int32_t dtype);
int32_t insight_dtype_alignment(int32_t dtype);
int32_t insight_dtype_kind(int32_t dtype);
uint32_t insight_dtype_flags(int32_t dtype);
int32_t insight_dtype_promotion_rank(int32_t dtype);
int insight_dtype_is_float(int32_t dtype);
int insight_dtype_is_int(int32_t dtype);
int insight_dtype_is_complex(int32_t dtype);
int insight_dtype_is_signed(int32_t dtype);

#ifdef __cplusplus
}
#endif