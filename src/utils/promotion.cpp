// insight/utils/promotion.cpp
#include "insight/utils/promotion.h"
#include "insight/core/exception.h"

namespace ins {

namespace {

bool is_real_numeric(DType dtype) {
  DTypeKind kind = dtype_kind(dtype);
  return kind == DTypeKind::Bool || kind == DTypeKind::UInt ||
         kind == DTypeKind::Int || kind == DTypeKind::Float;
}

} // namespace

bool can_promote(DType from, DType to) {
  if (from == to)
    return true;

  int from_rank = dtype_promotion_rank(from);
  int to_rank = dtype_promotion_rank(to);
  if (from_rank <= 0 || to_rank <= 0 || from_rank >= to_rank)
    return false;

  DTypeKind from_kind = dtype_kind(from);
  DTypeKind to_kind = dtype_kind(to);
  if (to_kind == DTypeKind::Complex) {
    if (to == DType::C32)
      return from == DType::F32;
    return from == DType::F32 || from == DType::F64 ||
           from_kind == DTypeKind::Complex;
  }
  if (from_kind == DTypeKind::Complex)
    return false;
  return is_real_numeric(from) && is_real_numeric(to);
}

DType promote_types(DType a, DType b) {
  if (a == b)
    return a;

  int pa = dtype_promotion_rank(a);
  int pb = dtype_promotion_rank(b);
  INS_CHECK(pa > 0 && pb > 0, "Invalid dtype for promotion: ", dtype_name(a),
            " vs ", dtype_name(b));

  DType higher = (pa > pb) ? a : b;
  DType lower = (pa > pb) ? b : a;

  INS_CHECK(can_promote(lower, higher), "Cannot promote from ",
            dtype_name(lower), " to ", dtype_name(higher));

  return higher;
}

Place promote_places(const Place &a, const Place &b) {
  // Both CPU → stay on CPU
  if (a.is_cpu() && b.is_cpu()) {
    return a;
  }

  // Both GPU → align to a's device
  if (a.is_gpu() && b.is_gpu()) {
    return a;
  }

  // Mixed CPU/GPU → promote to GPU (use the GPU side's device)
  if (a.is_gpu())
    return a;
  return b;
}

} // namespace ins