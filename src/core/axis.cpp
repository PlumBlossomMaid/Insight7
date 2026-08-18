// src/core/axis.cpp
#include "insight/core/axis.h"
#include "insight/core/exception.h"

namespace ins {

int normalize_axis(int axis, int ndim, const char *context) {
  int normalized = axis;
  if (normalized < 0)
    normalized += ndim;
  INS_CHECK(normalized >= 0 && normalized < ndim, context,
            ": axis out of range: ", axis, " (ndim=", ndim, ")");
  return normalized;
}

int normalize_axis_insert(int axis, int ndim, const char *context) {
  int normalized = axis;
  if (normalized < 0)
    normalized += ndim + 1;
  INS_CHECK(normalized >= 0 && normalized <= ndim, context,
            ": axis out of range: ", axis, " (ndim=", ndim, ")");
  return normalized;
}

} // namespace ins
