// insight/core/axis.h
#pragma once

namespace ins {

int normalize_axis(int axis, int ndim, const char *context);
int normalize_axis_insert(int axis, int ndim, const char *context);

} // namespace ins
