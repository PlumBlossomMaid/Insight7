// src/ops/signal/demod.cpp
#include "insight/ops/signal/demod.h"
#include "insight/core/axis.h"
#include "insight/core/exception.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/signal.h"
#include "insight/ops/unary.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

// Compute angle(z) = atan2(imag(z), real(z)) using composite ops
static Array complex_angle(const Array &z) {
  Array re = real(z);
  Array im = imag(z);

  // angle = atan(im/re) with quadrant correction
  // For re > 0: atan(im/re)
  // For re < 0, im >= 0: atan(im/re) + pi
  // For re < 0, im < 0: atan(im/re) - pi
  // For re == 0, im > 0: pi/2
  // For re == 0, im < 0: -pi/2
  Array ratio = div(im, re);
  Array base_angle = atan(ratio);

  DType dtype = base_angle.dtype();
  auto place = base_angle.place();
  Array pi = full(base_angle.shape(), M_PI, dtype, place);
  Array neg_pi = full(base_angle.shape(), -M_PI, dtype, place);
  Array half_pi = full(base_angle.shape(), M_PI / 2.0, dtype, place);
  Array neg_half_pi = full(base_angle.shape(), -M_PI / 2.0, dtype, place);
  Array zero = zeros(base_angle.shape(), dtype, place);

  // re < 0 && im >= 0 → add pi
  Array re_neg = less_than(re, zero);
  Array im_nneg = greater_equal(im, zero);
  Array case1 = logical_and(re_neg, im_nneg);
  base_angle = where(case1, add(base_angle, pi), base_angle);

  // re < 0 && im < 0 → subtract pi
  Array im_neg = less_than(im, zero);
  Array case2 = logical_and(re_neg, im_neg);
  base_angle = where(case2, add(base_angle, neg_pi), base_angle);

  // re == 0 && im > 0 → pi/2
  Array re_zero = equal(re, zero);
  Array im_pos = greater_than(im, zero);
  Array case3 = logical_and(re_zero, im_pos);
  base_angle = where(case3, half_pi, base_angle);

  // re == 0 && im < 0 → -pi/2
  Array case4 = logical_and(re_zero, im_neg);
  base_angle = where(case4, neg_half_pi, base_angle);

  return base_angle;
}

Array fm_demod(const Array &x, int axis) {
  INS_CHECK(x.defined(), "fm_demod: input is undefined");
  INS_CHECK(x.dtype() == DType::C32 || x.dtype() == DType::C64,
            "fm_demod: input must be complex-valued");

  int ax = normalize_axis(axis, x.shape().ndim(), "fm_demod");

  // Compute angle(x) = atan2(imag, real)
  Array x_angle = complex_angle(x);

  // Unwrap along axis
  Array x_unwrapped = unwrap(x_angle, ax);

  // Diff along axis
  Array result = diff(x_unwrapped, 1, ax);

  return result;
}

} // namespace signal
} // namespace ins
