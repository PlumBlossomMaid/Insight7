// src/ops/fft.cpp
/**
 * @file fft.cpp
 * @brief Fast Fourier Transform operations.
 *
 * Provides 1D, 2D, and N-dimensional FFTs for real and complex inputs.
 * Supports various normalization modes and frequency domain utilities.
 */

#include "insight/ops/fft.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/slice.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/manipulation.h"
#include <climits>
#include <cmath>
#include <functional>
#include <vector>

namespace ins {
namespace fft {

// ============================================================================
// Helper: Normalization code mapping
// ============================================================================

static int norm_to_code(const std::string &norm) {
  if (norm == "backward")
    return 0;
  if (norm == "forward")
    return 1;
  if (norm == "ortho")
    return 2;
  return 0; // default to backward
}

static std::string code_to_norm(int code) {
  switch (code) {
  case 0:
    return "backward";
  case 1:
    return "forward";
  case 2:
    return "ortho";
  default:
    return "backward";
  }
}

// ============================================================================
// Helper: Get device kind
// ============================================================================

static DeviceKind get_device_kind(const Place &place) {
  return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
}

// ============================================================================
// FFT Plan Structure
// ============================================================================

struct FFTPlan {
  int64_t fft_len;           // FFT length after padding/truncation
  int64_t batch_size;        // Number of 1D FFTs to perform
  std::vector<int> inv_perm; // Inverse permutation for result
  int input_is_complex;      // 1 if complex, 0 if real
  int output_is_complex;     // 1 if complex, 0 if real
  int inverse;               // 1 for ifft/irfft, 0 for fft/rfft
  int real_input;            // 1 for rfft/irfft, 0 otherwise
  int fft_ndim;              // Number of dimensions excluding complex dim
  int transformed_axis;      // Original axis being transformed
  DType dtype;               // F32 or F64
  int norm_code;             // 0=backward, 1=forward, 2=ortho
};

// ============================================================================
// FFT Preparation
// ============================================================================

static FFTPlan prepare_fft(const Array &x, int n, int axis, bool inverse,
                           bool real_input, const std::string &norm) {
  FFTPlan plan;
  plan.inverse = static_cast<int>(inverse) ? 1 : 0;
  plan.real_input = real_input ? 1 : 0;
  plan.norm_code = norm_to_code(norm);

  if (is_complex(x.dtype())) {
    plan.dtype = x.dtype();
    plan.input_is_complex = 1;
  } else {
    plan.dtype = (x.dtype() == DType::F64) ? DType::F64 : DType::F32;
    plan.input_is_complex = 0;
  }

  int ndim = x.shape().ndim();
  plan.fft_ndim = ndim;

  plan.transformed_axis = axis;
  if (plan.transformed_axis < 0)
    plan.transformed_axis += plan.fft_ndim;
  INS_CHECK(plan.transformed_axis >= 0 && plan.transformed_axis < plan.fft_ndim,
            "prepare_fft: axis out of range");

  int64_t current_len = x.shape().dim(plan.transformed_axis);
  plan.fft_len = (n > 0) ? n : current_len;
  INS_CHECK(plan.fft_len > 0, "prepare_fft: fft_len must be positive");

  // Build inverse permutation for moving axis back
  plan.inv_perm.clear();
  if (plan.transformed_axis != plan.fft_ndim - 1) {
    std::vector<int> perm;
    for (int i = 0; i < plan.fft_ndim; ++i) {
      if (i != plan.transformed_axis)
        perm.push_back(i);
    }
    perm.push_back(plan.transformed_axis);
    plan.inv_perm.resize(ndim);
    for (size_t i = 0; i < perm.size(); ++i) {
      plan.inv_perm[perm[i]] = static_cast<int>(i);
    }
  }

  // Compute batch size (product of non-transformed dimensions)
  plan.batch_size = 1;
  for (int i = 0; i < plan.fft_ndim; ++i) {
    if (i != plan.transformed_axis) {
      plan.batch_size *= x.shape().dim(i);
    }
  }
  INS_CHECK(plan.batch_size > 0, "prepare_fft: batch_size must be positive");

  if (real_input) {
    plan.output_is_complex = !static_cast<int>(inverse) ? 1 : 0;
  } else {
    plan.output_is_complex =
        plan.input_is_complex != static_cast<int>(inverse) ? 1 : 0;
  }

  return plan;
}

// ============================================================================
// Input Preparation
// ============================================================================

static Array prepare_input(const Array &x, const FFTPlan &plan, int n) {
  Array result = x;

  // Make contiguous
  if (!result.is_contiguous()) {
    result = result.contiguous();
  }

  // Transpose to move transformed axis to last
  if (!plan.inv_perm.empty()) {
    std::vector<int> perm(plan.inv_perm.size());
    for (size_t i = 0; i < plan.inv_perm.size(); ++i) {
      perm[plan.inv_perm[i]] = static_cast<int>(i);
    }
    result = result.transpose(perm);
    result = result.contiguous();
  }

  // Convert to working dtype if needed
  if (!is_complex(result.dtype())) {
    if (result.dtype() != plan.dtype) {
      result = result.to(plan.dtype);
    }
  }

  // Pad or truncate on the transformed axis (now last dimension)
  int last_axis_idx = result.shape().ndim() - 1;
  int64_t current_len = result.shape().dim(last_axis_idx);

  int64_t target_len = plan.fft_len;
  if (plan.real_input && plan.inverse) {
    // For irfft, n specifies output real length
    target_len = (n > 0) ? n : (current_len - 1) * 2;
  }

  if (target_len != current_len) {
    if (target_len > current_len) {
      // Pad with zeros
      std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
      pad_width[2 * last_axis_idx + 1] = target_len - current_len;
      result = pad(result, pad_width, 0.0);
    } else {
      // Truncate
      result = slice(result, last_axis_idx, 0, target_len, 1);
    }
    result = result.contiguous();
  }

  return result;
}

// ============================================================================
// Output Creation
// ============================================================================

static Array create_output(const FFTPlan &plan, const Array &input, int n) {
  std::vector<int64_t> out_dims;
  int ndim = input.shape().ndim();

  if (plan.real_input) {
    if (plan.inverse) {
      // irfft: complex -> real
      for (int i = 0; i < ndim - 1; ++i) {
        out_dims.push_back(input.shape().dim(i));
      }
      int64_t out_len;
      if (n > 0) {
        out_len = n;
      } else {
        int64_t in_complex_len = input.shape().dim(ndim - 1);
        out_len = (in_complex_len - 1) * 2;
      }
      out_dims.push_back(out_len);
      DType out_dtype = (input.dtype() == DType::C32) ? DType::F32 : DType::F64;
      return Array(Shape(out_dims), out_dtype, input.place());
    } else {
      // rfft: real -> complex
      for (int i = 0; i < ndim; ++i) {
        out_dims.push_back(input.shape().dim(i));
      }
      int64_t out_len = plan.fft_len / 2 + 1;
      out_dims.back() = out_len;
      DType out_dtype = (plan.dtype == DType::F32) ? DType::C32 : DType::C64;
      return Array(Shape(out_dims), out_dtype, input.place());
    }
  } else {
    // Standard complex FFT (C2C)
    for (int i = 0; i < ndim; ++i) {
      out_dims.push_back(input.shape().dim(i));
    }
    return Array(Shape(out_dims), plan.dtype, input.place());
  }
}

// ============================================================================
// Apply Normalization
// ============================================================================

static void apply_norm(Array &result, int64_t fft_len, int norm_code,
                       bool inverse) {
  double factor = 1.0;
  if (norm_code == 0) { // backward
    if (inverse)
      factor = 1.0 / fft_len;
  } else if (norm_code == 1) { // forward
    if (!inverse)
      factor = 1.0 / fft_len;
  } else if (norm_code == 2) { // ortho
    factor = 1.0 / std::sqrt(static_cast<double>(fft_len));
  } else {
    return;
  }

  if (std::abs(factor - 1.0) > 1e-12) {
    Array scalar = full(Shape({1}), factor, result.dtype(), result.place());
    result = mul(result, scalar);
  }
}

// ============================================================================
// 1D Complex FFT (C2C)
// ============================================================================

Array fft(const Array &x, int n, int axis, const std::string &norm) {
  // If input is real, convert to complex first
  if (!is_complex(x.dtype())) {
    Array x_complex = to_complex(x);
    return fft(x_complex, n, axis, norm);
  }

  INS_CHECK(x.defined(), "fft: input is undefined");

  FFTPlan plan = prepare_fft(x, n, axis, false, false, norm);
  Array input = prepare_input(x, plan, n);
  Array result = create_output(plan, input, n);

  int norm_code = plan.norm_code;
  ops().launch("fft", x.place(), plan.dtype,
               {result.layout_ptr(), input.layout_ptr(), &plan.fft_len,
                &plan.batch_size, &plan.inverse, &plan.real_input, &norm_code},
               {result.layout_ptr()});

  if (!plan.inv_perm.empty()) {
    result = result.transpose(plan.inv_perm);
  }

  apply_norm(result, plan.fft_len, norm_code, false);
  return result;
}

// ============================================================================
// 1D Inverse Complex FFT (C2C)
// ============================================================================

Array ifft(const Array &x, int n, int axis, const std::string &norm) {
  if (!is_complex(x.dtype())) {
    Array x_complex = to_complex(x);
    return ifft(x_complex, n, axis, norm);
  }

  INS_CHECK(x.defined(), "ifft: input is undefined");

  FFTPlan plan = prepare_fft(x, n, axis, true, false, norm);
  Array input = prepare_input(x, plan, n);
  Array result = create_output(plan, input, n);

  int norm_code = plan.norm_code;
  ops().launch("fft", x.place(), plan.dtype,
               {result.layout_ptr(), input.layout_ptr(), &plan.fft_len,
                &plan.batch_size, &plan.inverse, &plan.real_input, &norm_code},
               {result.layout_ptr()});

  if (!plan.inv_perm.empty()) {
    result = result.transpose(plan.inv_perm);
  }

  apply_norm(result, plan.fft_len, norm_code, true);
  return result;
}

// ============================================================================
// 1D Real FFT (R2C)
// ============================================================================

Array rfft(const Array &x, int n, int axis, const std::string &norm) {
  INS_CHECK(x.defined(), "rfft: input is undefined");
  INS_CHECK(!is_complex(x.dtype()), "rfft: input must be real");

  FFTPlan plan = prepare_fft(x, n, axis, false, true, norm);
  Array input = prepare_input(x, plan, n);
  Array result = create_output(plan, input, n);

  int norm_code = plan.norm_code;
  ops().launch("rfft", x.place(), plan.dtype,
               {result.layout_ptr(), input.layout_ptr(), &plan.fft_len,
                &plan.batch_size, &plan.inverse, &plan.real_input, &norm_code},
               {result.layout_ptr()});

  if (!plan.inv_perm.empty()) {
    result = result.transpose(plan.inv_perm);
  }

  apply_norm(result, plan.fft_len, norm_code, false);
  return result;
}

// ============================================================================
// 1D Inverse Real FFT (C2R)
// ============================================================================

Array irfft(const Array &x, int n, int axis, const std::string &norm) {
  INS_CHECK(x.defined(), "irfft: input is undefined");
  INS_CHECK(is_complex(x.dtype()), "irfft: input must be complex");

  FFTPlan plan = prepare_fft(x, n, axis, true, true, norm);

  // Compute expected real output length
  int64_t out_len;
  if (n > 0) {
    out_len = n;
  } else {
    int64_t in_len = x.shape().dim(axis);
    out_len = (in_len - 1) * 2;
  }

  // Compute expected complex input length for hermitian symmetry
  int64_t expected_input_len = out_len / 2 + 1;

  Array input = prepare_input(x, plan, expected_input_len);
  Array result = create_output(plan, input, n);

  int norm_code = plan.norm_code;
  ops().launch("irfft", x.place(), plan.dtype,
               {result.layout_ptr(), input.layout_ptr(), &out_len,
                &plan.batch_size, &plan.inverse, &plan.real_input, &norm_code},
               {result.layout_ptr()});

  if (!plan.inv_perm.empty()) {
    result = result.transpose(plan.inv_perm);
  }

  apply_norm(result, out_len, norm_code, true);
  return result;
}

// ============================================================================
// Hermitian FFT (Aliases)
// ============================================================================

Array hfft(const Array &x, int n, int axis, const std::string &norm) {
  return irfft(x, n, axis, norm);
}

Array ihfft(const Array &x, int n, int axis, const std::string &norm) {
  return rfft(x, n, axis, norm);
}

// ============================================================================
// 2D FFTs
// ============================================================================

Array fft2(const Array &x, const std::vector<int64_t> &s,
           const std::vector<int> &axes, const std::string &norm) {
  if (!is_complex(x.dtype())) {
    INS_THROW("fft2: input must be complex (dtype C32 or C64). "
              "For real input, use rfft2() instead.");
  }

  std::vector<int> target_axes = axes;
  if (target_axes.empty())
    target_axes = {-2, -1};
  return fftn(x, s, target_axes, norm);
}

Array ifft2(const Array &x, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
  if (!is_complex(x.dtype())) {
    INS_THROW("ifft2: input must be complex (dtype C32 or C64). "
              "For real output, use irfft2() instead.");
  }

  std::vector<int> target_axes = axes;
  if (target_axes.empty())
    target_axes = {-2, -1};
  return ifftn(x, s, target_axes, norm);
}

Array rfft2(const Array &x, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
  if (is_complex(x.dtype())) {
    INS_THROW(
        "rfft2: input must be real. For complex input, use fft2() instead.");
  }

  std::vector<int> target_axes = axes;
  if (target_axes.empty())
    target_axes = {-2, -1};
  return rfftn(x, s, target_axes, norm);
}

Array irfft2(const Array &x, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
  if (!is_complex(x.dtype())) {
    INS_THROW("irfft2: input must be complex (dtype C32 or C64). "
              "For real output, use rfft2() instead.");
  }

  std::vector<int> target_axes = axes;
  if (target_axes.empty())
    target_axes = {-2, -1};
  return irfftn(x, s, target_axes, norm);
}

Array hfft2(const Array &x, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
  return irfft2(x, s, axes, norm);
}

Array ihfft2(const Array &x, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
  return rfft2(x, s, axes, norm);
}

// ============================================================================
// N-D FFTs
// ============================================================================

Array fftn(const Array &x, const std::vector<int64_t> &s,
           const std::vector<int> &axes, const std::string &norm) {
  INS_CHECK(x.defined(), "fftn: input is undefined");

  std::vector<int> target_axes = axes;
  int ndim = x.shape().ndim();

  if (target_axes.empty()) {
    for (int i = 0; i < ndim; ++i) {
      target_axes.push_back(i);
    }
  }

  Array result = x;

  // Pad or truncate along axes
  if (!s.empty()) {
    INS_CHECK(s.size() == target_axes.size(),
              "fftn: s must have same length as axes");
    for (size_t i = 0; i < target_axes.size(); ++i) {
      int64_t target_len = s[i];
      int axis = target_axes[i];
      if (target_len > 0) {
        int64_t current_len = result.shape().dim(axis);
        if (target_len < current_len) {
          result = slice(result, axis, 0, target_len, 1);
        } else if (target_len > current_len) {
          std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
          pad_width[2 * axis + 1] = target_len - current_len;
          result = pad(result, pad_width, 0.0);
        }
      }
    }
    result = result.contiguous();
  }

  // Apply 1D FFT along each axis
  for (int ax : target_axes) {
    result = fft(result, -1, ax, norm);
    result = result.contiguous();
  }

  return result;
}

Array ifftn(const Array &x, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
  INS_CHECK(x.defined(), "ifftn: input is undefined");

  std::vector<int> target_axes = axes;
  int ndim = x.shape().ndim();

  if (target_axes.empty()) {
    for (int i = 0; i < ndim; ++i) {
      target_axes.push_back(i);
    }
  }

  Array result = x;

  // Pad or truncate along axes
  if (!s.empty()) {
    INS_CHECK(s.size() == target_axes.size(),
              "ifftn: s must have same length as axes");
    for (size_t i = 0; i < target_axes.size(); ++i) {
      int64_t target_len = s[i];
      int axis = target_axes[i];
      if (target_len > 0) {
        int64_t current_len = result.shape().dim(axis);
        if (target_len < current_len) {
          result = slice(result, axis, 0, target_len, 1);
        } else if (target_len > current_len) {
          std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
          pad_width[2 * axis + 1] = target_len - current_len;
          result = pad(result, pad_width, 0.0);
        }
      }
    }
    result = result.contiguous();
  }

  // Apply 1D IFFT along each axis
  for (int ax : target_axes) {
    result = ifft(result, -1, ax, norm);
    result = result.contiguous();
  }

  return result;
}

Array rfftn(const Array &x, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
  INS_CHECK(x.defined(), "rfftn: input is undefined");
  INS_CHECK(!is_complex(x.dtype()), "rfftn: input must be real");

  std::vector<int> target_axes = axes;
  int ndim = x.shape().ndim();

  if (target_axes.empty()) {
    for (int i = 0; i < ndim; ++i) {
      target_axes.push_back(i);
    }
  }

  Array result = x;

  // Pad or truncate along axes
  if (!s.empty()) {
    INS_CHECK(s.size() == target_axes.size(),
              "rfftn: s must have same length as axes");
    for (size_t i = 0; i < target_axes.size(); ++i) {
      int64_t target_len = s[i];
      int axis = target_axes[i];
      if (target_len > 0) {
        int64_t current_len = result.shape().dim(axis);
        if (target_len < current_len) {
          result = slice(result, axis, 0, target_len, 1);
        } else if (target_len > current_len) {
          std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
          pad_width[2 * axis + 1] = target_len - current_len;
          result = pad(result, pad_width, 0.0);
        }
      }
    }
    result = result.contiguous();
  }

  // Apply rfft on last axis, then fft on others
  int last_axis = target_axes.back();
  result = rfft(result, -1, last_axis, norm);
  result = result.contiguous();

  for (int i = static_cast<int>(target_axes.size()) - 2; i >= 0; --i) {
    result = fft(result, -1, target_axes[i], norm);
    result = result.contiguous();
  }

  return result;
}

Array irfftn(const Array &x, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
  INS_CHECK(x.defined(), "irfftn: input is undefined");
  INS_CHECK(is_complex(x.dtype()), "irfftn: input must be complex");

  std::vector<int> target_axes = axes;
  int ndim = x.shape().ndim();

  if (target_axes.empty()) {
    for (int i = 0; i < ndim; ++i) {
      target_axes.push_back(i);
    }
  }

  Array result = x;

  // Apply ifft on all axes except the last
  for (size_t i = 0; i < target_axes.size() - 1; ++i) {
    result = ifft(result, -1, target_axes[i], norm);
    result = result.contiguous();
  }

  // Apply irfft on the last axis
  int last_axis = target_axes.back();
  int64_t out_len =
      (s.size() > static_cast<size_t>(last_axis)) ? s[last_axis] : -1;
  result = irfft(result, out_len, last_axis, norm);

  return result;
}

Array hfftn(const Array &x, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
  return irfftn(x, s, axes, norm);
}

Array ihfftn(const Array &x, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
  return rfftn(x, s, axes, norm);
}

// ============================================================================
// Utility Functions
// ============================================================================

Array fftfreq(int64_t n, double d) {
  INS_CHECK(n > 0, "fftfreq: n must be positive");
  INS_CHECK(d > 0, "fftfreq: d must be positive");

  std::vector<double> data(n);
  double inv = 1.0 / (d * n);
  int64_t mid = (n + 1) / 2;

  for (int64_t i = 0; i < mid; ++i) {
    data[i] = i * inv;
  }
  for (int64_t i = mid; i < n; ++i) {
    data[i] = (i - n) * inv;
  }

  return to_array(data, Shape({n}));
}

Array rfftfreq(int64_t n, double d) {
  INS_CHECK(n > 0, "rfftfreq: n must be positive");
  INS_CHECK(d > 0, "rfftfreq: d must be positive");

  int64_t len = n / 2 + 1;
  std::vector<double> data(len);
  double inv = 1.0 / (d * n);

  for (int64_t i = 0; i < len; ++i) {
    data[i] = i * inv;
  }

  return to_array(data, Shape({len}));
}

Array fftshift(const Array &x, int axis) {
  INS_CHECK(x.defined(), "fftshift: input is undefined");

  int ndim = x.shape().ndim();

  if (axis == -1) {
    Array result = x;
    for (int i = 0; i < ndim; ++i) {
      result = fftshift(result, i);
    }
    return result;
  }

  int ax = axis;
  if (ax < 0)
    ax += ndim;
  INS_CHECK(ax >= 0 && ax < ndim, "fftshift: axis out of range");

  int64_t n = x.shape().dim(ax);
  int64_t mid = n / 2;

  auto last = x.slice(ax, mid, n, 1);
  auto first = x.slice(ax, 0, mid, 1);

  Array result = concat({last, first}, ax);

  return result;
}

Array ifftshift(const Array &x, int axis) {
  INS_CHECK(x.defined(), "ifftshift: input is undefined");

  int ndim = x.shape().ndim();

  if (axis == -1) {
    Array result = x;
    for (int i = 0; i < ndim; ++i) {
      result = ifftshift(result, i);
    }
    return result;
  }

  int ax = axis;
  if (ax < 0)
    ax += ndim;
  INS_CHECK(ax >= 0 && ax < ndim, "ifftshift: axis out of range");

  int64_t n = x.shape().dim(ax);
  int64_t mid = (n + 1) / 2; // Ceiling division for odd lengths

  auto last = x.slice(ax, mid, n, 1);
  auto first = x.slice(ax, 0, mid, 1);

  return concat({last, first}, ax);
}

int next_fast_len(int target) {
  if (target <= 0)
    return 1;

  const int primes[] = {2, 3, 5, 7};

  std::function<int(int, int)> impl = [&](int n, int idx) -> int {
    if (n == 1)
      return 1;
    if (idx == 4)
      return INT_MAX;

    int p = primes[idx];
    int branch1 = impl((n + p - 1) / p, idx);
    if (branch1 != INT_MAX)
      branch1 *= p;
    int branch2 = impl(n, idx + 1);

    return std::min(branch1, branch2);
  };

  return impl(target, 0);
}

} // namespace fft
} // namespace ins