// src/ops/signal/filtering.cpp
#include "insight/ops/signal/filtering.h"
#include "insight/core/axis.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#include "insight/ops/signal/convolution.h"
#include "insight/ops/signal/windows.h"
#include "insight/ops/unary.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

namespace {

static OpSchema signal_schema(const char *name, size_t input_count,
                              size_t output_count = 1) {
  std::vector<OpArgumentSchema> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back({"x" + std::to_string(i), OpArgumentKind::Array,
                      OpArgumentAccess::ReadOnly});
  }

  std::vector<OpArgumentSchema> outputs;
  outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    outputs.push_back({output_count == 1 ? "out" : "out" + std::to_string(i),
                       OpArgumentKind::Array, OpArgumentAccess::WriteOnly});
  }

  return OpSchema(name, OpKind::Signal, inputs, outputs)
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static void launch_signal_kernel(const char *name, const Place &place,
                                 DType dtype,
                                 const std::vector<Array> &inputs,
                                 const std::vector<Array> &outputs) {
  OpSchema schema = signal_schema(name, inputs.size(), outputs.size());
  ArrayIterator iter = schema.make_transform_iterator(inputs, outputs);
  auto arrays = iter.arrays();

  std::vector<OpRegistry::Arg> launch_inputs;
  launch_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    launch_inputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }

  std::vector<OpRegistry::Arg> launch_outputs;
  launch_outputs.reserve(outputs.size());
  for (size_t i = inputs.size(); i < arrays.size(); ++i) {
    launch_outputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }

  OpRegistry::launch_schema(name, place, dtype, launch_inputs, launch_outputs);
}

static void launch_signal_kernel(const char *name, const Place &place,
                                 DType dtype,
                                 const std::vector<Array> &inputs,
                                 const Array &output) {
  launch_signal_kernel(name, place, dtype, inputs, std::vector<Array>{output});
}

} // namespace

// ============================================================================
// hilbert
// ============================================================================

Array hilbert(const Array &x, int64_t N) {
  INS_CHECK(x.defined(), "hilbert: input is undefined");
  INS_CHECK(x.dtype() == DType::F32 || x.dtype() == DType::F64,
            "hilbert: only float types supported");

  if (N < 0)
    N = x.numel();
  INS_CHECK(N >= 1, "hilbert: N must be >= 1");

  Place cpu = CPUPlace();
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

  // FFT of input
  Array Xf = fft::fft(x, N);

  // Build frequency-domain Hilbert transform filter:
  // h[0] = 1, h[1..N/2-1] = 2, h[N/2] = 1 (if N even), h[N/2+1..N-1] = 0
  std::vector<std::complex<double>> h_data(N, {0.0, 0.0});
  h_data[0] = {1.0, 0.0};
  for (int64_t i = 1; i < (N + 1) / 2; ++i) {
    h_data[i] = {2.0, 0.0};
  }
  if (N % 2 == 0) {
    h_data[N / 2] = {1.0, 0.0};
  }

  Array h = to_array(h_data, cdtype, cpu);
  if (Xf.place().kind() != DeviceKind::CPU) {
    h = h.to(Xf.place());
  }

  // Multiply in frequency domain and inverse FFT
  Array result = fft::ifft(mul(Xf, h), N);
  return result;
}

// ============================================================================
// hilbert2
// ============================================================================

Array hilbert2(const Array &x, int64_t N) {
  INS_CHECK(x.defined(), "hilbert2: input is undefined");
  INS_CHECK(x.shape().ndim() == 2, "hilbert2: input must be 2D");

  int64_t rows = x.shape().dim(0);
  int64_t cols = x.shape().dim(1);

  if (N < 0)
    N = std::max(rows, cols);

  Place cpu = CPUPlace();
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

  // Build 1D Hilbert filter
  auto make_h = [](int64_t n) {
    std::vector<std::complex<double>> h(n, {0.0, 0.0});
    h[0] = {1.0, 0.0};
    for (int64_t i = 1; i < (n + 1) / 2; ++i)
      h[i] = {2.0, 0.0};
    if (n % 2 == 0)
      h[n / 2] = {1.0, 0.0};
    return h;
  };

  // Build 2D filter as outer product
  auto h_row = make_h(N);
  auto h_col = make_h(N);
  std::vector<std::complex<double>> h2d(N * N);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      h2d[i * N + j] = h_row[i] * h_col[j];
    }
  }

  Array h_arr = to_array(h2d, Shape({N, N}), cdtype, cpu);
  Array Xf = fft::fft2(x, {N, N});

  if (Xf.place().kind() != DeviceKind::CPU) {
    h_arr = h_arr.to(Xf.place());
  }

  Array result = fft::ifft2(mul(Xf, h_arr));
  // Crop back to original size
  result = slice(result, {0}, {0}, {rows});
  result = slice(result, {1}, {0}, {cols});
  return result;
}

// ============================================================================
// detrend
// ============================================================================

Array detrend(const Array &data, int axis, const std::string &type) {
  INS_CHECK(data.defined(), "detrend: input is undefined");
  INS_CHECK(type == "linear" || type == "constant",
            "detrend: type must be 'linear' or 'constant'");

  int ndim = data.shape().ndim();
  int ax = normalize_axis(axis, ndim, "detrend");

  if (type == "constant") {
    Array m = mean(data, ax, true);
    return sub(data, m);
  }

  // Linear detrend: subtract best-fit line along axis
  int64_t n = data.shape().dim(ax);
  if (n <= 1)
    return data.copy();

  double t_mean = (n - 1.0) / 2.0;
  double t_var = (n * n - 1.0) / 12.0;

  Array x_mean = mean(data, ax, true);

  // Build t array with correct shape for broadcasting
  Array t = arange(0.0, static_cast<double>(n), 1.0, DType::F64);
  std::vector<int64_t> t_shape(ndim, 1);
  t_shape[ax] = n;
  t = reshape(t, Shape(t_shape));

  DType work_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (t.dtype() != work_dtype)
    t = t.to(work_dtype);

  Array t_centered = sub(t, full(Shape(t_shape), t_mean, work_dtype));
  Array x_centered =
      sub(data.dtype() == work_dtype ? data : data.to(work_dtype),
          x_mean.dtype() == work_dtype ? x_mean : x_mean.to(work_dtype));

  Array product = mul(t_centered, x_centered);
  Array sum_product = sum(product, ax, true);

  double denom = n * t_var;
  Array slope = div(sum_product, full(sum_product.shape(), denom, work_dtype));
  Array trend = add(mul(slope, t_centered), x_mean);

  Array result =
      sub(data.dtype() == work_dtype ? data : data.to(work_dtype), trend);
  if (result.dtype() != data.dtype()) {
    result = result.to(data.dtype());
  }
  return result;
}

// ============================================================================
// firfilter — FIR filtering via convolution
// ============================================================================

Array firfilter(const Array &b, const Array &x, int axis) {
  INS_CHECK(b.defined() && x.defined(), "firfilter: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1, "firfilter: b must be 1D");

  // For 1D input, use convolve directly
  if (x.shape().ndim() == 1) {
    return convolve(x, b, "full");
  }

  // For multi-dimensional: apply convolve along each slice of the axis
  int ndim = x.shape().ndim();
  int ax = normalize_axis(axis, ndim, "firfilter");

  // Move target axis to last position
  Array x_moved = x;
  if (ax != ndim - 1) {
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    x_moved = permute(x_moved, perm);
  }

  // Reshape to 2D: (batch, signal_len)
  int64_t batch_size = x_moved.numel() / x.shape().dim(ax);
  int64_t signal_len = x.shape().dim(ax);
  Array x_2d = reshape(x_moved, {batch_size, signal_len});

  // Apply filter to each row using slice + convolve
  int64_t out_len = signal_len + b.numel() - 1;

  Place cpu = CPUPlace();
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array x_cpu = (x_2d.place().kind() == DeviceKind::CPU) ? x_2d : x_2d.to(cpu);

  // Process each row using composite ops
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  std::vector<Array> row_results;

  for (int64_t row = 0; row < batch_size; ++row) {
    // Extract row as 1D array
    Array row_arr =
        slice(x_cpu, {0}, {static_cast<int>(row)}, {static_cast<int>(row + 1)});
    row_arr = reshape(row_arr, {signal_len});
    if (row_arr.dtype() != work_dtype)
      row_arr = row_arr.to(work_dtype);

    // Convolve
    Array conv = convolve(row_arr, b_cpu, "full");
    if (conv.dtype() != work_dtype)
      conv = conv.to(work_dtype);
    row_results.push_back(conv);
  }

  // Stack all rows into a 2D array
  Array stacked = stack(row_results, 0);

  // Reshape back
  Array result = stacked;

  // Move axis back if needed
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = permute(result, inv_perm);
  }

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// firfilter_zi_state — FIR filter with initial/final state
// ============================================================================

std::pair<Array, Array> firfilter_zi_state(const Array &b, const Array &x,
                                           const Array &zi, int axis) {
  INS_CHECK(b.defined() && x.defined() && zi.defined(),
            "firfilter_zi_state: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1, "firfilter_zi_state: b must be 1D");
  INS_CHECK(zi.shape().ndim() == 1, "firfilter_zi_state: zi must be 1D");
  INS_CHECK(zi.numel() == b.numel() - 1,
            "firfilter_zi_state: zi length must be len(b)-1");

  int64_t nb = b.numel();
  int64_t zi_len = zi.numel();
  int64_t n = x.numel();

  // Work on CPU
  Place cpu = CPUPlace();
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array zi_cpu = (zi.place().kind() == DeviceKind::CPU) ? zi : zi.to(cpu);

  // Ensure consistent dtype
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (b_cpu.dtype() != work_dtype)
    b_cpu = b_cpu.to(work_dtype);
  if (x_cpu.dtype() != work_dtype)
    x_cpu = x_cpu.to(work_dtype);
  if (zi_cpu.dtype() != work_dtype)
    zi_cpu = zi_cpu.to(work_dtype);

  // Build extended input: [zi, x]
  Array x_ext = concat({zi_cpu, x_cpu}, 0);
  int64_t n_ext = zi_len + n;

  // Full convolution: y = convolve(x_ext, b, "full")
  // Output length = n_ext + nb - 1 = zi_len + n + nb - 1
  Array y_full = convolve(x_ext, b_cpu, "full");

  // Extract the relevant portion: we want the "same" part relative to x_ext
  // For lfilter-style output, we want the last n elements (matching x length)
  // But firfilter_zi_state returns the full convolution result
  // Output length should be n + nb - 1 (full convolution of x with b,
  // but with zi providing initial conditions)
  // Actually, the output should be length n (same as input) for lfilter
  // compatibility The kernel computes full convolution of [zi, x] with b,
  // output length = n_ext + nb - 1 But we want just the portion corresponding
  // to x: indices [zi_len-1 .. zi_len-1+n-1] which is length n
  Array y = slice(y_full, {0}, {static_cast<int>(zi_len - 1)},
                  {static_cast<int>(zi_len - 1 + n)});

  // Final state: last zi_len elements of extended input x_ext
  Array zf = slice(x_ext, {0}, {static_cast<int>(n_ext - zi_len)},
                   {static_cast<int>(n_ext)});

  // Transfer back to original device
  if (x.place().kind() != DeviceKind::CPU) {
    y = y.to(x.place());
    zf = zf.to(x.place());
  }
  if (y.dtype() != x.dtype())
    y = y.to(x.dtype());
  if (zf.dtype() != x.dtype())
    zf = zf.to(x.dtype());

  return {y, zf};
}

// ============================================================================
// lfilter — IIR/FIR digital filter (sequential, requires backend kernel)
// ============================================================================

Array lfilter(const Array &b, const Array &a, const Array &x, int axis) {
  INS_CHECK(b.defined() && a.defined() && x.defined(),
            "lfilter: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1 && a.shape().ndim() == 1,
            "lfilter: b and a must be 1D");

  // FIR case (a has only one element)
  if (a.numel() == 1) {
    return firfilter(b, x, axis);
  }

  // IIR case: dispatch to backend kernel
  Place cpu = CPUPlace();
  Array b_cpu = b.contiguous().to(cpu).to(DType::F64);
  Array a_cpu = a.contiguous().to(cpu).to(DType::F64);
  Array x_cpu = x.contiguous().to(cpu);

  DType work_dtype = (x_cpu.dtype() == DType::F32) ? DType::F32 : DType::F64;
  x_cpu = x_cpu.to(work_dtype);

  // Normalize coefficients by a[0]
  Array a0 = slice(a_cpu, 0, 0, 1);
  INS_CHECK(std::abs(a0.item<double>()) > 1e-15,
            "lfilter: a[0] must not be zero");
  b_cpu = div(b_cpu, a0);
  a_cpu = div(a_cpu, a0);
  // Ensure F64 for kernel
  b_cpu = b_cpu.to(DType::F64);
  a_cpu = a_cpu.to(DType::F64);

  int ndim = x_cpu.shape().ndim();
  int ax = normalize_axis(axis, ndim, "lfilter");
  int64_t n = x_cpu.shape().dim(ax);

  // Move signal axis to last
  Array x_moved = x_cpu;
  if (ax != ndim - 1) {
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    x_moved = permute(x_moved, perm);
  }

  // Flatten to 2D [batch, signal_len] for kernel
  int64_t batch = x_moved.numel() / n;
  Array x_flat = reshape(x_moved, {batch, n}).to(DType::F64);

  // Pre-allocate output
  Array y_flat({batch, n}, DType::F64, cpu);

  // Dispatch to backend kernel: inputs=[b, a, x], outputs=[y]
  launch_signal_kernel("lfilter", cpu, DType::F64, {b_cpu, a_cpu, x_flat},
                       y_flat);

  // Reshape back to original shape
  std::vector<int64_t> out_shape;
  for (int i = 0; i < ndim; ++i)
    out_shape.push_back((i == ax) ? n : x_cpu.shape().dim(i));
  Array result = reshape(y_flat, Shape(out_shape));

  // Permute signal axis back if needed
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = permute(result, inv_perm);
  }

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// lfilter_zi
// ============================================================================

Array lfilter_zi(const Array &b, const Array &a) {
  INS_CHECK(b.defined() && a.defined(), "lfilter_zi: inputs are undefined");

  Place cpu = CPUPlace();
  Array b_cpu = b.contiguous().to(cpu).to(DType::F64);
  Array a_cpu = a.contiguous().to(cpu).to(DType::F64);

  int64_t nb = b_cpu.numel();
  int64_t na = a_cpu.numel();
  int64_t n = std::max(nb, na);
  int64_t m = n - 1;

  if (m == 0) {
    return zeros({0}, DType::F64, cpu);
  }

  // Normalize by a[0]
  Array a0 = slice(a_cpu, 0, 0, 1);
  b_cpu = div(b_cpu, a0);
  a_cpu = div(a_cpu, a0);

  // Pre-allocate output
  Array zi({m}, DType::F64, cpu);

  // Dispatch to backend kernel: inputs=[b_norm, a_norm], outputs=[zi]
  launch_signal_kernel("lfilter_zi", cpu, DType::F64, {b_cpu, a_cpu}, zi);

  return zi;
}

// ============================================================================
// filtfilt — zero-phase filtering
// ============================================================================

Array filtfilt(const Array &b, const Array &a, const Array &x, int axis) {
  INS_CHECK(b.defined() && a.defined() && x.defined(),
            "filtfilt: inputs are undefined");

  int ndim = x.shape().ndim();
  int ax = normalize_axis(axis, ndim, "filtfilt");

  Array y = lfilter(b, a, x, ax);

  int64_t orig_len = x.shape().dim(ax);
  if (y.shape().dim(ax) > orig_len) {
    int64_t nb = b.numel();
    int64_t start = nb - 1;
    y = slice(y, {ax}, {start}, {start + orig_len});
  }

  Array y_rev = flip(y, ax);
  y_rev = lfilter(b, a, y_rev, ax);

  if (y_rev.shape().dim(ax) > orig_len) {
    int64_t nb = b.numel();
    int64_t start = nb - 1;
    y_rev = slice(y_rev, {ax}, {start}, {start + orig_len});
  }

  return flip(y_rev, ax);
}

// ============================================================================
// decimate — downsample with anti-aliasing
// ============================================================================

Array decimate(const Array &x, int64_t q, int axis, bool zero_phase) {
  INS_CHECK(x.defined(), "decimate: input is undefined");
  INS_CHECK(q >= 1, "decimate: q must be >= 1");
  if (q == 1)
    return x.copy();

  int64_t n = 10 * q + 1;
  double cutoff = 1.0 / q;
  Array h = firwin(n, {cutoff}, "hamming", "lowpass", true);
  Array a = to_array(std::vector<double>{1.0});

  Array filtered =
      zero_phase ? filtfilt(h, a, x, axis) : lfilter(h, a, x, axis);

  int ax = normalize_axis(axis, x.shape().ndim(), "decimate");

  return slice(filtered, {ax}, {0}, {x.shape().dim(ax)}, {static_cast<int>(q)});
}

// ============================================================================
// resample — FFT-based resampling
// ============================================================================

Array resample(const Array &x, int64_t num, int axis) {
  INS_CHECK(x.defined(), "resample: input is undefined");
  INS_CHECK(num >= 1, "resample: num must be >= 1");

  int ax = normalize_axis(axis, x.shape().ndim(), "resample");
  int64_t n = x.shape().dim(ax);

  if (num == n)
    return x.copy();

  Place cpu = CPUPlace();
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

  // For 1D case
  Array Xf = fft::fft(x, n);
  Array Xf_cpu = (Xf.place().kind() == DeviceKind::CPU) ? Xf : Xf.to(cpu);

  // Build new spectrum using composite ops
  // Extract positive frequencies: Xf[0:half+1]
  int64_t half_n = n / 2;
  Array X_pos = slice(Xf_cpu, {0}, {0}, {static_cast<int>(half_n + 1)});

  // Build negative frequencies: Xf[n-half:n] reversed = Xf[n-half], ...,
  // Xf[n-1]
  Array X_neg;
  if (half_n > 1) {
    X_neg = slice(Xf_cpu, {0}, {static_cast<int>(n - half_n)},
                  {static_cast<int>(n)});
    X_neg = flip(X_neg, 0);
  }

  std::vector<Array> parts;
  // DC
  parts.push_back(slice(X_pos, {0}, {0}, {1}));

  if (num > n) {
    // Zero-pad positive frequencies
    int64_t pad_pos = (num / 2) - half_n;
    if (pad_pos > 0) {
      Array pad_zeros_pos = zeros({pad_pos}, cdtype, cpu);
      parts.push_back(slice(X_pos, {0}, {1}, {static_cast<int>(half_n + 1)}));
      parts.push_back(pad_zeros_pos);
    } else {
      parts.push_back(slice(X_pos, {0}, {1}, {static_cast<int>(half_n + 1)}));
    }
    // Nyquist
    if (n % 2 == 0) {
      parts.push_back(slice(X_pos, {0}, {static_cast<int>(half_n)},
                            {static_cast<int>(half_n + 1)}));
    }
    // Zero-pad
    if (pad_pos > 0) {
      Array pad_zeros_neg = zeros({pad_pos}, cdtype, cpu);
      parts.push_back(pad_zeros_neg);
    }
    // Negative frequencies (flipped back)
    if (X_neg.defined() && X_neg.numel() > 0) {
      parts.push_back(X_neg);
    }
  } else {
    // Truncate positive frequencies
    int64_t half_num = num / 2;
    parts.push_back(slice(X_pos, {0}, {1}, {static_cast<int>(half_num + 1)}));
    // Nyquist
    if (num % 2 == 0) {
      parts.push_back(slice(X_pos, {0}, {static_cast<int>(half_num)},
                            {static_cast<int>(half_num + 1)}));
    }
    // Truncated negative frequencies
    if (X_neg.defined() && X_neg.numel() > 0) {
      int64_t neg_keep = std::min(half_num - 1, X_neg.numel());
      if (neg_keep > 0) {
        parts.push_back(slice(X_neg, {0}, {0}, {static_cast<int>(neg_keep)}));
      }
    }
  }

  Array new_X_arr = concat(parts, 0);
  // Ensure correct size
  if (new_X_arr.numel() != num) {
    // Pad or truncate to exact size
    if (new_X_arr.numel() < num) {
      Array pad = zeros({num - new_X_arr.numel()}, cdtype, cpu);
      new_X_arr = concat({new_X_arr, pad}, 0);
    } else {
      new_X_arr = slice(new_X_arr, {0}, {0}, {static_cast<int>(num)});
    }
  }
  Array result = fft::ifft(new_X_arr, num);
  result = mul(result, full(result.shape(), static_cast<double>(num) / n,
                            result.dtype()));

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// resample_poly — polyphase resampling
// ============================================================================

Array resample_poly(const Array &x, int64_t up, int64_t down, int axis) {
  INS_CHECK(x.defined(), "resample_poly: input is undefined");
  INS_CHECK(up >= 1 && down >= 1, "resample_poly: up and down must be >= 1");

  if (up == 1 && down == 1)
    return x.copy();

  // Design anti-aliasing filter
  int64_t max_rate = std::max(up, down);
  double f_c = 1.0 / max_rate;
  int64_t n = 10 * max_rate + 1;
  Array h = firwin(n, {f_c}, "hamming", "lowpass", true);

  Place cpu = CPUPlace();
  Array h_up =
      mul(h, full(h.shape(), static_cast<double>(up), DType::F64, cpu));

  int64_t input_len = x.numel();
  int64_t n_out = static_cast<int64_t>(
      std::ceil(static_cast<double>(input_len) * up / down));

  // Upsample by inserting zeros using scatter
  int64_t upsampled_len = input_len * up;
  Array up_arr = zeros({upsampled_len}, x.dtype(), cpu);

  // Set every up-th element to x[i] * up
  Array indices = arange(0.0, static_cast<double>(upsampled_len),
                         static_cast<double>(up), DType::I64, cpu);
  Array x_scaled =
      mul(x, full(x.shape(), static_cast<double>(up), x.dtype(), cpu));
  if (x_scaled.place().kind() != DeviceKind::CPU)
    x_scaled = x_scaled.to(cpu);
  put(up_arr, indices, x_scaled);

  // Apply FIR filter
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (up_arr.dtype() != work_dtype)
    up_arr = up_arr.to(work_dtype);
  if (h_up.dtype() != work_dtype)
    h_up = h_up.to(work_dtype);
  Array filtered = convolve(up_arr, h_up, "full");

  // Downsample by taking every down-th sample
  int64_t offset = (h_up.numel() - 1) / 2;
  Array result = slice(filtered, {0}, {static_cast<int>(offset)},
                       {static_cast<int>(offset + n_out * down)},
                       {static_cast<int>(down)});

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// freq_shift
// ============================================================================

Array freq_shift(const Array &x, double freq, double fs) {
  INS_CHECK(x.defined(), "freq_shift: input is undefined");
  INS_CHECK(fs > 0, "freq_shift: fs must be > 0");

  int64_t n = x.numel();
  Place cpu = CPUPlace();

  // Build complex exponential: exp(j*2*pi*freq/fs * [0,1,...,n-1])
  Array t = arange(0.0, static_cast<double>(n), 1.0, DType::F64, cpu);
  double phase_inc = 2.0 * M_PI * freq / fs;
  Array phase = mul(t, full(t.shape(), phase_inc, DType::F64, cpu));

  // cos(phase) + j*sin(phase)
  Array cos_val = cos(phase);
  Array sin_val = sin(phase);
  Array exp_arr = to_complex(cos_val, sin_val);

  // Convert x to complex if needed
  Array x_cpx;
  if (x.dtype() == DType::C64 || x.dtype() == DType::C32) {
    x_cpx = x;
  } else {
    x_cpx = to_complex(x);
  }

  // Element-wise multiply
  if (x_cpx.place().kind() != DeviceKind::CPU) {
    exp_arr = exp_arr.to(x_cpx.place());
  }

  Array result = mul(x_cpx, exp_arr);
  return result;
}

// ============================================================================
// wiener
// ============================================================================

Array wiener(const Array &im, const std::vector<int64_t> &mysize,
             double noise) {
  INS_CHECK(im.defined(), "wiener: input is undefined");

  int ndim = im.shape().ndim();

  std::vector<int64_t> sz = mysize;
  if (sz.empty()) {
    sz.assign(ndim, 3);
  }
  INS_CHECK(static_cast<int>(sz.size()) == ndim,
            "wiener: mysize must have same length as ndim");

  Place cpu = CPUPlace();
  DType work_dtype = (im.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array im_work = im.dtype() == work_dtype ? im : im.to(work_dtype);

  // Compute local mean using separable convolution
  Array local_mean = im_work;
  for (int d = 0; d < ndim; ++d) {
    int64_t k_size = sz[d];
    std::vector<double> kernel(k_size, 1.0 / k_size);
    Array k_arr = to_array(kernel, work_dtype, cpu);
    local_mean = convolve(local_mean, k_arr, "same");
  }

  // Compute local variance: E[x^2] - E[x]^2
  Array im_sq = mul(im_work, im_work);
  Array local_sq_mean = im_sq;
  for (int d = 0; d < ndim; ++d) {
    int64_t k_size = sz[d];
    std::vector<double> kernel(k_size, 1.0 / k_size);
    Array k_arr = to_array(kernel, work_dtype, cpu);
    local_sq_mean = convolve(local_sq_mean, k_arr, "same");
  }

  Array local_var = sub(local_sq_mean, mul(local_mean, local_mean));

  // Estimate noise if not provided
  double noise_est = noise;
  if (noise_est < 0) {
    Array var_mean = mean(local_var);
    noise_est = (work_dtype == DType::F64)
                    ? var_mean.item<double>()
                    : static_cast<double>(var_mean.item<float>());
    if (noise_est < 0)
      noise_est = 0;
  }

  // Wiener filter: result = local_mean + max(0, local_var - noise) / local_var
  // * (im - local_mean)
  Array noise_arr = full(local_var.shape(), noise_est, work_dtype, cpu);
  Array var_minus_noise = sub(local_var, noise_arr);
  Array zero = zeros_like(local_var);
  Array numerator = maximum(var_minus_noise, zero);

  Array eps = full(local_var.shape(), 1e-10, work_dtype, cpu);
  Array denominator = maximum(local_var, eps);

  Array gain = div(numerator, denominator);
  Array residual = sub(im_work, local_mean);
  Array result = add(local_mean, mul(gain, residual));

  if (result.dtype() != im.dtype())
    result = result.to(im.dtype());
  if (im.place().kind() != DeviceKind::CPU)
    result = result.to(im.place());
  return result;
}

// ============================================================================
// sosfilt — second-order sections filtering
// ============================================================================

Array sosfilt(const Array &x, const Array &sos) {
  INS_CHECK(x.defined(), "sosfilt: x is undefined");
  INS_CHECK(sos.defined(), "sosfilt: sos is undefined");
  INS_CHECK(x.shape().ndim() == 1, "sosfilt: x must be 1D");
  INS_CHECK(sos.shape().ndim() == 2, "sosfilt: sos must be 2D");
  INS_CHECK(sos.shape().dims()[1] == 6,
            "sosfilt: sos must have 6 columns [b0,b1,b2,a0,a1,a2]");

  Place cpu = CPUPlace();
  Array x_cpu = x.to(cpu).to(DType::F64);
  Array sos_cpu = sos.to(cpu).to(DType::F64);

  int64_t n = x.numel();
  int64_t n_sections = sos.shape().dims()[0];

  // Get raw data pointers
  const double *x_data = x_cpu.data<double>();
  const double *sos_data = sos_cpu.data<double>();

  // Create output array
  std::vector<double> y(n);
  std::vector<double> temp(n);

  // Copy input to working buffer
  for (int64_t i = 0; i < n; ++i)
    y[i] = x_data[i];

  // Apply each biquad section in cascade
  for (int64_t s = 0; s < n_sections; ++s) {
    const double *section = sos_data + s * 6;
    double b0 = section[0], b1 = section[1], b2 = section[2];
    double a0 = section[3], a1 = section[4], a2 = section[5];

    // Normalize by a0
    if (std::abs(a0) > 1e-30) {
      b0 /= a0;
      b1 /= a0;
      b2 /= a0;
      a1 /= a0;
      a2 /= a0;
    }

    // Direct Form II transposed
    double w1 = 0, w2 = 0;
    for (int64_t i = 0; i < n; ++i) {
      double xi = y[i];
      double wi = xi - a1 * w1 - a2 * w2;
      temp[i] = b0 * wi + b1 * w1 + b2 * w2;
      w2 = w1;
      w1 = wi;
    }

    // Swap buffers
    y.swap(temp);
  }

  // Convert back to original dtype
  if (x.dtype() == DType::F32) {
    std::vector<float> y_f32(n);
    for (int64_t i = 0; i < n; ++i)
      y_f32[i] = static_cast<float>(y[i]);
    Array result = to_array(y_f32, DType::F32, cpu);
    if (x.place().kind() != DeviceKind::CPU)
      result = result.to(x.place());
    return result;
  }

  Array result = to_array(y, DType::F64, cpu);
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// upfirdn — upsample, FIR filter, downsample
// ============================================================================

Array upfirdn(const Array &h, const Array &x, int64_t p, int64_t q) {
  INS_CHECK(h.defined(), "upfirdn: h is undefined");
  INS_CHECK(x.defined(), "upfirdn: x is undefined");
  INS_CHECK(h.shape().ndim() == 1, "upfirdn: h must be 1D");
  INS_CHECK(x.shape().ndim() == 1, "upfirdn: x must be 1D");
  INS_CHECK(p >= 1, "upfirdn: p must be >= 1");
  INS_CHECK(q >= 1, "upfirdn: q must be >= 1");

  Place cpu = CPUPlace();
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array h_cpu = h.to(cpu).to(work_dtype);
  Array x_cpu = x.to(cpu).to(work_dtype);

  int64_t h_len = h.numel();
  int64_t x_len = x.numel();

  // Upsample: insert zeros between samples
  int64_t up_len = x_len * p;
  std::vector<double> up_data(up_len, 0.0);
  const double *x_data = x_cpu.data<double>();
  for (int64_t i = 0; i < x_len; ++i) {
    up_data[i * p] = x_data[i];
  }

  // Apply FIR filter (direct convolution)
  const double *h_data = h_cpu.data<double>();
  int64_t conv_len = up_len + h_len - 1;
  std::vector<double> conv_data(conv_len, 0.0);

  for (int64_t i = 0; i < conv_len; ++i) {
    double sum = 0.0;
    int64_t k_start = std::max(int64_t(0), i - up_len + 1);
    int64_t k_end = std::min(h_len - 1, i);
    for (int64_t k = k_start; k <= k_end; ++k) {
      int64_t j = i - k;
      if (j >= 0 && j < up_len) {
        sum += h_data[k] * up_data[j];
      }
    }
    conv_data[i] = sum;
  }

  // Downsample: take every q-th sample, with offset for filter delay
  int64_t offset = h_len - 1;
  int64_t n_out = (conv_len - offset + q - 1) / q;
  std::vector<double> result_data;
  result_data.reserve(n_out);

  for (int64_t i = offset; i < conv_len; i += q) {
    result_data.push_back(conv_data[i]);
  }

  if (work_dtype == DType::F32) {
    std::vector<float> result_f32(result_data.size());
    for (size_t i = 0; i < result_data.size(); ++i)
      result_f32[i] = static_cast<float>(result_data[i]);
    Array result = to_array(result_f32, DType::F32, cpu);
    if (x.place().kind() != DeviceKind::CPU)
      result = result.to(x.place());
    return result;
  }

  Array result = to_array(result_data, DType::F64, cpu);
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// channelize_poly — polyphase channelizer
// ============================================================================

Array channelize_poly(const Array &x, const Array &h, int64_t n_channels) {
  INS_CHECK(x.defined(), "channelize_poly: x is undefined");
  INS_CHECK(h.defined(), "channelize_poly: h is undefined");
  INS_CHECK(x.shape().ndim() == 1, "channelize_poly: x must be 1D");
  INS_CHECK(h.shape().ndim() == 1, "channelize_poly: h must be 1D");
  INS_CHECK(n_channels >= 1, "channelize_poly: n_channels must be >= 1");

  Place cpu = CPUPlace();
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array x_cpu = x.to(cpu).to(work_dtype);
  Array h_cpu = h.to(cpu).to(work_dtype);

  int64_t x_len = x.numel();
  int64_t h_len = h.numel();

  // Number of output frames
  int64_t n_frames = x_len / n_channels;

  // Reshape prototype filter into polyphase matrix [n_channels x n_taps]
  int64_t n_taps = (h_len + n_channels - 1) / n_channels;
  // Pad h to n_channels * n_taps
  int64_t h_padded_len = n_channels * n_taps;
  std::vector<double> h_padded(h_padded_len, 0.0);
  const double *h_data = h_cpu.data<double>();
  for (int64_t i = 0; i < h_len; ++i)
    h_padded[i] = h_data[i];

  // Reshape into [n_channels x n_taps] (row-major: row k = h_padded[k],
  // k+n_channels, k+2*n_channels, ...)
  // polyphase[k][j] = h_padded[k + j * n_channels]

  // Reshape input into [n_frames x n_channels]
  const double *x_data = x_cpu.data<double>();

  // Output: [n_channels x n_frames]
  int64_t out_rows = n_channels;
  int64_t out_cols = n_frames;
  std::vector<double> out_data(out_rows * out_cols, 0.0);

  // For each channel k, convolve polyphase filter k with input column k
  for (int64_t k = 0; k < n_channels; ++k) {
    // Extract polyphase filter for channel k
    std::vector<double> h_k(n_taps);
    for (int64_t j = 0; j < n_taps; ++j) {
      int64_t idx = k + j * n_channels;
      h_k[j] = (idx < h_padded_len) ? h_padded[idx] : 0.0;
    }

    // Extract input samples for channel k: x[k], x[k+n_channels],
    // x[k+2*n_channels], ...
    std::vector<double> x_k;
    for (int64_t i = k; i < x_len; i += n_channels) {
      x_k.push_back(x_data[i]);
    }

    // Convolve h_k with x_k
    int64_t x_k_len = static_cast<int64_t>(x_k.size());
    int64_t conv_len = x_k_len + n_taps - 1;

    for (int64_t i = 0; i < out_cols; ++i) {
      double sum = 0.0;
      for (int64_t j = 0; j < n_taps; ++j) {
        int64_t idx = i - j;
        if (idx >= 0 && idx < x_k_len) {
          sum += h_k[j] * x_k[idx];
        }
      }
      out_data[k * out_cols + i] = sum;
    }
  }

  if (work_dtype == DType::F32) {
    std::vector<float> out_f32(out_data.size());
    for (size_t i = 0; i < out_data.size(); ++i)
      out_f32[i] = static_cast<float>(out_data[i]);
    Array result = to_array(out_f32, DType::F32, cpu);
    result = reshape(result, {out_rows, out_cols});
    if (x.place().kind() != DeviceKind::CPU)
      result = result.to(x.place());
    return result;
  }

  Array result = to_array(out_data, DType::F64, cpu);
  result = reshape(result, {out_rows, out_cols});
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

} // namespace signal
} // namespace ins
