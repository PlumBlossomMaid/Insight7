// src/ops/signal/radar.cpp
#include "insight/ops/signal/radar.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/cast.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
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
// pulse_compression — batched FFT-based pulse compression (GPU optimized)
// ============================================================================

Array pulse_compression(const Array &x, const Array &template_tx,
                        bool normalize, const std::string &window,
                        int64_t nfft) {
  INS_CHECK(x.defined(), "pulse_compression: x is undefined");
  INS_CHECK(template_tx.defined(),
            "pulse_compression: template_tx is undefined");
  INS_CHECK(x.shape().ndim() == 2,
            "pulse_compression: x must be 2D [num_pulses, samples_per_pulse]");
  INS_CHECK(template_tx.shape().ndim() == 1,
            "pulse_compression: template_tx must be 1D");

  int64_t num_pulses = x.shape().dim(0);
  int64_t samples_per_pulse = x.shape().dim(1);
  bool is_complex = (x.dtype() == DType::C32 || x.dtype() == DType::C64);

  // Determine FFT length (must be >= samples + template_len - 1 for full conv)
  int64_t tpl_len = template_tx.numel();
  int64_t min_n = samples_per_pulse + tpl_len - 1;
  if (nfft == 0 || nfft < min_n) {
    nfft = 1;
    while (nfft < min_n)
      nfft <<= 1;
  }

  // Apply window to template if specified
  Array tpl = template_tx;
  if (!window.empty()) {
    Array W = get_window(window, tpl_len, false);
    tpl = mul(tpl, W);
  }

  // Normalize template if requested
  if (normalize) {
    double norm_val = std::sqrt(sum(square(tpl)).item<double>());
    tpl = div(tpl, full(tpl.shape(), norm_val, tpl.dtype(), tpl.place()));
  }

  // Work on the same device as input
  Place dev = x.place();

  // Pad template to nfft and compute matched filter in freq domain
  DType work_dtype = is_complex ? x.dtype() : DType::F64;
  if (tpl.dtype() != work_dtype)
    tpl = tpl.to(work_dtype);

  Array tpl_padded = zeros({nfft}, work_dtype, dev);
  {
    Array tpl_view = tpl_padded.slice({Slice(0, tpl_len)});
    tpl_view.copy_from_(tpl);
  }

  // Matched filter in frequency domain: conj(flip(template))
  // Correct: fft(conj(flip(template_padded))) = conj(fft(template) *
  // exp(j*2π*k*(N-1)/N)) We compute it directly by flipping the template,
  // conjugating, then FFT.
  Array mf_fft;
  if (is_complex) {
    // For complex input: flip → conj → FFT (full complex path)
    Array tpl_flip = flip(tpl);
    Array mf_padded = zeros({nfft}, work_dtype, dev);
    {
      Array mf_view = mf_padded.slice({Slice(0, tpl_len)});
      mf_view.copy_from_(conj(tpl_flip));
    }
    mf_fft = fft::fft(mf_padded, nfft, 0);
  } else {
    // For real input: conj(rfft(template)) is correct (real signals are
    // symmetric)
    mf_fft = conj(fft::rfft(tpl_padded, nfft));
  }

  // Pad input and do batched FFT along axis 1
  Array x_work = x;
  if (x_work.dtype() != work_dtype)
    x_work = x_work.to(work_dtype);

  Array x_padded = zeros({num_pulses, nfft}, work_dtype, dev);
  {
    Array x_view = x_padded.slice({Slice(), Slice(0, samples_per_pulse)});
    x_view.copy_from_(x_work);
  }

  // Batched FFT along axis 1
  Array x_fft;
  if (is_complex) {
    x_fft = fft::fft(x_padded, nfft, 1);
  } else {
    x_fft = fft::rfft(x_padded, nfft, 1);
  }

  // Broadcast matched filter: [1, fft_len] × [num_pulses, fft_len]
  int64_t fft_len = is_complex ? nfft : (nfft / 2 + 1);
  Array mf_2d = reshape(mf_fft, {1, fft_len});
  Array prod = mul(x_fft, mf_2d);

  // Batched IFFT along axis 1
  Array result_full;
  if (is_complex) {
    result_full = fft::ifft(prod, nfft, 1, "backward");
  } else {
    // irfft returns real; wrap in to_complex for consistent complex output
    result_full = to_complex(fft::irfft(prod, nfft, 1));
  }

  // Crop to original signal length (matched filter 'same' mode)
  int64_t start = tpl_len / 2;
  int64_t end = start + samples_per_pulse;
  Array result = result_full.slice({Slice(), Slice(start, end)});
  return result;
}

// ============================================================================
// pulse_doppler — batched Doppler FFT along axis 0 (GPU optimized)
// ============================================================================

Array pulse_doppler(const Array &x, const std::string &window, int64_t nfft) {
  INS_CHECK(x.defined(), "pulse_doppler: x is undefined");
  INS_CHECK(x.shape().ndim() == 2,
            "pulse_doppler: x must be 2D [num_pulses, samples_per_pulse]");

  int64_t num_pulses = x.shape().dim(0);
  int64_t samples_per_pulse = x.shape().dim(1);

  if (nfft == 0)
    nfft = num_pulses;

  // Work on the same device as input
  Place dev = x.place();

  // Apply window along pulse dimension (axis 0)
  Array x_work = x;
  if (!window.empty()) {
    Array W = get_window(window, num_pulses, false);
    W = reshape(W, {num_pulses,
                    1}); // broadcast: [num_pulses, 1] × [num_pulses, samples]
    x_work = mul(x_work, W);
  }

  // Ensure complex for FFT
  bool is_complex =
      (x_work.dtype() == DType::C32 || x_work.dtype() == DType::C64);
  if (!is_complex) {
    x_work = to_complex(x_work);
  }

  // Batched FFT along axis 0: [num_pulses, samples] → [nfft, samples]
  Array result = fft::fft(x_work, nfft, 0);

  // It's conventional to fftshift along the Doppler (pulse) axis
  result = fft::fftshift(result, 0);

  return result;
}

// ============================================================================
// cfar_alpha
// ============================================================================

double cfar_alpha(double pfa, int N) {
  return N * (std::pow(pfa, -1.0 / N) - 1.0);
}

// ============================================================================
// ca_cfar — Cell-Averaging CFAR (sequential, requires backend kernel)
// ============================================================================

std::pair<Array, Array> ca_cfar(const Array &data,
                                const std::vector<int> &guard_cells,
                                const std::vector<int> &reference_cells,
                                double pfa) {
  INS_CHECK(data.defined(), "ca_cfar: input is undefined");

  Place cpu = CPUPlace();
  Array data_cpu = data.contiguous().to(cpu);

  int ndim = data.shape().ndim();
  INS_CHECK(ndim <= 2, "ca_cfar: supports 1D and 2D arrays only");

  DType work_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
  data_cpu = data_cpu.to(work_dtype);

  double alpha = cfar_alpha(
      pfa, (int)reference_cells.size() > 0 ? (2 * reference_cells[0]) : 2);

  if (ndim == 2) {
    int N_row = 2 * reference_cells[0];
    int N_col = reference_cells.size() > 1 ? 2 * reference_cells[1] : N_row;
    alpha = cfar_alpha(pfa, N_row * N_col);
  }

  // Wrap scalar parameters as 1-element arrays for kernel dispatch
  Array alpha_arr =
      to_array(std::vector<double>{alpha}, Shape({1}), DType::F64, cpu);
  int g = guard_cells.empty() ? 1 : guard_cells[0];
  int r = reference_cells.empty() ? 1 : reference_cells[0];

  Array gc_arr = to_array(std::vector<int>{g}, Shape({1}), DType::I32, cpu);
  Array rc_arr = to_array(std::vector<int>{r}, Shape({1}), DType::I32, cpu);

  // Pre-allocate outputs
  Array threshold = zeros(data_cpu.shape(), work_dtype, cpu);
  Array detections = zeros(data_cpu.shape(), DType::BOOL, cpu);

  // Dispatch to backend kernel
  launch_signal_kernel("ca_cfar", cpu, work_dtype,
                       {data_cpu, alpha_arr, gc_arr, rc_arr},
                       {threshold, detections});

  if (data.place().kind() != DeviceKind::CPU) {
    threshold = threshold.to(data.place());
    detections = detections.to(data.place());
  }
  return {threshold, detections};
}

// ============================================================================
// mvdr — MVDR Beamformer using composite ops
// ============================================================================

Array mvdr(const Array &x, const Array &sv, bool calc_cov) {
  INS_CHECK(x.defined(), "mvdr: x is undefined");
  INS_CHECK(sv.defined(), "mvdr: sv is undefined");

  INS_CHECK(x.shape().dim(0) <= x.shape().dim(1),
            "mvdr: matrix has more sensors than samples");
  INS_CHECK(x.shape().dim(0) == sv.numel(),
            "mvdr: steering vector and input data do not align");

  Place cpu = CPUPlace();
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;

  // Compute covariance matrix if needed: R = (1/N) * X * X^T
  Array R;
  if (calc_cov) {
    int64_t N = x.shape().dim(1);
    Array x_work = x.dtype() == work_dtype ? x : x.to(work_dtype);
    Array xT = transpose(x_work);
    R = div(matmul(x_work, xT), full({1}, static_cast<double>(N), work_dtype));
  } else {
    R = x.dtype() == work_dtype ? x : x.to(work_dtype);
  }

  // R_inv = inv(R)
  Array R_inv = inv(R);

  // sv as column vector
  Array sv_work = sv.dtype() == work_dtype ? sv : sv.to(work_dtype);
  Array sv_col;
  if (sv_work.shape().ndim() == 1) {
    sv_col = reshape(sv_work, {sv_work.numel(), 1});
  } else {
    sv_col = sv_work;
  }

  // w = R_inv * sv / (sv^T * R_inv * sv)
  Array wB = matmul(R_inv, sv_col);
  Array sv_t = transpose(sv_col);
  Array wA = matmul(sv_t, wB);

  // Scalar division using item()
  double wA_val = (work_dtype == DType::F64)
                      ? wA.item<double>()
                      : static_cast<double>(wA.item<float>());

  Array result = div(wB, full(wB.shape(), wA_val, work_dtype));
  result = reshape(result, {sv.numel()});

  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// ambgfun — Ambiguity Function (dispatches to backend kernel)
// ============================================================================

Array ambgfun(const Array &x, double fs, double prf, const Array &y,
              const std::string &cut, double cutValue) {
  INS_CHECK(x.defined(), "ambgfun: x is undefined");
  INS_CHECK(cut == "2d" || cut == "delay" || cut == "doppler",
            "ambgfun: cut must be '2d', 'delay', or 'doppler', got '", cut,
            "'");

  Place cpu = CPUPlace();
  Array x_cpu = x.contiguous().to(cpu);
  Array y_cpu = y.defined() ? y.contiguous().to(cpu) : x_cpu;

  int64_t N = x_cpu.numel();
  int64_t M = y_cpu.numel();

  // Convert to complex128
  Array x_cpx = x_cpu;
  if (x_cpu.dtype() != DType::C64)
    x_cpx = to_complex(x_cpu);
  Array y_cpx = y_cpu;
  if (y_cpu.dtype() != DType::C64)
    y_cpx = to_complex(y_cpu);

  // Normalize using composite ops: x /= sqrt(sum(|x|^2))
  // Compute energy in real domain to avoid sqrt(C64) which is not supported
  Array x_re = real(x_cpx);
  Array x_im = imag(x_cpx);
  Array x_energy = sum(add(square(x_re), square(x_im)));
  x_cpx = div(x_cpx, sqrt(x_energy));
  Array y_re = real(y_cpx);
  Array y_im = imag(y_cpx);
  Array y_energy = sum(add(square(y_re), square(y_im)));
  y_cpx = div(y_cpx, sqrt(y_energy));

  // Flatten to 1D
  x_cpx = reshape(x_cpx, {N});
  y_cpx = reshape(y_cpx, {M});

  // Build params array: [fs, N, M]
  Array params = to_array(std::vector<double>{fs, (double)N, (double)M},
                          Shape({3}), DType::F64, cpu);

  int64_t delay_len = N + M - 1;
  int64_t doppler_len = N;

  // Pre-allocate output
  Array result({doppler_len, delay_len}, DType::F64, cpu);

  // Dispatch to backend kernel
  launch_signal_kernel("ambgfun", cpu, DType::F64, {x_cpx, y_cpx, params},
                       result);

  // Extract 1D cut if requested
  if (cut == "delay") {
    // Cut along delay axis at a specific doppler index
    int64_t doppler_idx = static_cast<int64_t>(cutValue);
    if (doppler_idx < 0)
      doppler_idx += doppler_len;
    INS_CHECK(doppler_idx >= 0 && doppler_idx < doppler_len,
              "ambgfun: delay cut doppler index ", cutValue, " out of range [",
              -doppler_len, ", ", doppler_len, ")");
    // Extract row: result[doppler_idx, :]
    result = slice(result,
                   {Slice(doppler_idx, doppler_idx + 1), Slice(0, delay_len)});
    result = reshape(result, {delay_len});
  } else if (cut == "doppler") {
    // Cut along doppler axis at a specific delay index
    int64_t delay_idx = static_cast<int64_t>(cutValue);
    if (delay_idx < 0)
      delay_idx += delay_len;
    INS_CHECK(delay_idx >= 0 && delay_idx < delay_len,
              "ambgfun: doppler cut delay index ", cutValue, " out of range [",
              -delay_len, ", ", delay_len, ")");
    // Extract column: result[:, delay_idx]
    result =
        slice(result, {Slice(0, doppler_len), Slice(delay_idx, delay_idx + 1)});
    result = reshape(result, {doppler_len});
  }

  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

} // namespace signal
} // namespace ins
