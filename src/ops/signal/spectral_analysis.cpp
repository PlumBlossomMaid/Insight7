// src/ops/signal/spectral_analysis.cpp
#include "insight/ops/signal/spectral_analysis.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
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

Array detrend_segment(const Array &seg, const std::string &method) {
  if (method == "none" || method.empty())
    return seg;
  if (method == "constant") {
    Array m = mean(seg);
    return sub(seg, m);
  }
  return seg;
}

Array get_window_for_spectral(const std::string &window, int64_t nperseg) {
  return get_window(window, nperseg, false);
}

int64_t onesided_fft_len(int64_t n, bool onesided) {
  return onesided ? (n / 2 + 1) : n;
}

Array fft_freqs(int64_t nfft, double fs, bool onesided) {
  int64_t n = onesided ? (nfft / 2 + 1) : nfft;
  std::vector<double> freqs(n);
  double df = fs / nfft;
  for (int64_t i = 0; i < n; ++i) {
    freqs[i] = i * df;
  }
  return to_array(freqs, DType::F64, CPUPlace());
}

double compute_psd_scale(int64_t nfft, double fs, const Array &window,
                         const std::string &scaling, bool onesided) {
  double scale = (scaling == "density") ? fs : 1.0;

  // Window normalization: 1/sum(w^2)
  Array win_sq = square(window);
  double win_sum_sq = sum(win_sq).item<double>();
  scale /= win_sum_sq;

  return scale;
}

// Process one segment: window, detrend, FFT, return cross-spectrum
Array process_segment(const Array &x_seg, const Array &y_seg, const Array &win,
                      int64_t nfft, const std::string &detrend_method) {
  Place cpu = CPUPlace();

  // Window the segments
  Array x_win = mul(x_seg, win);
  Array y_win = mul(y_seg, win);

  // Detrend
  if (detrend_method == "constant") {
    x_win = sub(x_win, mean(x_win));
    y_win = sub(y_win, mean(y_win));
  }

  // Convert to complex for FFT
  DType cdtype = DType::C64;
  Array x_cpx = to_complex(x_win);
  Array y_cpx = to_complex(y_win);

  // FFT
  Array Xf = fft::fft(x_cpx, nfft);
  Array Yf = fft::fft(y_cpx, nfft);

  // Pxy = conj(X) * Y
  Array Xf_conj = conj(Xf);
  return mul(Xf_conj, Yf);
}

} // anonymous namespace

// ============================================================================
// csd — Cross Power Spectral Density
// ============================================================================

SpectralResult csd(const Array &x, const Array &y, double fs,
                   const std::string &window, int64_t nperseg, int64_t noverlap,
                   int64_t nfft, const std::string &detrend,
                   bool return_onesided, const std::string &scaling) {
  INS_CHECK(x.defined() && y.defined(), "csd: inputs are undefined");

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array y_cpu = (y.place().kind() == DeviceKind::CPU) ? y : y.to(cpu);

  if (nfft == 0)
    nfft = nperseg;
  if (noverlap == 0)
    noverlap = nperseg / 2;

  int64_t n = x_cpu.numel();
  int64_t step = nperseg - noverlap;

  Array win = get_window_for_spectral(window, nperseg);

  int64_t n_segments = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  INS_CHECK(n_segments >= 1, "csd: signal too short for given nperseg");

  int64_t freq_len = onesided_fft_len(nfft, return_onesided);

  // Accumulate cross-spectrum using Array composites
  Array Pxx_sum = zeros({freq_len}, DType::F64, cpu);

  for (int64_t seg = 0; seg < n_segments; ++seg) {
    int64_t start = seg * step;

    // Extract segments
    Array x_seg = slice(x_cpu, {0}, {start}, {start + nperseg});
    Array y_seg = slice(y_cpu, {0}, {start}, {start + nperseg});

    // Window and compute cross-spectrum
    if (win.dtype() != x_seg.dtype())
      win = win.to(x_seg.dtype());
    Array Pxy = process_segment(x_seg, y_seg, win, nfft, detrend);

    // Accumulate cross-spectrum using composite ops (no per-element extraction)
    Array Pxy_cpu = (Pxy.place().kind() == DeviceKind::CPU) ? Pxy : Pxy.to(cpu);
    Array Pxy_real_full = real(Pxy_cpu);
    Array Pxy_real = slice(Pxy_real_full, 0, 0, static_cast<int>(freq_len));
    Pxx_sum = add(Pxx_sum, Pxy_real.to(Pxx_sum.dtype()));
  }

  // Average and scale using composite ops
  double scale = compute_psd_scale(nfft, fs, win, scaling, return_onesided);
  double avg_scale = scale / n_segments;
  Array Pxx = mul(Pxx_sum, full({freq_len}, avg_scale, DType::F64, cpu));

  // For one-sided, double the non-DC and non-Nyquist components
  if (return_onesided && nfft > 1) {
    Array mask = ones({freq_len}, DType::F64, cpu);
    // Set first and last to 1.0, middle to 2.0
    // Use put: mask[1:-1] = 2.0
    Array idx =
        arange(1.0, static_cast<double>(freq_len - 1), 1.0, DType::I64, cpu);
    Array twos = full({freq_len - 2}, 2.0, DType::F64, cpu);
    mask = put(mask, idx, twos);
    Pxx = mul(Pxx, mask);
  }

  Array f = fft_freqs(nfft, fs, return_onesided);

  return {f, Pxx};
}

// ============================================================================
// welch — Power Spectral Density
// ============================================================================

SpectralResult welch(const Array &x, double fs, const std::string &window,
                     int64_t nperseg, int64_t noverlap, int64_t nfft,
                     const std::string &detrend, bool return_onesided,
                     const std::string &scaling) {
  return csd(x, x, fs, window, nperseg, noverlap, nfft, detrend,
             return_onesided, scaling);
}

// ============================================================================
// periodogram — Convenience wrapper
// ============================================================================

SpectralResult periodogram(const Array &x, double fs, const std::string &window,
                           int64_t nfft, const std::string &detrend,
                           bool return_onesided, const std::string &scaling) {
  return welch(x, fs, window, 256, 0, nfft, detrend, return_onesided, scaling);
}

// ============================================================================
// coherence
// ============================================================================

SpectralResult coherence(const Array &x, const Array &y, double fs,
                         const std::string &window, int64_t nperseg,
                         int64_t noverlap, int64_t nfft,
                         const std::string &detrend) {
  SpectralResult Pxx = welch(x, fs, window, nperseg, noverlap, nfft, detrend);
  SpectralResult Pyy = welch(y, fs, window, nperseg, noverlap, nfft, detrend);
  SpectralResult Pxy = csd(x, y, fs, window, nperseg, noverlap, nfft, detrend);

  // Cxy = |Pxy|^2 / (Pxx * Pyy) using composite ops
  Array pxy_sq = square(Pxy.Pxx);
  Array pxx_pyy = mul(Pxx.Pxx, Pyy.Pxx);

  // Avoid division by zero
  Array eps = full(pxx_pyy.shape(), 1e-30, pxx_pyy.dtype(), pxx_pyy.place());
  Array denominator = maximum(pxx_pyy, eps);
  Array cxy = div(pxy_sq, denominator);

  // Where denominator is too small, set to 0
  Array too_small = less_equal(pxx_pyy, eps);
  cxy = where(too_small, zeros_like(cxy), cxy);

  return {Pxx.f, cxy};
}

// ============================================================================
// spectrogram
// ============================================================================

SpectrogramResult spectrogram(const Array &x, double fs,
                              const std::string &window, int64_t nperseg,
                              int64_t noverlap, int64_t nfft,
                              const std::string &detrend, bool return_onesided,
                              const std::string &mode) {
  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  if (nfft == 0)
    nfft = nperseg;
  if (noverlap == 0)
    noverlap = nperseg / 8;

  int64_t n = x_cpu.numel();
  int64_t step = nperseg - noverlap;
  int64_t n_segments = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  INS_CHECK(n_segments >= 1, "spectrogram: signal too short");

  int64_t freq_len = onesided_fft_len(nfft, return_onesided);

  Array win = get_window_for_spectral(window, nperseg);

  // Compute time array
  std::vector<double> t_arr(n_segments);
  for (int64_t i = 0; i < n_segments; ++i) {
    t_arr[i] = (i * step + nperseg / 2.0) / fs;
  }

  // Compute spectrogram segment by segment, accumulate into 2D Array
  Array Sxx_2d = zeros({freq_len, n_segments}, DType::F64, cpu);

  for (int64_t seg = 0; seg < n_segments; ++seg) {
    int64_t start = seg * step;

    // Extract and window segment
    Array seg_arr = slice(x_cpu, {0}, {start}, {start + nperseg});
    if (win.dtype() != seg_arr.dtype())
      win = win.to(seg_arr.dtype());
    Array windowed = mul(seg_arr, win);

    // Detrend
    if (detrend == "constant") {
      windowed = sub(windowed, mean(windowed));
    }

    // FFT
    Array seg_cpx = to_complex(windowed);
    Array Xf = fft::fft(seg_cpx, nfft);
    Array Xf_cpu = (Xf.place().kind() == DeviceKind::CPU) ? Xf : Xf.to(cpu);

    // Extract values based on mode using composite ops
    Array Xf_real;
    if (mode == "psd") {
      // |Xf|^2 = real^2 + imag^2
      Array r = real(Xf_cpu);
      Array im = imag(Xf_cpu);
      Xf_real = add(square(r), square(im));
    } else if (mode == "complex") {
      Xf_real = real(Xf_cpu);
    } else {
      Xf_real = abs(Xf_cpu);
    }

    // Store column using put or direct accumulation
    Array col = slice(Xf_real, 0, 0, static_cast<int>(freq_len));
    // Accumulate into Sxx_2d column by column
    // Build index for this column's elements
    Array row_idx =
        arange(0.0, static_cast<double>(freq_len), 1.0, DType::I64, cpu);
    Array col_idx =
        full({freq_len}, static_cast<int64_t>(seg), DType::I64, cpu);
    // Use scatter to write column
    Sxx_2d = scatter(Sxx_2d, 1, col_idx, col);
  }

  // Scale PSD using composite ops
  if (mode == "psd") {
    double scale = compute_psd_scale(nfft, fs, win, "density", return_onesided);
    Sxx_2d = mul(Sxx_2d, full({1}, scale, DType::F64, cpu));
    if (return_onesided && nfft > 1) {
      Array mask = ones({freq_len}, DType::F64, cpu);
      Array idx_mid =
          arange(1.0, static_cast<double>(freq_len - 1), 1.0, DType::I64, cpu);
      Array twos = full({freq_len - 2}, 2.0, DType::F64, cpu);
      mask = put(mask, idx_mid, twos);
      // Broadcast mask across segments: mask is [freq_len], Sxx_2d is
      // [freq_len, n_segments]
      Sxx_2d = mul(Sxx_2d, unsqueeze(mask, 1));
    }
  }

  Array f = fft_freqs(nfft, fs, return_onesided);
  Array t = to_array(t_arr, DType::F64, cpu);
  // Transpose from [freq_len, n_segments] to [n_segments, freq_len] for output
  Array S = transpose(Sxx_2d);

  return {f, t, S};
}

// ============================================================================
// stft
// ============================================================================

SpectrogramResult stft(const Array &x, double fs, const std::string &window,
                       int64_t nperseg, int64_t noverlap, int64_t nfft) {
  return spectrogram(x, fs, window, nperseg, noverlap, nfft, "none", true,
                     "complex");
}

// ============================================================================
// istft
// ============================================================================

Array istft(const Array &Zxx, double fs, const std::string &window,
            int64_t nperseg, int64_t noverlap, int64_t nfft) {
  INS_CHECK(Zxx.defined(), "istft: Zxx is undefined");
  INS_CHECK(Zxx.shape().ndim() == 2, "istft: Zxx must be 2D");

  int64_t n_freq = Zxx.shape().dims()[0];   // frequency bins
  int64_t n_frames = Zxx.shape().dims()[1]; // time frames

  // Infer nfft from frequency bins: onesided -> nfft = 2*(n_freq-1)
  if (nfft <= 0)
    nfft = 2 * (n_freq - 1);
  if (nperseg <= 0)
    nperseg = nfft;
  if (noverlap <= 0)
    noverlap = nperseg / 2;

  // Get analysis window
  Array win = get_window(window, nperseg, false);

  // Inverse FFT each frame
  int64_t hop = nperseg - noverlap;
  Place dev = Zxx.place();

  // Create full spectrum (mirror conjugate for onesided)
  // onesided: n_freq = nfft/2+1, need to mirror to get full nfft
  std::vector<Array> segments;
  segments.reserve(n_frames);

  for (int64_t i = 0; i < n_frames; ++i) {
    // Extract column i as a 1D array
    // Zxx is [n_freq x n_frames], column i = Zxx[0:n_freq, i]
    std::vector<std::complex<double>> col_data(n_freq);
    Array Zxx_cpu = Zxx.to(CPUPlace());
    const std::complex<double> *z_data = Zxx_cpu.data<std::complex<double>>();
    for (int64_t j = 0; j < n_freq; ++j) {
      col_data[j] = z_data[j * n_frames + i];
    }
    Array col = to_array(col_data, DType::C64, CPUPlace());

    // Reconstruct full spectrum from one-sided
    Array full_spec;
    if (n_freq == nfft / 2 + 1 && nfft > 2) {
      // Mirror: conj(X[nfft-k]) = conj(X[k]) for k=1..nfft/2-1
      std::vector<Array> parts;
      parts.push_back(col); // [0..nfft/2]

      // Mirror part: reverse and conjugate, skip DC and Nyquist
      Array mirrored = col[Slice(1, n_freq - 1)]; // skip DC and Nyquist
      mirrored = conj(mirrored);
      // Reverse
      mirrored = flip(mirrored, 0);
      parts.push_back(mirrored);

      full_spec = concat(parts, 0);
    } else {
      full_spec = col;
    }

    // IFFT
    Array segment = fft::ifft(full_spec, nfft);
    // Take real part
    segment = real(segment);
    // Truncate to nperseg
    if (segment.numel() > nperseg) {
      segment = segment[Slice(0, nperseg)];
    }

    // Apply window (synthesis window / analysis window squared)
    segment = segment * win;
    segments.push_back(segment);
  }

  // Overlap-add synthesis
  int64_t n_total = (n_frames - 1) * hop + nperseg;
  Array output = zeros({n_total}, DType::F64, dev);
  Array norm = zeros({n_total}, DType::F64, dev);

  for (int64_t i = 0; i < n_frames; ++i) {
    int64_t start = i * hop;
    // Ensure segment is F64
    if (segments[i].dtype() != DType::F64)
      segments[i] = segments[i].to(DType::F64);
    // Add segment to output
    Array out_slice = output[Slice(start, start + nperseg)];
    out_slice = out_slice + segments[i];
    output[Slice(start, start + nperseg)] = out_slice;

    // Add window squared to norm
    Array win_sq = win * win;
    if (win_sq.dtype() != DType::F64)
      win_sq = win_sq.to(DType::F64);
    Array norm_slice = norm[Slice(start, start + nperseg)];
    norm_slice = norm_slice + win_sq;
    norm[Slice(start, start + nperseg)] = norm_slice;
  }

  // Normalize: divide by sum of squared windows
  // Avoid division by zero
  Array norm_cpu = norm.to(CPUPlace());
  Array out_cpu = output.to(CPUPlace());
  double *norm_data = norm_cpu.data<double>();
  double *out_data = out_cpu.data<double>();
  for (int64_t i = 0; i < n_total; ++i) {
    if (std::abs(norm_data[i]) > 1e-10) {
      out_data[i] /= norm_data[i];
    }
  }

  // Copy back to original device
  if (dev.kind() == DeviceKind::GPU) {
    output = to_array(std::vector<double>(out_data, out_data + n_total),
                      DType::F64, dev);
  } else {
    output = out_cpu;
  }

  return output;
}

// ============================================================================
// vectorstrength
// ============================================================================

std::pair<double, double> vectorstrength(const Array &events, double period) {
  INS_CHECK(events.defined(), "vectorstrength: events is undefined");
  INS_CHECK(period > 0, "vectorstrength: period must be > 0");

  // phasors = exp(2j*pi*events/period)
  Array t = mul(events, full(events.shape(), 2.0 * M_PI / period,
                             events.dtype(), events.place()));
  Array cos_val = cos(t);
  Array sin_val = sin(t);

  double real_sum = sum(cos_val).item<double>();
  double imag_sum = sum(sin_val).item<double>();

  int64_t n = events.numel();
  real_sum /= n;
  imag_sum /= n;

  double strength = std::sqrt(real_sum * real_sum + imag_sum * imag_sum);
  double angle = std::atan2(imag_sum, real_sum);

  return {strength, angle};
}

// ============================================================================
// lombscargle
// ============================================================================

Array lombscargle(const Array &x, const Array &y, const Array &freqs) {
  INS_CHECK(x.defined(), "lombscargle: x is undefined");
  INS_CHECK(y.defined(), "lombscargle: y is undefined");
  INS_CHECK(freqs.defined(), "lombscargle: freqs is undefined");
  INS_CHECK(x.shape().ndim() == 1, "lombscargle: x must be 1D");
  INS_CHECK(y.shape().ndim() == 1, "lombscargle: y must be 1D");
  INS_CHECK(freqs.shape().ndim() == 1, "lombscargle: freqs must be 1D");
  INS_CHECK(x.numel() == y.numel(),
            "lombscargle: x and y must have the same length");

  Place dev = x.place();
  DType dtype = x.dtype();
  INS_CHECK(dtype == DType::F32 || dtype == DType::F64,
            "lombscargle: x must be F32 or F64");

  // Ensure all inputs are on the same device and dtype
  Array x_c = x;
  Array y_c = y;
  Array f_c = freqs;
  if (x_c.dtype() != dtype)
    x_c = x_c.to(dtype);
  if (y_c.dtype() != dtype)
    y_c = y_c.to(dtype);
  if (f_c.dtype() != dtype)
    f_c = f_c.to(dtype);

  if (x_c.place() != dev)
    x_c = x_c.to(dev);
  if (y_c.place() != dev)
    y_c = y_c.to(dev);
  if (f_c.place() != dev)
    f_c = f_c.to(dev);

  // Allocate output
  Array out(f_c.shape(), dtype, dev);

  // Dispatch to backend
  launch_signal_kernel("signal_lombscargle", dev, dtype, {x_c, y_c, f_c}, out);

  return out;
}

} // namespace signal
} // namespace ins
