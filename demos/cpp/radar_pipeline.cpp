// demos/cpp/feature_extraction.cpp
// 多域特征提取与定位 — 比赛任务2 (Insight7 C++)
// 纯 Insight7 API，与 Python 版算法完全对齐
// 编译: cmake .. -DINSIGHT_BUILD_DEMOS=ON && make -j24
// 运行: ./feature_extraction --device gpu --seed 42 --iterations 50

#include "insight/core/profiler.h"
#include "insight/insight.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/random.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

// ============================================================
// 物理参数
// ============================================================
static constexpr double FS = 5000.0;
static constexpr double DURATION = 0.5;
static constexpr int64_t N_SAMPLES = int64_t(FS * DURATION); // 2500
static constexpr double F0_CHIRP = 50.0;
static constexpr double F1_CHIRP = 500.0;
static constexpr double FC_GAUSS = 200.0;
static constexpr double BW_GAUSS = 0.5;
static constexpr double FREQ_SAW = 80.0;
static constexpr double NOISE_STD = 0.15;
static constexpr double AMPLITUDE = 0.8;

static constexpr double BP_LOW = 40.0;
static constexpr double BP_HIGH = 300.0;
static constexpr int64_t NUMTAPS = 127;

static constexpr int64_t NPERSEG = 256;
static constexpr int64_t NOVERLAP = 192;

static constexpr int CWT_WIDTHS_START = 1;
static constexpr int CWT_WIDTHS_END = 64;
static constexpr int CWT_WIDTHS_STEP = 4;
static constexpr double MORLET_W = 5.0;

static constexpr double KALMAN_Q = 1e-4;
static constexpr double KALMAN_R = 0.5;
static constexpr int PEAK_ORDER = 15;

// ============================================================
// 全局缓存
// ============================================================
static Array _T;
static Array _COMPOSITE_BASE;
static Array _TAPS;
static Array _GAUSS_KERNEL;
static std::vector<std::pair<double, Array>> _CWT_WAVELETS;
static Place _place;

// ============================================================
// CLI 参数
// ============================================================
struct Args {
  std::string device = "default";
  int seed = 42;
  int iterations = 0;
  std::string plot;
  std::string output = ".";
  bool timer = false;
  bool info = false;
  bool profiler = false;
};

static Args parse_args(int argc, char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      args.device = argv[++i];
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      args.seed = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      args.iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--plot") == 0 && i + 1 < argc) {
      args.plot = argv[++i];
    } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      args.output = argv[++i];
    } else if (std::strcmp(argv[i], "--timer") == 0) {
      args.timer = true;
    } else if (std::strcmp(argv[i], "--info") == 0) {
      args.info = true;
    } else if (std::strcmp(argv[i], "--profiler") == 0) {
      args.profiler = true;
    }
  }
  return args;
}

// ============================================================
// 卡尔曼滤波 (纯 C++ 循环)
// ============================================================
static std::vector<double> kalman_smooth(const std::vector<double> &z,
                                         double q = KALMAN_Q,
                                         double r = KALMAN_R) {
  int n = (int)z.size();
  std::vector<double> x_hat(n, 0.0), P(n, 0.0);
  if (n == 0)
    return x_hat;
  x_hat[0] = z[0];
  P[0] = 1.0;
  for (int k = 1; k < n; ++k) {
    double x_pred = x_hat[k - 1];
    double p_pred = P[k - 1] + q;
    double K = p_pred / (p_pred + r);
    x_hat[k] = x_pred + K * (z[k] - x_pred);
    P[k] = (1 - K) * (1 - K) * p_pred + K * K * r;
  }
  return x_hat;
}

// ============================================================
// 中心差分梯度
// ============================================================
static Array gradient(const Array &x, double dx) {
  int64_t n = x.numel();
  if (n < 2)
    return zeros_like(x);
  Array result = zeros({n}, DType::F64, _place);
  if (n > 2) {
    result[Slice(1, n - 1)] =
        (x[Slice(2, n)] - x[Slice(0, n - 2)]) / (2.0 * dx);
  }
  // 边界: 用 full 创建标量数组
  Array cpu_x = x.to(CPUPlace());
  double v0 = cpu_x.data<double>()[0];
  double v1 = cpu_x.data<double>()[1];
  double vn1 = cpu_x.data<double>()[n - 1];
  double vn2 = cpu_x.data<double>()[n - 2];
  result[int64_t(0)] = full({1}, (v1 - v0) / dx, DType::F64, _place);
  result[n - 1] = full({1}, (vn1 - vn2) / dx, DType::F64, _place);
  return result;
}

// ============================================================
// 峰值查找 (简单线性扫描)
// ============================================================
static void find_peaks_simple(const std::vector<double> &sig,
                              std::vector<int> &peaks_max,
                              std::vector<int> &peaks_min,
                              int order = PEAK_ORDER) {
  int n = (int)sig.size();
  peaks_max.clear();
  peaks_min.clear();
  for (int i = order; i < n - order; ++i) {
    bool is_max = true, is_min = true;
    double val = sig[i];
    for (int j = 1; j <= order; ++j) {
      if (sig[i - j] >= val || sig[i + j] >= val)
        is_max = false;
      if (sig[i - j] <= val || sig[i + j] <= val)
        is_min = false;
      if (!is_max && !is_min)
        break;
    }
    if (is_max)
      peaks_max.push_back(i);
    if (is_min)
      peaks_min.push_back(i);
  }
}

// ============================================================
// 快速 CWT (预计算 wavelet + FFT 卷积)
// ============================================================
static Array cwt_fast(const Array &signal) {
  int64_t n = signal.numel();
  Array sig_cpx = to_complex(signal);
  Array sig_fft = fft::fft(sig_cpx, n);

  std::vector<Array> rows;
  for (auto &wp : _CWT_WAVELETS) {
    double width = wp.first;
    (void)width;
    Array &w_arr = wp.second;
    int64_t N_w = w_arr.numel();
    Array w_conj = conj(w_arr);
    Array w_rev = flip(w_conj, 0);
    if (w_rev.dtype() != DType::C64)
      w_rev = to_complex(w_rev);
    if (w_rev.place() != _place)
      w_rev = w_rev.to(_place);

    int64_t pad_len = n - N_w;
    Array w_padded;
    if (pad_len > 0) {
      w_padded = concat({w_rev, zeros({pad_len}, DType::C64, _place)}, 0);
    } else {
      w_padded = w_rev[Slice(0, n)];
    }
    Array w_fft_arr = fft::fft(w_padded, n);
    Array conv_fft = sig_fft * w_fft_arr;
    Array conv_result = fft::ifft(conv_fft, n);

    int64_t start = N_w / 2;
    Array part1 = conv_result[Slice(start, n)];
    Array part2 = conv_result[Slice(0, start)];
    Array conv_shifted = concat({part1, part2}, 0);
    Array conv_same = conv_shifted[Slice(0, n)];
    rows.push_back(real(conv_same));
  }
  return stack(rows, 0);
}

// ============================================================
// 初始化缓存
// ============================================================
static void init_cache(Place place) {
  _place = place;

  _T = arange(N_SAMPLES, DType::F64, place) / FS;

  // Chirp
  Array chirp_sig = signal::chirp(_T, F0_CHIRP, DURATION, F1_CHIRP,
                                  signal::ChirpMethod::Linear) *
                    AMPLITUDE;

  // Gausspulse
  int64_t center = int64_t(0.3 * N_SAMPLES);
  int64_t gauss_len = N_SAMPLES / 2;
  Array t_gauss = _T[Slice(0, gauss_len)] - _T[gauss_len / 2];
  Array g = signal::gausspulse(t_gauss, FC_GAUSS, BW_GAUSS);
  Array gauss_sig = zeros({N_SAMPLES}, DType::F64, place);
  int64_t clip_len = std::min(gauss_len, N_SAMPLES - center);
  gauss_sig[Slice(center, center + clip_len)] = g[Slice(0, clip_len)];

  // Sawtooth
  Array saw_sig = signal::sawtooth(_T * (2.0 * M_PI * FREQ_SAW), 0.5) * 0.4;

  _COMPOSITE_BASE = chirp_sig + gauss_sig + saw_sig;

  // FIR bandpass
  double nyquist = FS / 2.0;
  _TAPS = signal::firwin(NUMTAPS, {BP_LOW / nyquist, BP_HIGH / nyquist},
                         "hamming", "bandpass");

  // Gaussian smoothing kernel
  int64_t kernel_size = std::max(int64_t(3), int64_t(0.1 * N_SAMPLES));
  if (kernel_size % 2 == 0)
    kernel_size++;
  Array x_kernel = linspace(-3.0, 3.0, kernel_size, DType::F64, place);
  Array kernel = exp(-0.5 * x_kernel * x_kernel);
  _GAUSS_KERNEL = kernel / sum(kernel);

  // Pre-compute CWT wavelets
  _CWT_WAVELETS.clear();
  for (int w = CWT_WIDTHS_START; w < CWT_WIDTHS_END; w += CWT_WIDTHS_STEP) {
    int64_t N_wavelet = std::min(int64_t(10 * w), N_SAMPLES);
    if (N_wavelet < 1)
      N_wavelet = 1;
    Array w_arr = signal::morlet2(N_wavelet, double(w), MORLET_W);
    _CWT_WAVELETS.emplace_back(double(w), w_arr);
  }
}

// ============================================================
// 帧结果
// ============================================================
struct FrameResult {
  double total_ms;
  double t_gen_ms, t_filt_ms, t_spline_ms, t_demod_ms;
  double t_stft_ms, t_cwt_ms, t_corr_ms, t_peak_ms, t_kalman_ms;
  int n_peaks_max, n_peaks_min;
  std::vector<std::pair<int, double>> params;
  Array smoothed;
  Array inst_freq;
  Array Zxx;
  Array cwt_matrix;
  Array autocorr;
  std::vector<double> smoothed_freq;
  std::vector<int> peaks_max, peaks_min;
};

static FrameResult run_frame(int seed_val, Place place,
                             Array *external_noise = nullptr) {
  FrameResult r;
  auto t_start = std::chrono::high_resolution_clock::now();

  // [1] 合成信号
  auto t0 = std::chrono::high_resolution_clock::now();
  Array noise;
  if (external_noise) {
    noise = *external_noise;
    if (noise.place() != place)
      noise = noise.to(place);
  } else {
    seed(seed_val);
    noise = randn({N_SAMPLES}, DType::F64, place) * NOISE_STD;
  }
  Array composite = _COMPOSITE_BASE + noise;
  auto t1 = std::chrono::high_resolution_clock::now();
  r.t_gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [2] 去趋势 + 带通滤波
  t0 = std::chrono::high_resolution_clock::now();
  Array detrended = signal::detrend(composite);
  Array filtered_full = signal::fftconvolve(detrended, _TAPS, "full");
  int64_t half = NUMTAPS / 2;
  Array filtered = filtered_full[Slice(half, half + N_SAMPLES)];
  t1 = std::chrono::high_resolution_clock::now();
  r.t_filt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [3] 高斯平滑
  t0 = std::chrono::high_resolution_clock::now();
  r.smoothed = signal::fftconvolve(filtered, _GAUSS_KERNEL, "same");
  t1 = std::chrono::high_resolution_clock::now();
  r.t_spline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [4] FM 解调
  t0 = std::chrono::high_resolution_clock::now();
  Array analytic = signal::hilbert(r.smoothed);
  Array inst_phase = unwrap(angle(analytic));
  r.inst_freq = gradient(inst_phase, 1.0 / FS) / (2.0 * M_PI);
  t1 = std::chrono::high_resolution_clock::now();
  r.t_demod_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [5] STFT
  t0 = std::chrono::high_resolution_clock::now();
  auto stft_result = signal::stft(r.smoothed, FS, "hann", NPERSEG, NOVERLAP);
  r.Zxx = stft_result.Sxx;
  t1 = std::chrono::high_resolution_clock::now();
  r.t_stft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [6] CWT
  t0 = std::chrono::high_resolution_clock::now();
  r.cwt_matrix = cwt_fast(r.smoothed);
  t1 = std::chrono::high_resolution_clock::now();
  r.t_cwt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [7] 自相关
  t0 = std::chrono::high_resolution_clock::now();
  int64_t seg_len = std::min(N_SAMPLES, int64_t(512));
  Array seg = r.smoothed[Slice(0, seg_len)];
  Array autocorr_full = signal::correlate(seg, seg, "full");
  int64_t mid = autocorr_full.numel() / 2;
  r.autocorr = autocorr_full[Slice(mid, autocorr_full.numel())];
  double norm_val = r.autocorr.to(CPUPlace()).data<double>()[0];
  if (std::isfinite(norm_val) && std::abs(norm_val) > 1e-12)
    r.autocorr = r.autocorr / norm_val;
  t1 = std::chrono::high_resolution_clock::now();
  r.t_corr_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [8] 峰值查找
  t0 = std::chrono::high_resolution_clock::now();
  Array smoothed_cpu = r.smoothed.to(CPUPlace());
  std::vector<double> smoothed_vec(smoothed_cpu.data<double>(),
                                   smoothed_cpu.data<double>() + N_SAMPLES);
  find_peaks_simple(smoothed_vec, r.peaks_max, r.peaks_min, PEAK_ORDER);
  r.n_peaks_max = (int)r.peaks_max.size();
  r.n_peaks_min = (int)r.peaks_min.size();
  t1 = std::chrono::high_resolution_clock::now();
  r.t_peak_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // [9] 卡尔曼滤波
  t0 = std::chrono::high_resolution_clock::now();
  Array inst_freq_cpu = r.inst_freq.to(CPUPlace());
  std::vector<double> inst_freq_vec(inst_freq_cpu.data<double>(),
                                    inst_freq_cpu.data<double>() + N_SAMPLES);
  r.smoothed_freq = kalman_smooth(inst_freq_vec);
  for (int p : r.peaks_max) {
    if (p >= 0 && p < (int)r.smoothed_freq.size())
      r.params.emplace_back(p, r.smoothed_freq[p]);
  }
  t1 = std::chrono::high_resolution_clock::now();
  r.t_kalman_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  auto t_end = std::chrono::high_resolution_clock::now();
  r.total_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return r;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char *argv[]) {
  init();
  Args args = parse_args(argc, argv);

  // Device resolution
  std::string eff_device = args.device;
  if (eff_device == "default") {
    eff_device = has_device(DeviceKind::GPU) ? "gpu" : "cpu";
  }
  bool use_gpu = (eff_device == "gpu") && has_device(DeviceKind::GPU);
  bool do_all = (eff_device == "all");
  Place place = CPUPlace();
  if (use_gpu) {
    init();
    place = GPUPlace(0);
  }
  if (!do_all)
    set_device(place);

  int n_frames = args.iterations > 0 ? args.iterations : 1;

  printf("============================================================\n");
  printf("  多域特征提取与定位 (Insight7 C++)\n");
  printf("============================================================\n");
  printf("\n[配置]  采样率: %.1f kHz  信号长度: %lld 点  时长: %.1fs  "
         "设备: %s  帧数: %d\n",
         FS / 1e3, (long long)N_SAMPLES, DURATION,
         do_all ? "all" : (use_gpu ? "GPU" : "CPU"), n_frames);

  if (do_all && !has_device(DeviceKind::GPU)) {
    printf("[警告] GPU 不可用，--device all 退化为仅 CPU\n");
    do_all = false;
    eff_device = "cpu";
    place = CPUPlace();
  }

  // 初始化缓存 (单设备模式)
  if (!do_all) {
    init_cache(place);

    printf("\n[波形]  chirp(%.0f-%.0fHz) + gausspulse(%.0fHz) "
           "+ sawtooth(%.0fHz) + noise\n",
           F0_CHIRP, F1_CHIRP, FC_GAUSS, FREQ_SAW);
    printf("[滤波]  FIR 带通 %.0f-%.0fHz, %lld 阶\n", BP_LOW, BP_HIGH,
           (long long)NUMTAPS);
    printf("[特征]  FM解调 + STFT + CWT(morlet2, %d scales) + 自相关\n",
           (int)_CWT_WAVELETS.size());
    printf("[检测]  峰值查找(order=%d) + 卡尔曼滤波\n", PEAK_ORDER);
  }

  // --info 模式
  if (args.info) {
    auto cpu_mem = device_memory_info(DeviceKind::CPU, 0);
    printf("[Memory] CPU: total=%zuMB used=%zuMB\n",
           cpu_mem.total / (1024 * 1024),
           (cpu_mem.total - cpu_mem.free) / (1024 * 1024));
    if (has_device(DeviceKind::GPU)) {
      auto gpu_mem = device_memory_info(DeviceKind::GPU, 0);
      printf("[Memory] GPU: total=%zuMB used=%zuMB\n",
             gpu_mem.total / (1024 * 1024),
             (gpu_mem.total - gpu_mem.free) / (1024 * 1024));
    }
  }

  // Profiler
  ins::Profiler prof(place);
  if (args.profiler)
    prof.start();

  // ========================================================
  // --device all: 双设备比较
  // ========================================================
  if (do_all) {
    printf("\n[双设备比较] CPU vs GPU\n\n");

    Place cpu_place = CPUPlace();
    Place gpu_place = GPUPlace(0);

    std::vector<double> cpu_times, gpu_times;

    for (int frame = 0; frame < n_frames; ++frame) {
      // Generate noise once on CPU
      init_cache(cpu_place);
      seed(args.seed + frame);
      Array cpu_noise = randn({N_SAMPLES}, DType::F64, cpu_place) * NOISE_STD;

      // CPU
      FrameResult cpu_result =
          run_frame(args.seed + frame, cpu_place, &cpu_noise);
      cpu_times.push_back(cpu_result.total_ms);

      // GPU with same noise
      Array gpu_noise = cpu_noise.to(gpu_place);
      init_cache(gpu_place);
      FrameResult gpu_result =
          run_frame(args.seed + frame, gpu_place, &gpu_noise);
      gpu_times.push_back(gpu_result.total_ms);

      // 比较 smoothed
      Array gpu_smoothed_cpu = gpu_result.smoothed;
      if (gpu_smoothed_cpu.place().kind() != DeviceKind::CPU)
        gpu_smoothed_cpu = gpu_result.smoothed.to(CPUPlace());
      Array diff = abs(cpu_result.smoothed - gpu_smoothed_cpu);
      double max_diff_val = max(diff).item<double>();

      bool print_frame = (n_frames == 1 || frame == 0 ||
                          (frame + 1) % 10 == 0 || frame == n_frames - 1);

      if (print_frame) {
        if (args.timer) {
          printf("  帧 %4d/%d | CPU: %8.2f ms (gen %5.1f filt %5.1f spl %5.1f "
                 "demod %5.1f stft %5.1f cwt %5.1f corr %5.1f peak %5.1f "
                 "kalman %5.1f) | GPU: %8.2f ms (gen %5.1f filt %5.1f "
                 "spl %5.1f demod %5.1f stft %5.1f cwt %5.1f corr %5.1f "
                 "peak %5.1f kalman %5.1f) | max_diff=%.2e\n",
                 frame, n_frames, cpu_result.total_ms, cpu_result.t_gen_ms,
                 cpu_result.t_filt_ms, cpu_result.t_spline_ms,
                 cpu_result.t_demod_ms, cpu_result.t_stft_ms,
                 cpu_result.t_cwt_ms, cpu_result.t_corr_ms,
                 cpu_result.t_peak_ms, cpu_result.t_kalman_ms,
                 gpu_result.total_ms, gpu_result.t_gen_ms, gpu_result.t_filt_ms,
                 gpu_result.t_spline_ms, gpu_result.t_demod_ms,
                 gpu_result.t_stft_ms, gpu_result.t_cwt_ms,
                 gpu_result.t_corr_ms, gpu_result.t_peak_ms,
                 gpu_result.t_kalman_ms, max_diff_val);
        } else {
          printf("  帧 %4d/%d | CPU: %8.2f ms | GPU: %8.2f ms | "
                 "max_diff=%.2e\n",
                 frame, n_frames, cpu_result.total_ms, gpu_result.total_ms,
                 max_diff_val);
        }
      }
    }

    double cpu_sum = 0, gpu_sum = 0;
    for (auto t : cpu_times)
      cpu_sum += t;
    for (auto t : gpu_times)
      gpu_sum += t;
    double cpu_avg = cpu_sum / cpu_times.size();
    double gpu_avg = gpu_sum / gpu_times.size();

    printf("\n  ==================================================\n");
    printf("  性能总结 (CPU vs GPU)\n");
    printf("  ==================================================\n");
    printf("  总帧数: %d\n", n_frames);
    printf("  CPU 平均每帧: %.2f ms  FPS: %.2f\n", cpu_avg, 1000.0 / cpu_avg);
    printf("  GPU 平均每帧: %.2f ms  FPS: %.2f\n", gpu_avg, 1000.0 / gpu_avg);
    printf("  加速比: %.2fx\n", cpu_avg / gpu_avg);
    if (args.profiler) {
      prof.stop();
      prof.report();
    }
    printf("\n完成！\n");
    return 0;
  }

  // ========================================================
  // 单设备模式 (原有逻辑)
  // ========================================================

  std::vector<double> times;
  for (int frame = 0; frame < n_frames; ++frame) {
    FrameResult r = run_frame(args.seed + frame, place);
    times.push_back(r.total_ms);

    bool print_frame = (n_frames == 1 || frame == 0 || (frame + 1) % 10 == 0 ||
                        frame == n_frames - 1);
    if (print_frame) {
      std::string params_str = "无";
      if (!r.params.empty()) {
        char buf[128];
        snprintf(buf, sizeof(buf), "峰值@%d f=%.1fHz", r.params[0].first,
                 r.params[0].second);
        params_str = buf;
      }
      if (args.timer) {
        printf("  帧 %4d/%d | %8.2f ms | "
               "gen %5.1f filt %5.1f spl %5.1f demod %5.1f "
               "stft %5.1f cwt %5.1f corr %5.1f peak %5.1f kalman %5.1f | "
               "极大值 %d 极小值 %d | %s\n",
               frame, n_frames, r.total_ms, r.t_gen_ms, r.t_filt_ms,
               r.t_spline_ms, r.t_demod_ms, r.t_stft_ms, r.t_cwt_ms,
               r.t_corr_ms, r.t_peak_ms, r.t_kalman_ms, r.n_peaks_max,
               r.n_peaks_min, params_str.c_str());
      } else {
        printf("  帧 %4d/%d | %8.2f ms | 极大值 %d 极小值 %d | %s\n", frame,
               n_frames, r.total_ms, r.n_peaks_max, r.n_peaks_min,
               params_str.c_str());
      }
    }
  }

  double sum = 0;
  for (auto t : times)
    sum += t;
  double avg_ms = sum / times.size();
  double fps = 1000.0 / avg_ms;

  printf("\n  ==================================================\n");
  printf("  性能总结 (%s)\n", use_gpu ? "GPU" : "CPU");
  printf("  ==================================================\n");
  printf("  总帧数: %d\n", n_frames);
  printf("  平均每帧: %.2f ms  FPS: %.2f\n", avg_ms, fps);

  if (args.profiler) {
    prof.stop();
    prof.report();
  }

  if (has_device(DeviceKind::GPU) && !use_gpu) {
    printf("\n[提示] GPU 可用, 使用 --device gpu 运行 GPU 版本");
  }
  printf("\n完成！\n");
  return 0;
}
