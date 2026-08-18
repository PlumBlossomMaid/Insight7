"""
demos/python/signal_pipeline.py
多域特征提取与信号分析流水线 (纯 Insight7)

波形生成 → FIR 滤波 → 高斯平滑 → 特征提取 (STFT/CWT/自相关)
         → FM 解调 → Kalman 滤波 → 峰值查找

纯 Insight7 实现，不导入 numpy / matplotlib。
绘图由 --plot 参数可选启用，动态导入 _plot_pipeline.py。

用法:
    python signal_pipeline.py --device cpu
    python signal_pipeline.py --device cpu --plot
    python signal_pipeline.py --device gpu --iterations 5
"""

import argparse
import math
import os
import time
import insight as ins

PI = math.pi


# ============================================================
# 1. 波形生成
# ============================================================
def generate_waveforms(fs, duration):
    n = int(fs * duration)
    t = ins.arange(n, dtype=ins.float64) / fs

    sig_chirp = ins.signal.chirp(t, 50.0, duration, 500.0, method="linear") * 0.8

    center = int(0.3 * n)
    seg_len = n // 4
    t_seg = t[0:seg_len] - t[seg_len // 2]
    g = ins.signal.gausspulse(t_seg, fc=200, bw=0.5)
    sig_gauss = ins.zeros([n], ins.float64)
    clip_len = min(seg_len, n - center)
    sig_gauss[center : center + clip_len] = g[0:clip_len]

    sig_saw = ins.signal.sawtooth(t * (2 * PI * 80), width=0.5) * 0.4
    noise = ins.randn([n], ins.float64) * 0.15
    composite = sig_chirp + sig_gauss + sig_saw + noise

    return {
        "t": t,
        "fs": fs,
        "n": n,
        "composite": composite,
        "chirp": sig_chirp,
        "gausspulse": sig_gauss,
        "sawtooth": sig_saw,
        "noise": noise,
    }


# ============================================================
# 2. 滤波器设计
# ============================================================
def design_filters(fs):
    numtaps = 127
    nyq = fs / 2.0
    taps = ins.signal.firwin(numtaps, [40.0 / nyq, 300.0 / nyq], pass_zero="bandpass")
    return taps, numtaps


# ============================================================
# 3. 滤波
# ============================================================
def apply_filter(sig, taps, numtaps):
    full = ins.signal.fftconvolve(sig, taps, "full")
    half = numtaps // 2
    n = int(sig.shape[0])
    return full[half : half + n]


# ============================================================
# 4. 高斯平滑
# ============================================================
def gaussian_smooth(sig, smooth_factor=0.1):
    n = int(sig.shape[0])
    kernel_size = max(3, int(smooth_factor * n))
    if kernel_size % 2 == 0:
        kernel_size += 1

    x_k = ins.linspace(-3.0, 3.0, kernel_size, dtype=ins.float64)
    kernel = ins.exp(-0.5 * x_k**2)
    kernel = kernel / ins.sum(kernel).numpy().item()

    pad = kernel_size // 2
    padded = ins.zeros([n + 2 * pad], ins.float64)
    padded[pad : pad + n] = sig
    for i in range(pad):
        padded[i] = sig.at([pad - i]).numpy().item()
        padded[n + pad + i] = sig.at([n - 2 - i]).numpy().item()

    result = ins.signal.fftconvolve(padded, kernel, "valid")
    return result[0:n]


# ============================================================
# 5. 特征提取
# ============================================================
def extract_features(sig, fs):
    features = {}

    analytic = ins.signal.hilbert(sig)
    features["fm_demod"] = ins.signal.fm_demod(analytic) * fs

    stft_r = ins.signal.stft(sig, fs=fs, nperseg=256, noverlap=192)
    features["stft_f"] = stft_r.f
    features["stft_t"] = stft_r.t
    features["stft_Sxx"] = stft_r.Sxx

    spec_r = ins.signal.spectrogram(sig, fs=fs, nperseg=256, noverlap=192)
    features["spec_f"] = spec_r.f
    features["spec_t"] = spec_r.t
    features["spec_Sxx"] = spec_r.Sxx

    widths_list = [float(w) for w in range(1, 32)]
    features["cwt_widths"] = widths_list
    features["cwt_matrix"] = ins.abs(ins.signal.cwt(sig, ins.signal.morlet2, widths_list))

    seg_len = min(int(sig.shape[0]), 2048)
    seg = sig[0:seg_len]
    autocorr_full = ins.signal.correlate(seg, seg, "full")
    half = int(autocorr_full.shape[0]) // 2
    autocorr = autocorr_full[half:]
    norm_val = autocorr.at([0]).numpy().item()
    if norm_val != 0:
        autocorr = autocorr / norm_val
    features["autocorr"] = autocorr

    return features


# ============================================================
# 6. 峰值查找
# ============================================================
def find_peaks(sig, order=15):
    peaks_max = ins.signal.argrelextrema(sig, comparator="greater", order=order)
    peaks_min = ins.signal.argrelextrema(sig, comparator="less", order=order)

    max_arr = []
    if peaks_max and int(peaks_max[0].shape[1]) > 0:
        m = peaks_max[0]
        max_arr = [int(m.at([0, i]).numpy().item()) for i in range(int(m.shape[1]))]

    min_arr = []
    if peaks_min and int(peaks_min[0].shape[1]) > 0:
        m = peaks_min[0]
        min_arr = [int(m.at([0, i]).numpy().item()) for i in range(int(m.shape[1]))]

    return max_arr, min_arr


# ============================================================
# 7. Kalman 滤波
# ============================================================
def estimate_parameters(inst_freq, fs, peaks):
    n_peaks = len(peaks)
    if n_peaks == 0:
        return [], []

    peak_vals = [float(inst_freq.at([int(p)]).numpy().item()) for p in peaks]
    n = n_peaks

    kf = ins.signal.KalmanFilter(dim_x=1, dim_z=1, points=n, dtype=ins.float64)

    F = ins.zeros([n, 1, 1], ins.float64)
    H = ins.zeros([n, 1, 1], ins.float64)
    Q = ins.zeros([n, 1, 1], ins.float64)
    R = ins.zeros([n, 1, 1], ins.float64)
    for i in range(n):
        F[i, 0, 0] = 1.0
        H[i, 0, 0] = 1.0
        Q[i, 0, 0] = 1e-4
        R[i, 0, 0] = 0.5
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R

    x0 = ins.zeros([n, 1, 1], ins.float64)
    P0 = ins.zeros([n, 1, 1], ins.float64)
    for i in range(n):
        x0[i, 0, 0] = peak_vals[i]
        P0[i, 0, 0] = 1.0
    kf.x = x0
    kf.P = P0

    kf.predict()
    z = ins.zeros([n, 1, 1], ins.float64)
    for i in range(n):
        z[i, 0, 0] = peak_vals[i]
    kf.update(z)

    smoothed = ins.flatten(kf.x)
    smoothed_list = [smoothed.at([i]).numpy().item() for i in range(int(smoothed.shape[0]))]

    params = []
    for i, p in enumerate(peaks):
        if 0 <= int(p) < int(inst_freq.shape[0]):
            params.append(
                {
                    "index": int(p),
                    "time": int(p) / fs,
                    "inst_freq": smoothed_list[i],
                }
            )

    return smoothed_list, params


# ============================================================
# 主流水线
# ============================================================
def run_pipeline(device="cpu"):
    if device == "gpu" and ins.has_device("gpu"):
        ins.init()
        ins.set_device(ins.GPUPlace(0))
    else:
        if device == "gpu":
            print("  [警告] GPU 不可用，回退到 CPU")
        ins.set_device(ins.CPUPlace())
        device = "cpu"

    fs = 5000.0
    duration = 0.5
    timings = {}

    print("[1/6] 波形生成...")
    t0 = time.time()
    wavs = generate_waveforms(fs, duration)
    timings["waveform"] = time.time() - t0
    print(f"  信号: {wavs['n']} 采样点, {duration}s, " f"耗时: {timings['waveform']:.4f}s")

    print("[2/6] 滤波器设计...")
    t0 = time.time()
    taps_arr, numtaps = design_filters(fs)
    timings["filter_design"] = time.time() - t0
    print(f"  带通: {numtaps} 阶, [40, 300] Hz")

    print("[3/6] 滤波处理...")
    t0 = time.time()
    detrended = ins.signal.detrend(wavs["composite"])
    signal_filtered = apply_filter(detrended, taps_arr, numtaps)
    timings["filtering"] = time.time() - t0

    print("[4/6] 高斯平滑...")
    t0 = time.time()
    signal_smooth = gaussian_smooth(signal_filtered)
    timings["smooth"] = time.time() - t0

    print("[5/6] 特征提取...")
    t0 = time.time()
    features = extract_features(signal_smooth, fs)
    timings["features"] = time.time() - t0

    def _shape_str(arr):
        s = arr.shape
        return f"[{s[0]}, {s[1]}]" if s.ndim() > 1 else f"[{s[0]}]"

    print(f"  FM demod: {int(features['fm_demod'].shape[0])} 点")
    print(f"  STFT: {_shape_str(features['stft_Sxx'])}")
    print(f"  CWT: {_shape_str(features['cwt_matrix'])}")
    print(f"  自相关: {int(features['autocorr'].shape[0])} 延迟")

    print("[6/6] 峰值查找 + Kalman...")
    t0 = time.time()
    peaks_max, peaks_min = find_peaks(signal_smooth, order=15)
    print(f"  极值: {len(peaks_max)} 极大, {len(peaks_min)} 极小")
    smoothed_freq, params = estimate_parameters(features["fm_demod"], fs, peaks_max)
    print(f"  Kalman: {len(params)} 个峰值")
    if params:
        for p in params[:5]:
            print(
                f"    索引 {p['index']:5d}  时间 {p['time']:.4f}s  " f"频率 {p['inst_freq']:.2f}Hz"
            )
        if len(params) > 5:
            print(f"    ... 还有 {len(params) - 5} 个")

    timings["peaks_kalman"] = time.time() - t0
    timings["total"] = sum(timings.values())

    return {
        "wavs": wavs,
        "signal_filtered": signal_filtered,
        "signal_smooth": signal_smooth,
        "features": features,
        "smoothed_freq": smoothed_freq,
        "peaks_max": peaks_max,
        "params": params,
        "timings": timings,
        "device": device,
        "fs": fs,
        "duration": duration,
    }


def print_timings(r):
    print("\n" + "=" * 60)
    print("  性能总结")
    print("=" * 60)
    for k, v in r["timings"].items():
        print(f"  {k:>20s}: {v * 1000:8.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="信号分析流水线 (Insight7)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--iterations", type=int, default=0, help="纯计算轮数, 报告 FPS (0=不运行)")
    parser.add_argument("--plot", action="store_true", help="启用 matplotlib 绘图 (仅 Python)")
    parser.add_argument("--output", type=str, default=".", help="图片输出目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ins.init()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  多域特征提取与信号分析 (Insight7)")
    print("=" * 60)

    result = run_pipeline(args.device)
    print_timings(result)

    if args.plot:
        from _plot_pipeline import save_plots

        save_plots(result, args.output)

    if args.iterations > 0:
        print(f"\n[基准测试] {args.iterations} 轮纯计算...")
        times = []
        for i in range(args.iterations):
            t0 = time.time()
            run_pipeline(args.device)
            times.append(time.time() - t0)
        avg_ms = sum(times) / len(times) * 1000
        fps = 1.0 / (sum(times) / len(times))
        print(f"  平均: {avg_ms:.2f} ms/轮  FPS: {fps:.2f}")

    if ins.has_device("gpu") and args.device == "cpu":
        print("\n[提示] GPU 可用, 使用 --device gpu 运行 GPU 版本")
    print("\n完成！")
