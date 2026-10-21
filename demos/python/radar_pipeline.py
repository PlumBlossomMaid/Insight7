"""
demos/python/feature_extraction.py
多域特征提取与定位 (纯 Insight7, 全 GPU 优化)

特征:
  - 设备选择 (--device cpu|gpu)
  - 随机种子 (--seed, 默认 42)
  - 多帧计算 + FPS (--iterations N)
  - 可选 Matplotlib 绘图 (--plot DIR)
  - 纯 Insight7 计算，无 numpy 依赖

管线:
  1. 波形生成 (chirp + gausspulse + sawtooth + noise)
  2. 滤波器设计 (firwin 带通)
  3. 去趋势 + 带通滤波
  4. B样条平滑 (高斯核卷积)
  5. 特征提取 (FM解调, STFT, 频谱图, CWT, 自相关)
  6. 峰值查找 + 卡尔曼滤波估计

优化:
  - 所有波形/滤波器/窗函数一次性缓存
  - CWT 预计算所有 wavelet 数组，消除 Python 回调
  - 峰值查找用简单线性扫描，避免 argrelmax 开销
  - STFT/spectrogram 合并计算
  - 自相关用 FFT 卷积

用法:
    python feature_extraction.py --device cpu
    python feature_extraction.py --device gpu --iterations 200 --plot output/
"""

import argparse
import math
import os
import time
import insight as ins

PI = math.pi

# ============================================================
# 物理参数
# ============================================================
FS = 5000.0
DURATION = 0.5
N_SAMPLES = int(FS * DURATION)  # 2500
F0_CHIRP = 50.0
F1_CHIRP = 500.0
FC_GAUSS = 200.0
BW_GAUSS = 0.5
FREQ_SAW = 80.0
NOISE_STD = 0.15
AMPLITUDE = 0.8

# 滤波器参数
BP_LOW = 40.0
BP_HIGH = 300.0
NUMTAPS = 127

# STFT 参数
NPERSEG = 256
NOVERLAP = 192

# CWT 参数 (减少尺度数以优化性能)
CWT_WIDTHS_START = 1
CWT_WIDTHS_END = 64
CWT_WIDTHS_STEP = 4  # 每 4 个尺度取一个，63→16
MORLET_W = 5.0

# 卡尔曼滤波参数
KALMAN_Q = 1e-4
KALMAN_R = 0.5

# 峰值查找参数
PEAK_ORDER = 15

# ============================================================
# 全局缓存
# ============================================================
_T = None
_COMPOSITE_BASE = None  # chirp + gausspulse + sawtooth (无噪声)
_TAPS = None
_GAUSS_KERNEL = None  # 高斯平滑核
_CWT_WAVELETS = None  # 预计算的 wavelet 数组列表
_PLACE = None


def _init_cache(device="cpu"):
    """一次性初始化: 生成波形、滤波器、窗函数、CWT wavelet。"""
    global _T, _COMPOSITE_BASE, _TAPS, _GAUSS_KERNEL, _CWT_WAVELETS, _PLACE

    if device == "gpu" and ins.has_device("gpu"):
        ins.init()
        _PLACE = ins.GPUPlace(0)
    else:
        _PLACE = ins.CPUPlace()
    ins.set_device(_PLACE)

    # 时间轴
    _T = ins.arange(N_SAMPLES, dtype=ins.float64, place=_PLACE) / FS

    # Chirp 信号
    chirp_sig = ins.signal.chirp(_T, F0_CHIRP, DURATION, F1_CHIRP, method="linear") * AMPLITUDE

    # Gausspulse 信号
    center = int(0.3 * N_SAMPLES)
    gauss_len = N_SAMPLES // 4
    t_gauss = _T[:gauss_len] - _T[gauss_len // 2]
    g = ins.signal.gausspulse(t_gauss, fc=FC_GAUSS, bw=BW_GAUSS)
    gauss_sig = ins.zeros([N_SAMPLES], dtype=ins.float64, place=_PLACE)
    clip_len = min(gauss_len, N_SAMPLES - center)
    gauss_sig[center : center + clip_len] = g[:clip_len]

    # 锯齿波信号
    saw_sig = ins.signal.sawtooth(2.0 * PI * FREQ_SAW * _T, width=0.5) * 0.4

    # 合成基信号 (无噪声)
    _COMPOSITE_BASE = chirp_sig + gauss_sig + saw_sig

    # FIR 带通滤波器
    nyquist = FS / 2.0
    _TAPS = ins.signal.firwin(NUMTAPS, [BP_LOW / nyquist, BP_HIGH / nyquist], pass_zero="bandpass")

    # 高斯平滑核
    kernel_size = max(3, int(0.1 * N_SAMPLES))
    if kernel_size % 2 == 0:
        kernel_size += 1
    x_kernel = ins.linspace(-3.0, 3.0, kernel_size, dtype=ins.float64, place=_PLACE)
    kernel = ins.exp(-0.5 * x_kernel**2)
    _GAUSS_KERNEL = kernel / ins.sum(kernel)

    # 预计算 CWT wavelet 数组 (消除 Python 回调开销)
    _CWT_WAVELETS = []
    for w in range(CWT_WIDTHS_START, CWT_WIDTHS_END, CWT_WIDTHS_STEP):
        N_wavelet = min(10 * w, N_SAMPLES)
        if N_wavelet < 1:
            N_wavelet = 1
        w_arr = ins.signal.morlet2(int(N_wavelet), s=float(w), w=MORLET_W)
        _CWT_WAVELETS.append((float(w), w_arr))


def _gradient(x, dx):
    """中心差分梯度。"""
    n = int(x.shape[0])
    if n < 2:
        return ins.zeros_like(x)
    result = ins.zeros([n], dtype=ins.float64, place=_PLACE)
    if n > 2:
        result["1:-1"] = (x[2:] - x[: n - 2]) / (2.0 * dx)
    result["0"] = (float(x[1]) - float(x[0])) / dx
    result[f"{n - 1}"] = (float(x[n - 1]) - float(x[n - 2])) / dx
    return result


def _kalman_smooth(z_list, q=KALMAN_Q, r=KALMAN_R):
    """1D 卡尔曼滤波 (纯 Python)。"""
    n = len(z_list)
    if n == 0:
        return z_list
    x_hat = [0.0] * n
    P = [0.0] * n
    x_hat[0] = z_list[0]
    P[0] = 1.0
    for k in range(1, n):
        x_pred = x_hat[k - 1]
        p_pred = P[k - 1] + q
        K = p_pred / (p_pred + r)
        x_hat[k] = x_pred + K * (z_list[k] - x_pred)
        P[k] = (1 - K) ** 2 * p_pred + K**2 * r
    return x_hat


def _find_peaks_simple(signal_list, order=PEAK_ORDER):
    """简单峰值查找: 线性扫描，O(n) 复杂度。"""
    n = len(signal_list)
    peaks_max = []
    peaks_min = []
    for i in range(order, n - order):
        is_max = True
        is_min = True
        val = signal_list[i]
        for j in range(1, order + 1):
            if signal_list[i - j] >= val or signal_list[i + j] >= val:
                is_max = False
            if signal_list[i - j] <= val or signal_list[i + j] <= val:
                is_min = False
            if not is_max and not is_min:
                break
        if is_max:
            peaks_max.append(i)
        if is_min:
            peaks_min.append(i)
    return peaks_max, peaks_min


def _cwt_fast(signal_arr):
    """快速 CWT: 用预计算的 wavelet 做 FFT 卷积。"""
    n = int(signal_arr.shape[0])

    # FFT of signal (once)
    sig_cpx = ins.to_complex(signal_arr)
    sig_fft = ins.fft(sig_cpx, n)

    result_rows = []
    for width, w_arr in _CWT_WAVELETS:
        N_w = int(w_arr.shape[0])
        # Conjugate and reverse
        w_conj = ins.conj(w_arr)
        w_rev = ins.flip(w_conj, 0)
        # Convert to complex
        if w_rev.dtype != ins.complex128:
            w_rev = ins.to_complex(w_rev)
        # Move to same device
        if str(w_rev.place) != str(_PLACE):
            w_rev = w_rev.to(_PLACE)
        # Zero-pad to n
        pad_len = n - N_w
        if pad_len > 0:
            w_padded = ins.concat([w_rev, ins.zeros([pad_len], dtype=ins.complex128, place=_PLACE)])
        else:
            w_padded = w_rev[:n]
        # FFT convolution
        w_fft = ins.fft(w_padded, n)
        conv_fft = sig_fft * w_fft
        conv_result = ins.ifft(conv_fft, n)
        # Extract "same" mode: circular shift
        start = N_w // 2
        part1 = conv_result[start:]
        part2 = conv_result[:start]
        conv_shifted = ins.concat([part1, part2])
        conv_same = conv_shifted[:n]
        result_rows.append(ins.real(conv_same))

    # Stack rows
    return ins.stack(result_rows, 0)


def _run_frame(rng_seed=42, external_noise=None, prof=None):
    """运行一帧特征提取管线。
    external_noise: 可选外部噪声数组，用于 --device all 模式共享随机噪声。
    """
    # [1] 合成信号
    t0 = time.time()
    ins.seed(rng_seed)
    if prof:
        prof.begin_event("randn")
    if external_noise is not None:
        noise = external_noise
    else:
        noise = ins.randn([N_SAMPLES], dtype=ins.float64, place=_PLACE) * NOISE_STD
    if prof:
        prof.end_event()
    composite = _COMPOSITE_BASE + noise
    t_gen = time.time() - t0

    # [2] 去趋势 + 带通滤波
    t0 = time.time()
    if prof:
        prof.begin_event("detrend")
    detrended = ins.signal.detrend(composite)
    if prof:
        prof.end_event()
    if prof:
        prof.begin_event("fir_filter")
    filtered_full = ins.signal.fftconvolve(detrended, _TAPS, mode="full")
    half = NUMTAPS // 2
    filtered = filtered_full[half : half + N_SAMPLES]
    if prof:
        prof.end_event()
    t_filt = time.time() - t0

    # [3] 高斯平滑
    t0 = time.time()
    if prof:
        prof.begin_event("gauss_smooth")
    smoothed = ins.signal.fftconvolve(filtered, _GAUSS_KERNEL, mode="same")
    if prof:
        prof.end_event()
    smoothed_list = smoothed.list()
    t_spline = time.time() - t0

    # [4] FM 解调
    t0 = time.time()
    if prof:
        prof.begin_event("hilbert")
    analytic = ins.signal.hilbert(smoothed)
    if prof:
        prof.end_event()
    inst_phase = ins.unwrap(ins.angle(analytic))
    inst_freq = _gradient(inst_phase, 1.0 / FS) / (2.0 * PI)
    inst_freq_list = inst_freq.list()
    t_demod = time.time() - t0

    # [5] STFT
    t0 = time.time()
    if prof:
        prof.begin_event("stft")
    stft_result = ins.signal.stft(smoothed, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    if prof:
        prof.end_event()
    f_stft = stft_result.f
    t_stft_arr = stft_result.t
    Zxx = stft_result.Sxx
    t_stft_time = time.time() - t0

    # [6] 频谱图 (复用 STFT 结果, 跳过)

    # [7] CWT (快速版本)
    t0 = time.time()
    if prof:
        prof.begin_event("cwt")
    cwt_matrix = _cwt_fast(smoothed)
    if prof:
        prof.end_event()
    t_cwt = time.time() - t0

    # [8] 自相关 (简化: 取前 512 点)
    t0 = time.time()
    if prof:
        prof.begin_event("correlate")
    seg_len = min(N_SAMPLES, 512)
    seg = smoothed[:seg_len]
    autocorr_full = ins.signal.correlate(seg, seg, mode="full")
    if prof:
        prof.end_event()
    mid = int(autocorr_full.shape[0]) // 2
    autocorr = autocorr_full[mid:]
    norm_val = float(autocorr[0])
    if math.isfinite(norm_val) and abs(norm_val) > 1e-12:
        autocorr = autocorr / norm_val
    autocorr_list = autocorr.list()
    t_corr = time.time() - t0

    # [9] 峰值查找 (简单线性扫描)
    t0 = time.time()
    if prof:
        prof.begin_event("peak_finding")
    peaks_max, peaks_min = _find_peaks_simple(smoothed_list, PEAK_ORDER)
    if prof:
        prof.end_event()
    n_max = len(peaks_max)
    n_min = len(peaks_min)
    t_peak = time.time() - t0

    # [10] 卡尔曼滤波估计
    t0 = time.time()
    if prof:
        prof.begin_event("kalman")
    smoothed_freq_list = _kalman_smooth(inst_freq_list)
    if prof:
        prof.end_event()
    params = []
    for p in peaks_max:
        if 0 <= p < len(smoothed_freq_list):
            params.append((p, p / FS, smoothed_freq_list[p]))
    t_kalman = time.time() - t0

    total_ms = (
        t_gen + t_filt + t_spline + t_demod + t_stft_time + t_cwt + t_corr + t_peak + t_kalman
    ) * 1000

    return {
        "composite": composite,
        "filtered": filtered,
        "smoothed": smoothed,
        "inst_freq": inst_freq,
        "smoothed_freq": smoothed_freq_list,
        "f_stft": f_stft,
        "t_stft": t_stft_arr,
        "Zxx": Zxx,
        "cwt_matrix": cwt_matrix,
        "autocorr": autocorr,
        "autocorr_list": autocorr_list,
        "peaks_max": peaks_max,
        "peaks_min": peaks_min,
        "params": params,
        "n_peaks_max": n_max,
        "n_peaks_min": n_min,
        "total_ms": total_ms,
        "t_gen_ms": t_gen * 1000,
        "t_filt_ms": t_filt * 1000,
        "t_spline_ms": t_spline * 1000,
        "t_demod_ms": t_demod * 1000,
        "t_stft_ms": t_stft_time * 1000,
        "t_cwt_ms": t_cwt * 1000,
        "t_corr_ms": t_corr * 1000,
        "t_peak_ms": t_peak * 1000,
        "t_kalman_ms": t_kalman * 1000,
    }


def parse_args():
    p = argparse.ArgumentParser(description="多域特征提取与定位 (Insight7)")
    p.add_argument(
        "--device",
        choices=["cpu", "gpu", "all"],
        default="cpu",
        help="设备: cpu/gpu/all (all=双后端对比)",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--iterations", type=int, default=0, help="帧数 (0=单帧)")
    p.add_argument("--plot", type=str, default=None, help="图片输出目录")
    p.add_argument("--output", type=str, default=".", help="其他输出目录")
    p.add_argument("--timer", action="store_true", help="显示详细时序")
    p.add_argument("--info", action="store_true", help="显示设备内存信息")
    p.add_argument("--profiler", action="store_true", help="显示算子级时序")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ins.init()
    os.makedirs(args.output, exist_ok=True)

    n_frames = args.iterations if args.iterations > 0 else 1

    print("=" * 60)
    print("  多域特征提取与定位 (Insight7)")
    print("=" * 60)
    print(
        f"\n[配置]  采样率: {FS/1e3:.1f} kHz  "
        f"信号长度: {N_SAMPLES} 点  时长: {DURATION}s  "
        f"设备: {args.device}  帧数: {n_frames}"
    )

    _init_cache(args.device)

    print(
        f"\n[波形]  chirp({F0_CHIRP}-{F1_CHIRP}Hz) + gausspulse({FC_GAUSS}Hz) "
        f"+ sawtooth({FREQ_SAW}Hz) + noise"
    )
    print(f"[滤波]  FIR 带通 {BP_LOW}-{BP_HIGH}Hz, {NUMTAPS} 阶")
    n_scales = len(_CWT_WAVELETS)
    print(f"[特征]  FM解调 + STFT + CWT(morlet2, {n_scales} scales) + 自相关")
    print(f"[检测]  峰值查找(order={PEAK_ORDER}) + 卡尔曼滤波")

    prof = None
    if args.profiler:
        prof = ins.Profiler("cpu", 0)  # CPU device
        prof.start()

    # --info: 打印设备内存信息
    has_gpu = ins.has_device("gpu")
    if args.info:
        cpu_total, cpu_free = ins.device_memory_info(0, 0)
        print(
            f"[Memory] CPU: total={cpu_total//1024//1024}MB used={(cpu_total-cpu_free)//1024//1024}MB"
        )
        if has_gpu:
            gpu_total, gpu_free = ins.device_memory_info(1, 0)
            print(
                f"[Memory] GPU: total={gpu_total//1024//1024}MB used={(gpu_total-gpu_free)//1024//1024}MB"
            )

    device_all = args.device == "all"
    times_cpu = []
    times_gpu = []

    for frame in range(n_frames):
        seed_val = args.seed + frame

        if device_all:
            # Generate noise on CPU once so both runs use identical noise
            _init_cache("cpu")
            cpu_noise = ins.randn([N_SAMPLES], ins.float64, place=ins.CPUPlace()) * NOISE_STD

            # CPU run
            cpu_result = _run_frame(rng_seed=seed_val, external_noise=cpu_noise, prof=prof)
            times_cpu.append(cpu_result["total_ms"])

            # GPU run with same noise
            if has_gpu:
                gpu_noise = cpu_noise.to(ins.GPUPlace(0))
                _init_cache("gpu")
                gpu_result = _run_frame(rng_seed=seed_val, external_noise=gpu_noise, prof=prof)
                times_gpu.append(gpu_result["total_ms"])
                cpu_sig = cpu_result["smoothed"]
                gpu_sig_cpu = gpu_result["smoothed"].to(ins.CPUPlace())
                diff = ins.max(ins.abs(cpu_sig - gpu_sig_cpu))
                max_diff = float(diff)
            else:
                print("  [警告] GPU 不可用，仅运行 CPU")
                gpu_result = cpu_result
                times_gpu.append(cpu_result["total_ms"])
                max_diff = 0.0

            if n_frames == 1 or frame == 0 or (frame + 1) % 10 == 0 or frame == n_frames - 1:
                if args.timer:
                    print(
                        f"  帧 {frame:4d}/{n_frames} | "
                        f"CPU: {cpu_result['total_ms']:8.2f} ms "
                        f"(gen {cpu_result['t_gen_ms']:5.1f} filt {cpu_result['t_filt_ms']:5.1f} "
                        f"spl {cpu_result['t_spline_ms']:5.1f} demod {cpu_result['t_demod_ms']:5.1f} "
                        f"stft {cpu_result['t_stft_ms']:5.1f} cwt {cpu_result['t_cwt_ms']:5.1f} "
                        f"corr {cpu_result['t_corr_ms']:5.1f}) | "
                        f"GPU: {gpu_result['total_ms']:8.2f} ms "
                        f"(gen {gpu_result['t_gen_ms']:5.1f} filt {gpu_result['t_filt_ms']:5.1f} "
                        f"spl {gpu_result['t_spline_ms']:5.1f} demod {gpu_result['t_demod_ms']:5.1f} "
                        f"stft {gpu_result['t_stft_ms']:5.1f} cwt {gpu_result['t_cwt_ms']:5.1f} "
                        f"corr {gpu_result['t_corr_ms']:5.1f}) | "
                        f"max_diff={max_diff:.2e}"
                    )
                else:
                    print(
                        f"  帧 {frame:4d}/{n_frames} | "
                        f"CPU: {cpu_result['total_ms']:8.2f} ms | "
                        f"GPU: {gpu_result['total_ms']:8.2f} ms | "
                        f"max_diff={max_diff:.2e}"
                    )
        else:
            _init_cache(args.device)
            result = _run_frame(rng_seed=seed_val, prof=prof)
            t = result["total_ms"]
            if args.device == "gpu":
                times_gpu.append(t)
            else:
                times_cpu.append(t)

            if n_frames == 1 or frame == 0 or (frame + 1) % 10 == 0 or frame == n_frames - 1:
                params_str = (
                    f"峰值@{result['params'][0][0]} f={result['params'][0][2]:.1f}Hz"
                    if result["params"]
                    else "无"
                )
                if args.timer:
                    print(
                        f"  帧 {frame:4d}/{n_frames} | {t:8.2f} ms | "
                        f"gen {result['t_gen_ms']:5.1f} "
                        f"filt {result['t_filt_ms']:5.1f} "
                        f"spl {result['t_spline_ms']:5.1f} "
                        f"demod {result['t_demod_ms']:5.1f} "
                        f"stft {result['t_stft_ms']:5.1f} "
                        f"cwt {result['t_cwt_ms']:5.1f} "
                        f"corr {result['t_corr_ms']:5.1f} "
                        f"peak {result['t_peak_ms']:5.1f} "
                        f"kalman {result['t_kalman_ms']:5.1f} | "
                        f"极大值 {result['n_peaks_max']} "
                        f"极小值 {result['n_peaks_min']} | {params_str}"
                    )
                else:
                    print(
                        f"  帧 {frame:4d}/{n_frames} | {t:8.2f} ms | "
                        f"极大值 {result['n_peaks_max']} "
                        f"极小值 {result['n_peaks_min']} | {params_str}"
                    )

        if args.plot and (not device_all) and (frame == 0 or frame == n_frames - 1):
            os.makedirs(args.plot, exist_ok=True)
            from _plot_radar_pipeline import save_feature_plot

            save_feature_plot(result, args.plot, frame_idx=frame if n_frames > 1 else None)

    # 性能总结
    print(f"\n  {'=' * 50}")
    if device_all:
        cpu_avg = sum(times_cpu) / max(len(times_cpu), 1) if times_cpu else 0
        gpu_avg = sum(times_gpu) / max(len(times_gpu), 1) if times_gpu else 0
        print("  性能总结 (CPU vs GPU)")
        print(f"  {'=' * 50}")
        print(f"  总帧数: {n_frames}")
        print(f"  CPU 平均每帧: {cpu_avg:.2f} ms  FPS: {1000/max(cpu_avg,0.001):.2f}")
        if has_gpu:
            print(f"  GPU 平均每帧: {gpu_avg:.2f} ms  FPS: {1000/max(gpu_avg,0.001):.2f}")
            print(f"  加速比: {cpu_avg/max(gpu_avg,0.001):.2f}x")
    else:
        avg_ms = sum(times_cpu + times_gpu) / max(len(times_cpu + times_gpu), 1)
        fps = 1000.0 / max(avg_ms, 0.001)
        print(f"  性能总结 ({args.device.upper()})")
        print(f"  {'=' * 50}")
        print(f"  总帧数: {n_frames}")
        print(f"  平均每帧: {avg_ms:.2f} ms  FPS: {fps:.2f}")

    if has_gpu and args.device == "cpu":
        print("\n[提示] GPU 可用, 使用 --device gpu 运行 GPU 版本")

    if prof:
        prof.stop()
        prof.report()

    print("\n完成！")
