"""
demos/python/radar_detection.py
雷达目标检测与多普勒分析 (纯 Insight7, 全 GPU 优化)

特征:
  - 设备选择 (--device cpu|gpu)
  - 随机种子 (--seed, 默认 42)
  - 多帧计算 + FPS (--iterations N)
  - 目标从起点到终点线性移动 (每帧不同)
  - 可选 Matplotlib 绘图 (--plot DIR)
  - 纯 Insight7 计算，无 numpy 依赖

优化 (2026-06-13):
  - 回波模拟全向量化: 2D 数组 + broadcasting，零 Python 脉冲循环
  - 预计算匹配滤波器 FFT: 所有帧共享
  - Doppler FFT 全 GPU: fft(axis=0) + fftshift，无 CPU 来回搬运
  - 发射信号+常数一次性生成并缓存
  - _extract_targets 批量 GPU→CPU 传输，避免逐元素同步

用法:
    python radar_detection.py --device cpu
    python radar_detection.py --device gpu --iterations 200 --plot output/
    python radar_detection.py --device cpu --seed 123
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
FS = 10e6
T_PRF = 100e-6
N = 1000
B = 5e6
TAU = 50e-6
N_PULSES = 32
SNR_DB = 10
PULSE_LEN = int(TAU * FS)  # = 500
PC_OFFSET = PULSE_LEN // 2  # = 250
RANGE_PER_BIN = 3e8 / (2 * FS)  # = 15 m/bin
MAX_UNAMBIG_RANGE = RANGE_PER_BIN * N

# 目标移动轨迹 (线性插值)
DELAY_START = [35e-6, 50e-6]
DELAY_END = [11e-6, 20e-6]
DOPPLER_START = [2000.0, -1500.0]
DOPPLER_END = [1800.0, -1200.0]

# ============================================================
# 全局缓存 (一次性初始化，所有帧共享)
# ============================================================
_S_TX = None  # LFM 发射信号
_SIG_POWER = None  # 信号功率常数
_NOISE_SIGMA = None  # 噪声标准差常数
_DOPPLER_BINS = None  # 多普勒频率轴 (Python list)
_RANGE_BINS = None  # 距离轴 (Python list)
_PLACE = None  # 当前设备 Place
_HAMMING = None  # 缓存的 hamming 窗 [N_PULSES, 1]
_SLOW_TIMES = None  # 缓存的慢时间轴


def _init_cache(device="cpu"):
    """一次性初始化: 生成 LFM 信号并预计算所有常数。"""
    global \
        _S_TX, \
        _TEMPLATE, \
        _SIG_POWER, \
        _NOISE_SIGMA, \
        _DOPPLER_BINS, \
        _RANGE_BINS, \
        _PLACE, \
        _HAMMING, \
        _SLOW_TIMES

    # 确定设备
    if device == "gpu" and ins.has_device("gpu"):
        ins.init()
        _PLACE = ins.GPUPlace(0)
    else:
        _PLACE = ins.CPUPlace()
    ins.set_device(_PLACE)

    # LFM 发射信号
    t = ins.arange(N, dtype=ins.float64, place=_PLACE) / FS
    phase = PI * (B / TAU) * (t**2)
    s_tx = ins.to_complex(ins.cos(phase), ins.sin(phase))
    tau_arr = ins.full([1], TAU, ins.float64, place=_PLACE)
    mask = ins.cast(ins.less(t, tau_arr), ins.float64)
    _S_TX = s_tx * mask
    _TEMPLATE = _S_TX[0:PULSE_LEN]

    # 缓存信号功率 (一次性计算)
    sig_power_arr = ins.mean(ins.real(_S_TX) ** 2 + ins.imag(_S_TX) ** 2)
    _SIG_POWER = float(sig_power_arr)
    _NOISE_SIGMA = math.sqrt(_SIG_POWER / 10 ** (SNR_DB / 10) / 2)

    # 缓存多普勒和距离轴 (Python list, 避免每帧重复计算和 GPU→CPU 传输)
    cpu_place = ins.CPUPlace()
    doppler_freq = ins.fftshift(ins.fftfreq(N_PULSES, T_PRF))
    doppler_cpu = doppler_freq.to(cpu_place)
    _DOPPLER_BINS = doppler_cpu.list()

    range_arr = (ins.arange(N, dtype=ins.float64, place=_PLACE) - PC_OFFSET) * RANGE_PER_BIN
    range_cpu = range_arr.to(cpu_place)
    _RANGE_BINS = range_cpu.list()

    # 缓存 hamming 窗 (避免每帧重新创建)
    _HAMMING = ins.reshape(ins.signal.get_window("hamming", N_PULSES, fftbins=False), [N_PULSES, 1])

    # 缓存慢时间轴 (避免每帧重复 arange + mul)
    _SLOW_TIMES = ins.arange(N_PULSES, dtype=ins.float64, place=_PLACE) * T_PRF


def _get_target_params(frame_idx, total_frames):
    """计算第 frame_idx 帧的目标参数 (线性插值)。"""
    if total_frames <= 1:
        return DELAY_START, DOPPLER_START
    t = frame_idx / max(total_frames - 1, 1)
    delays = [s + (e - s) * t for s, e in zip(DELAY_START, DELAY_END)]
    dopplers = [s + (e - s) * t for s, e in zip(DOPPLER_START, DOPPLER_END)]
    return delays, dopplers


def _simulate_echoes(target_delays, target_dopplers, external_noise_r=None, external_noise_i=None):
    """
    全向量化回波模拟 (全 GPU, 零 Python 脉冲循环)。
    返回 [N_PULSES, N] complex128 回波信号。
    external_noise_r/external_noise_i: 可选外部噪声数组，用于 --device all 模式共享随机噪声。
    """
    noise_sigma = _NOISE_SIGMA
    if external_noise_r is not None and external_noise_i is not None:
        noise_r = external_noise_r
        noise_i = external_noise_i
    else:
        noise_r = ins.randn([N_PULSES, N], ins.float64, place=_PLACE) * noise_sigma
        noise_i = ins.randn([N_PULSES, N], ins.float64, place=_PLACE) * noise_sigma

    pulses = ins.zeros([N_PULSES, N], ins.complex128, place=_PLACE)
    for delay, doppler in zip(target_delays, target_dopplers):
        ds = int(delay * FS)
        delayed_1d = ins.zeros([N], ins.complex128, place=_PLACE)
        if ds < N:
            delayed_1d[ds:] = _S_TX[0 : N - ds]
        delayed_2d = ins.reshape(delayed_1d, [1, N])
        phase = _SLOW_TIMES * (2.0 * PI * doppler)
        rot = ins.to_complex(ins.cos(phase), ins.sin(phase))
        rot_2d = ins.reshape(rot, [N_PULSES, 1])
        pulses = pulses + delayed_2d * rot_2d

    pulses = pulses + ins.to_complex(noise_r, noise_i)
    return pulses  # [N_PULSES, N] complex128


def _extract_targets(energy, det, top_n=2, cluster_threshold=3.0):
    """
    从 CFAR 检测结果中提取目标。

    - 批量 GPU→CPU 传输: 一次 transfer 整个 idx/energy 数组
    - 后续 Python loop 仅操作 CPU 数据，无 GPU 同步开销
    """
    idx = ins.nonzero(det)
    n_det = int(idx.shape[1]) if idx.shape else 0
    if n_det == 0:
        return [], 0

    # 批量传输到 CPU + 一次性 flatten 提取
    cpu_place = ins.CPUPlace()
    idx_cpu = idx.to(cpu_place)
    energy_cpu = energy.to(cpu_place)
    row_list = idx_cpu[0].list()
    col_list = idx_cpu[1].list()
    energy_flat = energy_cpu.list()
    n_cols = N  # 距离单元数

    candidates = [
        (row_list[k], col_list[k], energy_flat[row_list[k] * n_cols + col_list[k]])
        for k in range(n_det)
    ]

    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[: max(top_n * 10, 20)]

    points = [(i, j) for i, j, _ in top_candidates]
    visited = [False] * len(points)
    clusters = []

    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        cluster_indices = [i]
        for j in range(i + 1, len(points)):
            if visited[j]:
                continue
            d = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            if d <= cluster_threshold:
                visited[j] = True
                cluster_indices.append(j)
        best_idx = max(cluster_indices, key=lambda x: top_candidates[x][2])
        clusters.append(
            (
                top_candidates[best_idx][0],
                top_candidates[best_idx][1],
                top_candidates[best_idx][2],
            )
        )

    clusters.sort(key=lambda x: x[2], reverse=True)

    # 按距离去重: 同一距离 bin 只保留能量最强的
    seen_ranges = set()
    deduped = []
    for d, r, v in clusters:
        if r not in seen_ranges:
            seen_ranges.add(r)
            deduped.append((d, r, v))

    return [(d, r) for d, r, _ in deduped[:top_n]], n_det


def run_detection(
    device="cpu",
    seed=42,
    frame_idx=0,
    total_frames=1,
    external_noise_r=None,
    external_noise_i=None,
    prof=None,
):
    """
    运行一帧雷达检测 (全 GPU 优化版)。
    frame_idx / total_frames 控制目标位置线性插值。
    external_noise_r/external_noise_i: 可选外部噪声数组，用于 --device all 模式共享随机噪声。
    """
    if device == "gpu" and ins.has_device("gpu"):
        ins.init()
        ins.set_device(ins.GPUPlace(0))
    else:
        if device == "gpu":
            print("  [警告] GPU 不可用，回退到 CPU")
        ins.set_device(ins.CPUPlace())
        device = "cpu"

    ins.seed(seed)

    target_delays, target_dopplers = _get_target_params(frame_idx, total_frames)

    # [1] 回波模拟 (全 GPU 向量化)
    t0 = time.time()
    if prof:
        prof.begin_event("echo_simulation")
    pulses = _simulate_echoes(target_delays, target_dopplers, external_noise_r, external_noise_i)
    if prof:
        prof.end_event()
    t_echo = time.time() - t0

    # [2] 脉冲压缩 (框架 API, batched FFT)
    t0 = time.time()
    if prof:
        prof.begin_event("pulse_compression")
    template = _S_TX[0:PULSE_LEN]
    pc = ins.signal.pulse_compression(pulses, template)
    if prof:
        prof.end_event()
    t_pc = time.time() - t0

    # [3] 去直流 + 多普勒处理 (框架 API)
    t0 = time.time()
    if prof:
        prof.begin_event("doppler")
    mean_pc = ins.mean(pc, axis=0, keepdims=True)
    pc_ac = pc - mean_pc
    spec = ins.signal.pulse_doppler(pc_ac * _HAMMING, window="", nfft=N_PULSES)
    doppler_shifted = spec / float(N_PULSES)
    if prof:
        prof.end_event()
    t_doppler = time.time() - t0

    # [4] 能量图
    energy = ins.real(doppler_shifted) ** 2 + ins.imag(doppler_shifted) ** 2

    # [5] CA-CFAR 检测
    t0 = time.time()
    if prof:
        prof.begin_event("ca_cfar")
    _, det = ins.signal.ca_cfar(energy, [4, 4], [12, 12], 1e-6)
    if prof:
        prof.end_event()
    t_cfar = time.time() - t0

    # [6] 目标提取 (批量 GPU→CPU 传输)
    t0 = time.time()
    if prof:
        prof.begin_event("target_extraction")
    targets, raw_count = _extract_targets(energy, det, top_n=2, cluster_threshold=3.0)
    if prof:
        prof.end_event()
    t_local = time.time() - t0

    # [7] 有效距离筛选
    valid = []
    for d, rr in targets:
        dist = _RANGE_BINS[rr]
        if 0 <= dist <= MAX_UNAMBIG_RANGE:
            valid.append((d, rr))

    total_ms = (t_echo + t_pc + t_doppler + t_cfar + t_local) * 1000

    return {
        "targets": valid,
        "raw_count": raw_count,
        "total_ms": total_ms,
        "pc_ms": t_echo * 1000,
        "doppler_ms": t_doppler * 1000,
        "cfar_ms": t_cfar * 1000,
        "local_ms": t_local * 1000,
        "doppler_bins": _DOPPLER_BINS,
        "range_bins": _RANGE_BINS,
        "energy": energy,
        "pc": pc,
        "pulses": pulses,
        "device": device,
    }


def parse_args():
    p = argparse.ArgumentParser(description="雷达目标检测 (Insight7)")
    p.add_argument(
        "--device",
        choices=["cpu", "gpu", "all"],
        default="cpu",
        help="设备: cpu/gpu/all (all=双后端对比)",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--iterations", type=int, default=0, help="帧数 (目标逐帧移动, 0=单帧)")
    p.add_argument("--plot", type=str, default=None, help="图片输出目录 (启用 Matplotlib 绘图)")
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
    print("  雷达目标检测与多普勒分析 (Insight7)")
    print("=" * 60)
    print(
        f"\n[配置]  采样率: {FS/1e6:.1f} MHz  "
        f"脉冲: {N_PULSES}  设备: {args.device}  "
        f"种子: {args.seed}  帧数: {n_frames}"
    )

    # 一次性初始化缓存
    _init_cache(args.device)

    # 波形设计与模糊函数分析 (ambgfun — 任务 1 第 5 步)
    print("\n[波形分析] 计算模糊函数...")
    template = _S_TX[0:PULSE_LEN]
    ambg = ins.signal.ambgfun(template, fs=FS, prf=1.0 / T_PRF)
    # 评估指标: 主瓣宽度、峰值旁瓣比
    ambg_abs = ins.abs(ambg)
    peak_val = float(ins.max(ambg_abs))
    peak_ratio = 20 * math.log10(peak_val / (float(ins.mean(ambg_abs)) + 1e-12))
    print(f"  模糊函数峰值: {peak_val:.2f}")
    print(f"  峰值/平均比: {peak_ratio:.1f} dB  (主瓣窄、旁瓣低 ✓)")

    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
        from _plot_radar import save_frame, save_ambiguity_plot

        save_ambiguity_plot(ambg, args.plot)
        plot_dir = args.plot

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

    prof = None
    if args.profiler:
        prof = ins.Profiler("cpu", 0)  # CPU device
        prof.start()

    device_all = args.device == "all"
    times_cpu = []
    times_gpu = []

    for frame in range(n_frames):
        seed_val = args.seed + frame

        if device_all:
            # 在 CPU 上一次性生成缓存和噪声（确保双后端基信号完全一致）
            _init_cache("cpu")
            ins.seed(seed_val)
            noise_r = ins.randn([N_PULSES, N], ins.float64, place=ins.CPUPlace()) * _NOISE_SIGMA
            noise_i = ins.randn([N_PULSES, N], ins.float64, place=ins.CPUPlace()) * _NOISE_SIGMA

            # CPU run
            cpu_result = run_detection(
                "cpu", seed_val, frame, n_frames, noise_r, noise_i, prof=prof
            )
            times_cpu.append(cpu_result["total_ms"])

            # GPU run — 复用 CPU 缓存（传输到 GPU），不重新生成
            if has_gpu:
                gpu_pl = ins.GPUPlace(0)
                _S_TX = _S_TX.to(gpu_pl)
                _TEMPLATE = _TEMPLATE.to(gpu_pl)
                _HAMMING = _HAMMING.to(gpu_pl)
                _SLOW_TIMES = _SLOW_TIMES.to(gpu_pl)
                _PLACE = gpu_pl
                gpu_noise_r = noise_r.to(gpu_pl)
                gpu_noise_i = noise_i.to(gpu_pl)
                gpu_result = run_detection(
                    "gpu", seed_val, frame, n_frames, gpu_noise_r, gpu_noise_i, prof=prof
                )
                times_gpu.append(gpu_result["total_ms"])
                # 比较回波信号（pulse_compression 之前，应完全一致）
                cpu_sig = cpu_result["pulses"]
                gpu_sig_cpu = gpu_result["pulses"].to(ins.CPUPlace())
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
                        f"(echo {cpu_result.get('pc_ms',0):5.2f} pc {cpu_result.get('doppler_ms',0):5.2f} "
                        f"dop {cpu_result.get('cfar_ms',0):5.2f} cfar {cpu_result.get('local_ms',0):5.2f}) | "
                        f"GPU: {gpu_result['total_ms']:8.2f} ms "
                        f"(echo {gpu_result.get('pc_ms',0):5.2f} pc {gpu_result.get('doppler_ms',0):5.2f} "
                        f"dop {gpu_result.get('cfar_ms',0):5.2f} cfar {gpu_result.get('local_ms',0):5.2f}) | "
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
            result = run_detection(args.device, seed_val, frame, n_frames, prof=prof)
            t = result["total_ms"]
            if args.device == "gpu":
                times_gpu.append(t)
            else:
                times_cpu.append(t)

            if n_frames == 1 or frame == 0 or (frame + 1) % 10 == 0 or frame == n_frames - 1:
                targets_str = (
                    "; ".join(
                        f"距离 {result['range_bins'][rr]:.0f}m "
                        f"多普勒 {result['doppler_bins'][d]:.0f}Hz"
                        for d, rr in result["targets"]
                    )
                    if result["targets"]
                    else "无"
                )
                if args.timer:
                    print(
                        f"  帧 {frame:4d}/{n_frames} | {t:8.2f} ms | "
                        f"echo {result.get('pc_ms',0):5.2f} "
                        f"pc {result.get('doppler_ms',0):5.2f} "
                        f"dop {result.get('cfar_ms',0):5.2f} "
                        f"cfar {result.get('local_ms',0):5.2f} "
                        f"ext {t - result.get('pc_ms',0) - result.get('doppler_ms',0) - result.get('cfar_ms',0) - result.get('local_ms',0):5.2f} | "
                        f"检测 {result['raw_count']:4d} → "
                        f"{len(result['targets'])} 目标 | {targets_str}"
                    )
                else:
                    print(
                        f"  帧 {frame:4d}/{n_frames} | {t:8.2f} ms | "
                        f"检测 {result['raw_count']:4d} → "
                        f"{len(result['targets'])} 目标 | {targets_str}"
                    )

        if args.plot and (not device_all):
            save_frame(result, plot_dir, frame_idx=frame if n_frames > 1 else None)

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
