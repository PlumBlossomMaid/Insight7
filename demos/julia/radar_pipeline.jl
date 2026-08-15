# demos/julia/feature_extraction.jl
# 多域特征提取与定位 — 比赛任务2 (Insight7 Julia)
# 纯 Insight7 API，与 Python/C++/Lua 版算法完全对齐
# 运行: LD_LIBRARY_PATH=build/backends/cpu julia demos/julia/feature_extraction.jl --device cpu

include("../../bindings/julia/Insight.jl")
using .Insight

# ============================================================
# 物理参数
# ============================================================
const FS = 5000.0
const DURATION = 0.5
const N_SAMPLES = Int64(FS * DURATION)  # 2500
const F0_CHIRP = 50.0
const F1_CHIRP = 500.0
const FC_GAUSS = 200.0
const BW_GAUSS = 0.5
const FREQ_SAW = 80.0
const NOISE_STD = 0.15
const AMPLITUDE = 0.8

const BP_LOW = 40.0
const BP_HIGH = 300.0
const NUMTAPS = Int64(127)

const NPERSEG = Int64(256)
const NOVERLAP = Int64(192)

const CWT_WIDTHS_START = 1
const CWT_WIDTHS_END = 64
const CWT_WIDTHS_STEP = 4
const MORLET_W = 5.0

const KALMAN_Q = 1e-4
const KALMAN_R = 0.5
const PEAK_ORDER = 15

# ============================================================
# 全局缓存
# ============================================================
_T = nothing
_COMPOSITE_BASE = nothing
_TAPS = nothing
_GAUSS_KERNEL = nothing
_CWT_WAVELETS = []

# ============================================================
# 卡尔曼滤波
# ============================================================
function kalman_smooth(z::Vector{Float64}; q=KALMAN_Q, r=KALMAN_R)
    n = length(z)
    if n == 0 return z end
    x_hat = Base.zeros(n)
    P = Base.zeros(n)
    x_hat[1] = z[1]
    P[1] = 1.0
    for k in 2:n
        x_pred = x_hat[k-1]
        p_pred = P[k-1] + q
        K = p_pred / (p_pred + r)
        x_hat[k] = x_pred + K * (z[k] - x_pred)
        P[k] = (1 - K)^2 * p_pred + K^2 * r
    end
    return x_hat
end

# ============================================================
# 中心差分梯度 (用 Insight.slice)
# ============================================================
function gradient(x, dx)
    n = Insight.numel(x)
    if n < 2 return Insight.zeros(Int64[n], Insight.float64) end
    result = Insight.zeros(Int64[n], Insight.float64)
    if n > 2
        # slice(dim_1based, start_1based_inclusive, stop_1based_exclusive)
        interior = (Insight.slice(x, 1, 3, n + 1) - Insight.slice(x, 1, 1, n - 1)) / (2.0 * dx)
        Insight.copy_from_!(Insight.slice(result, 1, 2, n), interior)
    end
    # 边界
    cpu_x = Insight.to(x, 0)
    v0 = Insight.item(cpu_x, 0)
    v1 = Insight.item(cpu_x, 1)
    vn1 = Insight.item(cpu_x, n-1)
    vn2 = Insight.item(cpu_x, n-2)
    Insight.copy_from_!(Insight.slice(result, 1, 1, 2), Insight.full(Int64[1], (v1 - v0) / dx, Insight.float64))
    Insight.copy_from_!(Insight.slice(result, 1, n, n + 1), Insight.full(Int64[1], (vn1 - vn2) / dx, Insight.float64))
    return result
end

# ============================================================
# 峰值查找
# ============================================================
function find_peaks_simple(sig::Vector{Float64}; order=PEAK_ORDER)
    n = length(sig)
    peaks_max = Int[]
    peaks_min = Int[]
    for i in (order+1):(n-order)
        is_max = true
        is_min = true
        val = sig[i]
        for j in 1:order
            if sig[i-j] >= val || sig[i+j] >= val is_max = false end
            if sig[i-j] <= val || sig[i+j] <= val is_min = false end
            if !is_max && !is_min break end
        end
        if is_max push!(peaks_max, i) end
        if is_min push!(peaks_min, i) end
    end
    return peaks_max, peaks_min
end

# ============================================================
# 快速 CWT (用 correlate 代替 FFT 卷积)
# ============================================================
function cwt_fast(signal)
    n = Insight.numel(signal)
    all_data = Float64[]
    for wp in _CWT_WAVELETS
        w_arr = wp[2]
        N_w = Insight.numel(w_arr)
        w_conj = Insight.conj_fn(w_arr)
        w_rev = Insight.flipud(w_conj)
        if Insight.dtype(w_rev) != Insight.complex128
            w_rev = Insight.to_complex(w_rev)
        end
        if Insight.dtype(signal) != Insight.complex128
            sig_cpx = Insight.to_complex(signal)
        else
            sig_cpx = signal
        end
        conv_full = Insight.signal.correlate(sig_cpx, w_rev, mode="same")
        row = Insight.real_part(conv_full)
        append!(all_data, Insight.to_data(Insight.to(row, 0)))
    end
    n_scales = length(_CWT_WAVELETS)
    return Insight.from_data(reshape(all_data, n, n_scales)')
end

# ============================================================
# 初始化缓存
# ============================================================
function init_cache(device)
    global _T, _COMPOSITE_BASE, _TAPS, _GAUSS_KERNEL, _CWT_WAVELETS

    if device == "gpu" && Insight.has_device(Int64(1))
        Insight.load_backend("rocm")
        Insight.set_device(Insight.GPUPlace(0))
    else
        Insight.set_device(Insight.CPUPlace())
    end

    _T = Insight.arange(Float64(0.0), Float64(N_SAMPLES), Float64(1.0), Insight.float64) / FS

    # Chirp
    chirp_sig = Insight.signal.chirp(_T, F0_CHIRP, DURATION, F1_CHIRP, method=Int32(0)) * AMPLITUDE

    # Gausspulse (手动实现, Julia 无 gausspulse 绑定)
    center = Int64(0.3 * N_SAMPLES)
    gauss_len = fld(N_SAMPLES, 2)
    center_val = Insight.item(_T, fld(gauss_len, 2))
    t_gauss = Insight.slice(_T, 1, 1, gauss_len + 1) - Insight.full(Int64[1], center_val, Insight.float64)
    # gausspulse(t, fc, bw) = exp(-pi^2*fc^2*bw^2*t^2/(4*ln2)) * cos(2*pi*fc*t)
    a = (π * FC_GAUSS * BW_GAUSS)^2 / (4.0 * log(2.0))
    t_sq = t_gauss * t_gauss
    g = Insight.exp(t_sq * (-a)) * Insight.cos(t_gauss * (2.0 * π * FC_GAUSS))
    gauss_sig = Insight.zeros(Int64[N_SAMPLES], Insight.float64)
    clip_len = Base.min(gauss_len, N_SAMPLES - center)
    Insight.copy_from_!(Insight.slice(gauss_sig, 1, center + 1, center + clip_len + 1), Insight.slice(g, 1, 1, clip_len + 1))

    # Sawtooth
    saw_sig = Insight.signal.sawtooth(_T * (2.0 * π * FREQ_SAW), width=0.5) * 0.4

    _COMPOSITE_BASE = chirp_sig + gauss_sig + saw_sig

    # FIR bandpass
    nyquist = FS / 2.0
    _TAPS = Insight.signal.firwin(NUMTAPS, Float64[BP_LOW/nyquist, BP_HIGH/nyquist], window="hamming", pass_zero="bandpass")

    # Gaussian smoothing kernel
    kernel_size = Base.max(3, Int64(0.1 * N_SAMPLES))
    if kernel_size % 2 == 0 kernel_size += 1 end
    x_kernel = Insight.linspace(-3.0, 3.0, kernel_size, Insight.float64)
    kernel = Insight.exp(x_kernel * x_kernel * (-0.5))
    _GAUSS_KERNEL = kernel / Insight.sum(kernel)

    # Pre-compute CWT wavelets
    _CWT_WAVELETS = []
    for w in CWT_WIDTHS_START:CWT_WIDTHS_STEP:(CWT_WIDTHS_END-1)
        N_wavelet = Base.min(Int64(10 * w), N_SAMPLES)
        if N_wavelet < 1 N_wavelet = 1 end
        w_arr = Insight.signal.morlet2(N_wavelet, Float64(w), w=MORLET_W)
        push!(_CWT_WAVELETS, (Float64(w), w_arr))
    end
end

# ============================================================
# 运行一帧
# ============================================================
function run_frame(seed_val; timer=false, noise=nothing)
    GC.enable(false)

    # [1] 合成信号
    t0 = time()
    if noise !== nothing
        composite = _COMPOSITE_BASE + noise
    else
        Insight.seed(seed_val)
        noise = Insight.randn(Int64[N_SAMPLES], Insight.float64) * NOISE_STD
        composite = _COMPOSITE_BASE + noise
    end
    t_gen = time() - t0

    # [2] 去趋势 + 带通滤波
    t0 = time()
    detrended = Insight.signal.detrend(composite)
    filtered_full = Insight.signal.fftconvolve(detrended, _TAPS, mode="full")
    half = fld(NUMTAPS, 2)
    filtered = Insight.slice(filtered_full, 1, half + 1, half + N_SAMPLES + 1)
    t_filt = time() - t0

    # [3] 高斯平滑
    t0 = time()
    smoothed = Insight.signal.fftconvolve(filtered, _GAUSS_KERNEL, mode="same")
    t_spline = time() - t0

    # [4] FM 解调
    t0 = time()
    analytic = Insight.signal.hilbert(smoothed)
    inst_phase = Insight.unwrap(Insight.angle_fn(analytic))
    inst_freq = gradient(inst_phase, 1.0 / FS) / (2.0 * π)
    t_demod = time() - t0

    # [5] STFT
    t0 = time()
    stft_result = Insight.signal.stft(smoothed, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    t_stft = time() - t0

    # [6] CWT
    t0 = time()
    cwt_matrix = cwt_fast(smoothed)
    t_cwt = time() - t0

    # [7] 自相关
    t0 = time()
    seg_len = Base.min(N_SAMPLES, 512)
    seg = Insight.slice(smoothed, 1, 1, seg_len + 1)
    autocorr_full = Insight.signal.correlate(seg, seg, mode="full")
    mid = fld(Insight.numel(autocorr_full), 2)
    autocorr = Insight.slice(autocorr_full, 1, mid + 1, Insight.numel(autocorr_full) + 1)
    norm_val = Insight.item(Insight.to(autocorr, 0), 0)
    if Base.abs(norm_val) > 1e-12
        autocorr = autocorr ./ norm_val
    end
    t_corr = time() - t0

    # [8] 峰值查找
    t0 = time()
    smoothed_vec = Insight.to_data(Insight.to(smoothed, 0))
    peaks_max, peaks_min = find_peaks_simple(smoothed_vec)
    t_peak = time() - t0

    # [9] 卡尔曼滤波
    t0 = time()
    inst_freq_vec = Insight.to_data(Insight.to(inst_freq, 0))
    smoothed_freq = kalman_smooth(inst_freq_vec)
    params = []
    for p in peaks_max
        if p >= 1 && p <= length(smoothed_freq)
            push!(params, (p, smoothed_freq[p]))
        end
    end
    t_kalman = time() - t0

    GC.enable(true)
    GC.gc()

    total_ms = (t_gen + t_filt + t_spline + t_demod + t_stft +
                t_cwt + t_corr + t_peak + t_kalman) * 1000

    t_gen_ms = t_gen * 1000
    t_filt_ms = t_filt * 1000
    t_spline_ms = t_spline * 1000
    t_demod_ms = t_demod * 1000
    t_stft_ms = t_stft * 1000
    t_cwt_ms = t_cwt * 1000
    t_corr_ms = t_corr * 1000
    t_peak_ms = t_peak * 1000
    t_kalman_ms = t_kalman * 1000

    if timer
        println("    gen: $(Base.round(t_gen_ms, digits=2)) ms")
        println("    filt: $(Base.round(t_filt_ms, digits=2)) ms")
        println("    spline: $(Base.round(t_spline_ms, digits=2)) ms")
        println("    demod: $(Base.round(t_demod_ms, digits=2)) ms")
        println("    stft: $(Base.round(t_stft_ms, digits=2)) ms")
        println("    cwt: $(Base.round(t_cwt_ms, digits=2)) ms")
        println("    corr: $(Base.round(t_corr_ms, digits=2)) ms")
        println("    peak: $(Base.round(t_peak_ms, digits=2)) ms")
        println("    kalman: $(Base.round(t_kalman_ms, digits=2)) ms")
    end

    return (
        total_ms = total_ms,
        t_gen_ms = t_gen_ms,
        t_filt_ms = t_filt_ms,
        t_spline_ms = t_spline_ms,
        t_demod_ms = t_demod_ms,
        t_stft_ms = t_stft_ms,
        t_cwt_ms = t_cwt_ms,
        t_corr_ms = t_corr_ms,
        t_peak_ms = t_peak_ms,
        t_kalman_ms = t_kalman_ms,
        n_peaks_max = length(peaks_max),
        n_peaks_min = length(peaks_min),
        params = params,
        smoothed = smoothed,
    )
end

# ============================================================
# CLI 参数解析
# ============================================================
function parse_args()
    args = Dict("device" => "cpu", "seed" => 42, "iterations" => 0, "timer" => false, "info" => false, "profiler" => false)
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--device" && i < length(ARGS)
            args["device"] = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--seed" && i < length(ARGS)
            args["seed"] = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--iterations" && i < length(ARGS)
            args["iterations"] = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--timer"
            args["timer"] = true
            i += 1
        elseif ARGS[i] == "--info"
            args["info"] = true
            i += 1
        elseif ARGS[i] == "--profiler"
            args["profiler"] = true
            i += 1
        else
            i += 1
        end
    end
    return args
end

# ============================================================
# MAIN
# ============================================================
args = parse_args()

n_frames = args["iterations"] > 0 ? args["iterations"] : 1

println("=" ^ 60)
println("  多域特征提取与定位 (Insight7 Julia)")
println("=" ^ 60)
println("\n[配置]  采样率: $(FS/1e3) kHz  信号长度: $N_SAMPLES 点  时长: $(DURATION)s  设备: $(args["device"])  帧数: $n_frames")

init_cache(args["device"])

if args["profiler"]
    global prof = Insight.Profiler(Int32(0), Int32(0))
    Insight.profiler_start(prof)
end

if args["device"] == "all"
    if !Insight.has_device(Int64(1))
        println("[WARN] --device all 但无 GPU 可用, 降级为 CPU 模式")
        args["device"] = "cpu"
    end
end

if args["info"]
    cpu_total, cpu_free = Insight.device_memory_info(Int64(0), Int64(0))
    println("[Memory] CPU: total=$(cpu_total ÷ (1024*1024))MB used=$((cpu_total - cpu_free) ÷ (1024*1024))MB")
    if Insight.has_device(Int64(1))
        gpu_total, gpu_free = Insight.device_memory_info(Int64(1), Int64(0))
        println("[Memory] GPU: total=$(gpu_total ÷ (1024*1024))MB used=$((gpu_total - gpu_free) ÷ (1024*1024))MB")
    end
end

println("\n[波形]  chirp($F0_CHIRP-$F1_CHIRP Hz) + gausspulse($FC_GAUSS Hz) + sawtooth($FREQ_SAW Hz) + noise")
println("[滤波]  FIR 带通 $BP_LOW-$BP_HIGH Hz, $NUMTAPS 阶")
println("[特征]  FM解调 + STFT + CWT(morlet2, $(_CWT_WAVELETS |> length) scales) + 自相关")
println("[检测]  峰值查找(order=$PEAK_ORDER) + 卡尔曼滤波")

device_flag = args["device"]
timer_flag = args["timer"]
times = Float64[]
for frame in 0:(n_frames-1)
    if device_flag == "all"
        # Generate noise on CPU once so CPU and GPU use identical input
        Insight.seed(args["seed"] + frame)
        cpu_noise = Insight.randn(Int64[N_SAMPLES], Insight.float64) * NOISE_STD

        cpu_r = run_frame(args["seed"] + frame; timer=timer_flag, noise=cpu_noise)
        gpu_noise = Insight.to(cpu_noise, Int64(1))
        gpu_r = run_frame(args["seed"] + frame; timer=timer_flag, noise=gpu_noise)
        push!(times, cpu_r.total_ms)
        cpu_scpu = Insight.to(cpu_r.smoothed, Int64(0))
        gpu_scpu = Insight.to(gpu_r.smoothed, Int64(0))
        diff_arr = Insight.abs(cpu_scpu - gpu_scpu)
        max_diff = Insight.item(Insight.max(diff_arr), 0)
        println("  帧 $(lpad(frame, 4))/$n_frames | CPU: $(lpad(round(cpu_r.total_ms, digits=2), 8)) ms | GPU: $(lpad(round(gpu_r.total_ms, digits=2), 8)) ms | max_diff=$(max_diff)")
    else
        r = run_frame(args["seed"] + frame; timer=timer_flag)
        push!(times, r.total_ms)

        if n_frames == 1 || frame == 0 || (frame + 1) % 10 == 0 || frame == n_frames - 1
            params_str = "无"
            if length(r.params) > 0
                params_str = "峰值@$(r.params[1][1]) f=$(round(r.params[1][2], digits=1))Hz"
            end

            if timer_flag
                println("  帧 $(lpad(frame, 4))/$n_frames | $(lpad(round(r.total_ms, digits=2), 8)) ms | " *
                        "gen $(lpad(round(r.t_gen_ms, digits=1), 5)) " *
                        "filt $(lpad(round(r.t_filt_ms, digits=1), 5)) " *
                        "spl $(lpad(round(r.t_spline_ms, digits=1), 5)) " *
                        "demod $(lpad(round(r.t_demod_ms, digits=1), 5)) " *
                        "stft $(lpad(round(r.t_stft_ms, digits=1), 5)) " *
                        "cwt $(lpad(round(r.t_cwt_ms, digits=1), 5)) " *
                        "corr $(lpad(round(r.t_corr_ms, digits=1), 5)) " *
                        "peak $(lpad(round(r.t_peak_ms, digits=1), 5)) " *
                        "kalman $(lpad(round(r.t_kalman_ms, digits=1), 5)) | " *
                        "极大值 $(r.n_peaks_max) 极小值 $(r.n_peaks_min) | $params_str")
            else
                println("  帧 $(lpad(frame, 4))/$n_frames | $(lpad(round(r.total_ms, digits=2), 8)) ms | " *
                        "极大值 $(r.n_peaks_max) 极小值 $(r.n_peaks_min) | $params_str")
            end
        end
    end
end

avg_ms = sum(times) / length(times)
fps = 1000.0 / avg_ms

summary_device = device_flag == "all" ? "CPU+GPU" : (device_flag == "gpu" ? "GPU" : "CPU")
println("\n  " * "=" ^ 50)
println("  性能总结 ($summary_device)")
println("  " * "=" ^ 50)
println("  总帧数: $n_frames")
println("  平均每帧: $(round(avg_ms, digits=2)) ms  FPS: $(round(fps, digits=2))")

if Insight.has_device(Int64(1)) && device_flag != "gpu" && device_flag != "all"
    println("\n[提示] GPU 可用, 使用 --device gpu 运行 GPU 版本")
end
if args["profiler"]
    Insight.profiler_stop(prof)
    Insight.profiler_report(prof)
end
println("\n完成！")
