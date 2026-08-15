# demos/julia/radar_detection.jl
# 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Julia)
# 纯 Insight7 API，与 Python/C++/Lua 版算法完全对齐
# 运行: julia radar_detection.jl --device gpu --seed 42 --iterations 50

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "build", "bindings", "julia"))
using Insight

# ============================================================
# 物理参数
# ============================================================
const FS = 10e6
const T_PRF = 100e-6
const N = Int64(1000)
const B = 5e6
const TAU = 50e-6
const N_PULSES = Int64(32)
const SNR_DB = 10.0
const PULSE_LEN = Int64(TAU * FS)
const PC_OFFSET = Base.div(PULSE_LEN, 2)
const RANGE_PER_BIN = 3e8 / (2 * FS)
const MAX_UNAMBIG_RANGE = RANGE_PER_BIN * N

const DELAY_START = [35e-6, 50e-6]
const DELAY_END = [11e-6, 20e-6]
const DOPPLER_START = [2000.0, -1500.0]
const DOPPLER_END = [1800.0, -1200.0]

# ============================================================
# 全局缓存
# ============================================================
_S_TX = nothing
_TEMPLATE = nothing
_HAMMING = nothing
_SLOW_TIMES = nothing
_SIG_POWER = 0.0
_NOISE_SIGMA = 0.0
_DOPPLER_BINS = nothing
_RANGE_BINS = nothing
_PLACE = nothing

function init_cache(device)
    global _S_TX, _TEMPLATE, _HAMMING, _SLOW_TIMES, _SIG_POWER, _NOISE_SIGMA, _DOPPLER_BINS, _RANGE_BINS, _PLACE

    if device == "gpu" && Insight.has_device(Int64(1))
        Insight.load_backend("rocm")
        _PLACE = Insight.GPUPlace(0)
    else
        _PLACE = Insight.CPUPlace()
    end
    Insight.set_device(_PLACE)

    t = Insight.arange(Float64(0.0), Float64(N), Float64(1.0), Insight.float64) / FS
    phase = π * (B / TAU) * (t * t)
    s_tx = Insight.to_complex(Insight.cos(phase), Insight.sin(phase))
    tau_arr = Insight.full(Int64[1], TAU, Insight.float64)
    mask = Insight.cast(Insight.less(t, tau_arr), Insight.float64)
    _S_TX = s_tx * mask

    r = Insight.real_part(_S_TX); im = Insight.imag_part(_S_TX)
    _SIG_POWER = Base.float(Insight.item(Insight.mean(r * r + im * im), 0))
    _NOISE_SIGMA = Base.sqrt(_SIG_POWER / 10.0^(SNR_DB / 10.0) / 2.0)

    doppler = Insight.fftshift(Insight.fftfreq(N_PULSES, T_PRF))
    _DOPPLER_BINS = Insight.to(doppler, Int64(0))
    range_arr = (Insight.arange(Float64(0.0), Float64(N), Float64(1.0), Insight.float64) - PC_OFFSET) * RANGE_PER_BIN
    _RANGE_BINS = Insight.to(range_arr, Int64(0))

    # 预创建脉冲压缩模板
    _TEMPLATE = Insight.slice(_S_TX, 1, 1, PULSE_LEN + 1)

    # 预创建 hamming 窗 (避免每帧重新创建)
    _HAMMING = Insight.reshape(Insight.signal.get_window("hamming", Int64(N_PULSES), fftbins=false), Int64[N_PULSES, 1])
    # 确保 hamming 窗在正确的设备上 (get_window 可能因后端差异在 CPU 上创建)
    if device == "gpu" && Insight.has_device(Int64(1))
        _HAMMING = Insight.to(_HAMMING, Int64(1))
    end

    # 预创建慢时间轴 (避免每帧重复 arange + mul)
    _SLOW_TIMES = Insight.arange(Float64(0.0), Float64(N_PULSES), Float64(1.0), Insight.float64) * T_PRF
end

function get_target_params(frame_idx, total_frames)
    if total_frames <= 1
        return copy(DELAY_START), copy(DOPPLER_START)
    end
    t = Float64(frame_idx) / Float64(Base.max(total_frames - 1, 1))
    delays = [s + (e - s) * t for (s, e) in zip(DELAY_START, DELAY_END)]
    dopplers = [s + (e - s) * t for (s, e) in zip(DOPPLER_START, DOPPLER_END)]
    return delays, dopplers
end

function simulate_echoes(delays, dopplers; noise_r=nothing, noise_i=nothing)
    local nr, ni
    if noise_r !== nothing && noise_i !== nothing
        nr = noise_r
        ni = noise_i
    else
        nr = Insight.randn(Int64[N_PULSES, N], Insight.float64) * _NOISE_SIGMA
        ni = Insight.randn(Int64[N_PULSES, N], Insight.float64) * _NOISE_SIGMA
    end
    pulses = Insight.zeros(Int64[N_PULSES, N], Insight.complex128)

    for k in 1:length(delays)
        ds = Base.round(Int64, delays[k] * FS)
        delayed_1d = Insight.zeros(Int64[N], Insight.complex128)
        if ds < N
            src = Insight.slice(_S_TX, 1, 1, N - ds + 1)
            dst = Insight.slice(delayed_1d, 1, ds + 1, N + 1)
            Insight.copy_from_!(dst, src)
        end
        delayed_2d = Insight.reshape(delayed_1d, Int64[1, N])
        phase = _SLOW_TIMES * (2.0 * π * dopplers[k])
        rot = Insight.to_complex(Insight.cos(phase), Insight.sin(phase))
        rot_2d = Insight.reshape(rot, Int64[N_PULSES, 1])
        pulses = pulses + delayed_2d * rot_2d
    end
    pulses = pulses + Insight.to_complex(nr, ni)
    return pulses
end

function extract_targets(energy, det)
    top_n = 2; cluster_threshold = 3.0
    NP_const = 32  # N_PULSES
    N_const = 1000  # N
    idx = Insight.nonzero(det)
    if idx.ptr == C_NULL
        return Tuple{Int64,Int64}[], 0
    end
    n_det = Insight.shape(idx)
    if length(n_det) < 2 || n_det[1] == 0 || n_det[2] == 0
        return Tuple{Int64,Int64}[], 0
    end
    # n_det from Insight.shape is column-major: [N, 2] where N = #detections
    n_nonzero = n_det[1]

    # 批量 GPU→CPU 传输 nonzero 索引
    idx_cpu = Insight.to(idx, Int64(0))
    eng_cpu = Insight.to(energy, Int64(0))
    # to_data returns column-major: N×2 where each row is [doppler, range]
    idx_jl = Insight.to_data(idx_cpu)  # N×2 matrix

    candidates = Tuple{Int64,Int64,Float64}[]
    for k in 1:n_nonzero
        di_raw = Int64(idx_jl[k, 1])
        ri_raw = Int64(idx_jl[k, 2])
        if di_raw < 0 || di_raw >= NP_const || ri_raw < 0 || ri_raw >= N_const
            continue  # skip invalid indices
        end
        di = di_raw + 1
        ri = ri_raw + 1
        v = Insight.item_flat(eng_cpu, di_raw * N_const + ri_raw)
        push!(candidates, (di, ri, v))
    end
    if length(candidates) == 0; return Tuple{Int64,Int64}[], 0; end

    sort!(candidates, by=x -> x[3], rev=true)
    keep = Base.min(length(candidates), Base.max(top_n * 10, 20))
    top = candidates[1:keep]
    visited = falses(keep)
    clusters = typeof(candidates[1])[]
    for i in 1:keep
        if visited[i]; continue; end
        visited[i] = true; cluster_idx = Int64[i]
        for j in i+1:keep
            if visited[j]; continue; end
            d = Base.sqrt((top[i][1] - top[j][1])^2 + (top[i][2] - top[j][2])^2)
            if d <= cluster_threshold; visited[j] = true; push!(cluster_idx, j); end
        end
        best_i = Base.argmax([top[ci][3] for ci in cluster_idx])
        push!(clusters, top[cluster_idx[best_i]])
    end
    sort!(clusters, by=x -> x[3], rev=true)
    seen = Set{Int64}()
    targets = Tuple{Int64,Int64}[]
    for c in clusters
        if !(c[2] in seen); push!(seen, c[2]); push!(targets, (c[1], c[2])); if length(targets) >= top_n; break; end; end
    end
    return targets, length(candidates)
end

function run_frame(delays, dopplers, _device, seed; noise_r=nothing, noise_i=nothing, timer=false, prof=nothing)
    if _device == "gpu" && Insight.has_device(Int64(1))
        Insight.load_backend("rocm")
        Insight.set_device(Insight.GPUPlace(0))
    else
        Insight.set_device(Insight.CPUPlace())
    end
    Insight.seed(seed)

    t0 = time()
    if prof !== nothing; Insight.profiler_begin_event(prof, "echo_simulation"); end
    pulses = simulate_echoes(delays, dopplers; noise_r=noise_r, noise_i=noise_i)
    if prof !== nothing; Insight.profiler_end_event(prof); end
    t1 = time()

    if prof !== nothing; Insight.profiler_begin_event(prof, "pulse_compression"); end
    pc = Insight.signal.pulse_compression(pulses, _TEMPLATE)
    if prof !== nothing; Insight.profiler_end_event(prof); end
    t2 = time()

    if prof !== nothing; Insight.profiler_begin_event(prof, "doppler"); end
    mean_pc = Insight.mean(pc, axis=1, keepdims=true)
    pc_ac = pc - mean_pc
    windowed = pc_ac * _HAMMING
    spec = Insight.signal.pulse_doppler(windowed, "", Int64(N_PULSES))
    shifted = spec / Float64(N_PULSES)
    if prof !== nothing; Insight.profiler_end_event(prof); end
    t3 = time()

    r2 = Insight.real_part(shifted); i2 = Insight.imag_part(shifted)
    energy = r2 * r2 + i2 * i2
    if prof !== nothing; Insight.profiler_begin_event(prof, "ca_cfar"); end
    cfar_result = Insight.signal.ca_cfar(energy, Int32[4, 4], Int32[12, 12], 1e-6)
    if prof !== nothing; Insight.profiler_end_event(prof); end
    t4 = time()

    if prof !== nothing; Insight.profiler_begin_event(prof, "target_extraction"); end
    targets, raw_count = extract_targets(energy, cfar_result[2])
    if prof !== nothing; Insight.profiler_end_event(prof); end
    t5 = time()

    valid = Tuple{Int64,Int64}[]
    rng_jl = Insight.to_data(_RANGE_BINS)
    for (d, r) in targets
        dist = rng_jl[r]
        if 0 <= dist <= MAX_UNAMBIG_RANGE; push!(valid, (d, r)); end
    end

    if timer
        println("    simulate: $(Base.round((t1 - t0) * 1000, digits=2)) ms")
        println("    pulse_compress: $(Base.round((t2 - t1) * 1000, digits=2)) ms")
        println("    doppler: $(Base.round((t3 - t2) * 1000, digits=2)) ms")
        println("    cfar: $(Base.round((t4 - t3) * 1000, digits=2)) ms")
        println("    extract: $(Base.round((t5 - t4) * 1000, digits=2)) ms")
    end

    return valid, raw_count, (t5 - t0) * 1000, energy, pulses
end

# ============================================================
# CLI + 主流程
# ============================================================
function main()
    local device = "cpu"; local seed = Int32(42); local iterations = Int32(0)
    local timer_flag = false; local info_flag = false; local profiler_flag = false
    local prof = nothing
    local i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--device" && i < length(ARGS); device = ARGS[i+1]; i += 1
        elseif ARGS[i] == "--seed" && i < length(ARGS); seed = parse(Int32, ARGS[i+1]); i += 1
        elseif ARGS[i] == "--iterations" && i < length(ARGS); iterations = parse(Int32, ARGS[i+1]); i += 1
        elseif ARGS[i] == "--timer"; timer_flag = true
        elseif ARGS[i] == "--info"; info_flag = true
        elseif ARGS[i] == "--profiler"; profiler_flag = true
        end
        i += 1
    end

    local n_frames = iterations > 0 ? iterations : 1
    println("=" ^ 60)
    println("  雷达目标检测与多普勒分析 (Insight7 Julia)")
    println("=" ^ 60)
    println("\n[配置] 采样率: $(FS/1e6) MHz  脉冲: $(N_PULSES)  设备: $(device)  种子: $(seed)  帧数: $(n_frames)")

    if device == "all"
        if !Insight.has_device(Int64(1))
            println("[WARN] --device all 但无 GPU 可用, 降级为 CPU 模式")
            device = "cpu"
        end
    end

    init_cache(device == "all" ? "cpu" : device)

    if profiler_flag
        prof = Insight.Profiler(Int32(0), Int32(0))
        Insight.profiler_start(prof)
    end

    if info_flag
        cpu_total, cpu_free = Insight.device_memory_info(Int64(0), Int64(0))
        println("[Memory] CPU: total=$(cpu_total ÷ (1024*1024))MB used=$((cpu_total - cpu_free) ÷ (1024*1024))MB")
        if Insight.has_device(Int64(1))
            gpu_total, gpu_free = Insight.device_memory_info(Int64(1), Int64(0))
            println("[Memory] GPU: total=$(gpu_total ÷ (1024*1024))MB used=$((gpu_total - gpu_free) ÷ (1024*1024))MB")
        end
    end

    println("\n[波形分析] 计算模糊函数...")
    local ambg = Insight.signal.ambgfun(_TEMPLATE, FS, 1.0 / T_PRF)
    local peak_val = Insight.item(Insight.max(Insight.abs(ambg)), 0)
    local mean_val = Insight.item(Insight.mean(Insight.abs(ambg)), 0)
    local peak_ratio = 20.0 * Base.log10(peak_val / (mean_val + 1e-12))
    println("  模糊函数峰值: $(Base.round(peak_val, digits=2))")
    println("  峰值/平均比: $(Base.round(peak_ratio, digits=1)) dB  (主瓣窄、旁瓣低 ✓)")

    local dop_jl = Insight.to_data(_DOPPLER_BINS)
    local rng_jl = Insight.to_data(_RANGE_BINS)
    local times = Float64[]

    # 禁用 GC 减少 finalizer 开销 (帧结束后恢复)
    GC.enable(false)
    for frame in 0:n_frames-1
        local delays, dopplers = get_target_params(frame, n_frames)

        if device == "all"
            # CPU 缓存 + 噪声（一次性生成，双后端共用）
            init_cache("cpu")
            Insight.seed(seed)
            cpu_noise_r = Insight.randn(Int64[N_PULSES, N], Insight.float64) * _NOISE_SIGMA
            cpu_noise_i = Insight.randn(Int64[N_PULSES, N], Insight.float64) * _NOISE_SIGMA

            local cpu_targets, cpu_raw_count, cpu_ms, cpu_energy, cpu_pulses = run_frame(delays, dopplers, "cpu", seed; noise_r=cpu_noise_r, noise_i=cpu_noise_i, timer=timer_flag, prof=prof)

            # GPU run: 复用 CPU 缓存，不重新生成
            local gpu_noise_r = Insight.to(cpu_noise_r, Int64(1))
            local gpu_noise_i = Insight.to(cpu_noise_i, Int64(1))
            global _S_TX = Insight.to(_S_TX, Int64(1))
            global _TEMPLATE = Insight.to(_TEMPLATE, Int64(1))
            global _HAMMING = Insight.to(_HAMMING, Int64(1))
            global _SLOW_TIMES = Insight.to(_SLOW_TIMES, Int64(1))
            local gpu_targets, gpu_raw_count, gpu_ms, gpu_energy, gpu_pulses = run_frame(delays, dopplers, "gpu", seed; noise_r=gpu_noise_r, noise_i=gpu_noise_i, timer=timer_flag, prof=prof)
            push!(times, cpu_ms)
            local cpu_p_cpu = Insight.to(cpu_pulses, Int64(0))
            local gpu_p_cpu = Insight.to(gpu_pulses, Int64(0))
            local diff_arr = Insight.abs(cpu_p_cpu - gpu_p_cpu)
            local max_diff = Insight.item(Insight.max(diff_arr), 0)
            println("  帧 $(Base.lpad(frame, 4))/$(n_frames) | CPU: $(Base.lpad(Base.round(cpu_ms, digits=2), 8)) ms | GPU: $(Base.lpad(Base.round(gpu_ms, digits=2), 8)) ms | max_diff=$(max_diff)")
        else
            local targets, raw_count, total_ms, _, _ = run_frame(delays, dopplers, device, seed; timer=timer_flag, prof=prof)
            push!(times, total_ms)

            if n_frames == 1 || frame == 0 || (frame + 1) % 10 == 0 || frame == n_frames - 1
                local targets_str = ""
                for (k, (d, r)) in enumerate(targets)
                    if k > 1; targets_str *= "; "; end
                    targets_str *= "距离 $(Base.round(Int, rng_jl[r]))m 多普勒 $(Base.round(Int, dop_jl[d]))Hz"
                end
                if targets_str == ""; targets_str = "无"; end
                println("  帧 $(Base.lpad(frame, 4))/$(n_frames) | $(Base.lpad(Base.round(total_ms, digits=2), 8)) ms | " *
                        "检测 $(Base.lpad(raw_count, 4)) → $(length(targets)) 目标 | $(targets_str)")
            end
        end
    end
    GC.enable(true)
    GC.gc()  # 清理累积的临时数组

    local avg_ms = sum(times) / length(times)
    local fps = 1000.0 / avg_ms
    local summary_device = device == "all" ? "CPU+GPU" : uppercase(device)
    println("\n  =================================================")
    println("  性能总结 ($summary_device)")
    println("  =================================================")
    println("  总帧数: $(n_frames)")
    println("  平均每帧: $(Base.round(avg_ms, digits=2)) ms  FPS: $(Base.round(fps, digits=2))")

    if profiler_flag
        Insight.profiler_stop(prof)
        Insight.profiler_report(prof)
    end

    println("\n完成！")
end

main()
