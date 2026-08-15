-- demos/lua/feature_extraction.lua
-- 多域特征提取与定位 — 比赛任务2 (Insight7 Lua)
-- 纯 Insight7 API，与 Python/C++/Julia 版算法完全对齐
-- 运行: LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--       LD_LIBRARY_PATH=build/backends/cpu luarocks exec lua demos/lua/feature_extraction.lua --device cpu

local ins = require("_insight")

-- ============================================================
-- 物理参数
-- ============================================================
local FS = 5000.0
local DURATION = 0.5
local N_SAMPLES = math.floor(FS * DURATION) -- 2500
local F0_CHIRP = 50.0
local F1_CHIRP = 500.0
local FC_GAUSS = 200.0
local BW_GAUSS = 0.5
local FREQ_SAW = 80.0
local NOISE_STD = 0.15
local AMPLITUDE = 0.8
local PI = math.pi

local BP_LOW = 40.0
local BP_HIGH = 300.0
local NUMTAPS = 127

local NPERSEG = 256
local NOVERLAP = 192

local CWT_WIDTHS_START = 1
local CWT_WIDTHS_END = 64
local CWT_WIDTHS_STEP = 4
local MORLET_W = 5.0

local KALMAN_Q = 1e-4
local KALMAN_R = 0.5
local PEAK_ORDER = 15

-- ============================================================
-- 全局缓存
-- ============================================================
local _T = nil
local _COMPOSITE_BASE = nil
local _TAPS = nil
local _GAUSS_KERNEL = nil
local _CWT_WIDTHS_LIST = {}
local _PLACE = nil

-- ============================================================
-- 卡尔曼滤波
-- ============================================================
local function kalman_smooth(z, q, r)
  q = q or KALMAN_Q
  r = r or KALMAN_R
  local n = #z
  if n == 0 then
    return z
  end
  local x_hat = { [1] = z[1] }
  local P = { [1] = 1.0 }
  for k = 2, n do
    local x_pred = x_hat[k - 1]
    local p_pred = P[k - 1] + q
    local K = p_pred / (p_pred + r)
    x_hat[k] = x_pred + K * (z[k] - x_pred)
    P[k] = (1 - K) * (1 - K) * p_pred + K * K * r
  end
  return x_hat
end

-- ============================================================
-- 中心差分梯度
-- ============================================================
local function gradient(x, dx)
  local n = x.numel
  if n < 2 then
    return ins.zeros({ n }, ins.float64, _PLACE)
  end
  local result = ins.zeros({ n }, ins.float64, _PLACE)
  if n > 2 then
    -- ins.slice(a, dim_1based, start_1based_inclusive, stop_1based_exclusive)
    local interior = (ins.slice(x, 1, 3, n + 1) - ins.slice(x, 1, 1, n - 1)) / (2.0 * dx)
    ins.slice(result, 1, 2, n):copy_from_(interior)
  end
  -- 边界
  local cpu = ins.CPUPlace()
  local v0 = x:to(cpu):get(0)
  local v1 = x:to(cpu):get(1)
  local vn1 = x:to(cpu):get(n - 1)
  local vn2 = x:to(cpu):get(n - 2)
  -- Use fill_ for scalar assignment
  ins.slice(result, 1, 1, 2):fill_((v1 - v0) / dx)
  ins.slice(result, 1, n, n + 1):fill_((vn1 - vn2) / dx)
  return result
end

-- ============================================================
-- 峰值查找 (简单线性扫描)
-- ============================================================
local function find_peaks_simple(sig, order)
  order = order or PEAK_ORDER
  local n = #sig
  local peaks_max = {}
  local peaks_min = {}
  for i = order + 1, n - order do
    local is_max = true
    local is_min = true
    local val = sig[i]
    for j = 1, order do
      if sig[i - j] >= val or sig[i + j] >= val then
        is_max = false
      end
      if sig[i - j] <= val or sig[i + j] <= val then
        is_min = false
      end
      if not is_max and not is_min then
        break
      end
    end
    if is_max then
      peaks_max[#peaks_max + 1] = i
    end
    if is_min then
      peaks_min[#peaks_min + 1] = i
    end
  end
  return peaks_max, peaks_min
end

-- ============================================================
-- CWT (使用原生 ins.signal.cwt)
-- ============================================================
local function cwt_fast(signal)
  return ins.signal.cwt(signal, function(points, s)
    return ins.signal.morlet2(points, s, MORLET_W)
  end, _CWT_WIDTHS_LIST)
end

-- ============================================================
-- 初始化缓存
-- ============================================================
local function init_cache(device)
  if device == "gpu" and ins.has_device("gpu") then
    ins.load_backend("rocm")
    _PLACE = ins.GPUPlace(0)
  else
    _PLACE = ins.CPUPlace()
  end
  ins.set_device(_PLACE)

  _T = ins.arange(N_SAMPLES, ins.float64, _PLACE) / FS

  -- Chirp
  local chirp_sig = ins.signal.chirp(_T, F0_CHIRP, DURATION, F1_CHIRP, "linear") * AMPLITUDE

  -- Gausspulse
  local center = math.floor(0.3 * N_SAMPLES)
  local gauss_len = math.floor(N_SAMPLES / 2)
  local center_val = _T:to(ins.CPUPlace()):get(math.floor(gauss_len / 2))
  local t_gauss = ins.slice(_T, 1, 1, gauss_len + 1) - ins.full({ 1 }, center_val, ins.float64, _PLACE)
  local g = ins.signal.gausspulse(t_gauss, FC_GAUSS, BW_GAUSS)
  local gauss_sig = ins.zeros({ N_SAMPLES }, ins.float64, _PLACE)
  local clip_len = math.min(gauss_len, N_SAMPLES - center)
  ins.slice(gauss_sig, 1, center + 1, center + clip_len + 1):copy_from_(ins.slice(g, 1, 1, clip_len + 1))

  -- Sawtooth
  local saw_sig = ins.signal.sawtooth(_T * (2.0 * PI * FREQ_SAW), 0.5) * 0.4

  _COMPOSITE_BASE = chirp_sig + gauss_sig + saw_sig

  -- FIR bandpass
  local nyquist = FS / 2.0
  _TAPS = ins.signal.firwin(NUMTAPS, { BP_LOW / nyquist, BP_HIGH / nyquist }, "hamming", "bandpass")

  -- Gaussian smoothing kernel
  local kernel_size = math.max(3, math.floor(0.1 * N_SAMPLES))
  if kernel_size % 2 == 0 then
    kernel_size = kernel_size + 1
  end
  local x_kernel = ins.linspace(-3.0, 3.0, kernel_size, ins.float64, _PLACE)
  local kernel = ins.exp(-0.5 * x_kernel * x_kernel)
  _GAUSS_KERNEL = kernel / ins.sum(kernel)

  -- Pre-compute CWT widths list
  _CWT_WIDTHS_LIST = {}
  for w = CWT_WIDTHS_START, CWT_WIDTHS_END - 1, CWT_WIDTHS_STEP do
    _CWT_WIDTHS_LIST[#_CWT_WIDTHS_LIST + 1] = w + 0.0
  end
end

-- ============================================================
-- 运行一帧
-- ============================================================
local function run_frame(seed_val, external_noise, prof)
  -- [1] 合成信号
  local t0 = os.clock()
  ins.seed(seed_val)
  local noise
  if prof then
    prof:begin_event("randn")
  end
  if external_noise then
    noise = external_noise
  else
    noise = ins.randn({ N_SAMPLES }, ins.float64, _PLACE) * NOISE_STD
  end
  if prof then
    prof:end_event()
  end
  local composite = _COMPOSITE_BASE + noise
  local t_gen = os.clock() - t0

  -- [2] 去趋势 + 带通滤波
  t0 = os.clock()
  if prof then
    prof:begin_event("detrend")
  end
  local detrended = ins.signal.detrend(composite)
  if prof then
    prof:end_event()
  end
  if prof then
    prof:begin_event("fir_filter")
  end
  local filtered_full = ins.signal.fftconvolve(detrended, _TAPS, "full")
  if prof then
    prof:end_event()
  end
  local half = math.floor(NUMTAPS / 2)
  local filtered = ins.slice(filtered_full, 1, half + 1, half + N_SAMPLES + 1)
  local t_filt = os.clock() - t0

  -- [3] 高斯平滑
  t0 = os.clock()
  if prof then
    prof:begin_event("gauss_smooth")
  end
  local smoothed = ins.signal.fftconvolve(filtered, _GAUSS_KERNEL, "same")
  if prof then
    prof:end_event()
  end
  local t_spline = os.clock() - t0

  -- [4] FM 解调
  t0 = os.clock()
  if prof then
    prof:begin_event("hilbert")
  end
  local analytic = ins.signal.hilbert(smoothed)
  if prof then
    prof:end_event()
  end
  local inst_phase = ins.unwrap(ins.angle(analytic))
  local inst_freq = gradient(inst_phase, 1.0 / FS) / (2.0 * PI)
  local t_demod = os.clock() - t0

  -- [5] STFT
  t0 = os.clock()
  if prof then
    prof:begin_event("stft")
  end
  local stft_result = ins.signal.stft(smoothed, FS, "hann", NPERSEG, NOVERLAP)
  if prof then
    prof:end_event()
  end
  local t_stft = os.clock() - t0

  -- [6] CWT
  t0 = os.clock()
  if prof then
    prof:begin_event("cwt")
  end
  local cwt_matrix = cwt_fast(smoothed)
  if prof then
    prof:end_event()
  end
  local t_cwt = os.clock() - t0

  -- [7] 自相关
  t0 = os.clock()
  if prof then
    prof:begin_event("correlate")
  end
  local seg_len = math.min(N_SAMPLES, 512)
  local seg = ins.slice(smoothed, 1, 1, seg_len + 1)
  local autocorr_full = ins.signal.correlate(seg, seg, "full")
  if prof then
    prof:end_event()
  end
  local mid = math.floor(autocorr_full.numel / 2)
  local autocorr = ins.slice(autocorr_full, 1, mid + 1, autocorr_full.numel + 1)
  local norm_val = autocorr:to(ins.CPUPlace()):get(0)
  if math.abs(norm_val) > 1e-12 then
    autocorr = autocorr / norm_val
  end
  local t_corr = os.clock() - t0

  -- [8] 峰值查找
  t0 = os.clock()
  if prof then
    prof:begin_event("peak_finding")
  end
  local smoothed_list = smoothed:to(ins.CPUPlace()):table()
  local peaks_max, peaks_min = find_peaks_simple(smoothed_list, PEAK_ORDER)
  if prof then
    prof:end_event()
  end
  local t_peak = os.clock() - t0

  -- [9] 卡尔曼滤波
  t0 = os.clock()
  if prof then
    prof:begin_event("kalman")
  end
  local inst_freq_list = inst_freq:to(ins.CPUPlace()):table()
  local smoothed_freq = kalman_smooth(inst_freq_list)
  if prof then
    prof:end_event()
  end
  local params = {}
  for _, p in ipairs(peaks_max) do
    if p >= 1 and p <= #smoothed_freq then
      params[#params + 1] = { p, smoothed_freq[p] }
    end
  end
  local t_kalman = os.clock() - t0

  local total_ms = (t_gen + t_filt + t_spline + t_demod + t_stft + t_cwt + t_corr + t_peak + t_kalman) * 1000

  return {
    total_ms = total_ms,
    t_gen_ms = t_gen * 1000,
    t_filt_ms = t_filt * 1000,
    t_spline_ms = t_spline * 1000,
    t_demod_ms = t_demod * 1000,
    t_stft_ms = t_stft * 1000,
    t_cwt_ms = t_cwt * 1000,
    t_corr_ms = t_corr * 1000,
    t_peak_ms = t_peak * 1000,
    t_kalman_ms = t_kalman * 1000,
    n_peaks_max = #peaks_max,
    n_peaks_min = #peaks_min,
    params = params,
    smoothed = smoothed,
  }
end

-- ============================================================
-- CLI 参数解析
-- ============================================================
local function parse_args()
  local args = { device = "cpu", seed = 42, iterations = 0, timer = false, info = false, profiler = false }
  local i = 1
  while i <= #arg do
    if arg[i] == "--device" and i < #arg then
      args.device = arg[i + 1]
      i = i + 2
    elseif arg[i] == "--seed" and i < #arg then
      args.seed = tonumber(arg[i + 1])
      i = i + 2
    elseif arg[i] == "--iterations" and i < #arg then
      args.iterations = tonumber(arg[i + 1])
      i = i + 2
    elseif arg[i] == "--timer" then
      args.timer = true
      i = i + 1
    elseif arg[i] == "--info" then
      args.info = true
      i = i + 1
    elseif arg[i] == "--profiler" then
      args.profiler = true
      i = i + 1
    else
      i = i + 1
    end
  end
  return args
end

-- ============================================================
-- MAIN
-- ============================================================
local args = parse_args()
ins.init()

local n_frames = args.iterations > 0 and args.iterations or 1

print("=" .. string.rep("=", 59))
print("  多域特征提取与定位 (Insight7 Lua)")
print("=" .. string.rep("=", 59))
print(
  string.format(
    "\n[配置]  采样率: %.1f kHz  信号长度: %d 点  时长: %.1fs  设备: %s  帧数: %d",
    FS / 1e3,
    N_SAMPLES,
    DURATION,
    args.device,
    n_frames
  )
)

init_cache(args.device)

print(
  string.format(
    "\n[波形]  chirp(%.0f-%.0fHz) + gausspulse(%.0fHz) + sawtooth(%.0fHz) + noise",
    F0_CHIRP,
    F1_CHIRP,
    FC_GAUSS,
    FREQ_SAW
  )
)
print(string.format("[滤波]  FIR 带通 %.0f-%.0fHz, %d 阶", BP_LOW, BP_HIGH, NUMTAPS))
print(string.format("[特征]  FM解调 + STFT + CWT(morlet2, %d scales) + 自相关", #_CWT_WIDTHS_LIST))
print(string.format("[检测]  峰值查找(order=%d) + 卡尔曼滤波", PEAK_ORDER))

-- Device info (--info flag)
if args.info then
  local cpu_total, cpu_free = ins.device_memory_info(0, 0)
  print(
    string.format(
      "[Memory] CPU: total=%dMB used=%dMB",
      math.floor(cpu_total / (1024 * 1024)),
      math.floor((cpu_total - cpu_free) / (1024 * 1024))
    )
  )
  -- Try loading GPU backend for info display (works even with --device cpu)
  if not ins.has_device("gpu") then
    pcall(ins.load_backend, "rocm")
  end
  if ins.has_device("gpu") then
    local gpu_total, gpu_free = ins.device_memory_info(1, 0)
    print(
      string.format(
        "[Memory] GPU: total=%dMB used=%dMB",
        math.floor(gpu_total / (1024 * 1024)),
        math.floor((gpu_total - gpu_free) / (1024 * 1024))
      )
    )
  end
end

-- Profiler
local prof = nil
if args.profiler then
  prof = ins.Profiler(0, 0)
  prof:start()
end

local times = {}
local cpu_times = {}
local gpu_times = {}

if args.device == "all" then
  -- Device comparison mode
  for frame = 0, n_frames - 1 do
    -- Generate noise on CPU once (same noise for both runs)
    init_cache("cpu")
    ins.seed(args.seed + frame)
    local cpu_noise = ins.randn({ N_SAMPLES }, ins.float64, _PLACE) * NOISE_STD

    -- CPU run
    local cpu_r = run_frame(args.seed + frame, cpu_noise, prof)

    if ins.has_device("gpu") then
      -- GPU run with same noise
      local gpu_noise = cpu_noise:to(ins.GPUPlace(0))
      init_cache("gpu")
      local gpu_r = run_frame(args.seed + frame, gpu_noise, prof)

      -- Compare smoothed signal
      local cpu_smoothed = cpu_r.smoothed
      local gpu_smoothed = gpu_r.smoothed:to(ins.CPUPlace())
      local diff = ins.max(ins.abs(cpu_smoothed - gpu_smoothed))
      local max_diff = diff:item()

      cpu_times[#cpu_times + 1] = cpu_r.total_ms
      gpu_times[#gpu_times + 1] = gpu_r.total_ms

      print(
        string.format(
          "  帧 %4d/%d | CPU: %8.2f ms | GPU: %8.2f ms | max_diff=%.1e",
          frame,
          n_frames,
          cpu_r.total_ms,
          gpu_r.total_ms,
          max_diff
        )
      )
    else
      print("  [WARNING] GPU not available, running on CPU only")
      cpu_times[#cpu_times + 1] = cpu_r.total_ms
      print(string.format("  帧 %4d/%d | CPU: %8.2f ms | (GPU unavailable)", frame, n_frames, cpu_r.total_ms))
    end
  end
  -- Restore CPU cache after comparison loop
  init_cache("cpu")
else
  -- Normal single-device mode
  for frame = 0, n_frames - 1 do
    local r = run_frame(args.seed + frame, nil, prof)
    times[#times + 1] = r.total_ms

    if n_frames == 1 or frame == 0 or (frame + 1) % 10 == 0 or frame == n_frames - 1 then
      local params_str = "无"
      if #r.params > 0 then
        params_str = string.format("峰值@%d f=%.1fHz", r.params[1][1], r.params[1][2])
      end

      if args.timer then
        print(
          string.format(
            "  帧 %4d/%d | %8.2f ms | gen %5.1f filt %5.1f spl %5.1f demod %5.1f "
              .. "stft %5.1f cwt %5.1f corr %5.1f peak %5.1f kalman %5.1f | "
              .. "极大值 %d 极小值 %d | %s",
            frame,
            n_frames,
            r.total_ms,
            r.t_gen_ms,
            r.t_filt_ms,
            r.t_spline_ms,
            r.t_demod_ms,
            r.t_stft_ms,
            r.t_cwt_ms,
            r.t_corr_ms,
            r.t_peak_ms,
            r.t_kalman_ms,
            r.n_peaks_max,
            r.n_peaks_min,
            params_str
          )
        )
      else
        print(
          string.format(
            "  帧 %4d/%d | %8.2f ms | 极大值 %d 极小值 %d | %s",
            frame,
            n_frames,
            r.total_ms,
            r.n_peaks_max,
            r.n_peaks_min,
            params_str
          )
        )
      end
    end
  end
end

-- 性能总结
if args.device == "all" then
  if #cpu_times > 0 then
    local cpu_sum = 0
    for _, t in ipairs(cpu_times) do
      cpu_sum = cpu_sum + t
    end
    local cpu_avg = cpu_sum / #cpu_times
    print(string.format("\n  %s", string.rep("=", 50)))
    print(string.format("  性能总结 (CPU vs GPU)"))
    print(string.format("  %s", string.rep("=", 50)))
    print(string.format("  总帧数: %d", #cpu_times))
    print(string.format("  CPU 平均每帧: %.2f ms  FPS: %.2f", cpu_avg, 1000.0 / cpu_avg))
    if #gpu_times > 0 then
      local gpu_sum = 0
      for _, t in ipairs(gpu_times) do
        gpu_sum = gpu_sum + t
      end
      local gpu_avg = gpu_sum / #gpu_times
      print(string.format("  GPU 平均每帧: %.2f ms  FPS: %.2f", gpu_avg, 1000.0 / gpu_avg))
      print(string.format("  加速比: %.2fx", cpu_avg / gpu_avg))
    end
  end
else
  local sum = 0
  for _, t in ipairs(times) do
    sum = sum + t
  end
  local avg_ms = sum / #times
  local fps = 1000.0 / avg_ms

  print(string.format("\n  %s", string.rep("=", 50)))
  print(string.format("  性能总结 (%s)", args.device == "gpu" and "GPU" or "CPU"))
  print(string.format("  %s", string.rep("=", 50)))
  print(string.format("  总帧数: %d", n_frames))
  print(string.format("  平均每帧: %.2f ms  FPS: %.2f", avg_ms, fps))
end

if ins.has_device("gpu") and args.device ~= "gpu" and args.device ~= "all" then
  print("\n[提示] GPU 可用, 使用 --device gpu 运行 GPU 版本")
end

if args.profiler then
  prof:stop()
  prof:report()
end

print("\n完成！")
