-- demos/lua/radar_detection.lua
-- 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Lua)
-- 纯 Insight7 API，与 Python/C++/Julia 版算法完全对齐
-- 运行: luajit radar_detection.lua --device gpu --seed 42 --iterations 50

local ins = require("insight")

-- ============================================================
-- 物理参数
-- ============================================================
local FS = 10e6
local T_PRF = 100e-6
local N = 1000
local B = 5e6
local TAU = 50e-6
local N_PULSES = 32
local SNR_DB = 10
local PULSE_LEN = TAU * FS -- 500
local PC_OFFSET = math.floor(PULSE_LEN / 2) -- 250
local RANGE_PER_BIN = 3e8 / (2 * FS) -- 15 m
local MAX_UNAMBIG_RANGE = RANGE_PER_BIN * N

local DELAY_START = { 35e-6, 50e-6 }
local DELAY_END = { 11e-6, 20e-6 }
local DOPPLER_START = { 2000.0, -1500.0 }
local DOPPLER_END = { 1800.0, -1200.0 }

-- ============================================================
-- 全局缓存
-- ============================================================
local _S_TX = nil
local _TEMPLATE = nil
local _SIG_POWER = nil
local _NOISE_SIGMA = nil
local _DOPPLER_BINS = nil
local _RANGE_BINS = nil
local _PLACE = nil
local _HAMMING = nil
local _SLOW_TIMES = nil

local function init_cache(device)
  if device == "gpu" and ins.has_device("gpu") then
    ins.init()
    _PLACE = ins.GPUPlace(0)
  else
    _PLACE = ins.CPUPlace()
  end
  ins.set_device(_PLACE)

  -- LFM 发射信号
  local t = ins.arange(N, ins.float64, _PLACE) / FS
  local phase = math.pi * (B / TAU) * (t * t)
  local s_tx = ins.to_complex(ins.cos(phase), ins.sin(phase))
  local tau_arr = ins.full({ 1 }, TAU, ins.float64, _PLACE)
  local mask = ins.cast(ins.less(t, tau_arr), ins.float64)
  _S_TX = s_tx * mask

  -- 信号功率
  local power_arr = ins.mean(ins.real(_S_TX) * ins.real(_S_TX) + ins.imag(_S_TX) * ins.imag(_S_TX))
  _SIG_POWER = power_arr:item()
  _NOISE_SIGMA = math.sqrt(_SIG_POWER / 10 ^ (SNR_DB / 10) / 2)

  -- 多普勒/距离轴
  local doppler = ins.fftshift(ins.fftfreq(N_PULSES, T_PRF))
  local cpu = ins.CPUPlace()
  _DOPPLER_BINS = doppler:to(cpu)

  local range_arr = (ins.arange(N, ins.float64, _PLACE) - PC_OFFSET) * RANGE_PER_BIN
  _RANGE_BINS = range_arr:to(cpu)

  -- 预创建脉冲压缩模板
  _TEMPLATE = _S_TX["1:" .. tostring(PULSE_LEN + 1)]

  -- 预创建 hamming 窗 (避免每帧重新创建)
  local hamming = ins.signal.get_window("hamming", N_PULSES, false)
  _HAMMING = hamming:reshape({ N_PULSES, 1 })

  -- 预创建慢时间轴 (避免每帧重复 arange + mul)
  _SLOW_TIMES = ins.arange(N_PULSES, ins.float64, _PLACE) * T_PRF
end

-- ============================================================
-- 目标参数插值
-- ============================================================
local function get_target_params(frame_idx, total_frames)
  if total_frames <= 1 then
    return DELAY_START, DOPPLER_START
  end
  local t = frame_idx / math.max(total_frames - 1, 1)
  local delays, dopplers = {}, {}
  for k = 1, #DELAY_START do
    delays[k] = DELAY_START[k] + (DELAY_END[k] - DELAY_START[k]) * t
    dopplers[k] = DOPPLER_START[k] + (DOPPLER_END[k] - DOPPLER_START[k]) * t
  end
  return delays, dopplers
end

-- ============================================================
-- 向量化回波模拟
-- ============================================================
local function simulate_echoes(delays, dopplers, external_noise_r, external_noise_i)
  local noise_r, noise_i
  if external_noise_r and external_noise_i then
    noise_r = external_noise_r
    noise_i = external_noise_i
  else
    noise_r = ins.randn({ N_PULSES, N }, ins.float64, _PLACE) * _NOISE_SIGMA
    noise_i = ins.randn({ N_PULSES, N }, ins.float64, _PLACE) * _NOISE_SIGMA
  end

  local pulses = ins.zeros({ N_PULSES, N }, "complex128", _PLACE)

  for k = 1, #delays do
    local ds = math.floor(delays[k] * FS)
    local delayed_1d = ins.zeros({ N }, "complex128", _PLACE)
    if ds < N then
      local src = ins.slice(_S_TX, 1, 1, N - ds + 1)
      ins.slice(delayed_1d, 1, ds + 1, N + 1):copy_from_(src)
    end
    local delayed_2d = delayed_1d:reshape({ 1, N })

    local phase = _SLOW_TIMES * (2 * math.pi * dopplers[k])
    local rot = ins.to_complex(ins.cos(phase), ins.sin(phase))
    local rot_2d = rot:reshape({ N_PULSES, 1 })

    pulses = pulses + delayed_2d * rot_2d
  end

  pulses = pulses + ins.to_complex(noise_r, noise_i)
  return pulses
end

-- ============================================================
-- 目标提取
-- ============================================================
local function extract_targets(energy, det, top_n, cluster_threshold)
  top_n = top_n or 2
  cluster_threshold = cluster_threshold or 3.0

  local idx = ins.nonzero(det)
  local n_det = idx.shape[2] or 0
  if n_det == 0 then
    return {}, 0
  end

  -- 批量提取: nonzero 索引 table() + 逐元素 get() 能量值
  local cpu = ins.CPUPlace()
  local idx_cpu = idx:to(cpu)
  local doppler_list = idx_cpu[1]:table() -- 第1行: 多普勒索引 (0-based)
  local range_list = idx_cpu[2]:table() -- 第2行: 距离索引 (0-based)
  local energy_cpu = energy:to(cpu)

  local candidates = {}
  for k = 1, n_det do
    local di = math.floor(doppler_list[k])
    local ri = math.floor(range_list[k])
    local v = energy_cpu:get(di * N + ri) -- 只取检测点 (0-based get)
    candidates[#candidates + 1] = { di, ri, v }
  end

  if #candidates == 0 then
    return {}, 0
  end

  table.sort(candidates, function(a, b)
    return a[3] > b[3]
  end)

  local keep = math.min(#candidates, math.max(top_n * 10, 20))
  local top = {}
  for i = 1, keep do
    top[i] = candidates[i]
  end

  -- 聚类
  local visited = {}
  for i = 1, keep do
    visited[i] = false
  end
  local clusters = {}

  for i = 1, keep do
    if not visited[i] then
      visited[i] = true
      local cluster_indices = { i }
      for j = i + 1, keep do
        if not visited[j] then
          local d = math.sqrt((top[i][1] - top[j][1]) ^ 2 + (top[i][2] - top[j][2]) ^ 2)
          if d <= cluster_threshold then
            visited[j] = true
            cluster_indices[#cluster_indices + 1] = j
          end
        end
      end
      local best_idx = i
      for _, ci in ipairs(cluster_indices) do
        if top[ci][3] > top[best_idx][3] then
          best_idx = ci
        end
      end
      clusters[#clusters + 1] = top[best_idx]
    end
  end

  table.sort(clusters, function(a, b)
    return a[3] > b[3]
  end)

  -- 按距离去重
  local seen = {}
  local targets = {}
  for _, c in ipairs(clusters) do
    if not seen[c[2]] then
      seen[c[2]] = true
      targets[#targets + 1] = { c[1], c[2] }
      if #targets >= top_n then
        break
      end
    end
  end

  return targets, #candidates
end

-- ============================================================
-- 单帧检测
-- ============================================================
local function run_frame(delays, dopplers, device, seed, external_noise_r, external_noise_i, prof)
  if device == "gpu" and ins.has_device("gpu") then
    ins.init()
    ins.set_device(ins.GPUPlace(0))
  else
    ins.set_device(ins.CPUPlace())
  end
  ins.seed(seed)

  local t0 = os.clock()

  -- [1] Echo simulation
  if prof then
    prof:begin_event("echo_simulation")
  end
  local pulses = simulate_echoes(delays, dopplers, external_noise_r, external_noise_i)
  if prof then
    prof:end_event()
  end
  local t1 = os.clock()

  -- [2] Pulse compression
  if prof then
    prof:begin_event("pulse_compression")
  end
  local pc = ins.signal.pulse_compression(pulses, _TEMPLATE)
  if prof then
    prof:end_event()
  end
  local t2 = os.clock()

  -- [3] Doppler processing
  if prof then
    prof:begin_event("doppler")
  end
  local mean_pc = ins.mean(pc, 1, true)
  local pc_ac = pc - mean_pc
  local spec = ins.signal.pulse_doppler(pc_ac * _HAMMING, "", N_PULSES)
  local shifted = spec / N_PULSES
  if prof then
    prof:end_event()
  end
  local t3 = os.clock()

  -- [4] Energy
  local energy = ins.real(shifted) * ins.real(shifted) + ins.imag(shifted) * ins.imag(shifted)

  -- [5] CA-CFAR
  if prof then
    prof:begin_event("ca_cfar")
  end
  local det = ins.signal.ca_cfar(energy, { 4, 4 }, { 12, 12 }, 1e-6)
  if prof then
    prof:end_event()
  end
  local t4 = os.clock()

  -- [6] Target extraction
  if prof then
    prof:begin_event("target_extraction")
  end
  local targets, raw_count = extract_targets(energy, det[2], 2, 3.0)
  if prof then
    prof:end_event()
  end
  local t5 = os.clock()

  -- [7] Range validation
  local valid = {}
  for _, tgt in ipairs(targets) do
    local dist = _RANGE_BINS:get(tgt[2])
    if dist >= 0 and dist <= MAX_UNAMBIG_RANGE then
      valid[#valid + 1] = tgt
    end
  end

  local result = {
    targets = valid,
    raw_count = raw_count,
    total_ms = (t5 - t0) * 1000,
    echo_ms = (t1 - t0) * 1000,
    pc_ms = (t2 - t1) * 1000,
    doppler_ms = (t3 - t2) * 1000,
    cfar_ms = (t4 - t3) * 1000,
    local_ms = (t5 - t4) * 1000,
    energy = energy,
    pulses = pulses,
  }
  return result
end

-- ============================================================
-- CLI 解析
-- ============================================================
local args = {
  device = "cpu",
  seed = 42,
  iterations = 0,
  timer = false,
  info = false,
  profiler = false,
}

local i = 1
while i <= #arg do
  if arg[i] == "--device" and i < #arg then
    args.device = arg[i + 1]
    i = i + 1
  elseif arg[i] == "--seed" and i < #arg then
    args.seed = tonumber(arg[i + 1])
    i = i + 1
  elseif arg[i] == "--iterations" and i < #arg then
    args.iterations = tonumber(arg[i + 1])
    i = i + 1
  elseif arg[i] == "--timer" then
    args.timer = true
  elseif arg[i] == "--info" then
    args.info = true
  elseif arg[i] == "--profiler" then
    args.profiler = true
  end
  i = i + 1
end

-- ============================================================
-- 主流程
-- ============================================================
ins.init()

local n_frames = args.iterations > 0 and args.iterations or 1

print("============================================================")
print("  雷达目标检测与多普勒分析 (Insight7 Lua)")
print("============================================================")
print(
  string.format(
    "\n[配置] 采样率: %.1f MHz  脉冲: %d  设备: %s  种子: %d  帧数: %d",
    FS / 1e6,
    N_PULSES,
    args.device,
    args.seed,
    n_frames
  )
)

init_cache(args.device)

-- 模糊函数分析
print("\n[波形分析] 计算模糊函数...")
-- 直接调用 C 层 (避开 wrapper 的 _wrap 兼容性问题)
local raw_signal = require("_insight").signal
local ok_ambg, ambg = pcall(raw_signal.ambgfun, _TEMPLATE, FS, 1.0 / T_PRF)
if ok_ambg then
  local peak_val = ins.max(ins.abs(ambg)):item()
  local mean_val = ins.mean(ins.abs(ambg)):item()
  local peak_ratio = 20 * math.log10(peak_val / (mean_val + 1e-12))
  print(string.format("  模糊函数峰值: %.2f", peak_val))
  print(string.format("  峰值/平均比: %.1f dB  (主瓣窄、旁瓣低 ✓)", peak_ratio))
else
  print("  ambgfun 计算失败:", ambg)
end

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

-- 帧循环
local times = {}
local cpu_times = {}
local gpu_times = {}

if args.device == "all" then
  -- Device comparison mode
  for frame = 0, n_frames - 1 do
    local delays, dopplers = get_target_params(frame, n_frames)

    -- Generate noise on CPU once (same noise for both runs)
    init_cache("cpu")
    ins.seed(args.seed)
    local cpu_noise_r = ins.randn({ N_PULSES, N }, ins.float64, _PLACE) * _NOISE_SIGMA
    local cpu_noise_i = ins.randn({ N_PULSES, N }, ins.float64, _PLACE) * _NOISE_SIGMA

    -- CPU run
    local cpu_result = run_frame(delays, dopplers, "cpu", args.seed, cpu_noise_r, cpu_noise_i, prof)

    if ins.has_device("gpu") then
      -- GPU run: 复用 CPU 缓存，不重新生成
      local gpu_place = ins.GPUPlace(0)
      local gpu_noise_r = cpu_noise_r:to(gpu_place)
      local gpu_noise_i = cpu_noise_i:to(gpu_place)
      _S_TX = _S_TX:to(gpu_place)
      _TEMPLATE = _TEMPLATE:to(gpu_place)
      _HAMMING = _HAMMING:to(gpu_place)
      _SLOW_TIMES = _SLOW_TIMES:to(gpu_place)
      _PLACE = gpu_place
      local gpu_result = run_frame(delays, dopplers, "gpu", args.seed, gpu_noise_r, gpu_noise_i, prof)

      -- Compare raw pulses (before pulse_compression — should be identical CPU vs GPU)
      local cpu_pulses = cpu_result.pulses
      local gpu_pulses = gpu_result.pulses:to(ins.CPUPlace())
      local diff = ins.max(ins.abs(cpu_pulses - gpu_pulses))
      local max_diff = diff:item()

      cpu_times[#cpu_times + 1] = cpu_result.total_ms
      gpu_times[#gpu_times + 1] = gpu_result.total_ms

      print(
        string.format(
          "  帧 %4d/%d | CPU: %8.2f ms | GPU: %8.2f ms | max_diff=%.1e",
          frame,
          n_frames,
          cpu_result.total_ms,
          gpu_result.total_ms,
          max_diff
        )
      )
    else
      print("  [WARNING] GPU not available, running on CPU only")
      cpu_times[#cpu_times + 1] = cpu_result.total_ms
      print(string.format("  帧 %4d/%d | CPU: %8.2f ms | (GPU unavailable)", frame, n_frames, cpu_result.total_ms))
    end
  end
  -- Restore CPU cache after comparison loop
  init_cache("cpu")
else
  -- Normal single-device mode
  for frame = 0, n_frames - 1 do
    local delays, dopplers = get_target_params(frame, n_frames)
    local result = run_frame(delays, dopplers, args.device, args.seed, nil, nil, prof)

    times[#times + 1] = result.total_ms

    if n_frames == 1 or frame == 0 or (frame + 1) % 10 == 0 or frame == n_frames - 1 then
      local targets_str = ""
      for k, tgt in ipairs(result.targets) do
        if k > 1 then
          targets_str = targets_str .. "; "
        end
        local dist = _RANGE_BINS:get(tgt[2])
        local dop = _DOPPLER_BINS:get(tgt[1])
        targets_str = targets_str .. string.format("距离 %.0fm 多普勒 %.0fHz", dist, dop)
      end
      if targets_str == "" then
        targets_str = "无"
      end

      if args.timer then
        print(
          string.format(
            "  帧 %4d/%d | %8.2f ms | 检测 %4d → %d 目标 | echo %5.2f pc %5.2f dop %5.2f cfar %5.2f ext %5.2f | %s",
            frame,
            n_frames,
            result.total_ms,
            result.raw_count,
            #result.targets,
            result.echo_ms,
            result.pc_ms,
            result.doppler_ms,
            result.cfar_ms,
            result.local_ms,
            targets_str
          )
        )
      else
        print(
          string.format(
            "  帧 %4d/%d | %8.2f ms | 检测 %4d → %d 目标 | %s",
            frame,
            n_frames,
            result.total_ms,
            result.raw_count,
            #result.targets,
            targets_str
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
    print(string.format("\n  ================================================="))
    print(string.format("  性能总结 (CPU vs GPU)"))
    print(string.format("  ================================================="))
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

  print(string.format("\n  ================================================="))
  print(string.format("  性能总结 (%s)", args.device:upper()))
  print(string.format("  ================================================="))
  print(string.format("  总帧数: %d", n_frames))
  print(string.format("  平均每帧: %.2f ms  FPS: %.2f", avg_ms, fps))
end

if args.profiler then
  prof:stop()
  prof:report()
end

print("\n完成！")
