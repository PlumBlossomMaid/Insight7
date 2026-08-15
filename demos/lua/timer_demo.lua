--- Timer Demo — Verifies the Timer API works correctly.
local ins = require("insight")

-- CPU Timer
local cpu_ok = false
local function test_cpu()
  local t = ins.Timer(0, 0)
  if t == nil then
    error("Failed to create CPU timer")
  end
  t:start()
  local start = os.clock()
  while os.clock() - start < 0.005 do
  end
  t:stop()
  local ms = t:elapsed()
  io.write(string.format("CPU timer: %.3f ms\n", ms))
  if ms < 0.0 or ms > 100.0 then
    error(string.format("CPU timer out of range: %.3f", ms))
  end
  cpu_ok = true
end

test_cpu()

-- GPU Timer (if available)
local gpu_ok, gpu_avail = pcall(ins.has_device, "gpu")
if gpu_avail then
  pcall(ins.load_backend, "rocm")
  local gpu_t = ins.Timer(1, 0)
  gpu_t:start()
  local a = ins.ones({ 256, 256 }, ins.float32, ins.GPUPlace(0))
  local b = ins.full({ 256, 256 }, 2.0, ins.float32, ins.GPUPlace(0))
  local c = a + b
  gpu_t:stop()
  local gpu_ms = gpu_t:elapsed()
  io.write(string.format("GPU timer: %.3f ms\n", gpu_ms))
else
  io.write("GPU timer: SKIPPED (GPU not available)\n")
end

io.write("OK\n")
