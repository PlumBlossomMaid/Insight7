-- Signal demod CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_demod.lua

local ok_ins, ins = pcall(require, "_insight")
if not ok_ins then
  print("SKIP: _insight module not available")
  os.exit(0)
end

local ok_gpu = pcall(function()
  ins.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

ins.set_device(ins.GPUPlace(0))

describe("Signal Demod CUDA Tests", function()
  -- Helper: build a simple FM signal
  local function make_fm(n, fs, fc, fm, dev)
    local t = {}
    local phase = 0
    local dt = 1.0 / fs
    for i = 0, n - 1 do
      local message = math.sin(2 * math.pi * fm * i * dt)
      phase = phase + 2 * math.pi * (fc + dev * message) * dt
      t[i + 1] = math.cos(phase)
    end
    return ins.from_table(t):to(ins.GPUPlace(0))
  end

  it("fm_demod", function()
    local x = make_fm(512, 256.0, 20.0, 2.0, 5.0)
    local result = ins.signal.fm_demod(x, 256.0)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("fm_demod_output_length", function()
    local x = make_fm(256, 256.0, 20.0, 2.0, 5.0)
    local result = ins.signal.fm_demod(x, 256.0)
    assert.is_not_nil(result)
    assert.are.equal(256, result.numel)
  end)

  it("fm_demod_short_signal", function()
    local x = make_fm(128, 256.0, 10.0, 1.0, 3.0)
    local result = ins.signal.fm_demod(x, 256.0)
    assert.is_not_nil(result)
    assert.are.equal(128, result.numel)
  end)
end)
