-- Signal wavelets CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_wavelets.lua

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

describe("Signal Wavelets CUDA Tests", function()
  it("morlet", function()
    local w = ins.signal.morlet(5.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)

  it("morlet_high_freq", function()
    local w = ins.signal.morlet(10.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)

  it("ricker", function()
    local w = ins.signal.ricker(100, 4.0)
    assert.is_not_nil(w)
    assert.are.equal(100, w.numel)
  end)

  it("ricker_narrow", function()
    local w = ins.signal.ricker(50, 2.0)
    assert.is_not_nil(w)
    assert.are.equal(50, w.numel)
  end)

  it("ricker_wide", function()
    local w = ins.signal.ricker(200, 10.0)
    assert.is_not_nil(w)
    assert.are.equal(200, w.numel)
  end)

  it("morlet2", function()
    local w = ins.signal.morlet2(5.0, 1.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)

  it("morlet2_scaled", function()
    local w = ins.signal.morlet2(5.0, 2.0)
    assert.is_not_nil(w)
    assert.is_true(w.numel > 0)
  end)
end)
