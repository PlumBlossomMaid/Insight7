-- Signal Waveforms CUDA binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_waveforms.lua

local ok_gpu = pcall(function()
  local native = require("_insight")
  native.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

describe("Signal Waveforms CUDA Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Sawtooth (3 tests)
  -- ========================================================================

  it("sawtooth_basic", function()
    local t = ins.from_table({ 0.0, math.pi, 2.0 * math.pi - 1e-10 })
    local y = ins.signal.sawtooth(t, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(-1.0, ins.item(y, 0), 1e-6)
    assert.near(0.0, ins.item(y, 1), 1e-6)
    assert.near(1.0, ins.item(y, 2), 1e-6)
  end)

  it("sawtooth_triangle", function()
    local t = ins.from_table({ 0.0, math.pi / 2.0, math.pi })
    local y = ins.signal.sawtooth(t, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(-1.0, ins.item(y, 0), 1e-6)
    assert.near(0.0, ins.item(y, 1), 1e-6)
    assert.near(1.0, ins.item(y, 2), 1e-6)
  end)

  it("sawtooth_periodicity", function()
    local t = ins.from_table({ 0.5, 0.5 + 2.0 * math.pi, 0.5 + 4.0 * math.pi })
    local y = ins.signal.sawtooth(t, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(ins.item(y, 0), ins.item(y, 1), 1e-6)
    assert.near(ins.item(y, 0), ins.item(y, 2), 1e-6)
  end)

  -- ========================================================================
  -- Square (2 tests)
  -- ========================================================================

  it("square_basic", function()
    local t = ins.from_table({ 0.0, math.pi, 2.0 * math.pi - 1e-10 })
    local y = ins.signal.square(t, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(1.0, ins.item(y, 0), 1e-6)
    assert.near(-1.0, ins.item(y, 1), 1e-6)
    assert.near(-1.0, ins.item(y, 2), 1e-6)
  end)

  it("square_periodicity", function()
    local t = ins.from_table({ 1.0, 1.0 + 2.0 * math.pi, 1.0 + 4.0 * math.pi })
    local y = ins.signal.square(t, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(ins.item(y, 0), ins.item(y, 1), 1e-6)
    assert.near(ins.item(y, 0), ins.item(y, 2), 1e-6)
  end)

  -- ========================================================================
  -- Gausspulse (2 tests)
  -- ========================================================================

  it("gausspulse_peak", function()
    local t = ins.from_table({ 0.0 })
    local y = ins.signal.gausspulse(t, 1000.0, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(1.0, ins.item(y, 0), 1e-6)
  end)

  it("gausspulse_decay", function()
    local t = ins.from_table({ 0.0, 0.001, 0.01 })
    local y = ins.signal.gausspulse(t, 1000.0, 0.5)
    assert.is_not_nil(y)
    assert.are.equal(3, y.numel)
    assert.near(1.0, ins.item(y, 0), 1e-6)
    -- Decay: |y[0]| > |y[2]|
    assert.is_true(math.abs(ins.item(y, 0)) > math.abs(ins.item(y, 2)))
  end)

  -- ========================================================================
  -- Chirp (2 tests)
  -- ========================================================================

  it("chirp_linear", function()
    local t = ins.from_table({ 0.0 })
    local y = ins.signal.chirp(t, 6.0, 1.0, 10.0, "linear", 0)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(1.0, ins.item(y, 0), 1e-6)
  end)

  it("chirp_with_phase", function()
    local t = ins.from_table({ 0.0 })
    local y = ins.signal.chirp(t, 6.0, 1.0, 10.0, "linear", math.pi / 2.0)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(0.0, ins.item(y, 0), 1e-6)
  end)

  -- ========================================================================
  -- Unit Impulse (1 test)
  -- ========================================================================

  it("unit_impulse_center", function()
    local y = ins.signal.unit_impulse({ 11 })
    assert.is_not_nil(y)
    assert.are.equal(11, y.numel)
    assert.near(1.0, ins.item(y, 5), 1e-10)
    assert.near(0.0, ins.item(y, 0), 1e-10)
    assert.near(0.0, ins.item(y, 10), 1e-10)
  end)
end)
