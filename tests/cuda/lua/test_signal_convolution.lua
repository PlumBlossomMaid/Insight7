-- Signal Convolution CUDA binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_convolution.lua

local ok_gpu = pcall(function()
  local native = require("_insight")
  native.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

describe("Signal Convolution CUDA Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- FFT Convolve (5 tests)
  -- ========================================================================

  it("fftconvolve_full", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 1.0 })
    local c = ins.signal.fftconvolve(a, b, "full")
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
    -- [1, 3, 5, 3]
    assert.near(1.0, ins.item(c, 0), 1e-10)
    assert.near(3.0, ins.item(c, 1), 1e-10)
    assert.near(5.0, ins.item(c, 2), 1e-10)
    assert.near(3.0, ins.item(c, 3), 1e-10)
  end)

  it("fftconvolve_same", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local b = ins.from_table({ 1.0, 1.0, 1.0 })
    local c = ins.signal.fftconvolve(a, b, "same")
    assert.is_not_nil(c)
    assert.are.equal(5, c.numel)
    -- [3, 6, 9, 12, 9]
    assert.near(3.0, ins.item(c, 0), 1e-10)
    assert.near(6.0, ins.item(c, 1), 1e-10)
    assert.near(9.0, ins.item(c, 2), 1e-10)
    assert.near(12.0, ins.item(c, 3), 1e-10)
    assert.near(9.0, ins.item(c, 4), 1e-10)
  end)

  it("fftconvolve_valid", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local b = ins.from_table({ 1.0, 1.0, 1.0 })
    local c = ins.signal.fftconvolve(a, b, "valid")
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
    -- [6, 9, 12]
    assert.near(6.0, ins.item(c, 0), 1e-10)
    assert.near(9.0, ins.item(c, 1), 1e-10)
    assert.near(12.0, ins.item(c, 2), 1e-10)
  end)

  it("fftconvolve_commutative", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 4.0, 5.0 })
    local c1 = ins.signal.fftconvolve(a, b, "full")
    local c2 = ins.signal.fftconvolve(b, a, "full")
    assert.is_not_nil(c1)
    assert.is_not_nil(c2)
    assert.are.equal(c1.numel, c2.numel)
    for i = 0, c1.numel - 1 do
      assert.near(ins.item(c1, i), ins.item(c2, i), 1e-10)
    end
  end)

  it("fftconvolve_impulse", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local b = ins.from_table({ 1.0 })
    local c = ins.signal.fftconvolve(a, b, "full")
    assert.is_not_nil(c)
    assert.are.equal(5, c.numel)
    for i = 0, 4 do
      assert.near(i + 1.0, ins.item(c, i), 1e-10)
    end
  end)

  -- ========================================================================
  -- Correlate (3 tests)
  -- ========================================================================

  it("correlate_full", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 1.0 })
    local c = ins.signal.correlate(a, b, "full")
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
    assert.near(1.0, ins.item(c, 0), 1e-10)
    assert.near(3.0, ins.item(c, 1), 1e-10)
    assert.near(5.0, ins.item(c, 2), 1e-10)
    assert.near(3.0, ins.item(c, 3), 1e-10)
  end)

  it("correlate_same", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local b = ins.from_table({ 1.0, 1.0, 1.0 })
    local c = ins.signal.correlate(a, b, "same")
    assert.is_not_nil(c)
    assert.are.equal(5, c.numel)
  end)

  it("correlate_autocorrelation", function()
    local a = ins.from_table({ 1.0, 0.0, -1.0 })
    local c = ins.signal.correlate(a, a, "full")
    assert.is_not_nil(c)
    assert.are.equal(5, c.numel)
    assert.near(-1.0, ins.item(c, 0), 1e-10)
    assert.near(2.0, ins.item(c, 2), 1e-10)
    assert.near(-1.0, ins.item(c, 4), 1e-10)
  end)

  -- ========================================================================
  -- Convolve2d (2 tests)
  -- ========================================================================

  it("convolve2d_basic", function()
    local a = ins.from_table({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } })
    local b = ins.from_table({ { 1.0, 0.0 }, { 0.0, 1.0 } })
    local c = ins.signal.convolve2d(a, b, "full")
    assert.is_not_nil(c)
    -- Full 2D: (3+2-1) x (3+2-1) = 4x4
    assert.are.equal(16, c.numel)
  end)

  it("convolve2d_same", function()
    local a = ins.from_table({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } })
    local b = ins.from_table({ { 1.0, 0.0 }, { 0.0, 1.0 } })
    local c = ins.signal.convolve2d(a, b, "same")
    assert.is_not_nil(c)
    assert.are.equal(9, c.numel)
  end)

  -- ========================================================================
  -- Correlate2d (1 test)
  -- ========================================================================

  it("correlate2d_valid", function()
    local a = ins.from_table({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } })
    local b = ins.from_table({ { 1.0, 1.0 }, { 1.0, 1.0 } })
    local c = ins.signal.correlate2d(a, b, "valid")
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  -- ========================================================================
  -- Choose Conv Method (1 test)
  -- ========================================================================

  it("choose_conv_method_small", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 1.0 })
    local method = ins.signal.choose_conv_method(a, b, "full")
    assert.are.equal("direct", method)
  end)

  -- ========================================================================
  -- Correlation Lags (2 tests)
  -- ========================================================================

  it("correlation_lags_full", function()
    local lags = ins.signal.correlation_lags(5, 3, "full")
    assert.is_not_nil(lags)
    assert.are.equal(7, lags.numel)
    assert.near(-2.0, ins.item(lags, 0), 1e-10)
    assert.near(4.0, ins.item(lags, 6), 1e-10)
  end)

  it("correlation_lags_same", function()
    local lags = ins.signal.correlation_lags(5, 3, "same")
    assert.is_not_nil(lags)
    assert.are.equal(5, lags.numel)
    assert.near(-1.0, ins.item(lags, 0), 1e-10)
    assert.near(3.0, ins.item(lags, 4), 1e-10)
  end)
end)
