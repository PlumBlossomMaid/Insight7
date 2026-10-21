-- Signal Filtering CUDA binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_filtering.lua

local ok_gpu = pcall(function()
  local native = require("_insight")
  native.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

describe("Signal Filtering CUDA Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Hilbert (3 tests)
  -- ========================================================================

  it("hilbert_basic", function()
    -- Build a cosine signal
    local N = 64
    local data = {}
    for i = 0, N - 1 do
      data[i + 1] = math.cos(2.0 * math.pi * i / N)
    end
    local x = ins.from_table(data)
    local y = ins.signal.hilbert(x, N)
    assert.is_not_nil(y)
    assert.are.equal(N, y.numel)
  end)

  it("hilbert_dc", function()
    local data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
    local x = ins.from_table(data)
    local y = ins.signal.hilbert(x, 8)
    assert.is_not_nil(y)
    assert.are.equal(8, y.numel)
  end)

  it("hilbert_default_n", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.signal.hilbert(x)
    assert.is_not_nil(y)
    assert.are.equal(4, y.numel)
  end)

  -- ========================================================================
  -- Detrend (3 tests)
  -- ========================================================================

  it("detrend_constant", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local y = ins.signal.detrend(x, -1, "constant")
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    -- Mean = 3.0, detrended = [-2, -1, 0, 1, 2]
    assert.near(-2.0, ins.item(y, 0), 1e-10)
    assert.near(0.0, ins.item(y, 2), 1e-10)
    assert.near(2.0, ins.item(y, 4), 1e-10)
  end)

  it("detrend_linear", function()
    local x = ins.from_table({ 0.0, 3.0, 6.0, 9.0, 12.0 })
    local y = ins.signal.detrend(x, -1, "linear")
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    for i = 0, 4 do
      assert.near(0.0, ins.item(y, i), 1e-10)
    end
  end)

  it("detrend_linear_noisy", function()
    local x = ins.from_table({ 0.1, 3.0, 5.9, 9.1, 12.0 })
    local y = ins.signal.detrend(x, -1, "linear")
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    for i = 0, 4 do
      assert.near(0.0, ins.item(y, i), 0.2)
    end
  end)

  -- ========================================================================
  -- Lfilter (3 tests)
  -- ========================================================================

  it("lfilter_fir_basic", function()
    local b = ins.from_table({ 1.0, 1.0 })
    local a = ins.from_table({ 1.0 })
    local x = ins.from_table({ 1.0, 0.0, 0.0, 0.0, 0.0 })
    local y = ins.signal.lfilter(b, a, x)
    assert.is_not_nil(y)
    assert.are.equal(6, y.numel)
    assert.near(1.0, ins.item(y, 0), 1e-10)
    assert.near(1.0, ins.item(y, 1), 1e-10)
    for i = 2, 5 do
      assert.near(0.0, ins.item(y, i), 1e-10)
    end
  end)

  it("lfilter_fir_moving_avg", function()
    local b = ins.from_table({ 1.0 / 3, 1.0 / 3, 1.0 / 3 })
    local a = ins.from_table({ 1.0 })
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local y = ins.signal.lfilter(b, a, x)
    assert.is_not_nil(y)
    assert.near(1.0, ins.item(y, 1), 1e-10)
    assert.near(2.0, ins.item(y, 2), 1e-10)
    assert.near(3.0, ins.item(y, 3), 1e-10)
    assert.near(4.0, ins.item(y, 4), 1e-10)
  end)

  it("lfilter_iir_impulse", function()
    local b = ins.from_table({ 1.0 })
    local a = ins.from_table({ 1.0, -0.5 })
    local x = ins.from_table({ 1.0, 0.0, 0.0, 0.0, 0.0 })
    local y = ins.signal.lfilter(b, a, x)
    assert.is_not_nil(y)
    assert.near(1.0, ins.item(y, 0), 1e-10)
    assert.near(0.5, ins.item(y, 1), 1e-10)
    assert.near(0.25, ins.item(y, 2), 1e-10)
    assert.near(0.125, ins.item(y, 3), 1e-10)
    assert.near(0.0625, ins.item(y, 4), 1e-10)
  end)

  -- ========================================================================
  -- Filtfilt (1 test)
  -- ========================================================================

  it("filtfilt_fir", function()
    local b = ins.from_table({ 1.0, 1.0 })
    local a = ins.from_table({ 1.0 })
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local y = ins.signal.filtfilt(b, a, x)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
  end)

  -- ========================================================================
  -- Freq Shift (2 tests)
  -- ========================================================================

  it("freq_shift_zero", function()
    local N = 64
    local data = {}
    for i = 1, N do
      data[i] = 1.0
    end
    local x = ins.from_table(data)
    local y = ins.signal.freq_shift(x, 0.0, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(N, y.numel)
  end)

  it("freq_shift_positive", function()
    local N = 16
    local data = {}
    for i = 1, N do
      data[i] = 1.0
    end
    local x = ins.from_table(data)
    local y = ins.signal.freq_shift(x, 0.25, 1.0)
    assert.is_not_nil(y)
    assert.are.equal(N, y.numel)
  end)

  -- ========================================================================
  -- Decimate (2 tests)
  -- ========================================================================

  it("decimate_basic", function()
    local N = 100
    local data = {}
    for i = 0, N - 1 do
      data[i + 1] = math.sin(2.0 * math.pi * i / N)
    end
    local x = ins.from_table(data)
    local y = ins.signal.decimate(x, 2)
    assert.is_not_nil(y)
    assert.are.equal(50, y.numel)
  end)

  it("decimate_zero_phase", function()
    local N = 100
    local data = {}
    for i = 0, N - 1 do
      data[i + 1] = math.sin(2.0 * math.pi * i / N)
    end
    local x = ins.from_table(data)
    local y = ins.signal.decimate(x, 4)
    assert.is_not_nil(y)
    assert.are.equal(25, y.numel)
  end)

  -- ========================================================================
  -- Resample (2 tests)
  -- ========================================================================

  it("resample_identity", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local y = ins.signal.resample(x, 5)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    for i = 0, 4 do
      assert.near(i + 1.0, ins.item(y, i), 1e-6)
    end
  end)

  it("resample_upsample", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 })
    local y = ins.signal.resample(x, 8)
    assert.is_not_nil(y)
    assert.are.equal(8, y.numel)
  end)

  -- ========================================================================
  -- Resample Poly (2 tests)
  -- ========================================================================

  it("resample_poly_identity", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local y = ins.signal.resample_poly(x, 1, 1)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    for i = 0, 4 do
      assert.near(i + 1.0, ins.item(y, i), 1e-10)
    end
  end)

  it("resample_poly_upsample", function()
    local x = ins.from_table({ 1.0, 0.0, 1.0, 0.0 })
    local y = ins.signal.resample_poly(x, 2, 1)
    assert.is_not_nil(y)
    assert.is_true(y.numel >= 7)
    assert.is_true(y.numel <= 9)
  end)

  -- ========================================================================
  -- Firfilter (1 test)
  -- ========================================================================

  it("firfilter_basic", function()
    local b = ins.from_table({ 1.0, 2.0, 1.0 })
    local x = ins.from_table({ 1.0, 0.0, 0.0, 0.0, 0.0 })
    local y = ins.signal.firfilter(b, x)
    assert.is_not_nil(y)
    assert.are.equal(7, y.numel)
    assert.near(1.0, ins.item(y, 0), 1e-10)
    assert.near(2.0, ins.item(y, 1), 1e-10)
    assert.near(1.0, ins.item(y, 2), 1e-10)
    for i = 3, 6 do
      assert.near(0.0, ins.item(y, i), 1e-10)
    end
  end)
end)
