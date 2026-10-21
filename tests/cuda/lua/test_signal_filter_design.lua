-- Signal Filter Design CUDA binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_filter_design.lua

local ok_gpu = pcall(function()
  local native = require("_insight")
  native.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

describe("Signal Filter Design CUDA Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Kaiser Beta (4 tests)
  -- ========================================================================

  it("kaiser_beta_low", function()
    -- a <= 21 -> beta = 0
    assert.are.equal(0.0, ins.signal.kaiser_beta(0.0))
    assert.are.equal(0.0, ins.signal.kaiser_beta(21.0))
  end)

  it("kaiser_beta_mid", function()
    local a = 40.0
    local expected = 0.5842 * (a - 21.0) ^ 0.4 + 0.07886 * (a - 21.0)
    assert.near(expected, ins.signal.kaiser_beta(a), 1e-10)
  end)

  it("kaiser_beta_high", function()
    local a = 60.0
    local expected = 0.1102 * (a - 8.7)
    assert.near(expected, ins.signal.kaiser_beta(a), 1e-10)
  end)

  it("kaiser_beta_boundary", function()
    local beta = ins.signal.kaiser_beta(50.0)
    assert.is_true(beta > 4.0)
    assert.is_true(beta < 6.0)
  end)

  -- ========================================================================
  -- Kaiser Atten (1 test)
  -- ========================================================================

  it("kaiser_atten_basic", function()
    local atten = ins.signal.kaiser_atten(101, 0.1)
    local expected = 2.285 * 100.0 * math.pi * 0.1 + 7.95
    assert.near(expected, atten, 1e-10)
  end)

  -- ========================================================================
  -- Firwin (5 tests)
  -- ========================================================================

  it("firwin_lowpass_basic", function()
    local h = ins.signal.firwin(11, { 0.3 }, "lowpass", true)
    assert.is_not_nil(h)
    assert.are.equal(11, h.numel)
    -- Symmetric
    for i = 0, 4 do
      assert.near(ins.item(h, i), ins.item(h, 10 - i), 1e-10)
    end
  end)

  it("firwin_highpass_basic", function()
    local h = ins.signal.firwin(11, { 0.3 }, "highpass", true)
    assert.is_not_nil(h)
    assert.are.equal(11, h.numel)
  end)

  it("firwin_bandpass", function()
    local h = ins.signal.firwin(21, { 0.2, 0.5 }, "bandpass", false)
    assert.is_not_nil(h)
    assert.are.equal(21, h.numel)
  end)

  it("firwin_bandstop", function()
    local h = ins.signal.firwin(21, { 0.2, 0.5 }, "bandstop", false)
    assert.is_not_nil(h)
    assert.are.equal(21, h.numel)
  end)

  it("firwin_different_windows", function()
    local h1 = ins.signal.firwin(11, { 0.3 }, "lowpass", false)
    local h2 = ins.signal.firwin(11, { 0.3 }, "highpass", false)
    assert.is_not_nil(h1)
    assert.is_not_nil(h2)
    assert.are.equal(11, h1.numel)
    assert.are.equal(11, h2.numel)
    -- Different filter types should give different results
    assert.are_not.equal(ins.item(h1, 0), ins.item(h2, 0))
  end)

  -- ========================================================================
  -- Firwin2 (2 tests)
  -- ========================================================================

  it("firwin2_lowpass", function()
    local h = ins.signal.firwin2(15, { 0.0, 0.5, 0.5, 1.0 }, { 1.0, 1.0, 0.0, 0.0 })
    assert.is_not_nil(h)
    assert.are.equal(15, h.numel)
  end)

  it("firwin2_different_lengths", function()
    local h_short = ins.signal.firwin2(7, { 0.0, 0.5, 1.0 }, { 1.0, 1.0, 0.0 })
    local h_long = ins.signal.firwin2(31, { 0.0, 0.5, 1.0 }, { 1.0, 1.0, 0.0 })
    assert.are.equal(7, h_short.numel)
    assert.are.equal(31, h_long.numel)
  end)

  -- ========================================================================
  -- Cmplx Sort (2 tests)
  -- ========================================================================

  it("cmplx_sort_real", function()
    local a = ins.from_table({ 3.0, -1.0, 2.0, -4.0 })
    local sorted = ins.signal.cmplx_sort(a)
    assert.is_not_nil(sorted)
    assert.are.equal(4, sorted.numel)
    -- Sorted by absolute value: |-1|, |2|, |3|, |-4|
    assert.near(-1.0, ins.item(sorted, 0), 1e-10)
    assert.near(2.0, ins.item(sorted, 1), 1e-10)
    assert.near(3.0, ins.item(sorted, 2), 1e-10)
    assert.near(-4.0, ins.item(sorted, 3), 1e-10)
  end)

  it("cmplx_sort_single", function()
    local a = ins.from_table({ 42.0 })
    local sorted = ins.signal.cmplx_sort(a)
    assert.is_not_nil(sorted)
    assert.are.equal(1, sorted.numel)
    assert.near(42.0, ins.item(sorted, 0), 1e-10)
  end)
end)
