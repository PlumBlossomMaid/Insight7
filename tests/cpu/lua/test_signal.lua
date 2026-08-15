-- Signal CPU binding tests (base signal functions).
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal.lua

describe("Signal CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Unwrap (5 tests)
  -- ========================================================================

  it("unwrap_basic", function()
    local a = ins.from_table({
      0.0,
      1.5707963267948966,
      3.141592653589793,
      4.71238898038469,
      6.283185307179586,
      9.42477796076938,
    })
    local u = ins.unwrap(a)
    assert.is_not_nil(u)
    assert.are.equal(6, u.numel)
  end)

  it("unwrap_scalar", function()
    local a = ins.from_table({ 3.141592653589793 })
    local u = ins.unwrap(a)
    assert.is_not_nil(u)
    assert.are.equal(1, u.numel)
  end)

  it("unwrap_2d_axis1", function()
    local a = ins.from_table({
      { 0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469 },
      { 0.1, 1.5707963267948966, 3.2, 3.3 },
    })
    local u = ins.unwrap(a, 2)
    assert.is_not_nil(u)
    assert.are.equal(8, u.numel)
  end)

  it("unwrap_2d_axis0", function()
    local a = ins.from_table({
      { 0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469 },
      { 0.1, 1.5707963267948966, 3.2, 3.3 },
    })
    local u = ins.unwrap(a, 1)
    assert.is_not_nil(u)
    assert.are.equal(8, u.numel)
  end)

  it("unwrap_with_jumps", function()
    local a = ins.from_table({ 0.0, 0.1, 3.2, 3.3, 6.4, 6.5 })
    local u = ins.unwrap(a)
    assert.is_not_nil(u)
    assert.are.equal(6, u.numel)
  end)

  -- ========================================================================
  -- Sinc (2 tests)
  -- ========================================================================

  it("sinc_values", function()
    local a = ins.from_table({ 0.0, 1.0, 2.0, -1.0, -2.0, 0.5 })
    local y = ins.sinc(a)
    assert.is_not_nil(y)
    assert.are.equal(6, y.numel)
    -- sinc(0) = 1
    assert.near(1.0, y:get(0), 1e-5)
    -- sinc(1) = 0
    assert.near(0.0, y:get(1), 1e-5)
    -- sinc(2) = 0
    assert.near(0.0, y:get(2), 1e-5)
    -- sinc(0.5) = 2/pi ≈ 0.6366
    assert.near(0.6366198, y:get(5), 1e-5)
  end)

  it("sinc_size1", function()
    local a = ins.from_table({ 0.0 })
    local y = ins.sinc(a)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(1.0, y:get(0), 1e-5)
  end)

  -- ========================================================================
  -- Convolve (2 tests)
  -- ========================================================================

  it("convolve_full", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local v = ins.from_table({ 1.0, 1.0 })
    local c = ins.convolve(a, v, "full")
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
    -- [1, 3, 5, 3]
    assert.near(1.0, c:get(0), 1e-5)
    assert.near(3.0, c:get(1), 1e-5)
    assert.near(5.0, c:get(2), 1e-5)
    assert.near(3.0, c:get(3), 1e-5)
  end)

  it("convolve_same", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local v = ins.from_table({ 1.0, 1.0 })
    local c = ins.convolve(a, v, "same")
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  -- ========================================================================
  -- Correlate (1 test)
  -- ========================================================================

  it("correlate_full", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 1.0 })
    local c = ins.signal.correlate(a, b, "full")
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)
end)
