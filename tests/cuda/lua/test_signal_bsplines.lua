-- Signal Bsplines CUDA binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_bsplines.lua

local ok_gpu = pcall(function()
  local native = require("_insight")
  native.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

describe("Signal B-Splines CUDA Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Gauss Spline (3 tests)
  -- ========================================================================

  it("gauss_spline_basic", function()
    local x = ins.from_table({ 0.0 })
    local y = ins.signal.gauss_spline(3, x)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    -- At x=0: 1/sqrt(2*pi*sigma^2), sigma^2=(n+1)/12 = 4/12
    local sigma_sq = 4.0 / 12.0
    local expected = 1.0 / math.sqrt(2.0 * math.pi * sigma_sq)
    assert.near(expected, ins.item(y, 0), 1e-10)
  end)

  it("gauss_spline_symmetry", function()
    local x = ins.from_table({ -2.0, -1.0, 0.0, 1.0, 2.0 })
    local y = ins.signal.gauss_spline(2, x)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    assert.near(ins.item(y, 0), ins.item(y, 4), 1e-10)
    assert.near(ins.item(y, 1), ins.item(y, 3), 1e-10)
    -- Peak at center
    assert.is_true(ins.item(y, 2) > ins.item(y, 0))
    assert.is_true(ins.item(y, 2) > ins.item(y, 1))
  end)

  it("gauss_spline_decay", function()
    local x = ins.from_table({ 0.0, 1.0, 2.0, 3.0, 5.0 })
    local y = ins.signal.gauss_spline(3, x)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    for i = 1, 4 do
      assert.is_true(ins.item(y, i) < ins.item(y, i - 1))
    end
  end)

  -- ========================================================================
  -- Cubic (5 tests)
  -- ========================================================================

  it("cubic_basic", function()
    local x = ins.from_table({ 0.0 })
    local y = ins.signal.cubic(x)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(2.0 / 3.0, ins.item(y, 0), 1e-10)
  end)

  it("cubic_symmetry", function()
    local x = ins.from_table({ -1.5, -0.5, 0.0, 0.5, 1.5 })
    local y = ins.signal.cubic(x)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    assert.near(ins.item(y, 0), ins.item(y, 4), 1e-10)
    assert.near(ins.item(y, 1), ins.item(y, 3), 1e-10)
  end)

  it("cubic_zero_outside", function()
    local x = ins.from_table({ -2.5, -2.0, 2.0, 2.5 })
    local y = ins.signal.cubic(x)
    assert.is_not_nil(y)
    assert.are.equal(4, y.numel)
    for i = 0, 3 do
      assert.near(0.0, ins.item(y, i), 1e-10)
    end
  end)

  it("cubic_region1", function()
    -- |x| < 1: 2/3 - 0.5*|x|^2*(2-|x|)
    local x = ins.from_table({ 0.5 })
    local y = ins.signal.cubic(x)
    assert.is_not_nil(y)
    local expected = 2.0 / 3.0 - 0.5 * 0.25 * 1.5
    assert.near(expected, ins.item(y, 0), 1e-10)
  end)

  it("cubic_region2", function()
    -- 1 <= |x| < 2: (1/6)*(2-|x|)^3
    local x = ins.from_table({ 1.5 })
    local y = ins.signal.cubic(x)
    assert.is_not_nil(y)
    local expected = (1.0 / 6.0) * 0.125
    assert.near(expected, ins.item(y, 0), 1e-10)
  end)

  -- ========================================================================
  -- Quadratic (3 tests)
  -- ========================================================================

  it("quadratic_basic", function()
    local x = ins.from_table({ 0.0 })
    local y = ins.signal.quadratic(x)
    assert.is_not_nil(y)
    assert.are.equal(1, y.numel)
    assert.near(0.75, ins.item(y, 0), 1e-10)
  end)

  it("quadratic_symmetry", function()
    local x = ins.from_table({ -1.0, -0.25, 0.0, 0.25, 1.0 })
    local y = ins.signal.quadratic(x)
    assert.is_not_nil(y)
    assert.are.equal(5, y.numel)
    assert.near(ins.item(y, 0), ins.item(y, 4), 1e-10)
    assert.near(ins.item(y, 1), ins.item(y, 3), 1e-10)
  end)

  it("quadratic_zero_outside", function()
    local x = ins.from_table({ -2.0, -1.5, 1.5, 2.0 })
    local y = ins.signal.quadratic(x)
    assert.is_not_nil(y)
    assert.are.equal(4, y.numel)
    for i = 0, 3 do
      assert.near(0.0, ins.item(y, i), 1e-10)
    end
  end)
end)
