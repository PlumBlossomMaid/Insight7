-- Per-module Lua wrapper tests (insight.indexing, insight.manipulation,
-- insight.signal.filtering) and Array assignment sugar.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_module_wrappers.lua

describe("Lua Module Wrapper Tests", function()
  local ins, indexing, manipulation, filtering, argrule, parse_args
  setup(function()
    ins = require("insight")
    indexing = require("insight.indexing")
    manipulation = require("insight.manipulation")
    filtering = require("insight.signal.filtering")
    argrule = require("argrule")
    parse_args = require("insight._cli")
  end)

  it("uses_argrule_for_dual_calling_convention", function()
    assert.are.equal(true, argrule.strict())
    local a = ins.from_table({ 1.0, 2.0 })
    local result = indexing.gather({ x = a, dim = 0, index = ins.from_table({ 0 }) })
    assert.are.equal(1, result.numel)
  end)

  it("parses_cli_with_argrule", function()
    local args = parse_args({ "--device", "gpu", "--seed", "7", "--iterations", "3", "--timer" })
    assert.are.equal("gpu", args.device)
    assert.are.equal(7, args.seed)
    assert.are.equal(3, args.iterations)
    assert.is_true(args.timer)
    assert.is_false(args.info)
  end)

  it("rejects_invalid_cli_values", function()
    assert.has_error(function()
      parse_args({ "--device", "tpu" })
    end)
    assert.has_error(function()
      parse_args({ "--iterations", "-1" })
    end)
  end)

  -- ========================================================================
  -- Assignment sugar
  -- ========================================================================

  it("integer_assignment_scalar", function()
    local a = ins.zeros({ 4 }, ins.float64)
    a[2] = 5.0
    assert.near(5.0, a[2]:item(), 1e-9)
    assert.near(0.0, a[1]:item(), 1e-9)
    assert.near(0.0, a[3]:item(), 1e-9)
  end)

  it("integer_assignment_row", function()
    local a = ins.zeros({ 2, 2 }, ins.float64)
    a[1] = ins.from_table({ 7.0, 8.0 })
    assert.near(7.0, a["1,1"]:item(), 1e-9)
    assert.near(8.0, a["1,2"]:item(), 1e-9)
    assert.near(0.0, a["2,1"]:item(), 1e-9)
  end)

  it("integer_assignment_rejects_zero", function()
    local a = ins.zeros({ 4 }, ins.float64)
    assert.has_error(function()
      a[0] = 1.0
    end)
  end)

  it("string_assignment", function()
    local a = ins.zeros({ 4 }, ins.float64)
    a["2:4"] = 3.0
    assert.near(0.0, a[1]:item(), 1e-9)
    assert.near(3.0, a[2]:item(), 1e-9)
    assert.near(3.0, a[3]:item(), 1e-9)
    assert.near(0.0, a[4]:item(), 1e-9)
  end)

  -- ========================================================================
  -- insight.indexing
  -- ========================================================================

  it("indexing_gather", function()
    local a = ins.from_table({ { 10.0, 20.0 }, { 30.0, 40.0 } })
    local index = ins.from_table({ { 0, 1 }, { 1, 0 } })
    local g = indexing.gather(a, 0, index)
    assert.are.equal(4, g.numel)
    assert.near(10.0, g["1,1"]:item(), 1e-9)
    assert.near(40.0, g["1,2"]:item(), 1e-9)
    assert.near(30.0, g["2,1"]:item(), 1e-9)
    assert.near(20.0, g["2,2"]:item(), 1e-9)
  end)

  it("indexing_gather_named_args", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0, 40.0 })
    local index = ins.from_table({ 0, 3 })
    local g = indexing.gather({ x = a, dim = 0, index = index })
    assert.are.equal(2, g.numel)
    assert.near(10.0, g[1]:item(), 1e-9)
    assert.near(40.0, g[2]:item(), 1e-9)
  end)

  it("indexing_scatter_gather_roundtrip", function()
    local a = ins.zeros({ 4 }, ins.float64)
    local index = ins.from_table({ 0, 2 })
    local src = ins.from_table({ 10.0, 20.0 })
    local s = indexing.scatter(a, 0, index, src)
    assert.are.equal(4, s.numel)
    assert.near(10.0, s[1]:item(), 1e-9)
    assert.near(0.0, s[2]:item(), 1e-9)
    assert.near(20.0, s[3]:item(), 1e-9)

    local back = indexing.gather(s, 0, index)
    assert.are.equal(2, back.numel)
    assert.near(10.0, back[1]:item(), 1e-9)
    assert.near(20.0, back[2]:item(), 1e-9)
  end)

  it("indexing_scatter_add", function()
    local a = ins.zeros({ 4 }, ins.float64)
    local index = ins.from_table({ 0, 0, 2 })
    local src = ins.from_table({ 1.0, 2.0, 3.0 })
    local r = indexing.scatter_add(a, 0, index, src)
    assert.near(3.0, r[1]:item(), 1e-9)
    assert.near(0.0, r[2]:item(), 1e-9)
    assert.near(3.0, r[3]:item(), 1e-9)
  end)

  it("indexing_scatter_reduce", function()
    local a = ins.zeros({ 4 }, ins.float64)
    local index = ins.from_table({ 0, 0, 2 })
    local src = ins.from_table({ 1.0, 2.0, 3.0 })
    local r = indexing.scatter_reduce(a, 0, index, src, "add")
    assert.are.equal(4, r.numel)
    assert.near(3.0, r[1]:item(), 1e-9)
    assert.near(3.0, r[3]:item(), 1e-9)
  end)

  it("indexing_indices_from_table", function()
    local idx = indexing.indices({ 2, 3 })
    assert.is_not_nil(idx)
    assert.are.equal(12, idx.numel)
  end)

  -- ========================================================================
  -- insight.manipulation
  -- ========================================================================

  it("manipulation_tile_from_table", function()
    local a = ins.from_table({ 1.0, 2.0 })
    local b = manipulation.tile(a, { 3 })
    assert.are.equal(6, b.numel)
    assert.near(1.0, b[1]:item(), 1e-9)
    assert.near(2.0, b[2]:item(), 1e-9)
    assert.near(1.0, b[3]:item(), 1e-9)
  end)

  -- ========================================================================
  -- insight.signal.filtering
  -- ========================================================================

  it("filtering_wiener_forwards_noise", function()
    local x = ins.from_table({ 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0 })
    -- noise = 0 makes the Wiener gain 1, i.e. the filter is an identity
    local y = filtering.wiener(x, { 3 }, 0.0)
    assert.are.equal(8, y.numel)
    for i = 1, 8 do
      assert.near(x[i]:item(), y[i]:item(), 1e-9)
    end
  end)
end)
