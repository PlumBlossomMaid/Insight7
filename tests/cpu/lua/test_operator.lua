-- Operator CPU binding tests — aligned with C++ test_operator.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_operator.lua

describe("Operator CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  it("add_operator", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 4.0, 5.0, 6.0 })
    local c = a + b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("sub_operator", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 })
    local b = ins.from_table({ 1.0, 2.0, 3.0 })
    local c = a - b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("mul_operator", function()
    local a = ins.from_table({ 2.0, 3.0, 4.0 })
    local b = ins.from_table({ 5.0, 6.0, 7.0 })
    local c = a * b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("div_operator", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 })
    local b = ins.from_table({ 2.0, 4.0, 5.0 })
    local c = a / b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("negation", function()
    local a = ins.from_table({ 1.0, -2.0, 3.0 })
    local c = -a
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("scalar_add", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local c = a + 10.0
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("scalar_mul", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local c = a * 2.0
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("eq_operator", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 0.0, 3.0 })
    local c = ins.equal(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("lt_operator", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0 })
    local b = ins.from_table({ 2.0, 4.0, 5.0 })
    local c = ins.less(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("le_operator", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0 })
    local b = ins.from_table({ 2.0, 4.0, 5.0 })
    local c = ins.less_equal(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("indexing", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0, 40.0 })
    local b = a["0"]
    assert.is_not_nil(b)
    assert.are.equal(1, b.numel)
  end)

  it("string_slice", function()
    local a = ins.ones({ 4, 4 }, ins.float32)
    local b = a[":,1:-1"]
    assert.is_not_nil(b)
    -- :,1:-1 selects columns 1..3 (Lua 1-based), giving 4*3=12 elements
    assert.are.equal(12, b.numel)
  end)

  it("string_slice_positive_step", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0, 40.0, 50.0 })
    -- Lua 1-based "1:5:2" → C++ [0:4:2] → elements 1 and 3
    local b = a["1:5:2"]
    assert.are.equal(2, b.numel)
    assert.near(10.0, b[1]:item(), 1e-9)
    assert.near(30.0, b[2]:item(), 1e-9)
  end)

  it("string_slice_step_only", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 })
    local b = a["::2"]
    assert.are.equal(3, b.numel)
    assert.near(10.0, b[1]:item(), 1e-9)
    assert.near(30.0, b[2]:item(), 1e-9)
    assert.near(50.0, b[3]:item(), 1e-9)
  end)

  it("string_slice_2d_step", function()
    local a = ins.from_table({ { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } })
    -- row 1, columns 1 and 3 (Lua 1-based)
    local b = a["1:2,1:5:2"]
    assert.are.equal(2, b.numel)
    assert.near(1.0, b["1,1"]:item(), 1e-9)
    assert.near(3.0, b["1,2"]:item(), 1e-9)
  end)

  it("get_scalar", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 })
    local v = a:get(0)
    assert.is_not_nil(v)
  end)

  it("eq_self", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 })
    local b = ins.from_table({ 1.0, 2.0, 3.0 })
    local c = (a == b)
    assert.is_not_nil(c)
  end)
end)
