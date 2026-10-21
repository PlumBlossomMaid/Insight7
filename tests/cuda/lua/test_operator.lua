-- Operator CUDA binding tests — aligned with C++ test_operator.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
--   ~/.luarocks/bin/busted tests/cuda/lua/test_operator.lua

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

describe("Operator CUDA Tests", function()
  it("add_operator", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 4.0, 5.0, 6.0 }):to(ins.GPUPlace(0))
    local c = a + b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("sub_operator", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = a - b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("mul_operator", function()
    local a = ins.from_table({ 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 5.0, 6.0, 7.0 }):to(ins.GPUPlace(0))
    local c = a * b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("div_operator", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 2.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local c = a / b
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("negation", function()
    local a = ins.from_table({ 1.0, -2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = -a
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("scalar_add", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = a + 10.0
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("scalar_mul", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = a * 2.0
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("eq_operator", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 1.0, 0.0, 3.0 }):to(ins.GPUPlace(0))
    local c = (a == b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("lt_operator", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 2.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local c = (a < b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("le_operator", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 2.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local c = (a <= b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("indexing", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0, 40.0 }):to(ins.GPUPlace(0))
    local b = a["0"]
    assert.is_not_nil(b)
    assert.are.equal(1, b.numel)
  end)

  it("string_slice", function()
    local a = ins.ones({ 4, 4 }, ins.float32):to(ins.GPUPlace(0))
    local b = a[":,1:-1"]
    assert.is_not_nil(b)
    assert.are.equal(8, b.numel)
  end)

  it("get_scalar", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    local v = a:get(0)
    assert.is_not_nil(v)
  end)

  it("eq_self", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = (a == b)
    assert.is_not_nil(c)
  end)
end)
