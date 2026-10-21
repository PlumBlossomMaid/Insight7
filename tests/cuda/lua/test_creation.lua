-- Creation CUDA binding tests — aligned with C++ test_creation.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
--   ~/.luarocks/bin/busted tests/cuda/lua/test_creation.lua

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

describe("Creation CUDA Tests", function()
  it("zeros_1d", function()
    local a = ins.zeros({ 5 }, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(5, a.numel)
  end)

  it("zeros_2d", function()
    local a = ins.zeros({ 3, 4 }, ins.float64):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(12, a.numel)
    assert.are.equal(2, a.ndim)
  end)

  it("ones_1d", function()
    local a = ins.ones({ 4 }, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(4, a.numel)
  end)

  it("ones_2d", function()
    local a = ins.ones({ 2, 5 }, ins.float64):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("full", function()
    local a = ins.full({ 3, 3 }, 7.0, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(9, a.numel)
  end)

  it("full_negative", function()
    local a = ins.full({ 2, 2 }, -3.5, ins.float64):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(4, a.numel)
  end)

  it("eye", function()
    local a = ins.eye(4):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(16, a.numel)
  end)

  it("arange", function()
    local a = ins.arange(10, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("linspace", function()
    local a = ins.linspace(0.0, 1.0, 11, ins.float64):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(11, a.numel)
  end)

  it("from_table_1d", function()
    local a = ins.from_table({ 1.5, 2.5, 3.5 }):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  it("from_table_2d", function()
    local a = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } }):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
    assert.are.equal(2, a.ndim)
  end)

  it("zeros_like", function()
    local a = ins.ones({ 3, 3 }, ins.float32):to(ins.GPUPlace(0))
    local b = ins.zeros_like(a)
    assert.is_not_nil(b)
    assert.are.equal(9, b.numel)
  end)

  it("ones_like", function()
    local a = ins.zeros({ 2, 4 }, ins.float64):to(ins.GPUPlace(0))
    local b = ins.ones_like(a)
    assert.is_not_nil(b)
    assert.are.equal(8, b.numel)
  end)

  it("dtype_float64", function()
    local a = ins.zeros({ 3 }, ins.float64):to(ins.GPUPlace(0))
    assert.are.equal(ins.float64, a.dtype)
  end)

  it("dtype_int32", function()
    local a = ins.ones({ 3 }, ins.int32):to(ins.GPUPlace(0))
    assert.are.equal(ins.int32, a.dtype)
  end)
end)
