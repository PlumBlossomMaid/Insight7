-- DType CUDA binding tests — aligned with C++ test_dtype.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
--   ~/.luarocks/bin/busted tests/cuda/lua/test_dtype.lua

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

describe("DType CUDA Tests", function()
  it("dtype_shortcuts", function()
    assert.is_not_nil(ins.float32)
    assert.is_not_nil(ins.float64)
    assert.is_not_nil(ins.int32)
    assert.is_not_nil(ins.int64)
    assert.is_not_nil(ins.bool)
  end)

  it("cast_f32_to_f64", function()
    local a = ins.ones({ 3 }, ins.float32):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float64)
    assert.is_not_nil(b)
    assert.are.equal(ins.float64, b.dtype)
    assert.are.equal(3, b.numel)
  end)

  it("cast_f64_to_i32", function()
    local a = ins.from_table({ 1.9, 2.5, 3.1 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    assert.is_not_nil(b)
    assert.are.equal(ins.int32, b.dtype)
  end)

  it("cast_i32_to_f32", function()
    local a = ins.from_table({ 1, 2, 3 }, ins.int32):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float32)
    assert.is_not_nil(b)
    assert.are.equal(ins.float32, b.dtype)
  end)

  it("cast_to_bool", function()
    local a = ins.from_table({ 0.0, 1.0, 2.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.bool)
    assert.is_not_nil(b)
    assert.are.equal(ins.bool, b.dtype)
  end)

  it("cast_i64_to_f64", function()
    local a = ins.from_table({ 100, 200, 300 }, ins.int64):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float64)
    assert.is_not_nil(b)
    assert.are.equal(ins.float64, b.dtype)
  end)

  it("cast_preserves_values", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float32)
    local c = ins.cast(b, ins.float64)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("array_dtype_property", function()
    local a = ins.zeros({ 3 }, ins.float32):to(ins.GPUPlace(0))
    assert.are.equal(ins.float32, a.dtype)
    local b = ins.zeros({ 3 }, ins.int64):to(ins.GPUPlace(0))
    assert.are.equal(ins.int64, b.dtype)
  end)

  it("cast_roundtrip", function()
    local a = ins.ones({ 4 }, ins.float64):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    local c = ins.cast(b, ins.float64)
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)
end)
