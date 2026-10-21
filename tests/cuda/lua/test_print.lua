-- Print CUDA binding tests — aligned with C++ test_print.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
--   ~/.luarocks/bin/busted tests/cuda/lua/test_print.lua

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

describe("Print CUDA Tests", function()
  it("tostring_1d", function()
    local a = ins.zeros({ 5 }, ins.float32):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_2d", function()
    local a = ins.ones({ 3, 4 }, ins.float64):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_int", function()
    local a = ins.zeros({ 3 }, ins.int32):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_from_table", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("tostring_2d_table", function()
    local a = ins.from_table({ { 1, 2 }, { 3, 4 } }):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
    assert.is_true(#s > 0)
  end)

  it("repr_does_not_crash_large", function()
    local a = ins.ones({ 100, 100 }, ins.float32):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
  end)

  it("repr_empty", function()
    local a = ins.zeros({ 0 }, ins.float32):to(ins.GPUPlace(0))
    local s = tostring(a)
    assert.is_not_nil(s)
  end)

  it("repr_after_cast", function()
    local a = ins.ones({ 3 }, ins.float32):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    local s = tostring(b)
    assert.is_not_nil(s)
  end)
end)
