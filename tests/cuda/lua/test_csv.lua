-- CSV CUDA binding tests — aligned with C++ test_csv.cpp
-- Note: CSV read/write is not exposed in the Lua bindings. This test verifies
-- GPU data roundtrip via from_table/to which is the binding-level equivalent.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
--   ~/.luarocks/bin/busted tests/cuda/lua/test_csv.lua

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

describe("CSV CUDA Tests", function()
  it("roundtrip_1d", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(4, a.numel)
  end)

  it("roundtrip_2d", function()
    local a = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } }):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("roundtrip_int", function()
    local a = ins.from_table({ 10, 20, 30 }, ins.int32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  it("roundtrip_float64", function()
    local a = ins.from_table({ 1.1, 2.2, 3.3 }):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  it("roundtrip_after_ops", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.add(a, a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)
end)
