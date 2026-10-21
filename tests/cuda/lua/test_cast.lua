-- Type casting CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_cast.lua

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

describe("Cast CUDA Tests", function()
  it("cast float64 to float32", function()
    local a = ins.from_table({ 1.5, 2.5, 3.5 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float32)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast float32 to float64", function()
    local a = ins.zeros({ 3 }, ins.float32):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float64)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast float64 to int32", function()
    local a = ins.from_table({ 1.9, 2.5, 3.1 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast float64 to int64", function()
    local a = ins.from_table({ 1.9, 2.5, 3.1 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int64)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast float64 to uint8", function()
    local a = ins.from_table({ 0.0, 128.0, 255.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.uint8)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast float64 to bool", function()
    local a = ins.from_table({ 0.0, 1.0, 0.5 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.bool)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast bool to float64", function()
    local a = ins.from_table({ 1, 0, 1 }, ins.bool):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float64)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast int32 to float64", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    local c = ins.cast(b, ins.float64)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("cast preserves shape", function()
    local a = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float32)
    assert.is_not_nil(b)
    assert.are.equal(4, b.numel)
  end)

  it("cast idempotent", function()
    local a = ins.from_table({ 1.5, 2.5 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.float64)
    assert.is_not_nil(b)
    assert.are.equal(2, b.numel)
  end)

  it("cast int to int", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    local c = ins.cast(b, ins.int64)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("cast roundtrip", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    local c = ins.cast(b, ins.float64)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("cast negative values", function()
    local a = ins.from_table({ -1.5, -2.5, 3.5 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("cast 2d array", function()
    local a = ins.from_table({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.int32)
    assert.is_not_nil(b)
    assert.are.equal(6, b.numel)
  end)

  it("cast uint8 to float32", function()
    local a = ins.from_table({ 0.0, 128.0, 255.0 }):to(ins.GPUPlace(0))
    local b = ins.cast(a, ins.uint8)
    local c = ins.cast(b, ins.float32)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)
end)
