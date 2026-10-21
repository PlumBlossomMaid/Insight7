-- Reduction CUDA binding tests — aligned with C++ test_reduction.cpp
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;"
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
--   ~/.luarocks/bin/busted tests/cuda/lua/test_reduction.lua

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

describe("Reduction CUDA Tests", function()
  it("sum", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local s = ins.sum(a)
    assert.is_not_nil(s)
    assert.are.equal(1, s.numel)
  end)

  it("sum_axis", function()
    local a = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } }):to(ins.GPUPlace(0))
    local s = ins.sum(a, 0)
    assert.is_not_nil(s)
  end)

  it("mean", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local m = ins.mean(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("mean_axis", function()
    local a = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } }):to(ins.GPUPlace(0))
    local m = ins.mean(a, 1)
    assert.is_not_nil(m)
  end)

  it("max", function()
    local a = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 }):to(ins.GPUPlace(0))
    local m = ins.max(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("min", function()
    local a = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 }):to(ins.GPUPlace(0))
    local m = ins.min(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("prod", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local p = ins.prod(a)
    assert.is_not_nil(p)
    assert.are.equal(1, p.numel)
  end)

  it("argmax", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0, 2.0 }):to(ins.GPUPlace(0))
    local m = ins.argmax(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("argmin", function()
    local a = ins.from_table({ 5.0, 1.0, 3.0, 2.0 }):to(ins.GPUPlace(0))
    local m = ins.argmin(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  it("cumsum", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local c = ins.cumsum(a, 0)
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  it("cumprod", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local c = ins.cumprod(a, 0)
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  it("cummax", function()
    local a = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 }):to(ins.GPUPlace(0))
    local c = ins.cummax(a, 0)
    assert.is_not_nil(c)
    assert.are.equal(5, c.numel)
  end)

  it("cummin", function()
    local a = ins.from_table({ 3.0, 1.0, 4.0, 1.0, 5.0 }):to(ins.GPUPlace(0))
    local c = ins.cummin(a, 0)
    assert.is_not_nil(c)
    assert.are.equal(5, c.numel)
  end)

  it("any", function()
    local a = ins.from_table({ 1, 0, 1 }, ins.bool):to(ins.GPUPlace(0))
    local r = ins.any(a)
    assert.is_not_nil(r)
  end)

  it("all", function()
    local a = ins.from_table({ 1, 0, 1 }, ins.bool):to(ins.GPUPlace(0))
    local r = ins.all(a)
    assert.is_not_nil(r)
  end)

  it("var", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local v = ins.var(a)
    assert.is_not_nil(v)
    assert.are.equal(1, v.numel)
  end)

  it("std", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local s = ins.std(a)
    assert.is_not_nil(s)
    assert.are.equal(1, s.numel)
  end)

  it("count_nonzero", function()
    local a = ins.from_table({ 0.0, 1.0, 0.0, 3.0, 0.0 }):to(ins.GPUPlace(0))
    local c = ins.count_nonzero(a)
    assert.is_not_nil(c)
  end)
end)
