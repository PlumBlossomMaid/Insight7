-- CUDA binding tests — 35 tests aligned with CPU Python/Lua/Julia
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_cuda_basic.lua

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

describe("CUDA Binding Tests", function()
  -- ========================================================================
  -- Creation (7 tests)
  -- ========================================================================

  it("zeros", function()
    local a = ins.zeros({ 2, 3 }, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("ones", function()
    local a = ins.ones({ 2, 3 }, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("full", function()
    local a = ins.full({ 2, 3 }, 7.0, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(6, a.numel)
  end)

  it("eye", function()
    local a = ins.eye(3):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(9, a.numel)
  end)

  it("arange", function()
    local a = ins.arange(10, ins.float32):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("linspace", function()
    local a = ins.linspace(0.0, 1.0, 5, ins.float64):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(5, a.numel)
  end)

  it("from_table", function()
    local a = ins.from_table({ 1.5, 2.5, 3.5 }):to(ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(3, a.numel)
  end)

  -- ========================================================================
  -- Arithmetic (5 tests)
  -- ========================================================================

  it("add", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 4.0, 5.0, 6.0 }):to(ins.GPUPlace(0))
    local c = ins.add(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("sub", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = ins.sub(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("mul", function()
    local a = ins.from_table({ 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 5.0, 6.0, 7.0 }):to(ins.GPUPlace(0))
    local c = ins.mul(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("div", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 2.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local c = ins.div(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("neg", function()
    local a = ins.from_table({ 1.0, -2.0, 3.0 }):to(ins.GPUPlace(0))
    local c = ins.negative(a)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  -- ========================================================================
  -- Unary (5 tests)
  -- ========================================================================

  it("abs", function()
    local a = ins.from_table({ -3.0, -1.0, 0.0, 2.0, 4.0 }):to(ins.GPUPlace(0))
    local b = ins.abs(a)
    assert.is_not_nil(b)
    assert.are.equal(5, b.numel)
  end)

  it("sqrt", function()
    local a = ins.from_table({ 1.0, 4.0, 9.0, 16.0 }):to(ins.GPUPlace(0))
    local b = ins.sqrt(a)
    assert.is_not_nil(b)
    assert.are.equal(4, b.numel)
  end)

  it("exp", function()
    local a = ins.from_table({ 0.0, 1.0, 2.0 }):to(ins.GPUPlace(0))
    local b = ins.exp(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("log", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.log(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  it("sin", function()
    local a = ins.from_table({ 0.0, 0.5, 1.0 }):to(ins.GPUPlace(0))
    local b = ins.sin(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  -- ========================================================================
  -- Reduction (5 tests)
  -- ========================================================================

  it("sum", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local s = ins.sum(a)
    assert.is_not_nil(s)
    assert.are.equal(1, s.numel)
  end)

  it("mean", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local m = ins.mean(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
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

  it("argmax", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0, 2.0 }):to(ins.GPUPlace(0))
    local m = ins.argmax(a)
    assert.is_not_nil(m)
    assert.are.equal(1, m.numel)
  end)

  -- ========================================================================
  -- Comparison (4 tests)
  -- ========================================================================

  it("equal", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 1.0, 0.0, 3.0 }):to(ins.GPUPlace(0))
    local c = ins.equal(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("greater", function()
    local a = ins.from_table({ 3.0, 1.0, 5.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 2.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local c = ins.greater(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("less", function()
    local a = ins.from_table({ 1.0, 5.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 2.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local c = ins.less(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  it("logical_and", function()
    local a = ins.from_table({ 1, 1, 0 }, ins.bool):to(ins.GPUPlace(0))
    local b = ins.from_table({ 1, 0, 0 }, ins.bool):to(ins.GPUPlace(0))
    local c = ins.logical_and(a, b)
    assert.is_not_nil(c)
    assert.are.equal(3, c.numel)
  end)

  -- ========================================================================
  -- Manipulation (3 tests)
  -- ========================================================================

  it("flatten", function()
    local a = ins.zeros({ 2, 3 }, ins.float64):to(ins.GPUPlace(0))
    local b = ins.flatten(a)
    assert.is_not_nil(b)
    assert.are.equal(6, b.numel)
  end)

  it("concat", function()
    local a = ins.from_table({ 1.0, 2.0 }):to(ins.GPUPlace(0))
    local b = ins.from_table({ 3.0, 4.0 }):to(ins.GPUPlace(0))
    local c = ins.concat({ a, b })
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  it("flip", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local b = ins.flip(a)
    assert.is_not_nil(b)
    assert.are.equal(3, b.numel)
  end)

  -- ========================================================================
  -- Linalg (3 tests)
  -- ========================================================================

  it("matmul", function()
    local a = ins.from_table({ { 1.0, 2.0 }, { 3.0, 4.0 } }):to(ins.GPUPlace(0))
    local b = ins.from_table({ { 5.0, 6.0 }, { 7.0, 8.0 } }):to(ins.GPUPlace(0))
    local c = ins.matmul(a, b)
    assert.is_not_nil(c)
    assert.are.equal(4, c.numel)
  end)

  it("det (skipped: needs cuBLAS)", function()
    pending("needs cuBLAS")
  end)
  it("inv (skipped: needs cuBLAS)", function()
    pending("needs cuBLAS")
  end)

  -- ========================================================================
  -- FFT (2 tests)
  -- ========================================================================

  it("fft", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local y = ins.fft(x)
    assert.is_not_nil(y)
    assert.are.equal(4, y.numel)
  end)

  it("ifft", function()
    local x = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    local y = ins.fft(x)
    local z = ins.ifft(y)
    assert.is_not_nil(z)
    assert.are.equal(4, z.numel)
  end)

  -- ========================================================================
  -- Signal (1 test)
  -- ========================================================================

  it("hann", function()
    local w = ins.signal.hann(16)
    assert.is_not_nil(w)
    assert.are.equal(16, w.numel)
  end)
end)
