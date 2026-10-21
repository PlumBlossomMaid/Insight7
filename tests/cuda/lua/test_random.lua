-- Random number generation CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_random.lua

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

describe("Random CUDA Tests", function()
  it("rand shape", function()
    local a = ins.rand({ 3, 4 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(12, a.numel)
  end)

  it("randn shape", function()
    local a = ins.randn({ 3, 4 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(12, a.numel)
  end)

  it("seed determinism", function()
    ins.seed(42)
    local a = ins.rand({ 5 }, ins.float64, ins.GPUPlace(0))
    ins.seed(42)
    local b = ins.rand({ 5 }, ins.float64, ins.GPUPlace(0))
    assert.are.equal(5, a.numel)
    assert.are.equal(5, b.numel)
  end)

  it("get_seed", function()
    ins.seed(12345)
    local s = ins.get_seed()
    assert.are.equal(12345, s)
  end)

  it("randint shape", function()
    local a = ins.randint(0, 10, { 3, 3 }, ins.int64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(9, a.numel)
  end)

  it("rand_like", function()
    local template = ins.zeros({ 3, 4 }, ins.float64):to(ins.GPUPlace(0))
    local a = ins.rand_like(template)
    assert.is_not_nil(a)
    assert.are.equal(12, a.numel)
  end)

  it("randn_like", function()
    local template = ins.zeros({ 3, 4 }, ins.float64):to(ins.GPUPlace(0))
    local a = ins.randn_like(template)
    assert.is_not_nil(a)
    assert.are.equal(12, a.numel)
  end)

  it("exponential shape", function()
    local a = ins.exponential(1.0, { 100 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(100, a.numel)
  end)

  it("gamma shape", function()
    local a = ins.gamma(2.0, { 100 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(100, a.numel)
  end)

  it("beta shape", function()
    local a = ins.beta(2.0, 5.0, { 100 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(100, a.numel)
  end)

  it("binomial shape", function()
    local a = ins.binomial(10, 0.5, { 100 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(100, a.numel)
  end)

  it("poisson shape", function()
    local a = ins.poisson(5.0, { 100 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(100, a.numel)
  end)

  it("randperm", function()
    local a = ins.randperm(10, ins.int64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("rand 1d", function()
    local a = ins.rand({ 10 }, ins.float32, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(10, a.numel)
  end)

  it("rand 3d", function()
    local a = ins.rand({ 2, 3, 4 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(24, a.numel)
  end)

  it("randn 1d", function()
    local a = ins.randn({ 20 }, ins.float64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(20, a.numel)
  end)

  it("randint 1d", function()
    local a = ins.randint(0, 100, { 50 }, ins.int64, ins.GPUPlace(0))
    assert.is_not_nil(a)
    assert.are.equal(50, a.numel)
  end)
end)
