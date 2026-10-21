-- Signal Windows CUDA binding tests.
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_windows.lua

local ok_gpu = pcall(function()
  local native = require("_insight")
  native.init()
end)
if not ok_gpu then
  print("SKIP: GPU backend not available")
  os.exit(0)
end

describe("Signal Windows CUDA Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- ========================================================================
  -- Boxcar (2 tests)
  -- ========================================================================

  it("boxcar_basic", function()
    local w = ins.signal.boxcar(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    for i = 0, 4 do
      assert.near(1.0, ins.item(w, i), 1e-10)
    end
  end)

  it("boxcar_size1", function()
    local w = ins.signal.boxcar(1)
    assert.is_not_nil(w)
    assert.are.equal(1, w.numel)
    assert.near(1.0, ins.item(w, 0), 1e-10)
  end)

  -- ========================================================================
  -- Triang (2 tests)
  -- ========================================================================

  it("triang_odd", function()
    local w = ins.signal.triang(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    -- [1/3, 2/3, 1, 2/3, 1/3]
    assert.near(1.0 / 3.0, ins.item(w, 0), 1e-10)
    assert.near(2.0 / 3.0, ins.item(w, 1), 1e-10)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    assert.near(2.0 / 3.0, ins.item(w, 3), 1e-10)
    assert.near(1.0 / 3.0, ins.item(w, 4), 1e-10)
  end)

  it("triang_even", function()
    local w = ins.signal.triang(4)
    assert.is_not_nil(w)
    assert.are.equal(4, w.numel)
    -- [0.25, 0.75, 0.75, 0.25]
    assert.near(0.25, ins.item(w, 0), 1e-10)
    assert.near(0.75, ins.item(w, 1), 1e-10)
    assert.near(0.75, ins.item(w, 2), 1e-10)
    assert.near(0.25, ins.item(w, 3), 1e-10)
  end)

  -- ========================================================================
  -- Parzen (1 test)
  -- ========================================================================

  it("parzen_basic", function()
    local w = ins.signal.parzen(8)
    assert.is_not_nil(w)
    assert.are.equal(8, w.numel)
    assert.near(0.00390625, ins.item(w, 0), 1e-6)
    assert.near(0.10546875, ins.item(w, 1), 1e-6)
    assert.near(0.47265625, ins.item(w, 2), 1e-6)
    assert.near(0.91796875, ins.item(w, 3), 1e-6)
  end)

  -- ========================================================================
  -- Bohman (1 test)
  -- ========================================================================

  it("bohman_endpoints", function()
    local w = ins.signal.bohman(11)
    assert.is_not_nil(w)
    assert.are.equal(11, w.numel)
    assert.near(0.0, ins.item(w, 0), 1e-10)
    assert.near(0.0, ins.item(w, 10), 1e-10)
    assert.near(1.0, ins.item(w, 5), 1e-6)
  end)

  -- ========================================================================
  -- Bartlett (1 test)
  -- ========================================================================

  it("bartlett_basic", function()
    local w = ins.signal.bartlett(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(0.0, ins.item(w, 0), 1e-10)
    assert.near(0.5, ins.item(w, 1), 1e-10)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    assert.near(0.5, ins.item(w, 3), 1e-10)
    assert.near(0.0, ins.item(w, 4), 1e-10)
  end)

  -- ========================================================================
  -- Cosine (1 test)
  -- ========================================================================

  it("cosine_basic", function()
    local w = ins.signal.cosine(4)
    assert.is_not_nil(w)
    assert.are.equal(4, w.numel)
    -- sin(pi/8), sin(3pi/8), sin(5pi/8), sin(7pi/8)
    assert.near(math.sin(math.pi / 8), ins.item(w, 0), 1e-10)
    assert.near(math.sin(3 * math.pi / 8), ins.item(w, 1), 1e-10)
  end)

  -- ========================================================================
  -- Exponential (1 test)
  -- ========================================================================

  it("exponential_basic", function()
    local w = ins.signal.exponential(5, 1.0, 2.0)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    -- Symmetric
    assert.near(ins.item(w, 0), ins.item(w, 4), 1e-10)
    assert.near(math.exp(-2.0), ins.item(w, 0), 1e-10)
  end)

  -- ========================================================================
  -- Blackman (1 test)
  -- ========================================================================

  it("blackman_basic", function()
    local w = ins.signal.blackman(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(0.0, ins.item(w, 0), 1e-10)
    assert.near(0.0, ins.item(w, 4), 1e-10)
    assert.near(1.0, ins.item(w, 2), 1e-6)
  end)

  -- ========================================================================
  -- Nuttall (1 test)
  -- ========================================================================

  it("nuttall_basic", function()
    local w = ins.signal.nuttall(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(0.0003628, ins.item(w, 0), 1e-4)
    assert.near(1.0, ins.item(w, 2), 1e-6)
  end)

  -- ========================================================================
  -- Blackmanharris (1 test)
  -- ========================================================================

  it("blackmanharris_basic", function()
    local w = ins.signal.blackmanharris(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(6.0e-5, ins.item(w, 0), 1e-4)
    assert.near(1.0, ins.item(w, 2), 1e-6)
  end)

  -- ========================================================================
  -- Flattop (1 test)
  -- ========================================================================

  it("flattop_basic", function()
    local w = ins.signal.flattop(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(ins.item(w, 0), ins.item(w, 4), 1e-10)
    assert.near(ins.item(w, 1), ins.item(w, 3), 1e-10)
  end)

  -- ========================================================================
  -- Hann (2 tests)
  -- ========================================================================

  it("hann_basic", function()
    local w = ins.signal.hann(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(0.0, ins.item(w, 0), 1e-10)
    assert.near(0.0, ins.item(w, 4), 1e-10)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    assert.near(0.5, ins.item(w, 1), 1e-10)
    assert.near(0.5, ins.item(w, 3), 1e-10)
  end)

  it("hann_size16", function()
    local w = ins.signal.hann(16)
    assert.is_not_nil(w)
    assert.are.equal(16, w.numel)
  end)

  -- ========================================================================
  -- Hamming (1 test)
  -- ========================================================================

  it("hamming_basic", function()
    local w = ins.signal.hamming(5)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(0.08, ins.item(w, 0), 1e-10)
    assert.near(0.08, ins.item(w, 4), 1e-10)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    assert.near(ins.item(w, 1), ins.item(w, 3), 1e-10)
  end)

  -- ========================================================================
  -- Tukey (2 tests)
  -- ========================================================================

  it("tukey_alpha0", function()
    local w = ins.signal.tukey(10, 0.0)
    assert.is_not_nil(w)
    assert.are.equal(10, w.numel)
    for i = 0, 9 do
      assert.near(1.0, ins.item(w, i), 1e-10)
    end
  end)

  it("tukey_alpha1", function()
    local w_tukey = ins.signal.tukey(10, 1.0)
    local w_hann = ins.signal.hann(10)
    assert.is_not_nil(w_tukey)
    assert.is_not_nil(w_hann)
    for i = 0, 9 do
      assert.near(ins.item(w_hann, i), ins.item(w_tukey, i), 1e-10)
    end
  end)

  -- ========================================================================
  -- Barthann (1 test)
  -- ========================================================================

  it("barthann_basic", function()
    local w = ins.signal.barthann(11)
    assert.is_not_nil(w)
    assert.are.equal(11, w.numel)
    assert.near(1.0, ins.item(w, 5), 0.02)
  end)

  -- ========================================================================
  -- Kaiser (1 test)
  -- ========================================================================

  it("kaiser_beta0", function()
    local w = ins.signal.kaiser(5, 0.0)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    for i = 0, 4 do
      assert.near(1.0, ins.item(w, i), 1e-10)
    end
  end)

  -- ========================================================================
  -- Gaussian (1 test)
  -- ========================================================================

  it("gaussian_basic", function()
    local w = ins.signal.gaussian(5, 1.0)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    assert.near(ins.item(w, 0), ins.item(w, 4), 1e-10)
    assert.near(math.exp(-2.0), ins.item(w, 0), 1e-10)
  end)

  -- ========================================================================
  -- General Gaussian (1 test)
  -- ========================================================================

  it("general_gaussian_basic", function()
    local w = ins.signal.general_gaussian(5, 1.0, 1.0)
    assert.is_not_nil(w)
    assert.are.equal(5, w.numel)
    assert.near(1.0, ins.item(w, 2), 1e-10)
    assert.near(math.exp(-2.0), ins.item(w, 0), 1e-10)
  end)

  -- ========================================================================
  -- Chebwin (1 test)
  -- ========================================================================

  it("chebwin_basic", function()
    local w = ins.signal.chebwin(11, 50.0)
    assert.is_not_nil(w)
    assert.are.equal(11, w.numel)
    -- Symmetric
    for i = 0, 4 do
      assert.near(ins.item(w, i), ins.item(w, 10 - i), 1e-6)
    end
  end)

  -- ========================================================================
  -- Taylor (1 test)
  -- ========================================================================

  it("taylor_basic", function()
    local w = ins.signal.taylor(32, 4, -30, true)
    assert.is_not_nil(w)
    assert.are.equal(32, w.numel)
    -- Symmetric
    for i = 0, 15 do
      assert.near(ins.item(w, i), ins.item(w, 31 - i), 1e-6)
    end
  end)

  -- ========================================================================
  -- Get Window (2 tests)
  -- ========================================================================

  it("get_window_boxcar", function()
    local w = ins.signal.get_window("boxcar", 8)
    assert.is_not_nil(w)
    assert.are.equal(8, w.numel)
    for i = 0, 7 do
      assert.near(1.0, ins.item(w, i), 1e-10)
    end
  end)

  it("get_window_hann", function()
    local w = ins.signal.get_window("hann", 5)
    local ref = ins.signal.hann(5)
    assert.is_not_nil(w)
    assert.is_not_nil(ref)
    for i = 0, 4 do
      assert.near(ins.item(ref, i), ins.item(w, i), 1e-10)
    end
  end)
end)
