-- Signal estimation CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_estimation.lua

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

describe("Signal Estimation CUDA Tests", function()
  it("kalman_filter_constructor", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    assert.is_not_nil(kf)
    assert.are.equal(2, kf.dim_x)
    assert.are.equal(1, kf.dim_z)
  end)

  it("kalman_filter_state_shape", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    local x = kf.x
    assert.is_not_nil(x)
    assert.are.equal(2, x.numel)
  end)

  it("kalman_filter_covariance_shape", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    local P = kf.P
    assert.is_not_nil(P)
    assert.are.equal(4, P.numel)
  end)

  it("kalman_filter_set_F", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    local F = ins.from_table({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.F = F
    assert.is_not_nil(kf.F)
    assert.are.equal(4, kf.F.numel)
  end)

  it("kalman_filter_set_H", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    local H = ins.from_table({ { 1.0, 0.0 } })
    kf.H = H
    assert.is_not_nil(kf.H)
    assert.are.equal(2, kf.H.numel)
  end)

  it("kalman_filter_predict", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    kf.F = ins.from_table({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.H = ins.from_table({ { 1.0, 0.0 } })
    kf:predict()
    assert.is_not_nil(kf.x)
  end)

  it("kalman_filter_update", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    kf.F = ins.from_table({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.H = ins.from_table({ { 1.0, 0.0 } })
    kf:predict()
    local z = ins.from_table({ 1.0 })
    kf:update(z)
    assert.is_not_nil(kf.x)
  end)

  it("kalman_filter_predict_update_cycle", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    kf.F = ins.from_table({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.H = ins.from_table({ { 1.0, 0.0 } })
    for _ = 1, 5 do
      kf:predict()
      kf:update(ins.from_table({ 1.0 }))
    end
    assert.is_not_nil(kf.x)
    assert.are.equal(2, kf.x.numel)
  end)
end)
