-- Signal estimation CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_estimation.lua

describe("Signal Estimation CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  -- Helper: wrap 2D table into 3D (points=1) unsqueezed array
  local function mat2d(tbl)
    return ins.from_table(tbl):unsqueeze(1)
  end

  -- Helper: wrap 1D table into 3D [1, n, 1] array
  local function vec3d(tbl)
    local a = ins.from_table(tbl)
    return a:reshape({ 1, #tbl, 1 })
  end

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
    local F = mat2d({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.F = F
    assert.is_not_nil(kf.F)
    assert.are.equal(4, kf.F.numel)
  end)

  it("kalman_filter_set_H", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    local H = mat2d({ { 1.0, 0.0 } })
    kf.H = H
    assert.is_not_nil(kf.H)
    assert.are.equal(2, kf.H.numel)
  end)

  it("kalman_filter_predict", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    kf.F = mat2d({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.H = mat2d({ { 1.0, 0.0 } })
    kf:predict()
    assert.is_not_nil(kf.x)
  end)

  it("kalman_filter_update", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    kf.F = mat2d({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.H = mat2d({ { 1.0, 0.0 } })
    -- Set initial covariance and noise matrices
    kf.P = mat2d({ { 1.0, 0.0 }, { 0.0, 1.0 } })
    kf.R = mat2d({ { 1.0 } })
    kf.Q = mat2d({ { 0.1, 0.0 }, { 0.0, 0.1 } })
    kf:predict()
    local z = vec3d({ 1.0 })
    kf:update(z)
    assert.is_not_nil(kf.x)
  end)

  it("kalman_filter_predict_update_cycle", function()
    local kf = ins.signal.KalmanFilter(2, 1)
    kf.F = mat2d({ { 1.0, 1.0 }, { 0.0, 1.0 } })
    kf.H = mat2d({ { 1.0, 0.0 } })
    -- Set non-zero initial covariance to avoid singular matrix in update
    kf.P = mat2d({ { 1.0, 0.0 }, { 0.0, 1.0 } })
    kf.R = mat2d({ { 1.0 } })
    kf.Q = mat2d({ { 0.1, 0.0 }, { 0.0, 0.1 } })
    for _ = 1, 5 do
      kf:predict()
      kf:update(vec3d({ 1.0 }))
    end
    assert.is_not_nil(kf.x)
    assert.are.equal(2, kf.x.numel)
  end)
end)
