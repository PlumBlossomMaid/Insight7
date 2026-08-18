-- Device information CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_device_info.lua

describe("DeviceInfo CPU Tests", function()
  local ins
  setup(function()
    ins = require("_insight")
    ins.set_device(ins.CPUPlace())
  end)

  it("device_name cpu", function()
    local name = ins.device_name("cpu", 0)
    assert.is_not_nil(name)
    assert.is_string(name)
  end)

  it("device_name returns string", function()
    local name = ins.device_name("cpu")
    assert.is_string(name)
  end)

  it("gpu_version returns number", function()
    local ver = ins.gpu_version()
    assert.is_number(ver)
    assert.is_true(ver >= 0)
  end)

  it("compute_capability returns number", function()
    local cc = ins.compute_capability(0)
    assert.is_number(cc)
    assert.is_true(cc >= 0)
  end)

  it("device_count returns number", function()
    local count = ins.gpu_count()
    assert.is_number(count)
    assert.is_true(count >= 0)
  end)

  it("device_name default args", function()
    local name = ins.device_name("cpu")
    assert.is_not_nil(name)
    assert.is_string(name)
  end)

  it("compute_capability default", function()
    local cc = ins.compute_capability()
    assert.is_number(cc)
  end)

  it("gpu_version nonneg", function()
    local ver = ins.gpu_version()
    assert.is_true(ver >= 0)
  end)

  it("device_count nonneg", function()
    local count = ins.gpu_count()
    assert.is_true(count >= 0)
  end)
end)
