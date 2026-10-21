-- Device information CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_device_info.lua

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

describe("DeviceInfo CUDA Tests", function()
  it("device_name gpu", function()
    local name = ins.device_name("gpu", 0)
    assert.is_not_nil(name)
    assert.is_string(name)
    assert.is_true(#name > 0)
  end)

  it("active_gpu_backend_name", function()
    local name = ins.active_gpu_backend_name()
    assert.is_string(name)
    assert.is_true(#name > 0)
  end)

  it("device_name rejects backend name", function()
    assert.has_error(function()
      ins.device_name(ins.active_gpu_backend_name(), 0)
    end)
  end)

  it("gpu_version positive", function()
    local ver = ins.gpu_version()
    assert.is_number(ver)
    assert.is_true(ver > 0)
  end)

  it("device_count positive", function()
    local count = ins.device_count()
    assert.is_number(count)
    assert.is_true(count > 0)
  end)

  it("compute_capability positive", function()
    local cc = ins.compute_capability(0)
    assert.is_number(cc)
    assert.is_true(cc > 0)
  end)

  it("compute_capability stays positive", function()
    local cc = ins.compute_capability(0)
    assert.is_true(cc > 0)
  end)

  it("gpu_version stays positive", function()
    local ver = ins.gpu_version()
    assert.is_true(ver > 0)
  end)

  it("device_name stable", function()
    local n1 = ins.device_name("gpu", 0)
    local n2 = ins.device_name("gpu", 0)
    assert.are.equal(n1, n2)
  end)

  it("compute_capability stable", function()
    local cc1 = ins.compute_capability(0)
    local cc2 = ins.compute_capability(0)
    assert.are.equal(cc1, cc2)
  end)

  it("device_name multiple calls", function()
    for i = 1, 5 do
      local name = ins.device_name("gpu", 0)
      assert.is_not_nil(name)
    end
  end)
end)
