-- Signal radar CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_radar.lua

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

describe("Signal Radar CUDA Tests", function()
  it("pulse_compression", function()
    local n = 64
    local tpl_t = {}
    local sig_t = {}
    for i = 1, n do
      tpl_t[i] = math.sin(2 * math.pi * i / n)
      sig_t[i] = tpl_t[i]
    end
    for i = n + 1, 2 * n do
      sig_t[i] = 0.0
    end
    local tpl = ins.from_table(tpl_t):to(ins.GPUPlace(0))
    local sig = ins.from_table(sig_t):to(ins.GPUPlace(0))
    local result = ins.signal.pulse_compression(sig, tpl)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("pulse_doppler", function()
    local rows = {}
    for r = 1, 16 do
      local row = {}
      for c = 1, 32 do
        row[c] = math.sin(2 * math.pi * r * c / 512)
      end
      rows[r] = row
    end
    local data = ins.from_table(rows):to(ins.GPUPlace(0))
    local result = ins.signal.pulse_doppler(data, 16)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("cfar_alpha", function()
    local alpha = ins.signal.cfar_alpha(20, 1e-3)
    assert.is_not_nil(alpha)
    assert.is_true(alpha > 0)
  end)

  it("cfar_alpha_different_pfa", function()
    local alpha1 = ins.signal.cfar_alpha(20, 1e-2)
    local alpha2 = ins.signal.cfar_alpha(20, 1e-6)
    assert.is_true(alpha1 < alpha2)
  end)

  it("ca_cfar", function()
    local t = {}
    for i = 1, 100 do
      t[i] = 0.0
    end
    t[30] = 10.0
    t[70] = 10.0
    local data = ins.from_table(t):to(ins.GPUPlace(0))
    local result = ins.signal.ca_cfar(data, 10, 2, 1e-2)
    assert.is_not_nil(result)
    assert.are.equal(100, result.numel)
  end)

  it("mvdr", function()
    local rows = {}
    for r = 1, 4 do
      local row = {}
      for c = 1, 32 do
        row[c] = math.sin(2 * math.pi * r * c / 128)
      end
      rows[r] = row
    end
    local x = ins.from_table(rows):to(ins.GPUPlace(0))
    local sv = ins.from_table({ 1.0, 1.0, 1.0, 1.0 }):to(ins.GPUPlace(0))
    local result = ins.signal.mvdr(x, sv)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("ambgfun", function()
    local t = {}
    for i = 1, 64 do
      t[i] = math.sin(2 * math.pi * i / 64)
    end
    local x = ins.from_table(t):to(ins.GPUPlace(0))
    local result = ins.signal.ambgfun(x, 32, 32)
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)
end)
