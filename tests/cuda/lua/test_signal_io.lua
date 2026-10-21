-- Signal io CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_io.lua

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

describe("Signal IO CUDA Tests", function()
  local tmpfile

  after_each(function()
    if tmpfile then
      os.remove(tmpfile)
      tmpfile = nil
    end
  end)

  it("write_read_bin_roundtrip", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    tmpfile = os.tmpname() .. ".bin"
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, "float64")
    assert.is_not_nil(result)
    assert.are.equal(5, result.numel)
  end)

  it("write_read_bin_float32", function()
    local data = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    tmpfile = os.tmpname() .. ".bin"
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, "float32")
    assert.is_not_nil(result)
    assert.are.equal(3, result.numel)
  end)

  it("write_read_bin_int16", function()
    local data = ins.from_table({ 100, 200, 300, 400 }, ins.int16):to(ins.GPUPlace(0))
    tmpfile = os.tmpname() .. ".bin"
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, "int16")
    assert.is_not_nil(result)
    assert.are.equal(4, result.numel)
  end)

  it("write_read_bin_large", function()
    local t = {}
    for i = 1, 1024 do
      t[i] = math.sin(i)
    end
    local data = ins.from_table(t):to(ins.GPUPlace(0))
    tmpfile = os.tmpname() .. ".bin"
    ins.signal.write_bin(data, tmpfile)
    local result = ins.signal.read_bin(tmpfile, "float64")
    assert.is_not_nil(result)
    assert.are.equal(1024, result.numel)
  end)

  it("pack_bin", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local packed = ins.signal.pack_bin(data, "float64")
    assert.is_not_nil(packed)
    assert.is_true(packed.numel > 0)
  end)

  it("unpack_bin", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    tmpfile = os.tmpname() .. ".bin"
    ins.signal.write_bin(data, tmpfile)
    local raw = ins.signal.read_bin(tmpfile, "uint8")
    local result = ins.signal.unpack_bin(raw, "float64")
    assert.is_not_nil(result)
    assert.is_true(result.numel > 0)
  end)

  it("write_sigmf", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0 }):to(ins.GPUPlace(0))
    tmpfile = os.tmpname() .. ".sigmf-data"
    ins.signal.write_sigmf(data, tmpfile, {})
    assert.is_not_nil(tmpfile)
  end)
end)
