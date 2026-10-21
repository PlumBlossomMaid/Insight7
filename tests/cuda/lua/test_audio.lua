-- Audio/I/O CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_audio.lua

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

describe("Audio CUDA Tests", function()
  it("write read roundtrip gpu float64", function()
    local a = ins.from_table({ 1.0, 2.5, 3.7, -1.2, 0.0 }):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_f64.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(5, result.numel)
    os.remove(path)
  end)

  it("write read roundtrip gpu float32", function()
    local a = ins.zeros({ 4 }, ins.float32):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_f32.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float32)
    assert.is_not_nil(result)
    assert.are.equal(4, result.numel)
    os.remove(path)
  end)

  it("write read roundtrip gpu int32", function()
    local a = ins.from_table({ 10.0, 20.0, 30.0 }):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_i32.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.int32)
    assert.is_not_nil(result)
    assert.are.equal(3, result.numel)
    os.remove(path)
  end)

  it("write read zeros gpu", function()
    local a = ins.zeros({ 100 }, ins.float64):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_zeros.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(100, result.numel)
    os.remove(path)
  end)

  it("write read ones gpu", function()
    local a = ins.ones({ 50 }, ins.float64):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_ones.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(50, result.numel)
    os.remove(path)
  end)

  it("write read negative gpu", function()
    local a = ins.from_table({ -100.5, -200.3, -300.1 }):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_neg.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(3, result.numel)
    os.remove(path)
  end)

  it("write read single gpu", function()
    local a = ins.from_table({ 42.0 }):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_single.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(1, result.numel)
    os.remove(path)
  end)

  it("write read large gpu", function()
    local t = {}
    for i = 1, 1000 do
      t[i] = i * 0.1
    end
    local a = ins.from_table(t):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_large.bin"
    ins.write_bin(path, a)
    local result = ins.read_bin(path, ins.float64)
    assert.is_not_nil(result)
    assert.are.equal(1000, result.numel)
    os.remove(path)
  end)

  it("write creates file gpu", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_create.bin"
    ins.write_bin(path, a)
    local f = io.open(path, "r")
    assert.is_not_nil(f)
    f:close()
    os.remove(path)
  end)

  it("write file size correct gpu", function()
    local a = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 }):to(ins.GPUPlace(0))
    local path = "/tmp/insight_lua_audio_gpu_size.bin"
    ins.write_bin(path, a)
    local f = io.open(path, "r")
    local size = f:seek("end")
    f:close()
    assert.are.equal(40, size)
    os.remove(path)
  end)
end)
