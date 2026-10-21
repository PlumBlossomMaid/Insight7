-- Signal acoustics CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_acoustics.lua

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

describe("Signal Acoustics CUDA Tests", function()
  it("mel2hz", function()
    local hz = ins.signal.mel2hz(1000.0)
    assert.is_not_nil(hz)
    assert.is_true(hz > 0)
  end)

  it("hz2mel", function()
    local mel = ins.signal.hz2mel(1000.0)
    assert.is_not_nil(mel)
    assert.is_true(mel > 0)
  end)

  it("mel_hz_roundtrip", function()
    local hz_orig = 1000.0
    local mel = ins.signal.hz2mel(hz_orig)
    local hz_back = ins.signal.mel2hz(mel)
    assert.is_true(math.abs(hz_back - hz_orig) < 1.0)
  end)

  it("mel_frequencies", function()
    local freqs = ins.signal.mel_frequencies(128, 0.0, 11025.0, false)
    assert.is_not_nil(freqs)
    assert.are.equal(128, freqs.numel)
  end)

  it("mel_frequencies_custom", function()
    local freqs = ins.signal.mel_frequencies(64, 20.0, 8000.0, false)
    assert.is_not_nil(freqs)
    assert.are.equal(64, freqs.numel)
  end)

  it("hz2bark", function()
    local bark = ins.signal.hz2bark(1000.0)
    assert.is_not_nil(bark)
    assert.is_true(bark > 0)
  end)

  it("bark2hz", function()
    local hz = ins.signal.bark2hz(10.0)
    assert.is_not_nil(hz)
    assert.is_true(hz > 0)
  end)

  it("bark_hz_roundtrip", function()
    local hz_orig = 1000.0
    local bark = ins.signal.hz2bark(hz_orig)
    local hz_back = ins.signal.bark2hz(bark)
    assert.is_true(math.abs(hz_back - hz_orig) < 10.0)
  end)
end)
