-- Signal spectral CUDA binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
--   ~/.luarocks/bin/busted tests/cuda/lua/test_signal_spectral.lua

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

describe("Signal Spectral CUDA Tests", function()
  -- Helper: build a sine wave of length n at freq Hz, sampled at fs Hz
  local function make_sine(freq, fs, n)
    local t = {}
    for i = 0, n - 1 do
      t[i + 1] = math.sin(2 * math.pi * freq * i / fs)
    end
    return ins.from_table(t):to(ins.GPUPlace(0))
  end

  it("welch", function()
    local x = make_sine(10, 256, 1024)
    local result = ins.signal.welch(x, 256.0, "hann", 256, 128, 256, "density")
    assert.is_not_nil(result)
  end)

  it("periodogram", function()
    local x = make_sine(10, 256, 512)
    local result = ins.signal.periodogram(x, 256.0, "hann", 256, "density")
    assert.is_not_nil(result)
  end)

  it("csd", function()
    local x = make_sine(10, 256, 1024)
    local y = make_sine(10, 256, 1024)
    local result = ins.signal.csd(x, y, 256.0, "hann", 256, 128, 256)
    assert.is_not_nil(result)
  end)

  it("coherence", function()
    local x = make_sine(10, 256, 1024)
    local y = make_sine(10, 256, 1024)
    local result = ins.signal.coherence(x, y, 256.0, "hann", 256, 128, 256)
    assert.is_not_nil(result)
  end)

  it("spectrogram", function()
    local x = make_sine(10, 256, 1024)
    local result = ins.signal.spectrogram(x, 256.0, "hann", 256, 128, 256)
    assert.is_not_nil(result)
  end)

  it("stft", function()
    local x = make_sine(10, 256, 1024)
    local result = ins.signal.stft(x, 256.0, "hann", 256, 128, 256)
    assert.is_not_nil(result)
  end)

  it("vectorstrength", function()
    local events = ins.from_table({ 0.0, 1.0, 2.0, 3.0 }):to(ins.GPUPlace(0))
    local result = ins.signal.vectorstrength(events, 1.0, 1.0)
    assert.is_not_nil(result)
  end)
end)
