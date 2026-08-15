-- Signal peak_finding CPU binding tests
-- Run with:
--   LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
--   LD_LIBRARY_PATH=build/backends/cpu ~/.luarocks/bin/busted tests/cpu/lua/test_signal_peak_finding.lua

describe("Signal Peak Finding CPU Tests", function()
  local ins
  setup(function()
    ins = require("insight")
  end)

  it("argrelmax", function()
    local data = ins.from_table({ 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0 })
    local result = ins.signal.argrelmax(data)
    assert.is_not_nil(result)
  end)

  it("argrelmin", function()
    local data = ins.from_table({ 3.0, 1.0, 4.0, 0.0, 5.0, 2.0, 6.0 })
    local result = ins.signal.argrelmin(data)
    assert.is_not_nil(result)
  end)

  it("argrelextrema_greater", function()
    local data = ins.from_table({ 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0 })
    local result = ins.signal.argrelextrema(data, "greater")
    assert.is_not_nil(result)
  end)

  it("argrelmax_order2", function()
    local data = ins.from_table({ 0.0, 1.0, 3.0, 2.0, 1.0, 5.0, 2.0, 1.0, 0.0 })
    local result = ins.signal.argrelmax(data, 1, 2)
    assert.is_not_nil(result)
  end)

  it("argrelmin_order2", function()
    local data = ins.from_table({ 5.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 4.0, 5.0 })
    local result = ins.signal.argrelmin(data, 1, 2)
    assert.is_not_nil(result)
  end)

  it("argrelmax_no_peaks", function()
    local data = ins.from_table({ 1.0, 2.0, 3.0, 4.0, 5.0 })
    local result = ins.signal.argrelmax(data)
    assert.is_not_nil(result)
  end)

  it("argrelmin_no_valleys", function()
    local data = ins.from_table({ 5.0, 4.0, 3.0, 2.0, 1.0 })
    local result = ins.signal.argrelmin(data)
    assert.is_not_nil(result)
  end)

  it("argrelmax_sine", function()
    local t = {}
    for i = 0, 99 do
      t[i + 1] = math.sin(4 * math.pi * i / 100)
    end
    local data = ins.from_table(t)
    local result = ins.signal.argrelmax(data)
    assert.is_not_nil(result)
  end)

  it("argrelmin_sine", function()
    local t = {}
    for i = 0, 99 do
      t[i + 1] = math.sin(4 * math.pi * i / 100)
    end
    local data = ins.from_table(t)
    local result = ins.signal.argrelmin(data)
    assert.is_not_nil(result)
  end)
end)
