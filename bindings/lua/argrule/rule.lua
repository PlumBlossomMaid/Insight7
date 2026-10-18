local argrule = require "argrule"
local state = require "argrule._state"

local table_unpack = table.unpack or unpack

local M = {}

local function clone_rule(raw, index)
  if type(raw) ~= "table" then
    error("rule #" .. index .. " must be a table", 3)
  end
  local name = raw.name or raw[1]
  if type(name) ~= "string" then
    error("rule #" .. index .. " must have a string name", 3)
  end
  local type_contract = raw.type
  if type_contract == nil then
    type_contract = raw[2]
  end
  local has_default = raw.default ~= nil or raw.defaulta ~= nil or raw.defaultf ~= nil
  return {
    name = name,
    type = type_contract,
    is = argrule.contract(type_contract),
    doc = raw.doc or raw.help,
    default = raw.default,
    defaulta = raw.defaulta,
    defaultf = raw.defaultf,
    has_default = has_default,
    opt = raw.opt == true,
    check = raw.check,
    optional = raw.opt == true or has_default,
  }
end

local function build_meta(spec)
  if type(spec) ~= "table" then
    error("rule spec must be a table", 3)
  end
  if spec.noordered and spec.nonamed then
    error("rule spec must allow positional or named calls", 3)
  end
  local rules = {}
  local i = 1
  while type(spec[i]) == "table" do
    rules[i] = clone_rule(spec[i], i)
    i = i + 1
  end
  return {
    rules = rules,
    doc = spec.doc or spec.help,
    nonamed = spec.nonamed == true,
    noordered = spec.noordered == true,
    source = "interpreted",
  }
end

local function has_self_rule(meta)
  return meta.rules[1] ~= nil and meta.rules[1].name == "self"
end

local function is_named_key(meta, key)
  if type(key) ~= "string" then
    return false
  end
  for i = 1, #meta.rules do
    if meta.rules[i].name == key then
      return true
    end
  end
  return false
end

local function has_named_key(meta, t)
  for key in pairs(t) do
    if is_named_key(meta, key) then
      return true
    end
  end
  return false
end

local function table_array_len(t)
  local n = 0
  while t[n + 1] ~= nil do
    n = n + 1
  end
  return n
end

local function bad(meta, message)
  local text = message .. "\n" .. require("argrule._usage").render(meta)
  error(text, 3)
end

local function fill_defaults(meta, args)
  for i = 1, #meta.rules do
    local rule = meta.rules[i]
    if args[i] == nil then
      if rule.default ~= nil then
        args[i] = rule.default
      elseif rule.defaulta ~= nil then
        for j = 1, #meta.rules do
          if meta.rules[j].name == rule.defaulta then
            args[i] = args[j]
            break
          end
        end
      elseif rule.defaultf ~= nil then
        args[i] = rule.defaultf(args)
      end
    end
  end
end

local function validate(meta, args)
  fill_defaults(meta, args)
  for i = 1, #meta.rules do
    local rule = meta.rules[i]
    local value = args[i]
    if value == nil then
      if not rule.optional then
        return false, "missing argument '" .. rule.name .. "'"
      end
    else
      if state.strict_enabled and not rule.is(value) then
        return false, "bad argument #" .. i .. " '" .. rule.name .. "'"
      end
      if rule.check and not rule.check(value, args) then
        return false, "bad argument #" .. i .. " '" .. rule.name .. "'"
      end
    end
  end
  return true
end

local function parse_named(meta, t)
  if meta.nonamed then
    return nil, "named-table calls are disabled"
  end
  local args = {}
  local positional = table_array_len(t)
  if positional > #meta.rules then
    return nil, "too many positional table arguments"
  end
  for i = 1, positional do
    args[i] = t[i]
  end
  for key, value in pairs(t) do
    if type(key) == "string" and is_named_key(meta, key) then
      for i = 1, #meta.rules do
        if meta.rules[i].name == key then
          if args[i] ~= nil then
            return nil, "argument '" .. key .. "' was given twice"
          end
          args[i] = value
          break
        end
      end
    elseif type(key) ~= "number" or key < 1 or key > positional or math.floor(key) ~= key then
      return nil, "unknown argument '" .. tostring(key) .. "'"
    end
  end
  local ok, err = validate(meta, args)
  if not ok then
    return nil, err
  end
  return args
end

local function parse_position_table(meta, t)
  if meta.nonamed then
    return nil, "table calls are disabled"
  end
  local args = {}
  local n = table_array_len(t)
  if n > #meta.rules then
    return nil, "too many positional table arguments"
  end
  for i = 1, n do
    args[i] = t[i]
  end
  local ok, err = validate(meta, args)
  if not ok then
    return nil, err
  end
  return args
end

local function parse_whole_first(meta, t)
  if #meta.rules == 0 then
    return nil, "unexpected table argument"
  end
  local args = { [1] = t }
  local ok, err = validate(meta, args)
  if not ok then
    return nil, err
  end
  return args
end

local function parse_position(meta, n, ...)
  if meta.noordered then
    return nil, "positional calls are disabled"
  end
  if n > #meta.rules then
    return nil, "too many arguments"
  end
  local args = {}
  for i = 1, n do
    args[i] = select(i, ...)
  end
  local ok, err = validate(meta, args)
  if not ok then
    return nil, err
  end
  return args
end

local function parse(meta, n, ...)
  if n == 1 then
    local t = ...
    if type(t) == "table" and getmetatable(t) == nil then
      if has_named_key(meta, t) then
        return parse_named(meta, t)
      end
      local args, err = parse_position_table(meta, t)
      if args then
        return args
      end
      args = parse_whole_first(meta, t)
      if args then
        return args
      end
      return nil, err
    end
  end
  return parse_position(meta, n, ...)
end

local function table_with_self(self, t)
  if t.self ~= nil then
    return nil, "argument 'self' was given twice"
  end

  local out = { [1] = self }
  for key, value in pairs(t) do
    if type(key) == "number" and key >= 1 and key == math.floor(key) then
      out[key + 1] = value
    else
      out[key] = value
    end
  end
  return out
end

local function parse_self_call(meta, n, ...)
  if not has_self_rule(meta) or n < 2 then
    return nil, nil, false
  end

  local self = ...
  local second = select(2, ...)
  if n == 2 and type(second) == "table" and getmetatable(second) == nil then
    local with_self, err = table_with_self(self, second)
    if not with_self then
      return nil, err, true
    end
    local args
    args, err = parse(meta, 1, with_self)
    return args, err, true
  end

  local args, err = parse(meta, n, ...)
  return args, err, true
end

local function invoke(fn, meta, ...)
  local n = select("#", ...)
  local args, err, handled_self = parse_self_call(meta, n, ...)
  if not handled_self then
    args, err = parse(meta, n, ...)
  end
  if not args then
    return bad(meta, err)
  end
  return fn(table_unpack(args, 1, #meta.rules))
end

function M.make()
  return function(spec)
    local meta = build_meta(spec)
    local call = spec.call
    if call ~= nil then
      if type(call) ~= "function" then
        error("spec.call must be a function", 2)
      end
      local wrapped = function(...)
        return invoke(call, meta, ...)
      end
      state.meta[wrapped] = meta
      return wrapped
    end
    return function(fn)
      if type(fn) ~= "function" then
        error("decorated value must be a function", 2)
      end
      local wrapped = function(...)
        return invoke(fn, meta, ...)
      end
      state.meta[wrapped] = meta
      return wrapped
    end
  end
end

M.rule = M.make()

setmetatable(M, {
  __call = function(_, ...)
    return M.rule(...)
  end,
})

return M
