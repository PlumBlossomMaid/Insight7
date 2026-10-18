local state = require "argrule._state"
local usage_mod = require "argrule._usage"

local M = {}

local contract_mt = {}

function contract_mt.__call(self, value)
  if
    self.__argrule_name == "Callable"
    and type(value) == "table"
    and getmetatable(value) == nil
    and next(value) == nil
  then
    return self
  end
  return self.__argrule_predicate(value)
end

function contract_mt.__tostring(self)
  return self.__argrule_name
end

local function is_contract(value)
  return type(value) == "table" and getmetatable(value) == contract_mt
end

local function make_contract(name, predicate)
  return setmetatable({
    __argrule_name = name,
    __argrule_predicate = predicate,
  }, contract_mt)
end

local function sequence(spec, name, min_count, max_count)
  if type(spec) ~= "table" then
    error(name .. " expects a table", 2)
  end

  local n = 0
  while spec[n + 1] ~= nil do
    n = n + 1
  end

  local count = 0
  for key in pairs(spec) do
    if type(key) ~= "number" or key < 1 or key ~= math.floor(key) then
      error(name .. " expects an array-like table", 2)
    end
    count = count + 1
  end

  if count ~= n then
    error(name .. " expects a contiguous array-like table", 2)
  end
  if n < min_count then
    error(name .. " expects at least " .. min_count .. " contract", 2)
  end
  if max_count and n > max_count then
    error(name .. " expects at most " .. max_count .. " contract", 2)
  end

  return n
end

local function type_object_name(value)
  if type(value) ~= "table" then
    return type(value)
  end

  local name = rawget(value, "__name") or rawget(value, "_name") or rawget(value, "__typename")
  if type(name) == "string" then
    return name
  end

  local mt = getmetatable(value)
  if type(mt) == "table" then
    name = rawget(mt, "__name") or rawget(mt, "_name") or rawget(mt, "__typename")
    if type(name) == "string" then
      return name
    end
  end

  return "object"
end

local function class_base(class)
  if type(class) ~= "table" then
    return nil
  end
  return rawget(class, "_base") or rawget(class, "_parent") or rawget(class, "_parent_with_init")
end

local function matches_class_chain(class, target)
  while type(class) == "table" do
    if class == target then
      return true
    end
    class = class_base(class)
  end
  return false
end

local function matches_type_object(value, target)
  if value == target then
    return true
  end

  if type(value) == "table" or type(value) == "userdata" then
    local is_a = value.is_a
    if type(is_a) == "function" then
      local ok, result = pcall(is_a, value, target)
      if ok and result then
        return true
      end
    end
  end

  if type(value) == "table" then
    local class = rawget(value, "class") or rawget(value, "_class")
    if matches_class_chain(class, target) then
      return true
    end
  end

  local mt = getmetatable(value)
  while type(mt) == "table" do
    if mt == target or rawget(mt, "__index") == target or rawget(mt, "_class") == target then
      return true
    end
    if matches_class_chain(mt, target) or matches_class_chain(rawget(mt, "_class"), target) then
      return true
    end
    mt = class_base(mt) or getmetatable(mt)
  end

  return false
end

local function is_callable(value)
  if type(value) == "function" then
    return true
  end
  local mt = getmetatable(value)
  return type(mt) == "table" and type(mt.__call) == "function"
end

local function builtin(name)
  return make_contract(name, function(value)
    return type(value) == name
  end)
end

M.any = make_contract("any", function()
  return true
end)
M.Nil = make_contract("nil", function(value)
  return value == nil
end)
M["nil"] = M.Nil
M.boolean = builtin "boolean"
M.number = builtin "number"
M.string = builtin "string"
M.table = builtin "table"
M.userdata = builtin "userdata"
M.thread = builtin "thread"
M.func = builtin "function"
M["function"] = M.func

M.Callable = make_contract("Callable", is_callable)

function M.contract(contract)
  local kind = type(contract)
  if contract == nil then
    return M.any
  end
  if is_contract(contract) then
    return contract
  end
  if kind == "string" then
    if state.is_lua_type(contract) then
      return M[contract]
    end
    error("unknown string contract '" .. contract .. "'", 3)
  end
  if kind == "function" then
    return make_contract("predicate", contract)
  end
  if kind == "table" then
    return make_contract(type_object_name(contract), function(value)
      return matches_type_object(value, contract)
    end)
  end
  error("invalid type contract", 3)
end

function M.alias(spec)
  sequence(spec, "alias", 1, 1)
  local base = M.contract(spec[1])
  return make_contract(tostring(base), function(value)
    return base(value)
  end)
end

function M.Union(spec)
  local n = sequence(spec, "Union", 1)
  local contracts = {}
  local names = {}
  for i = 1, n do
    contracts[i] = M.contract(spec[i])
    names[i] = tostring(contracts[i])
  end
  return make_contract(table.concat(names, "|"), function(value)
    for i = 1, n do
      if contracts[i](value) then
        return true
      end
    end
    return false
  end)
end

function M.Optional(spec)
  sequence(spec, "Optional", 1, 1)
  return M.Union { spec[1], M.Nil }
end

local function length_of(value)
  local ok, n = pcall(function()
    return #value
  end)
  if ok and type(n) == "number" and n >= 0 then
    return n
  end

  if type(value) == "table" or type(value) == "userdata" then
    local len = value.len
    if type(len) == "function" then
      ok, n = pcall(len, value)
      if ok and type(n) == "number" then
        return n
      end
    end

    local size = value.size
    if type(size) == "function" then
      ok, n = pcall(size, value)
      if ok and type(n) == "number" then
        return n
      end
    end
  end

  return nil
end

function M.List(spec)
  sequence(spec, "List", 1, 1)
  local elem_is = M.contract(spec[1])
  return make_contract("List[" .. tostring(elem_is) .. "]", function(value)
    local kind = type(value)
    if kind ~= "table" and kind ~= "userdata" then
      return false
    end
    local n = length_of(value)
    if n == nil then
      return false
    end
    for i = 1, n do
      if not elem_is(value[i]) then
        return false
      end
    end
    return true
  end)
end

function M.Tuple(spec)
  local n = sequence(spec, "Tuple", 1)
  local contracts = {}
  local names = {}
  for i = 1, n do
    contracts[i] = M.contract(spec[i])
    names[i] = tostring(contracts[i])
  end
  return make_contract("Tuple[" .. table.concat(names, ",") .. "]", function(value)
    local kind = type(value)
    if kind ~= "table" and kind ~= "userdata" then
      return false
    end
    if length_of(value) ~= n then
      return false
    end
    for i = 1, n do
      if not contracts[i](value[i]) then
        return false
      end
    end
    return true
  end)
end

function M.Dict(spec)
  sequence(spec, "Dict", 2, 2)
  local key_is = M.contract(spec[1])
  local value_is = M.contract(spec[2])
  return make_contract(
    "Dict[" .. tostring(key_is) .. "," .. tostring(value_is) .. "]",
    function(value)
      if type(value) ~= "table" then
        return false
      end
      for key, item in pairs(value) do
        if not key_is(key) or not value_is(item) then
          return false
        end
      end
      return true
    end
  )
end

function M.Literal(spec)
  local n = sequence(spec, "Literal", 1)
  local names = {}
  for i = 1, n do
    names[i] = tostring(spec[i])
  end
  return make_contract("Literal[" .. table.concat(names, ",") .. "]", function(value)
    for i = 1, n do
      if value == spec[i] then
        return true
      end
    end
    return false
  end)
end

function M.type_name(contract)
  return tostring(M.contract(contract))
end

function M.usage(fn)
  local meta = state.meta[fn]
  if meta == nil then
    return nil
  end
  return usage_mod.render(meta)
end

function M.source(fn)
  local meta = state.meta[fn]
  return meta and meta.source or nil
end

function M.strict(value)
  if value ~= nil then
    state.strict_enabled = not not value
  end
  return state.strict_enabled
end

M._state = state

return M
