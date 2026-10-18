local M = {}

local function append(out, text)
  out[#out + 1] = text
end

local function type_name(contract)
  local kind = type(contract)
  if contract == nil then
    return "any"
  end
  if kind == "table" then
    local name = rawget(contract, "__argrule_name")
    if type(name) == "string" then
      return name
    end
    local parts = {}
    local i = 1
    while contract[i] ~= nil do
      parts[#parts + 1] = type_name(contract[i])
      i = i + 1
    end
    if #parts > 0 then
      return table.concat(parts, "|")
    end
    return "object"
  end
  if kind == "string" then
    return contract
  end
  if kind == "function" then
    return "predicate"
  end
  return kind
end

function M.render(meta)
  local name = meta.name or "function"
  local out = {}
  append(out, "usage: " .. name .. "(")
  local width = 0
  for i = 1, #meta.rules do
    local rule = meta.rules[i]
    if #rule.name > width then
      width = #rule.name
    end
  end
  for i = 1, #meta.rules do
    local rule = meta.rules[i]
    local label = rule.name
    if rule.optional then
      label = "[" .. label .. "]"
    end
    local pad = string.rep(" ", width - #rule.name)
    local line = "  " .. label .. pad .. " = " .. type_name(rule.type)
    if rule.doc then
      line = line .. " -- " .. rule.doc
    end
    if rule.has_default then
      line = line .. " [default]"
    end
    append(out, line)
  end
  append(out, ")")
  if meta.doc then
    append(out, "  " .. meta.doc)
  end
  return table.concat(out, "\n")
end

return M
