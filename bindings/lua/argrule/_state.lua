local M = {}

M.meta = setmetatable({}, { __mode = "k" })
M.strict_enabled = true

local lua_types = {
  ["nil"] = true,
  boolean = true,
  number = true,
  string = true,
  table = true,
  ["function"] = true,
  userdata = true,
  thread = true,
}

function M.is_lua_type(name)
  return lua_types[name] == true
end

return M
