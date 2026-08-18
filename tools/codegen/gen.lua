#!/usr/bin/env lua

local function script_dir()
  local source = debug.getinfo(1, "S").source
  if source:sub(1, 1) == "@" then
    source = source:sub(2)
  end
  return source:match("^(.*)/[^/]*$") or "."
end

local root = script_dir()
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

local args = {...}
local out_dir = args[1] or "generated/codegen"

local function mkdir_p(path)
  local ok = os.execute(string.format("mkdir -p %q", path))
  if ok ~= true and ok ~= 0 then
    error("failed to create directory: " .. path)
  end
end

local function write_file(path, content)
  local file = assert(io.open(path, "w"))
  file:write(content)
  file:close()
end

local function validate_op(module_name, op)
  assert(type(op.name) == "string" and op.name ~= "", module_name .. ": op.name is required")
  assert(type(op.kind) == "string" and op.kind ~= "", op.name .. ": op.kind is required")
  assert(type(op.inputs) == "table" and #op.inputs > 0, op.name .. ": inputs are required")
  assert(type(op.outputs) == "table" and #op.outputs > 0, op.name .. ": outputs are required")
  assert(type(op.dtypes) == "table" and #op.dtypes > 0, op.name .. ": dtypes are required")
  if op.broadcast ~= nil then
    assert(type(op.broadcast) == "boolean", op.name .. ": broadcast must be boolean")
  end
  for _, backend in ipairs({"host", "device"}) do
    assert(type(op[backend]) == "table", op.name .. ": " .. backend .. " strategy is required")
  end
end

local module_names = {"cast", "creation", "elementwise", "reduction", "unary"}
local modules = {}
for _, module_name in ipairs(module_names) do
  modules[module_name] = dofile(root .. "/schema/" .. module_name .. ".lua")
end

local all_ops = {}
for _, module_name in ipairs(module_names) do
  local ops = modules[module_name]
  assert(type(ops) == "table", module_name .. " schema must return a table")
  for _, op in ipairs(ops) do
    validate_op(module_name, op)
    all_ops[#all_ops + 1] = op
  end
end

table.sort(all_ops, function(a, b)
  return a.name < b.name
end)

local manifest = require("templates.manifest")
local op_header = require("templates.op_header")
local kernel_plan = require("templates.kernel_plan")
local source_manifest = require("templates.source_manifest")

mkdir_p(out_dir .. "/include/insight/generated")
mkdir_p(out_dir .. "/docs")

write_file(out_dir .. "/docs/op_manifest.md", manifest.render(all_ops))
write_file(out_dir .. "/docs/kernel_plan.md", kernel_plan.render_markdown(all_ops))
write_file(out_dir .. "/docs/source_manifest.md", source_manifest.render_markdown(all_ops))
write_file(out_dir .. "/include/insight/generated/kernel_plan.h",
           kernel_plan.render(all_ops))
write_file(out_dir .. "/include/insight/generated/source_manifest.h",
           source_manifest.render(all_ops))
for _, module_name in ipairs(module_names) do
  write_file(out_dir .. "/include/insight/generated/" .. module_name .. "_ops.h",
             op_header.render(module_name, modules[module_name]))
end

print(string.format("generated %d ops into %s", #all_ops, out_dir))
