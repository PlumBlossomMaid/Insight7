local rule = require("argrule.rule")
local argrule = require("argrule")

local parse_options = rule({
  {
    name = "device",
    type = argrule.string,
    default = "cpu",
    check = function(value)
      return value == "cpu" or value == "gpu" or value == "all"
    end,
  },
  { name = "seed", type = argrule.number, default = 42 },
  {
    name = "iterations",
    type = argrule.number,
    default = 0,
    check = function(value)
      return value >= 0 and value == math.floor(value)
    end,
  },
  { name = "timer", type = argrule.boolean, default = false },
  { name = "info", type = argrule.boolean, default = false },
  { name = "profiler", type = argrule.boolean, default = false },
})(function(device, seed, iterations, timer, info, profiler)
  return {
    device = device,
    seed = seed,
    iterations = iterations,
    timer = timer,
    info = info,
    profiler = profiler,
  }
end)

local function parse_args(argv)
  local raw = {}
  local i = 1
  while i <= #argv do
    local option = argv[i]
    if option == "--device" or option == "--seed" or option == "--iterations" then
      if i == #argv then
        error("missing value for " .. option, 2)
      end
      local value = argv[i + 1]
      if option == "--device" then
        raw.device = value
      elseif option == "--seed" then
        raw.seed = tonumber(value)
        if raw.seed == nil then
          error("invalid value for --seed: " .. tostring(value), 2)
        end
      else
        raw.iterations = tonumber(value)
        if raw.iterations == nil then
          error("invalid value for --iterations: " .. tostring(value), 2)
        end
      end
      i = i + 2
    elseif option == "--timer" then
      raw.timer = true
      i = i + 1
    elseif option == "--info" then
      raw.info = true
      i = i + 1
    elseif option == "--profiler" then
      raw.profiler = true
      i = i + 1
    else
      error("unknown option " .. tostring(option), 2)
    end
  end
  return parse_options(raw)
end

return parse_args
