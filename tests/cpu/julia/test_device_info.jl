# Device information CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_device_info.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

passed = 0
failed = 0

function check(name, cond)
    global passed, failed
    if cond
        passed += 1
    else
        failed += 1
        println("FAIL: $name")
    end
end

println("=== Device Info ===")

# device_name cpu
name = Insight.device_name(0, 0)
check("device_name", typeof(name) == String)

# gpu_version
ver = Insight.gpu_version()
check("gpu_version", typeof(ver) == Int && ver >= 0)

# driver_version
dver = Insight.driver_version()
check("driver_version", typeof(dver) == Int && dver >= 0)

# compute_capability
cc = Insight.compute_capability(0)
check("compute_capability", typeof(cc) == Int && cc >= 0)

# gpu_count
gc = Insight.gpu_count()
check("gpu_count", typeof(gc) == Int && gc >= 0)

# device_memory
try
    mem = Insight.device_memory(0)
    check("device_memory", mem.total >= 0 && mem.free >= 0)
catch e
    println("SKIP: device_memory ($e)")
end

# device_name default
name2 = Insight.device_name(0, 0)
check("device_name_default", typeof(name2) == String)

# compute_capability default
cc2 = Insight.compute_capability()
check("compute_capability_default", typeof(cc2) == Int)

# gpu_version nonneg
check("gpu_version_nonneg", ver >= 0)

# driver_version nonneg
check("driver_version_nonneg", dver >= 0)

# gpu_count nonneg
check("gpu_count_nonneg", gc >= 0)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
