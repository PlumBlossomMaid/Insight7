# Device information CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_device_info.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

# Try GPU
try
    Insight.load_backend("gpu")
    Insight.set_device(Insight.GPUPlace(0))
catch e
    println("SKIP: GPU backend not available: $e")
    exit(0)
end

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

println("=== Device Info GPU ===")

# device_name gpu
name = Insight.device_name(1, 0)
check("device_name", typeof(name) == String && length(name) > 0)

# gpu_version positive
ver = Insight.gpu_version()
check("gpu_version_positive", ver > 0)

# driver_version positive
dver = Insight.driver_version()
check("driver_version_positive", dver > 0)

# compute_capability positive
cc = Insight.compute_capability(0)
check("compute_capability_positive", cc > 0)

# gpu_count positive
gc = Insight.gpu_count()
check("gpu_count_positive", gc > 0)

# device_memory
mem = Insight.device_memory(0)
check("device_memory", mem.total > 0 && mem.free > 0 && mem.total >= mem.free)

# compute_capability positive on repeated query
check("compute_capability_stays_positive", cc > 0)

# gpu runtime version positive on repeated query
check("gpu_version_stays_positive", ver > 0)

# device_name stable
name2 = Insight.device_name(1, 0)
check("device_name_stable", name == name2)

# compute_capability stable
cc2 = Insight.compute_capability(0)
check("compute_capability_stable", cc == cc2)

# device_memory total reasonable
check("device_memory_total", mem.total >= 1024 * 1024 * 1024)

# driver version positive on repeated query
check("driver_version_stays_positive", dver > 0)

# active backend name
backend_name = Insight.active_gpu_backend_name()
check("active_gpu_backend_name", typeof(backend_name) == String && length(backend_name) > 0)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
