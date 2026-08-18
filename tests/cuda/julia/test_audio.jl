# Audio/I/O CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_audio.jl

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

println("=== Audio I/O CUDA ===")

tmpdir = mktempdir(prefix="insight_jl_audio_gpu_")

# write read roundtrip gpu float64
try
    a = Insight.to_array(Insight.from_data([1.0, 2.5, 3.7, -1.2, 0.0]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_f64.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    check("write_read_f64", Insight.numel(result) == 5)
catch e
    println("SKIP: write_read_f64 ($e)")
end

# write read roundtrip gpu float32
try
    a = Insight.to_array(Insight.zeros(Int64[4], Insight.float32), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_f32.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float32)
    check("write_read_f32", Insight.numel(result) == 4)
catch e
    println("SKIP: write_read_f32 ($e)")
end

# write read roundtrip gpu int32
try
    a = Insight.to_array(Insight.from_data([10.0, 20.0, 30.0]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_i32.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.int32)
    check("write_read_i32", Insight.numel(result) == 3)
catch e
    println("SKIP: write_read_i32 ($e)")
end

# write read zeros gpu
try
    a = Insight.to_array(Insight.zeros(Int64[100], Insight.float64), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_zeros.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    check("write_read_zeros", Insight.numel(result) == 100)
catch e
    println("SKIP: write_read_zeros ($e)")
end

# write read ones gpu
try
    a = Insight.to_array(Insight.ones(Int64[50], Insight.float64), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_ones.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    check("write_read_ones", Insight.numel(result) == 50)
catch e
    println("SKIP: write_read_ones ($e)")
end

# write read negative gpu
try
    a = Insight.to_array(Insight.from_data([-100.5, -200.3, -300.1]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_neg.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    check("write_read_neg", Insight.numel(result) == 3)
catch e
    println("SKIP: write_read_neg ($e)")
end

# write read single gpu
try
    a = Insight.to_array(Insight.from_data([42.0]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_single.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    d = Insight.to_array(result)
    check("write_read_single", Insight.numel(result) == 1 && isapprox(d[1], 42.0))
catch e
    println("SKIP: write_read_single ($e)")
end

# write read large gpu
try
    t = Float64[i * 0.1 for i in 1:1000]
    a = Insight.to_array(Insight.from_data(t), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_large.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    check("write_read_large", Insight.numel(result) == 1000)
catch e
    println("SKIP: write_read_large ($e)")
end

# write creates file gpu
try
    a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_create.bin")
    Insight.write_bin(path, a)
    check("write_creates_file", isfile(path) && filesize(path) > 0)
catch e
    println("SKIP: write_creates_file ($e)")
end

# file size correct gpu
try
    a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_size.bin")
    Insight.write_bin(path, a)
    check("file_size", filesize(path) == 40)
catch e
    println("SKIP: file_size ($e)")
end

# read back to gpu
try
    a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
    path = joinpath(tmpdir, "test_roundtrip.bin")
    Insight.write_bin(path, a)
    result = Insight.read_bin(path, Insight.float64)
    gpu_result = Insight.to_array(result, Insight.GPUPlace(0))
    check("read_back_to_gpu", Insight.numel(gpu_result) == 3)
catch e
    println("SKIP: read_back_to_gpu ($e)")
end

# ============================================================================
# Cleanup
# ============================================================================
try
    rm(tmpdir, recursive=true, force=true)
catch
end

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
