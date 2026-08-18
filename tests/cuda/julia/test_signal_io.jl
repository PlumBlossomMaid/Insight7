# Signal io CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_io.jl

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

function approx(a, b; atol=1e-10)
    return abs(a - b) < atol
end

# ============================================================================
# Signal I/O
# ============================================================================
println("=== Signal IO ===")

# write_bin / read_bin roundtrip (GPU write, CPU read)
data = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".bin"
Insight.write_bin(tmpfile, data_gpu)
result = Insight.read_bin(tmpfile, dtype=Float64)
check("write_read_roundtrip", Insight.numel(result) == 5)
rm(tmpfile, force=true)

# write_bin / read_bin float32
data = Insight.from_data(Float32[10.0, 20.0, 30.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".bin"
Insight.write_bin(tmpfile, data_gpu)
result = Insight.read_bin(tmpfile, dtype=Float32)
check("write_read_float32", Insight.numel(result) == 3)
rm(tmpfile, force=true)

# write_bin / read_bin int16
data = Insight.from_data(Int16[100, 200, 300, 400])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".bin"
Insight.write_bin(tmpfile, data_gpu)
result = Insight.read_bin(tmpfile, dtype=Int16)
check("write_read_int16", Insight.numel(result) == 4)
rm(tmpfile, force=true)

# write_bin / read_bin large
large_data = [sin(i) for i in 1:1024]
data = Insight.from_data(large_data)
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".bin"
Insight.write_bin(tmpfile, data_gpu)
result = Insight.read_bin(tmpfile, dtype=Float64)
check("write_read_large", Insight.numel(result) == 1024)
rm(tmpfile, force=true)

# pack_bin
data = Insight.from_data([1.0, 2.0, 3.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
packed = Insight.pack_bin(data_gpu)
check("pack_bin", Insight.numel(packed) > 0)

# unpack_bin
data = Insight.from_data([1.0, 2.0, 3.0, 4.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".bin"
Insight.write_bin(tmpfile, data_gpu)
raw = Insight.read_bin(tmpfile, dtype=UInt8)
unpacked = Insight.unpack_bin(raw, Float64)
check("unpack_bin", Insight.numel(unpacked) > 0)
rm(tmpfile, force=true)

# write_sigmf
data = Insight.from_data([1.0, 2.0, 3.0, 4.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".sigmf-data"
Insight.write_sigmf(tmpfile, data_gpu)
check("write_sigmf", isfile(tmpfile))
rm(tmpfile, force=true)

# read_bin roundtrip data integrity
orig = [1.5, 2.5, 3.5, 4.5, 5.5]
data = Insight.from_data(orig)
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
tmpfile = tempname() * ".bin"
Insight.write_bin(tmpfile, data_gpu)
result = Insight.read_bin(tmpfile, dtype=Float64)
back = Insight.to_data(result)
check("roundtrip_integrity", all(approx.(orig, back)))
rm(tmpfile, force=true)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
