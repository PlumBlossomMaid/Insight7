# Complex number CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_complex.jl

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

println("=== Complex CUDA ===")

# is_complex on real
a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
check("is_complex_real", !Insight.is_complex(a))

# to_complex
real = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
c = Insight.to_complex(real)
check("to_complex", Insight.numel(c) == 3)

# to_complex with imag
real2 = Insight.to_array(Insight.from_data([1.0, 2.0]), Insight.GPUPlace(0))
imag2 = Insight.to_array(Insight.from_data([3.0, 4.0]), Insight.GPUPlace(0))
c2 = Insight.to_complex(real2, imag2)
check("to_complex_with_imag", Insight.numel(c2) == 2)

# real part
r = Insight.real_part(c)
check("real_part", Insight.numel(r) == 3)

# imag part
i = Insight.imag_part(c)
check("imag_part", Insight.numel(i) == 3)

# as_complex
a4 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
a4 = Insight.reshape(a4, [2, 2])
ac = Insight.as_complex(a4)
check("as_complex", Insight.numel(ac) == 2)

# as_real
ar = Insight.as_real(ac)
check("as_real", Insight.numel(ar) == 4)

# has_complex_shape
check("has_complex_shape", Insight.has_complex_shape(a4))

# complex abs
abs_c = Insight.abs(c)
check("complex_abs", Insight.numel(abs_c) == 3)

# complex conj
conj_c = Insight.conj(c)
check("complex_conj", Insight.numel(conj_c) == 3)

# complex angle
angle_c = Insight.angle(c)
check("complex_angle", Insight.numel(angle_c) == 3)

# complex add
c_a = Insight.to_complex(Insight.to_array(Insight.from_data([1.0, 2.0]), Insight.GPUPlace(0)))
c_b = Insight.to_complex(Insight.to_array(Insight.from_data([3.0, 4.0]), Insight.GPUPlace(0)))
s = Insight.add(c_a, c_b)
check("complex_add", Insight.numel(s) == 2)

# complex mul
p = Insight.mul(c_a, c_b)
check("complex_mul", Insight.numel(p) == 2)

# complex exp
e = Insight.exp(c_a)
check("complex_exp", Insight.numel(e) == 2)

# roundtrip
a5 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
a5 = Insight.reshape(a5, [2, 2])
c5 = Insight.as_complex(a5)
r5 = Insight.as_real(c5)
check("roundtrip", Insight.numel(r5) == 4)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
