#!/usr/bin/env julia
#=
Timer Demo — Verifies the Timer API works correctly.
=#
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia", "src"))

using Insight
using Insight: Timer, timer_start, timer_stop, timer_elapsed_ms, timer_destroy

# CPU Timer
let
    t = Timer(0, 0)
    timer_start(t)
    sleep(0.005)
    timer_stop(t)
    ms = timer_elapsed_ms(t)
    println(string("CPU timer: ", round(ms, digits=3), " ms"))
    if ms < 0.0 || ms > 100.0
        error(string("CPU timer out of range: ", ms))
    end
    timer_destroy(t)
end

# GPU Timer (if available)
if has_device(1)
    load_backend("rocm")
    t = Timer(1, 0)
    timer_start(t)
    a = Insight.ones(Int64[256, 256], Insight.float32, Int32(1))
    b = Insight.full(Int64[256, 256], 2.0, Insight.float32, Int32(1))
    c = Insight.add(a, b)
    timer_stop(t)
    ms = timer_elapsed_ms(t)
    println(string("GPU timer: ", round(ms, digits=3), " ms"))
    timer_destroy(t)
else
    println("GPU timer: SKIPPED (GPU not available)")
end

println("OK")
