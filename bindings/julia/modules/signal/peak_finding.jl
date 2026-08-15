# modules/signal/peak_finding.jl
# Peak finding functions.

"""
    argrelmax(data::InsightArray; axis=1, order=1) -> Vector{Int64}

Find local maxima indices along an axis.
"""
function argrelmax(data::InsightArray; axis::Int=1, order::Int=1)::Vector{Int64}
    out = Vector{Int64}(undef, numel(data))
    n = ccall((:insight_jl_argrelmax, LIB_INSIGHT), Int64,
              (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}),
              data, Int32(_julia_axis(data, axis)), Int32(order), out)
    return out[1:n]
end

"""
    argrelmin(data::InsightArray; axis=1, order=1) -> Vector{Int64}

Find local minima indices along an axis.
"""
function argrelmin(data::InsightArray; axis::Int=1, order::Int=1)::Vector{Int64}
    out = Vector{Int64}(undef, numel(data))
    n = ccall((:insight_jl_argrelmin, LIB_INSIGHT), Int64,
              (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}),
              data, Int32(_julia_axis(data, axis)), Int32(order), out)
    return out[1:n]
end
