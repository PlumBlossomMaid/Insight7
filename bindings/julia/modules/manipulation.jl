# modules/manipulation.jl
# Array manipulation functions.

"""
    reshape(x::InsightArray, dims::Vector{Int64}) -> InsightArray

Reshape an array to the given dimensions.
"""
function reshape end

"""
    transpose(x::InsightArray) -> InsightArray

Transpose a 2-D array.
"""
function transpose end

"""
    permute(x::InsightArray, axes::Vector{Int32}) -> InsightArray

Permute the axes of an array.
"""
function permute end

"""
    swapaxes(x::InsightArray, axis1::Int, axis2::Int) -> InsightArray

Swap two axes of an array.
"""
function swapaxes end

"""
    moveaxis(x::InsightArray, source::Int, destination::Int) -> InsightArray

Move an axis to a new position.
"""
function moveaxis end

"""
    fliplr(x::InsightArray) -> InsightArray

Flip an array left/right along the second Julia dimension.
"""
function fliplr end

"""
    flipud(x::InsightArray) -> InsightArray

Flip an array up/down along the first Julia dimension.
"""
function flipud end

"""
    rot90(x::InsightArray; k::Int=1, axes::Vector{Int32}=Int32[1, 2]) -> InsightArray

Rotate an array by 90 degrees in the plane specified by `axes`.
"""
function rot90 end

"""
    diag_fn(x::InsightArray; k::Int=0) -> InsightArray

Extract the k-th diagonal, or construct a diagonal array.
"""
function diag_fn end

"""
    diagonal(x::InsightArray; offset::Int=0, axis1::Int=1, axis2::Int=2) -> InsightArray

Return the specified diagonals.
"""
function diagonal end

"""
    tril(x::InsightArray; k::Int=0) -> InsightArray

Return the lower triangular part of an array.
"""
function tril end

"""
    triu(x::InsightArray; k::Int=0) -> InsightArray

Return the upper triangular part of an array.
"""
function triu end

"""
    diff_fn(x::InsightArray; n::Int=1, axis::Int=-1) -> InsightArray

Compute the n-th discrete difference along the given axis.
"""
function diff_fn end
