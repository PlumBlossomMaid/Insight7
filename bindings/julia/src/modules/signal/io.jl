# modules/signal/io.jl
# Signal I/O functions.

"""
    read_bin(file::String; dtype=UInt8, num_samples=0, offset=0) -> InsightArray

Read binary data from a file.
"""
function read_bin(file::String; dtype::DataType=UInt8,
                  num_samples::Int=0, offset::Int=0)::InsightArray
    dtype_code = _dtype_code(dtype)
    ptr = ccall((:insight_jl_read_bin, LIB_INSIGHT), Ptr{Cvoid},
                (Cstring, Int32, Int64, Int64),
                file, Int32(dtype_code), Int64(num_samples), Int64(offset))
    return _wrap_ptr(ptr, "read_bin")
end

"""
    write_bin(file::String, data::InsightArray; append=true)

Write binary data to a file.
"""
function write_bin(file::String, data::InsightArray; append::Bool=true)
    ccall((:insight_jl_write_bin, LIB_INSIGHT), Cvoid,
          (Cstring, Ptr{Cvoid}, Int32), file, data, Int32(append ? 1 : 0))
end

"""
    pack_bin(data::InsightArray) -> InsightArray

Pack array data into a binary representation.
"""
function pack_bin(data::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_pack_bin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), data)
    return _wrap_ptr(ptr, "pack_bin")
end

"""
    unpack_bin(binary::InsightArray, dtype::DataType; endianness="L") -> InsightArray

Unpack binary data into an array of the specified dtype.
"""
function unpack_bin(binary::InsightArray, dtype::DataType;
                    endianness::String="L")::InsightArray
    dtype_code = _dtype_code(dtype)
    ptr = ccall((:insight_jl_unpack_bin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Cstring),
                binary, Int32(dtype_code), endianness)
    return _wrap_ptr(ptr, "unpack_bin")
end

"""
    read_sigmf(data_file::String; meta_file="", num_samples=0, offset=0) -> InsightArray

Read data from a SigMF recording.
"""
function read_sigmf(data_file::String; meta_file::String="",
                    num_samples::Int=0, offset::Int=0)::InsightArray
    ptr = ccall((:insight_jl_read_sigmf, LIB_INSIGHT), Ptr{Cvoid},
                (Cstring, Cstring, Int64, Int64),
                data_file, meta_file, Int64(num_samples), Int64(offset))
    return _wrap_ptr(ptr, "read_sigmf")
end

"""
    write_sigmf(data_file::String, data::InsightArray; append=true)

Write data in SigMF format.
"""
function write_sigmf(data_file::String, data::InsightArray; append::Bool=true)
    ccall((:insight_jl_write_sigmf, LIB_INSIGHT), Cvoid,
          (Cstring, Ptr{Cvoid}, Int32), data_file, data, Int32(append ? 1 : 0))
end
