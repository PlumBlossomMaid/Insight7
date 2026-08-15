set(ROCM_HIPIFY_PERL "/opt/rocm/bin/hipify-perl" CACHE FILEPATH
    "HIP source conversion tool")
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
set(INSIGHT_ROCM_ARCHITECTURE "gfx1100" CACHE STRING
    "ROCm GPU architecture")

file(GLOB_RECURSE KERNEL_SOURCES "${CMAKE_SOURCE_DIR}/backends/cuda/kernels/*.cu")
file(GLOB_RECURSE DEVICE_SOURCES "${CMAKE_SOURCE_DIR}/backends/cuda/device/*.cpp")
file(GLOB_RECURSE REGISTRY_SOURCES "${CMAKE_SOURCE_DIR}/backends/cuda/registry/*.cpp")

file(GLOB_RECURSE HEADER_SOURCES "${CMAKE_SOURCE_DIR}/backends/cuda/kernels/*.cuh")

set(ROCM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ROCM_KERNEL_SOURCES "")
set(ROCM_KERNEL_OBJECTS "")
set(ROCM_HEADER_OUTPUTS "")
set(ROCM_KERNEL_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/backends/cuda/registry")

file(MAKE_DIRECTORY "${ROCM_GENERATED_DIR}")
file(CREATE_LINK "${CMAKE_SOURCE_DIR}/backends/cuda/registry"
     "${ROCM_GENERATED_DIR}/registry" SYMBOLIC)

foreach(_header ${HEADER_SOURCES})
    file(RELATIVE_PATH _header_rel "${CMAKE_SOURCE_DIR}/backends/cuda/kernels" "${_header}")
    set(_header_out "${ROCM_GENERATED_DIR}/kernels/${_header_rel}")
    list(APPEND ROCM_HEADER_OUTPUTS "${_header_out}")
    add_custom_command(
        OUTPUT "${_header_out}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${ROCM_GENERATED_DIR}/kernels"
        COMMAND "${ROCM_HIPIFY_PERL}" -quiet -o "${_header_out}.hipify" "${_header}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/strip_bom.py"
                "${_header_out}.hipify" "${_header_out}"
        DEPENDS "${_header}"
        COMMENT "HIPifying ROCm header ${_header_rel}")
endforeach()

foreach(_src ${KERNEL_SOURCES})
    get_filename_component(_src_dir "${_src}" DIRECTORY)
    file(RELATIVE_PATH _src_rel_dir "${CMAKE_SOURCE_DIR}/backends/cuda/kernels" "${_src_dir}")
    list(APPEND ROCM_KERNEL_INCLUDE_DIRS "${_src_dir}")
    file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}/backends/cuda/kernels" "${_src}")
    string(REPLACE "/" "_" _name "${_rel}")
    string(REPLACE ".cu" ".hip.cpp" _name "${_name}")
    set(_hip_src "${ROCM_GENERATED_DIR}/kernels/${_name}")
    set(_obj "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_name}.o")
    list(APPEND ROCM_KERNEL_SOURCES "${_hip_src}")
    list(APPEND ROCM_KERNEL_OBJECTS "${_obj}")
    set(_src_include_flags
        "-I${ROCM_GENERATED_DIR}/kernels/${_src_rel_dir}"
        "-I${_src_dir}"
        "-I${CMAKE_SOURCE_DIR}/backends/cuda/registry"
        "-I${ROCM_PATH}/include"
        "-I${ROCM_PATH}/include/hipblas"
        "-I${ROCM_PATH}/include/hipfft"
        "-I${ROCM_PATH}/include/hiprand"
        "-I${ROCM_PATH}/include/hipsolver")
    add_custom_command(
        OUTPUT "${_hip_src}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${ROCM_GENERATED_DIR}/kernels/${_src_rel_dir}"
        COMMAND "${ROCM_HIPIFY_PERL}" -quiet -o "${_hip_src}.hipify" "${_src}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/strip_bom.py"
                "${_hip_src}.hipify" "${_hip_src}"
        DEPENDS "${_src}"
        COMMENT "HIPifying ROCm kernel ${_rel}")
    add_custom_command(
        OUTPUT "${_obj}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND "${ROCM_HIPCC}" --offload-arch=${INSIGHT_ROCM_ARCHITECTURE}
                -std=c++17 -fPIC -I"${CMAKE_SOURCE_DIR}/include"
                -I"${CMAKE_SOURCE_DIR}/backends/cuda"
                ${_src_include_flags}
                -DINSIGHT_GPU_BACKEND_EXPORTS -c "${_hip_src}" -o "${_obj}"
        DEPENDS "${_hip_src}" ${ROCM_HEADER_OUTPUTS}
        COMMENT "Compiling ROCm kernel ${_rel}")
endforeach()

set(ROCM_DEVICE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/generated/device/cuda_device.hip.cpp")
set(ROCM_DEVICE_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/cuda_device.hip.cpp.o")
add_custom_command(
    OUTPUT "${ROCM_DEVICE_SOURCE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/generated/device"
    COMMAND "${ROCM_HIPIFY_PERL}" -quiet -o "${ROCM_DEVICE_SOURCE}.hipify"
            "${CMAKE_SOURCE_DIR}/backends/cuda/device/cuda_device.cpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/strip_bom.py"
            "${ROCM_DEVICE_SOURCE}.hipify" "${ROCM_DEVICE_SOURCE}"
    DEPENDS "${CMAKE_SOURCE_DIR}/backends/cuda/device/cuda_device.cpp")
add_custom_command(
    OUTPUT "${ROCM_DEVICE_OBJECT}"
    COMMAND "${ROCM_HIPCC}" --offload-arch=${INSIGHT_ROCM_ARCHITECTURE}
            -std=c++17 -fPIC -I"${CMAKE_SOURCE_DIR}/include"
            -I"${CMAKE_SOURCE_DIR}/backends/cuda"
            -I"${CMAKE_SOURCE_DIR}/backends/cuda/registry"
            -I"${ROCM_PATH}/include"
            -DINSIGHT_GPU_BACKEND_EXPORTS -c "${ROCM_DEVICE_SOURCE}"
            -o "${ROCM_DEVICE_OBJECT}"
    DEPENDS "${ROCM_DEVICE_SOURCE}"
    COMMENT "Compiling ROCm device support")

add_library(insight_rocm_backend SHARED
    ${ROCM_KERNEL_OBJECTS}
    ${ROCM_DEVICE_OBJECT}
    ${REGISTRY_SOURCES}
    ${CMAKE_SOURCE_DIR}/backends/cuda/init.cpp)

target_compile_definitions(insight_rocm_backend PRIVATE INSIGHT_GPU_BACKEND_EXPORTS)
target_include_directories(insight_rocm_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hipblas
    ${ROCM_PATH}/include/hipfft
    ${ROCM_PATH}/include/hiprand
    ${ROCM_PATH}/include/hipsolver)
target_link_directories(insight_rocm_backend PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(insight_rocm_backend PRIVATE
    insight_core
    amdhip64
    hipblas
    hipfft
    hiprand
    hipsolver)

set_target_properties(insight_rocm_backend PROPERTIES
    OUTPUT_NAME insight_rocm_backend)
