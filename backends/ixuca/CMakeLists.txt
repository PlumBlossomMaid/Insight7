# backends/ixuca/CMakeLists.txt
# IXUCA (天数智芯 IXUCA) GPU backend using CoreX SDK
#
# Custom compilation: use CoreX clang++ (not CMake CUDA integration,
# since CMake doesn't recognize it as a valid CUDA compiler).

file(GLOB_RECURSE KERNEL_SOURCES "kernels/*.cu")
file(GLOB_RECURSE DEVICE_SOURCES "device/*.cpp")
file(GLOB_RECURSE REGISTRY_SOURCES "registry/*.cpp")

set(IXUCA_CLANGXX "/usr/local/corex/bin/clang++")
set(IXUCA_INCLUDE_DIR "/usr/local/corex/include")
set(IXUCA_LIB_DIR "/usr/local/corex-4.3.8/lib64")

# Step 1: Compile each .cu kernel to .o (without RDC — each kernel file is self-contained)
set(KERNEL_OBJECTS "")
foreach(_src ${KERNEL_SOURCES})
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}/kernels" "${_src}")
    string(REPLACE "/" "_" _obj_name "${_rel}")
    string(REPLACE ".cu" ".o" _obj_name "${_obj_name}")
    set(_obj "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_obj_name}")
    list(APPEND KERNEL_OBJECTS ${_obj})
    add_custom_command(
        OUTPUT ${_obj}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
        COMMAND ${IXUCA_CLANGXX} -fPIC -std=c++17
            --cuda-gpu-arch=ivcore11
            -Xclang -fcuda-allow-variadic-functions -mllvm --skip-double
            -I${IXUCA_INCLUDE_DIR}
            -I${CMAKE_SOURCE_DIR}/include
            -DINSIGHT_GPU_BACKEND_EXPORTS
            -c ${_src} -o ${_obj}
        DEPENDS ${_src}
        COMMENT "Building IXUCA kernel ${_rel}"
    )
endforeach()

# Step 2: Build the shared library (no device link needed without RDC)
add_library(insight_ixuca_backend SHARED
    ${KERNEL_OBJECTS}
    ${DEVICE_SOURCES}
    ${REGISTRY_SOURCES}
    init.cpp
)
target_compile_definitions(insight_ixuca_backend PRIVATE INSIGHT_GPU_BACKEND_EXPORTS)

target_include_directories(insight_ixuca_backend
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${IXUCA_INCLUDE_DIR}
)

target_link_directories(insight_ixuca_backend PRIVATE
    ${IXUCA_LIB_DIR}
    "/usr/local/corex/lib"
)

target_link_libraries(insight_ixuca_backend
    PRIVATE insight_core
    cudart cublas cufft curand cusolver cusparse
    ixthunk
)
