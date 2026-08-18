# tests/CMakeLists.txt

# ========== 收集测试源文件 ==========
file(GLOB CPU_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpu/cpp/*.cpp")
file(GLOB CUDA_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/cpp/*.cpp")

add_executable(insight_tests_cpu ${CPU_TEST_SOURCES})
if(INSIGHT_WITH_CUDA)
    add_executable(insight_tests_cuda ${CUDA_TEST_SOURCES})
endif()

target_link_libraries(insight_tests_cpu
    insight_core
    insight_cpu_backend
    GTest::gtest_main
)

target_include_directories(insight_tests_cpu
    PRIVATE
        ${INSIGHT_INCLUDE_DIR}
        $<$<BOOL:${INSIGHT_LUA_CODEGEN_INTERPRETER}>:${INSIGHT_CODEGEN_OUT_DIR}/include>
)

if(INSIGHT_WITH_CUDA)
    target_link_libraries(insight_tests_cuda
        insight_core
        insight_cpu_backend
        insight_cuda_backend
        GTest::gtest_main
    )
endif()

if(INSIGHT_LUA_CODEGEN_INTERPRETER)
    add_dependencies(insight_tests_cpu insight_codegen)
endif()

# ========== 复制运行时依赖 ==========

function(copy_backend_dll TARGET_NAME BACKEND_TARGET)
    if(NOT BACKEND_TARGET)
        return()
    endif()
    
    add_dependencies(${TARGET_NAME} ${BACKEND_TARGET})
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${BACKEND_TARGET}>"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<TARGET_FILE_NAME:${BACKEND_TARGET}>"
        COMMENT "Copying ${BACKEND_TARGET} to ${TARGET_NAME}"
    )
endfunction()

function(copy_all_backend_dlls TARGET_NAME)
    if(INSIGHT_DYNAMIC_CPU_BACKEND)
        copy_backend_dll(${TARGET_NAME} insight_cpu_backend)
    endif()
    if(INSIGHT_WITH_CUDA AND INSIGHT_DYNAMIC_GPU_BACKEND)
        copy_backend_dll(${TARGET_NAME} insight_cuda_backend)
    endif()
    if(INSIGHT_WITH_IXUCA AND INSIGHT_DYNAMIC_GPU_BACKEND)
        copy_backend_dll(${TARGET_NAME} insight_ixuca_backend)
    endif()
    if(INSIGHT_WITH_SDAA AND INSIGHT_DYNAMIC_GPU_BACKEND)
        copy_backend_dll(${TARGET_NAME} insight_sdaa_backend)
    endif()
endfunction()

# CPU 测试
copy_all_backend_dlls(insight_tests_cpu)

# GPU 测试（如果需要）
if(INSIGHT_WITH_CUDA)
    copy_all_backend_dlls(insight_tests_cuda)
endif()

# IXUCA 测试
if(INSIGHT_WITH_IXUCA)
    add_executable(insight_tests_ixuca ${CUDA_TEST_SOURCES})
    target_link_libraries(insight_tests_ixuca
        insight_core
        insight_cpu_backend
        insight_ixuca_backend
        GTest::gtest_main
    )
    target_include_directories(insight_tests_ixuca
        PRIVATE ${INSIGHT_INCLUDE_DIR}
    )
    copy_all_backend_dlls(insight_tests_ixuca)
endif()

# SDAA 测试
if(INSIGHT_WITH_SDAA)
    add_executable(insight_tests_sdaa ${CUDA_TEST_SOURCES})
    target_link_libraries(insight_tests_sdaa
        insight_core
        insight_cpu_backend
        insight_sdaa_backend
        GTest::gtest_main
    )
    target_include_directories(insight_tests_sdaa
        PRIVATE ${INSIGHT_INCLUDE_DIR}
    )
    copy_all_backend_dlls(insight_tests_sdaa)
endif()

# ========== 自动发现并注册测试 ==========
include(GoogleTest)
gtest_discover_tests(insight_tests_cpu)
if(INSIGHT_WITH_CUDA)
    gtest_discover_tests(insight_tests_cuda
        PROPERTIES ENVIRONMENT "INSIGHT_GPU_BACKEND=cuda")
endif()
if(INSIGHT_WITH_IXUCA)
    gtest_discover_tests(insight_tests_ixuca
        PROPERTIES ENVIRONMENT "INSIGHT_GPU_BACKEND=ixuca")
endif()
if(INSIGHT_WITH_SDAA)
    gtest_discover_tests(insight_tests_sdaa
        PROPERTIES ENVIRONMENT "INSIGHT_GPU_BACKEND=sdaa")
endif()
