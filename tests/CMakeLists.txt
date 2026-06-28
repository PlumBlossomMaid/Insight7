# tests/CMakeLists.txt

# ========== 收集测试源文件 ==========
file(GLOB CPU_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpu/cpp/*.cpp")
file(GLOB CUDA_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cuda/cpp/*.cpp")

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
)

if(INSIGHT_WITH_CUDA)
    target_link_libraries(insight_tests_cuda
        insight_core
        insight_cpu_backend
        insight_cuda_backend
        GTest::gtest_main
    )
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
    if(INSIGHT_WITH_ILUVATAR AND INSIGHT_DYNAMIC_GPU_BACKEND)
        copy_backend_dll(${TARGET_NAME} insight_iluvatar_backend)
    endif()
endfunction()

# CPU 测试
copy_all_backend_dlls(insight_tests_cpu)

# GPU 测试（如果需要）
if(INSIGHT_WITH_CUDA)
    copy_all_backend_dlls(insight_tests_cuda)
endif()

# IXUCA 测试
if(INSIGHT_WITH_ILUVATAR)
    add_executable(insight_tests_iluvatar ${CUDA_TEST_SOURCES})
    target_link_libraries(insight_tests_iluvatar
        insight_core
        insight_cpu_backend
        insight_iluvatar_backend
        GTest::gtest_main
    )
    target_include_directories(insight_tests_iluvatar
        PRIVATE ${INSIGHT_INCLUDE_DIR}
    )
    copy_all_backend_dlls(insight_tests_iluvatar)
endif()

# ========== 自动发现并注册测试 ==========
include(GoogleTest)
gtest_discover_tests(insight_tests_cpu)
if(INSIGHT_WITH_CUDA)
    gtest_discover_tests(insight_tests_cuda)
endif()
if(INSIGHT_WITH_ILUVATAR)
    gtest_discover_tests(insight_tests_iluvatar)
endif()
