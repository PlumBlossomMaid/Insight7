# demos/CMakeLists.txt

# Helper function to create a demo target with backend DLL copying
function(add_insight_demo TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_link_libraries(${TARGET_NAME} PRIVATE insight_core)

    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${INSIGHT_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/demos
    )

    # Copy CPU backend DLL
    if(INSIGHT_DYNAMIC_CPU_BACKEND)
        add_dependencies(${TARGET_NAME} insight_cpu_backend)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:insight_cpu_backend>"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<TARGET_FILE_NAME:insight_cpu_backend>"
            COMMENT "Copying CPU backend DLL to ${TARGET_NAME}"
        )
    endif()

    # Copy GPU backend DLLs
    if(INSIGHT_DYNAMIC_GPU_BACKEND)
        foreach(_gpu_backend insight_cuda_backend insight_rocm_backend insight_ixuca_backend insight_sdaa_backend)
            if(TARGET ${_gpu_backend})
                add_dependencies(${TARGET_NAME} ${_gpu_backend})
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${_gpu_backend}>"
                        "$<TARGET_FILE_DIR:${TARGET_NAME}>/$<TARGET_FILE_NAME:${_gpu_backend}>"
                    COMMENT "Copying ${_gpu_backend} to ${TARGET_NAME}"
                )
            endif()
        endforeach()
    endif()
endfunction()

# Register each C++ demo (sources in cpp/ subdirectory)
add_insight_demo(demo_basic_ops    cpp/basic_ops.cpp)
add_insight_demo(demo_linalg       cpp/linalg_demo.cpp)
add_insight_demo(demo_fft          cpp/fft_demo.cpp)
add_insight_demo(demo_gpu_transfer cpp/gpu_transfer.cpp)

if(INSIGHT_USE_SNDFILE)
    add_insight_demo(demo_sndfile cpp/sndfile_demo.cpp)
    target_compile_definitions(demo_sndfile PRIVATE
        AUDIO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../audio"
    )
endif()

add_insight_demo(demo_radar_detection cpp/radar_detection.cpp)
add_insight_demo(demo_radar_pipeline cpp/radar_pipeline.cpp)
add_insight_demo(demo_timer cpp/timer_demo.cpp)
