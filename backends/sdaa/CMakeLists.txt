# backends/sdaa/CMakeLists.txt
# SDAA GPU backend.

set(SDAA_HOME "$ENV{SDAA_HOME}" CACHE PATH "SDAA SDK root")
if(NOT SDAA_HOME AND DEFINED ENV{SDAA_ROOT})
    set(SDAA_HOME "$ENV{SDAA_ROOT}")
endif()
if(NOT SDAA_HOME)
    set(SDAA_HOME "/opt/tecoai")
endif()

set(SDAA_INCLUDE_DIR "${SDAA_HOME}/include")
find_library(SDAA_RUNTIME_LIBRARY sdaart
    HINTS "${SDAA_HOME}/lib64" "${SDAA_HOME}/lib"
    NO_DEFAULT_PATH)
if(SDAA_RUNTIME_LIBRARY)
    get_filename_component(SDAA_LIBRARY_DIR "${SDAA_RUNTIME_LIBRARY}" DIRECTORY)
else()
    set(SDAA_LIBRARY_DIR "${SDAA_HOME}/lib64")
endif()

file(GLOB_RECURSE DEVICE_SOURCES CONFIGURE_DEPENDS "device/*.cpp")
file(GLOB_RECURSE REGISTRY_SOURCES CONFIGURE_DEPENDS "registry/*.cpp")

add_library(insight_sdaa_backend SHARED
    ${DEVICE_SOURCES}
    ${REGISTRY_SOURCES}
    init.cpp
)

target_compile_definitions(insight_sdaa_backend PRIVATE INSIGHT_GPU_BACKEND_EXPORTS)
if(SDAA_RUNTIME_LIBRARY AND EXISTS "${SDAA_INCLUDE_DIR}/sdaa_runtime.h")
    target_compile_definitions(insight_sdaa_backend PRIVATE INSIGHT_SDAA_RUNTIME)
    target_include_directories(insight_sdaa_backend PRIVATE "${SDAA_INCLUDE_DIR}")
    target_link_libraries(insight_sdaa_backend PRIVATE "${SDAA_RUNTIME_LIBRARY}" dl rt)
    target_link_directories(insight_sdaa_backend PRIVATE "${SDAA_LIBRARY_DIR}")
    set_target_properties(insight_sdaa_backend PROPERTIES
        BUILD_RPATH "${SDAA_LIBRARY_DIR}"
        INSTALL_RPATH "${SDAA_LIBRARY_DIR}"
    )
    message(STATUS "SDAA runtime backend enabled: ${SDAA_HOME}")
elseif(INSIGHT_SDAA_STUB_DEVICE)
    target_compile_definitions(insight_sdaa_backend PRIVATE INSIGHT_SDAA_STUB_DEVICE)
    message(STATUS "SDAA stub backend enabled")
else()
    message(FATAL_ERROR "SDAA runtime not found. Set SDAA_HOME or enable INSIGHT_SDAA_STUB_DEVICE=ON")
endif()

target_include_directories(insight_sdaa_backend
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(insight_sdaa_backend PRIVATE insight_core)

set_target_properties(insight_sdaa_backend PROPERTIES
    OUTPUT_NAME insight_sdaa_backend
)
