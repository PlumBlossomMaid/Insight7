file(GLOB_RECURSE KERNEL_SOURCES "kernels/*.cpp")
file(GLOB_RECURSE DEVICE_SOURCES "device/*.cpp")
file(GLOB_RECURSE REGISTRY_SOURCES "registry/*.cpp")

add_library(insight_cpu_backend SHARED
    ${KERNEL_SOURCES}
    ${DEVICE_SOURCES}
    ${REGISTRY_SOURCES}
)
target_compile_definitions(insight_cpu_backend PRIVATE INSIGHT_CPU_BACKEND_EXPORTS)
target_link_libraries(insight_cpu_backend PRIVATE insight_core)

target_include_directories(insight_cpu_backend
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

if(INSIGHT_USE_FFTW3)
    find_package(FFTW3)
    if(FFTW3_FOUND)
        target_link_libraries(insight_cpu_backend PRIVATE ${FFTW3_LIBRARIES})
        target_compile_definitions(insight_cpu_backend PUBLIC INSIGHT_USE_FFTW3)
        target_include_directories(insight_cpu_backend PRIVATE ${FFTW3_INCLUDE_DIRS})
        message(STATUS "FFTW3 found and enabled for CPU backend")
    else()
        message(WARNING "FFTW3 not found, disabling FFT support")
        set(INSIGHT_USE_FFTW3 OFF)
    endif()
endif()

# OpenBLAS (optional, for accelerated linear algebra)
option(INSIGHT_USE_OPENBLAS "Use OpenBLAS for BLAS/LAPACK operations" ON)
if(INSIGHT_USE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        # LAPACKE headers: check local OpenBLAS if not in system paths.
        # The linalg kernels include <lapacke.h> unconditionally under
        # INSIGHT_USE_OPENBLAS, so the header must be located before the
        # definition is set — otherwise configure succeeds and every linalg
        # translation unit fails to compile.
        find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h
            PATHS
                /usr/include
                /usr/local/include
                ${OpenBLAS_INCLUDE_DIR}
                ${OpenBLAS_INCLUDE_DIR}/..
                ${CMAKE_PREFIX_PATH}
                $ENV{OPENBLAS_HOME}/include
        )
        if(LAPACKE_INCLUDE_DIR)
            # Link OpenBLAS (BLAS + LAPACK Fortran interfaces)
            target_link_libraries(insight_cpu_backend PRIVATE ${OpenBLAS_LIBRARIES})

            # Also need LAPACKE (C interface for LAPACK)
            # OpenBLAS ships lapacke.h and compiles LAPACKE symbols directly into libopenblas.
            # A separate lapacke.lib may not exist — the symbols are already in libopenblas.
            find_library(LAPACKE_LIB NAMES lapacke liblapacke)
            if(LAPACKE_LIB)
                target_link_libraries(insight_cpu_backend PRIVATE ${LAPACKE_LIB})
                message(STATUS "LAPACKE found and linked: ${LAPACKE_LIB}")
            else()
                message(STATUS "LAPACKE not found as separate lib — using OpenBLAS for LAPACKE")
            endif()

            target_include_directories(insight_cpu_backend PRIVATE ${LAPACKE_INCLUDE_DIR})
            message(STATUS "LAPACKE headers found: ${LAPACKE_INCLUDE_DIR}")

            target_compile_definitions(insight_cpu_backend PUBLIC INSIGHT_USE_OPENBLAS)
            message(STATUS "OpenBLAS found and enabled for CPU backend")
        else()
            message(WARNING
                "OpenBLAS was found at ${OpenBLAS_INCLUDE_DIR} but lapacke.h was not. "
                "Falling back to internal implementations for linear algebra. "
                "Set LAPACKE_INCLUDE_DIR to the directory containing lapacke.h to enable OpenBLAS.")
            set(INSIGHT_USE_OPENBLAS OFF)
        endif()
    else()
        message(WARNING "OpenBLAS not found, falling back to internal implementations for linear algebra")
        set(INSIGHT_USE_OPENBLAS OFF)
    endif()
else()
    message(STATUS "OpenBLAS disabled by user option")
endif()

# On Windows, copy dependency DLLs to the backend output directory so that
# LoadLibraryA can find them alongside the backend DLL (standard Windows practice).
if(WIN32)
    set(_CPU_DEP_DLLS "")
    if(INSIGHT_USE_FFTW3 AND FFTW3_FOUND)
        foreach(_lib ${FFTW3_LIBRARIES})
            get_filename_component(_lib_dir "${_lib}" DIRECTORY)
            get_filename_component(_lib_base "${_lib}" NAME_WE)
            # vcpkg layout: lib/fftw3.lib → bin/fftw3.dll
            # local layout: lib/libfftw3.a → bin/libfftw3.dll (or fftw3.dll)
            set(_dll_candidates
                "${_lib_dir}/../bin/${_lib_base}.dll"
                "${_lib_dir}/${_lib_base}.dll"
            )
            foreach(_dll ${_dll_candidates})
                if(EXISTS "${_dll}")
                    list(APPEND _CPU_DEP_DLLS "${_dll}")
                    break()
                endif()
            endforeach()
        endforeach()
    endif()
    if(INSIGHT_USE_OPENBLAS AND OpenBLAS_FOUND)
        get_filename_component(_lib_dir "${OpenBLAS_LIBRARY}" DIRECTORY)
        set(_dll_candidates
            "${_lib_dir}/../bin/libopenblas.dll"
            "${_lib_dir}/../bin/openblas.dll"
            "${_lib_dir}/libopenblas.dll"
            "${_lib_dir}/openblas.dll"
        )
        foreach(_dll ${_dll_candidates})
            if(EXISTS "${_dll}")
                list(APPEND _CPU_DEP_DLLS "${_dll}")
                break()
            endif()
        endforeach()
    endif()
    if(_CPU_DEP_DLLS)
        add_custom_command(TARGET insight_cpu_backend POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_CPU_DEP_DLLS}
                $<TARGET_FILE_DIR:insight_cpu_backend>
            COMMENT "Copying CPU backend dependency DLLs to output directory"
        )
    endif()
endif()
