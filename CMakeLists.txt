# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# ========== Set policies to suppress warnings ==========
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

# ========== Project definition without hardcoding CUDA ==========
# CUDA will be enabled later if available and requested
project(insight VERSION 1.0.0 LANGUAGES CXX)

# ========== Options ==========
option(INSIGHT_BUILD_TESTS "Build unit tests" ON)
option(INSIGHT_WITH_CUDA "Enable CUDA backend" ON)
option(INSIGHT_DYNAMIC_CPU_BACKEND "CPU backend is dynamically loaded" ON)
option(INSIGHT_DYNAMIC_GPU_BACKEND "GPU backend is dynamically loaded" ON)

# ========== CUDA language support (optional) ==========
if(INSIGHT_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA backend enabled: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "CUDA compiler not found, disabling CUDA backend")
        set(INSIGHT_WITH_CUDA OFF)
    endif()
endif()

# ========== Global CUDA architecture settings ==========
if(INSIGHT_WITH_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75;80;86;89" CACHE STRING "CUDA architectures")
endif()

# ========== C++ standard ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: enable M_PI, M_E, etc. from <cmath>
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    # /utf-8 only for CXX (not nvcc, which doesn't understand it)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# ========== Add custom CMake module path ==========
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ========== Include auto-fetch function ==========
include(FetchDependency)

# ========== Directory Definitions ==========
set(INSIGHT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INSIGHT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INSIGHT_INTERNAL_DIR ${INSIGHT_SRC_DIR}/internal)
set(INSIGHT_UTILS_DIR ${INSIGHT_SRC_DIR}/utils)

# ========== Source Files Collection ==========
file(GLOB_RECURSE INSIGHT_SOURCES
    ${INSIGHT_SRC_DIR}/*/*.cpp
)

# ========== GoogleTest (for unit testing) ==========
if(INSIGHT_BUILD_TESTS)
    auto_fetch_dependency(googletest
        "https://github.com/google/googletest.git"
        "v1.15.0"
    )
endif()

# ========== Core Library ==========
add_library(insight_core STATIC
    ${INSIGHT_SOURCES}
    src/init.cpp
)
set_target_properties(insight_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ========== OpenMP ==========
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(insight_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(insight_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_OPENMP)
endif()

# libdl for dlopen/dlsym (dynamic backend loading)
target_link_libraries(insight_core PUBLIC ${CMAKE_DL_LIBS})

# ========== Thrust ==========
if(INSIGHT_WITH_CUDA)
    option(INSIGHT_USE_THRUST "Use Thrust for GPU algorithms (sort, scan, etc.)" ON)
else()
    set(INSIGHT_USE_THRUST OFF CACHE BOOL "" FORCE)
endif()

# ========== OpenBLAS ==========
option(INSIGHT_USE_OPENBLAS "Use OpenBLAS for linear algebra" ON)
if(INSIGHT_USE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_link_libraries(insight_core PRIVATE ${OpenBLAS_LIBRARIES})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_OPENBLAS)
        message(STATUS "OpenBLAS found and enabled")
    else()
        message(WARNING "OpenBLAS not found, disabling INSIGHT_USE_OPENBLAS")
        set(INSIGHT_USE_OPENBLAS OFF)
    endif()
else()
    message(STATUS "OpenBLAS support disabled")
endif()

# ========== FFTW3 ==========
option(INSIGHT_USE_FFTW3 "Use FFTW3 for FFT operations" ON)
if(INSIGHT_USE_FFTW3)
    find_package(FFTW3)
    if(FFTW3_FOUND)
        target_link_libraries(insight_core PRIVATE ${FFTW3_LIBRARIES})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_FFTW3)
        message(STATUS "FFTW3 found and enabled")
    else()
        message(WARNING "FFTW3 not found, disabling INSIGHT_USE_FFTW3")
        set(INSIGHT_USE_FFTW3 OFF)
    endif()
else()
    message(STATUS "FFTW3 support disabled")
endif()

# FFTW3 include dir needs to be visible to consumers (CPU backend kernels)
if(FFTW3_FOUND AND FFTW3_INCLUDE_DIRS)
    target_include_directories(insight_core PUBLIC ${FFTW3_INCLUDE_DIRS})
endif()

# ========== libsndfile (audio I/O) ==========
option(INSIGHT_USE_SNDFILE "Use libsndfile for audio file I/O" ON)
if(INSIGHT_USE_SNDFILE)
    find_package(LibSndFile QUIET)
    if(NOT SndFile_FOUND)
        message(STATUS "libsndfile not found on system, building from source...")

        # Determine source locations: prefer local third_party/, fall back to FetchContent
        set(_local "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
        foreach(_dep ogg vorbis FLAC opus sndfile)
            string(TOUPPER "${_dep}" _dep_upper)
            string(TOLOWER "${_dep}" _dep_lower)
            # Try exact name, lowercase, and lib-prefixed
            if(EXISTS "${_local}/${_dep}/CMakeLists.txt")
                set(${_dep_upper}_LOCAL "${_local}/${_dep}")
                message(STATUS "  ${_dep}: using local ${_local}/${_dep}")
            elseif(EXISTS "${_local}/${_dep_lower}/CMakeLists.txt")
                set(${_dep_upper}_LOCAL "${_local}/${_dep_lower}")
                message(STATUS "  ${_dep}: using local ${_local}/${_dep_lower}")
            elseif(EXISTS "${_local}/lib${_dep_lower}/CMakeLists.txt")
                set(${_dep_upper}_LOCAL "${_local}/lib${_dep_lower}")
                message(STATUS "  ${_dep}: using local ${_local}/lib${_dep_lower}")
            else()
                message(STATUS "  ${_dep}: will fetch from GitHub")
            endif()
        endforeach()

        include(FetchContent)

        # Ogg
        if(NOT DEFINED OGG_LOCAL)
            FetchContent_Declare(ogg
                GIT_REPOSITORY "https://github.com/xiph/ogg.git"
                GIT_TAG "v1.3.5" GIT_SHALLOW TRUE)
        endif()

        # Vorbis
        if(NOT DEFINED VORBIS_LOCAL)
            FetchContent_Declare(vorbis
                GIT_REPOSITORY "https://github.com/xiph/vorbis.git"
                GIT_TAG "v1.3.7" GIT_SHALLOW TRUE)
        endif()

        # FLAC
        if(NOT DEFINED FLAC_LOCAL)
            FetchContent_Declare(FLAC
                GIT_REPOSITORY "https://github.com/xiph/flac.git"
                GIT_TAG "1.4.3" GIT_SHALLOW TRUE)
        endif()

        # Opus
        if(NOT DEFINED OPUS_LOCAL)
            FetchContent_Declare(opus
                GIT_REPOSITORY "https://github.com/xiph/opus.git"
                GIT_TAG "v1.5.2" GIT_SHALLOW TRUE)
        endif()

        # libsndfile
        if(NOT DEFINED SNDFILE_LOCAL)
            FetchContent_Declare(sndfile
                GIT_REPOSITORY "https://github.com/libsndfile/libsndfile.git"
                GIT_TAG "1.2.2" GIT_SHALLOW TRUE)
        endif()

        # Common cache settings
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(INSTALL_CMAKE_PACKAGE_MODULE OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(INSTALL_MANPAGES OFF CACHE BOOL "" FORCE)
        set(INSTALL_PKGCONFIG_MODULES OFF CACHE BOOL "" FORCE)
        set(BUILD_CXXLIBS OFF CACHE BOOL "" FORCE)

        # Pre-set Ogg variables so find_package(Ogg) works for vorbis/FLAC/opus
        set(_ogg_src "${_local}/ogg")
        set(_ogg_bin "${CMAKE_BINARY_DIR}/_deps/ogg-build")
        set(OGG_INCLUDE_DIR "${_ogg_src}/include" CACHE PATH "" FORCE)
        set(OGG_LIBRARY "${_ogg_bin}/${CMAKE_STATIC_LIBRARY_PREFIX}ogg${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
        set(Ogg_FOUND TRUE CACHE BOOL "" FORCE)
        set(OGG_LIBRARIES "${OGG_LIBRARY}" CACHE STRING "" FORCE)
        set(OGG_INCLUDE_DIRS "${_ogg_src}/include" CACHE PATH "" FORCE)
        set(CMAKE_MODULE_PATH "${_local}/vorbis/cmake;${_local}/flac/cmake;${CMAKE_MODULE_PATH}")

        # Determine source dir for each dependency
        macro(_resolve_dep VAR NAME)
            set(_saved_skip ${CMAKE_SKIP_INSTALL_RULES})
            if(DEFINED ${VAR})
                set(_${NAME}_src "${${VAR}}")
                set(CMAKE_SKIP_INSTALL_RULES ON)
                add_subdirectory("${${VAR}}" "${CMAKE_BINARY_DIR}/_deps/${NAME}-build")
                set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
            else()
                set(CMAKE_SKIP_INSTALL_RULES ON)
                FetchContent_MakeAvailable(${NAME})
                set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
                set(_${NAME}_src "${${NAME}_SOURCE_DIR}")
            endif()
        endmacro()

        _resolve_dep(OGG_LOCAL ogg)

        _resolve_dep(FLAC_LOCAL FLAC)
        set(FLAC_INCLUDE_DIR "${_flac_src}/include" CACHE PATH "" FORCE)
        set(FLAC_LIBRARY "${CMAKE_BINARY_DIR}/_deps/flac-build/src/libFLAC/${CMAKE_STATIC_LIBRARY_PREFIX}FLAC${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
        set(FLAC_FOUND TRUE CACHE BOOL "" FORCE)
        set(FLAC_LIBRARIES "${FLAC_LIBRARY};${OGG_LIBRARY}" CACHE STRING "" FORCE)
        set(FLAC_INCLUDE_DIRS "${_flac_src}/include" CACHE PATH "" FORCE)

        _resolve_dep(VORBIS_LOCAL vorbis)
        set(Vorbis_Vorbis_INCLUDE_DIR "${_vorbis_src}/include" CACHE PATH "" FORCE)
        set(Vorbis_Vorbis_LIBRARY "${CMAKE_BINARY_DIR}/_deps/vorbis-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vorbis${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
        set(Vorbis_Enc_INCLUDE_DIR "${_vorbis_src}/include" CACHE PATH "" FORCE)
        set(Vorbis_Enc_LIBRARY "${CMAKE_BINARY_DIR}/_deps/vorbis-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vorbisenc${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
        set(Vorbis_File_INCLUDE_DIR "${_vorbis_src}/include" CACHE PATH "" FORCE)
        set(Vorbis_File_LIBRARY "${CMAKE_BINARY_DIR}/_deps/vorbis-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vorbisfile${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)

        _resolve_dep(OPUS_LOCAL opus)
        set(OPUS_INCLUDE_DIR "${_opus_src}/include" CACHE PATH "" FORCE)
        set(OPUS_LIBRARY "${CMAKE_BINARY_DIR}/_deps/opus-build/${CMAKE_STATIC_LIBRARY_PREFIX}opus${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)

        # Create imported targets (skip if already created by add_subdirectory)
        foreach(_spec
            "Ogg::ogg|OGG_LIBRARY|OGG_INCLUDE_DIRS"
            "FLAC::FLAC|FLAC_LIBRARY|FLAC_INCLUDE_DIRS"
            "Vorbis::vorbis|Vorbis_Vorbis_LIBRARY|Vorbis_Vorbis_INCLUDE_DIR"
            "Vorbis::vorbisenc|Vorbis_Enc_LIBRARY|Vorbis_Enc_INCLUDE_DIR"
            "Vorbis::vorbisfile|Vorbis_File_LIBRARY|Vorbis_File_INCLUDE_DIR"
            "Opus::opus|OPUS_LIBRARY|OPUS_INCLUDE_DIR")
            string(REPLACE "|" ";" _parts "${_spec}")
            list(GET _parts 0 _target)
            list(GET _parts 1 _lib_var)
            list(GET _parts 2 _inc_var)
            if(NOT TARGET ${_target})
                # Check if the base target exists (from add_subdirectory)
                # If so, create an ALIAS; otherwise create an IMPORTED target
                string(REPLACE "::" ";" _name_parts "${_target}")
                list(GET _name_parts 1 _base_name)
                if(TARGET ${_base_name})
                    add_library(${_target} ALIAS ${_base_name})
                else()
                    add_library(${_target} STATIC IMPORTED)
                    set_target_properties(${_target} PROPERTIES
                        IMPORTED_LOCATION "${${_lib_var}}"
                        INTERFACE_INCLUDE_DIRECTORIES "${${_inc_var}}")
                endif()
            endif()
        endforeach()
        # Link dependencies (only for imported targets we created, not add_subdirectory targets)
        get_target_property(_flac_type FLAC::FLAC TYPE)
        if(_flac_type STREQUAL "STATIC_LIBRARY")
            get_target_property(_flac_imp FLAC::FLAC IMPORTED)
            if(_flac_imp)
                set_target_properties(FLAC::FLAC PROPERTIES INTERFACE_LINK_LIBRARIES Ogg::ogg)
                set_target_properties(Vorbis::vorbis PROPERTIES INTERFACE_LINK_LIBRARIES Ogg::ogg)
                set_target_properties(Vorbis::vorbisenc PROPERTIES INTERFACE_LINK_LIBRARIES "Vorbis::vorbis;Ogg::ogg")
                set_target_properties(Vorbis::vorbisfile PROPERTIES INTERFACE_LINK_LIBRARIES "Vorbis::vorbis;Ogg::ogg")
            endif()
        endif()
        # Ogg binary dir for generated config_types.h
        set(_ogg_bin "${CMAKE_BINARY_DIR}/_deps/ogg-build")
        get_target_property(_ogg_imp Ogg::ogg IMPORTED)
        if(_ogg_imp)
            set_target_properties(Ogg::ogg PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_ogg_src}/include;${_ogg_bin}/include")
        endif()

        # Build remaining deps that weren't built by _resolve_dep
        set(_saved_skip_install ${CMAKE_SKIP_INSTALL_RULES})
        set(CMAKE_SKIP_INSTALL_RULES ON)
        # Only FetchContent_MakeAvailable deps that aren't local (local ones already resolved above)
        # The _resolve_dep macro already called MakeAvailable for non-local deps
        set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip_install})

        # Opus header symlink (Windows: copy dir, Unix: symlink)
        if(NOT EXISTS "${_opus_src}/include/opus")
            if(WIN32)
                file(COPY "${_opus_src}/include/" DESTINATION "${_opus_src}/include/opus")
            else()
                file(CREATE_LINK "${_opus_src}/include" "${_opus_src}/include/opus" SYMBOLIC)
            endif()
        endif()

        # libsndfile
        if(DEFINED SNDFILE_LOCAL)
            set(_sndfile_src "${SNDFILE_LOCAL}")
            set(_saved_skip ${CMAKE_SKIP_INSTALL_RULES})
            set(CMAKE_SKIP_INSTALL_RULES ON)
            add_subdirectory("${SNDFILE_LOCAL}" "${CMAKE_BINARY_DIR}/_deps/libsndfile-build")
            set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip})
        else()
            set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
            set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
            set(ENABLE_EXTERNAL_LIBS ON CACHE BOOL "" FORCE)
            set(HAVE_EXTERNAL_XIPH_LIBS ON CACHE BOOL "" FORCE)
            set(CMAKE_SKIP_INSTALL_RULES ON)
            FetchContent_MakeAvailable(sndfile)
            set(CMAKE_SKIP_INSTALL_RULES ${_saved_skip_install})
            set(_sndfile_src "${sndfile_SOURCE_DIR}")
        endif()
        set(SndFile_FOUND TRUE)
    endif()
    if(TARGET SndFile::sndfile-static)
        target_link_libraries(insight_core PRIVATE SndFile::sndfile-static)
    elseif(TARGET SndFile::sndfile)
        target_link_libraries(insight_core PRIVATE SndFile::sndfile)
        # sndfile's codec deps are PRIVATE, must link explicitly
        if(TARGET Vorbis::vorbisenc)
            target_link_libraries(insight_core PRIVATE Vorbis::vorbisenc Vorbis::vorbis Vorbis::vorbisfile)
        endif()
        if(TARGET FLAC::FLAC)
            target_link_libraries(insight_core PRIVATE FLAC::FLAC)
        endif()
        if(TARGET Opus::opus)
            target_link_libraries(insight_core PRIVATE Opus::opus)
        endif()
        if(TARGET Ogg::ogg)
            target_link_libraries(insight_core PRIVATE Ogg::ogg)
        endif()
    endif()
    target_compile_definitions(insight_core PUBLIC INSIGHT_USE_SNDFILE)
    message(STATUS "libsndfile enabled (with FLAC support)")
else()
    message(STATUS "libsndfile support disabled")
endif()

# ========== Include Directories ==========
target_include_directories(insight_core
    PUBLIC
        ${INSIGHT_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIRS}
    PRIVATE
        ${INSIGHT_INTERNAL_DIR}
        ${INSIGHT_UTILS_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ========== Matplotplusplus (C++ plotting, matplotlib-compatible) ==========
include(cmake/ApplyPatch.cmake)
option(INSIGHT_USE_MATPLOT "Use matplotplusplus for plotting" ON)
if(INSIGHT_USE_MATPLOT)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(LOCAL_MATPLOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplotplusplus")
    if(EXISTS "${LOCAL_MATPLOT}/CMakeLists.txt")
        message(STATUS "Using local matplotplusplus from ${LOCAL_MATPLOT}")
        # Apply patches BEFORE add_subdirectory (like PaddlePaddle PATCH_COMMAND)
        apply_patch("${LOCAL_MATPLOT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
        apply_patch("${LOCAL_MATPLOT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gnuplot-unknown-terminal.patch")
        add_subdirectory("${LOCAL_MATPLOT}" "${CMAKE_CURRENT_BINARY_DIR}/matplotplusplus")
    else()
        message(STATUS "Fetching matplotplusplus from GitHub...")
        include(FetchContent)
        FetchContent_Declare(matplotplusplus
            GIT_REPOSITORY "https://github.com/alandefreitas/matplotplusplus"
            GIT_TAG        "v1.2.1"
            GIT_SHALLOW    TRUE
        )
        set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MATPLOTPP_WITH_INTERACTIVE_BACKENDS OFF CACHE BOOL "" FORCE)
        FetchContent_Populate(matplotplusplus)
        apply_patch("${matplotplusplus_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gcc-compat.patch")
        apply_patch("${matplotplusplus_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/matplotplusplus/gnuplot-unknown-terminal.patch")
        add_subdirectory("${matplotplusplus_SOURCE_DIR}" "${matplotplusplus_BINARY_DIR}")
    endif()
    if(TARGET matplot)
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_MATPLOT)
        target_link_libraries(insight_core PUBLIC matplot)
        if(MSVC)
            # Suppress C4834 (discarding [[nodiscard]] return value) which matplot
            # triggers in its own source but is treated as error via /WX.
            target_compile_options(matplot PRIVATE /wd4834)
        endif()
        message(STATUS "Matplotplusplus enabled")
    else()
        message(WARNING "Matplotplusplus target not found, disabling")
        set(INSIGHT_USE_MATPLOT OFF)
    endif()
else()
    message(STATUS "Matplotplusplus support disabled")
endif()

# ========== CPU Backend (always dynamic) ==========
add_subdirectory(backends/cpu)

# ========== CUDA Backend ==========
if(INSIGHT_WITH_CUDA)
    if(INSIGHT_USE_THRUST)
        set(INSIGHT_CUDA_USE_THRUST ON)
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_THRUST)
    else()
        set(INSIGHT_CUDA_USE_THRUST OFF)
    endif()
    add_subdirectory(backends/cuda)
    target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_CUDA)
else()
    message(STATUS "CUDA backend disabled by user option")
endif()

# ========== IXUCA (天数智芯 Iluvatar) Backend ==========
# Auto-detect CoreX SDK (provides CUDA-compatible runtime for Iluvatar GPU)
# Set to ON to force, OFF to disable, or leave unset for auto-detection
set(INSIGHT_WITH_ILUVATAR "AUTO" CACHE STRING "Enable Iluvatar (天数智芯) GPU backend (ON/OFF/AUTO)")
set_property(CACHE INSIGHT_WITH_ILUVATAR PROPERTY STRINGS ON OFF AUTO)

if(INSIGHT_WITH_ILUVATAR STREQUAL "ON" OR INSIGHT_WITH_ILUVATAR STREQUAL "AUTO")
    # Check for CoreX clang++ compiler
    find_program(COREX_CLANGXX "clang++" PATHS "/usr/local/corex/bin" NO_DEFAULT_PATH)
    # Check for CoreX CUDA-compatible headers
    if(COREX_CLANGXX AND EXISTS "/usr/local/corex/include/cuda_runtime.h")
        set(INSIGHT_WITH_ILUVATAR ON)
        add_subdirectory(backends/iluvatar)
        target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_ILUVATAR)
        message(STATUS "IXUCA backend enabled (CoreX SDK found at /usr/local/corex)")
    elseif(INSIGHT_WITH_ILUVATAR STREQUAL "AUTO")
        # Auto mode: not found, silently disable
        set(INSIGHT_WITH_ILUVATAR OFF)
        message(STATUS "IXUCA backend: CoreX SDK not found, disabled")
    else()
        message(WARNING "IXUCA backend requested but CoreX SDK not found")
        set(INSIGHT_WITH_ILUVATAR OFF)
    endif()
else()
    set(INSIGHT_WITH_ILUVATAR OFF)
    message(STATUS "IXUCA backend disabled by user option")
endif()

# ========== Demos ==========
option(INSIGHT_BUILD_DEMOS "Build demo programs" ON)
if(INSIGHT_BUILD_DEMOS)
    add_subdirectory(demos)
endif()

# ========== Language Bindings (Python / Lua / Julia) ==========
option(INSIGHT_BUILD_BINDINGS "Build language bindings" ON)
if(INSIGHT_BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()

# ========== Unit Tests ==========
if(INSIGHT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ========== Configuration Summary ==========
message(STATUS "")
message(STATUS "Insight Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  OpenMP:    ${OpenMP_CXX_FOUND}")
message(STATUS "  OpenBLAS:  ${INSIGHT_USE_OPENBLAS}")
message(STATUS "  FFTW3:     ${INSIGHT_USE_FFTW3}")
message(STATUS "  SndFile:   ${INSIGHT_USE_SNDFILE}")
message(STATUS "  Matplot:   ${INSIGHT_USE_MATPLOT}")
message(STATUS "  CUDA:      ${INSIGHT_WITH_CUDA}")
message(STATUS "  IXUCA:     ${INSIGHT_WITH_ILUVATAR}")
message(STATUS "  Thrust:    ${INSIGHT_USE_THRUST}")
message(STATUS "  Tests:     ${INSIGHT_BUILD_TESTS}")
message(STATUS "")