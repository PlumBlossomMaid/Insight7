# copy_backends.cmake — copy all built backend libraries into the Lua package

file(GLOB _BACKEND_FILES
    "${SRC_DIR}/*/libinsight_*_backend.so"
    "${SRC_DIR}/*/libinsight_*_backend.dylib"
    "${SRC_DIR}/*/insight_*_backend.dll"
)

foreach(_f ${_BACKEND_FILES})
    get_filename_component(_name ${_f} NAME)
    message(STATUS "Copying ${_name} -> ${DST_DIR}/")
    file(COPY_FILE "${_f}" "${DST_DIR}/${_name}" ONLY_IF_DIFFERENT)
endforeach()
