# cmake/InsightPlatform.cmake
# Cross-platform shared library naming utilities

#[=======================================================================[.rst:
InsightPlatform
---------------

Provides helper functions for platform-specific shared library naming.

``insight_shared_lib_name(RESULT_VAR base_name)``
  Sets ``RESULT_VAR`` to the full shared library filename for the current
  platform:

  - Windows: ``<base_name>.dll``
  - Linux:   ``lib<base_name>.so``
  - macOS:   ``lib<base_name>.dylib``
#]=======================================================================]

function(insight_shared_lib_name RESULT_VAR base_name)
    if(WIN32)
        set(${RESULT_VAR} "${base_name}.dll" PARENT_SCOPE)
    elseif(APPLE)
        set(${RESULT_VAR} "lib${base_name}.dylib" PARENT_SCOPE)
    else()
        set(${RESULT_VAR} "lib${base_name}.so" PARENT_SCOPE)
    endif()
endfunction()
