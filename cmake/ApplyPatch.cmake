# cmake/ApplyPatch.cmake
# Idempotent patch application for third-party dependencies.
# Usage: apply_patch(<source_dir> <patch_file>)
#
# Tries multiple methods:
#   1. git apply --ignore-whitespace (most tolerant)
#   2. patch -p1 (works without git, Linux/macOS only)
#   3. Reverse check (already applied)
#
# FATAL_ERROR if all methods fail — patches MUST apply.

function(apply_patch SOURCE_DIR PATCH_FILE)
    if(NOT EXISTS "${PATCH_FILE}")
        message(FATAL_ERROR "Patch file not found: ${PATCH_FILE}")
    endif()

    if(NOT EXISTS "${SOURCE_DIR}")
        message(FATAL_ERROR "Source dir not found: ${SOURCE_DIR}")
    endif()

    get_filename_component(PATCH_NAME "${PATCH_FILE}" NAME_WE)
    set(PATCH_STAMP "${SOURCE_DIR}/.patch_applied_${PATCH_NAME}")

    if(EXISTS "${PATCH_STAMP}")
        execute_process(
            COMMAND git apply --reverse --check --ignore-whitespace "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE R_STAMP
            OUTPUT_QUIET ERROR_QUIET
        )
        if(R_STAMP EQUAL 0)
            message(STATUS "[patch] ${PATCH_NAME}: already applied (stamp verified)")
            return()
        endif()
        message(STATUS "[patch] ${PATCH_NAME}: stale stamp found, reapplying")
        file(REMOVE "${PATCH_STAMP}")
    endif()

    message(STATUS "[patch] ${PATCH_NAME}: applying to ${SOURCE_DIR} ...")

    # Method 1: git apply (preferred, works on all platforms)
    execute_process(
        COMMAND git apply --check --ignore-whitespace "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE R1
        OUTPUT_VARIABLE O1 ERROR_VARIABLE E1
    )
    message(STATUS "[patch] ${PATCH_NAME}: git apply --check result=${R1}")
    if(R1 EQUAL 0)
        execute_process(
            COMMAND git apply --ignore-whitespace "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE R1A
            ERROR_VARIABLE E1A
        )
        if(R1A EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Applied via git apply")
            message(STATUS "[patch] ${PATCH_NAME}: OK (git apply)")
            return()
        endif()
        message(STATUS "[patch] ${PATCH_NAME}: git apply failed: ${E1A}")
    else()
        message(STATUS "[patch] ${PATCH_NAME}: git apply --check failed: ${E1}")
    endif()

    # Method 2: patch -p1 (Linux/macOS only — not available on Windows by default)
    if(NOT WIN32)
        execute_process(
            COMMAND patch -p1 --forward --dry-run
            INPUT_FILE "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE R2
            OUTPUT_VARIABLE O2 ERROR_VARIABLE E2
        )
        message(STATUS "[patch] ${PATCH_NAME}: patch -p1 --dry-run result=${R2}")
        if(R2 EQUAL 0)
            execute_process(
                COMMAND patch -p1 --forward
                INPUT_FILE "${PATCH_FILE}"
                WORKING_DIRECTORY "${SOURCE_DIR}"
                RESULT_VARIABLE R2A
                ERROR_VARIABLE E2A
            )
            if(R2A EQUAL 0)
                file(WRITE "${PATCH_STAMP}" "Applied via patch -p1")
                message(STATUS "[patch] ${PATCH_NAME}: OK (patch -p1)")
                return()
            endif()
            message(STATUS "[patch] ${PATCH_NAME}: patch -p1 failed: ${E2A}")
        else()
            message(STATUS "[patch] ${PATCH_NAME}: patch dry-run failed: ${E2}")
        endif()

        # Method 3: Already applied? (reverse check, Linux/macOS only)
        execute_process(
            COMMAND patch -p1 --reverse --dry-run
            INPUT_FILE "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE R3
            OUTPUT_QUIET ERROR_QUIET
        )
        if(R3 EQUAL 0)
            file(WRITE "${PATCH_STAMP}" "Already applied")
            message(STATUS "[patch] ${PATCH_NAME}: OK (already applied)")
            return()
        endif()
    endif()

    # All methods failed — this is a build-breaking error
    message(FATAL_ERROR
        "[patch] ${PATCH_NAME}: FAILED to apply!\n"
        "  Source: ${SOURCE_DIR}\n"
        "  Patch:  ${PATCH_FILE}\n"
        "  git apply error: ${E1}\n"
        "  This must be fixed — the patch is required for headless CI.")
endfunction()
