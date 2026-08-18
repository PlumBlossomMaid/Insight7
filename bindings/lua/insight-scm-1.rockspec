package = "insight"
version = "scm-1"

source = {
    url = "https://github.com/PlumBlossomMaid/Insight7.git",
    branch = "main",
}

description = {
    summary = "Insight7 — lightweight scientific computing framework for Lua",
    detailed = [[
        Insight is a C++ array computing framework with GPU support.
        This package provides Lua/LuaJIT bindings via sol2 with
        Penlight-enhanced API wrappers.
    ]],
    homepage = "https://github.com/PlumBlossomMaid/Insight7",
    license = "MIT",
}

dependencies = {
    "lua >= 5.1",
    "penlight >= 1.5",
}

external_dependencies = {
    INSIGHT = {
        header = "insight/core/array.h",
        library = "insight_core",
    },
}

build = {
    type = "command",

    build_command = [[
        cmake -S . -B build.luarocks \
            -DCMAKE_BUILD_TYPE=Release \
            -DINSIGHT_BUILD_TESTS=OFF \
            -DINSIGHT_BUILD_DEMOS=OFF \
            -DINSIGHT_BUILD_BINDINGS=ON \
            -DINSIGHT_BUILD_PYTHON_BINDING=OFF \
            -DINSIGHT_BUILD_JULIA_BINDING=OFF \
            -DINSIGHT_BUILD_LUA_BINDING=ON \
            -DINSIGHT_WITH_CUDA=ON \
            -DINSIGHT_USE_FFTW3=ON \
            -DINSIGHT_USE_OPENBLAS=ON \
            -DLUA_INCLUDE_DIR="$(LUA_INCDIR)" \
            && cmake --build build.luarocks -j "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
    ]],

    install_command = [[
        mkdir -p "$(LIBDIR)" "$(LUADIR)/insight" && \
        cp build.luarocks/bindings/lua/_insight.so "$(LIBDIR)/" 2>/dev/null || \
        cp build.luarocks/bindings/lua/_insight.dll "$(LIBDIR)/" && \
        for f in build.luarocks/backends/*/*insight*backend*.so \
                 build.luarocks/backends/*/*insight*backend*.dll; do \
            [ -f "$f" ] && cp "$f" "$(LIBDIR)/" 2>/dev/null || true; \
        done && \
        cp bindings/lua/insight/init.lua "$(LUADIR)/insight/" && \
        cp bindings/lua/insight/*.lua "$(LUADIR)/insight/" 2>/dev/null || true
    ]],
}
