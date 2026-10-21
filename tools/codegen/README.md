# Insight Lua Codegen

This directory contains the first build-time Lua generator for the architecture refactor. It is intentionally small: the generator validates operator schema records and emits deterministic, reviewable metadata files before any generated kernels replace handwritten code.

## Run manually

```bash
lua tools/codegen/gen.lua build/generated/codegen
```

Generated output:

- `docs/op_manifest.md`
- `docs/kernel_plan.md`
- `docs/source_manifest.md`
- `include/insight/generated/kernel_plan.h`
- `include/insight/generated/source_manifest.h`
- `include/insight/generated/cast_ops.h`
- `include/insight/generated/creation_ops.h`
- `include/insight/generated/elementwise_ops.h`
- `include/insight/generated/reduction_ops.h`
- `include/insight/generated/unary_ops.h`

The generator is intentionally compatible with Lua 5.1, 5.2, 5.3, 5.4, and LuaJIT. It only uses the Lua 5.1 language/library subset: `dofile`, `require`, `io`, `os.execute`, `table`, and `string`.

The schema is backend-neutral: each operator appears once as a single IR record. The IR exposes only host and device adapters; concrete device backends consume the device adapter instead of adding backend-specific schema keys.


```bash
cmake --build build --target insight_codegen
```

The `insight_codegen` target is a developer target and is not part of the default build. Ordinary source builds should keep working when Lua is absent.
