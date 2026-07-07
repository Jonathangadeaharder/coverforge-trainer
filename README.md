# coverforge-trainer

MLX LoRA trainer for the **CoverSpec** contract — the renderer-agnostic cover
specification that mediates between an episode-analysis model and the cover
rendering engine.

## The Contract

The canonical contract lives at [`schema/cover-spec.schema.json`](schema/cover-spec.schema.json).
It is the single source of truth for the `CoverSpec` data model. The pydantic
model in `src/coverforge_trainer/cover_spec.py` mirrors it exactly (same enums,
optionality, field names, and constraints). `tests/test_schema_sync.py` guards
against drift between the pydantic model and the JSON Schema.

## Install

```bash
uv sync
```

> Note: `mlx` and `mlx-lm` require macOS on Apple Silicon (arm64). On other
> platforms `uv sync` will fail to resolve those wheels.

## Run tests

```bash
uv run pytest
uv run ruff check
uv run pyright
```

Branch coverage is enforced at >= 90%.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `COVERFORGE_BASE_MODEL` | `mlx-community/Qwen3.5-0.8B-MLX-4bit` | Base MLX model to fine-tune |
| `COVERFORGE_MAX_CONTEXT` | `8192` | Maximum context tokens |

LoRA training constants (`LORA_RANK=8`, `LORA_ALPHA=16`, `LORA_LAYERS`,
`EPOCHS=3`, `LEARNING_RATE=1e-4`, `BATCH_SIZE=4`) and the adapter output
directory (`runs/cover-lora`) are defined in `src/coverforge_trainer/config.py`.
