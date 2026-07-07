from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BASE_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
DEFAULT_MAX_CONTEXT_TOKENS = 8192

BASE_MODEL = os.environ.get("COVERFORGE_BASE_MODEL", DEFAULT_BASE_MODEL)
MAX_CONTEXT_TOKENS = int(os.environ.get("COVERFORGE_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT_TOKENS)))

ADAPTER_DIR = Path("runs/cover-lora")
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schema" / "cover-spec.schema.json"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 4


def get_base_model() -> str:
    return os.environ.get("COVERFORGE_BASE_MODEL", DEFAULT_BASE_MODEL)


def get_max_context() -> int:
    return int(os.environ.get("COVERFORGE_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT_TOKENS)))


__all__ = [
    "ADAPTER_DIR",
    "BASE_MODEL",
    "BATCH_SIZE",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "EPOCHS",
    "LEARNING_RATE",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_LAYERS",
    "LORA_RANK",
    "MAX_CONTEXT_TOKENS",
    "SCHEMA_PATH",
    "get_base_model",
    "get_max_context",
]
