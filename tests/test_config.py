from __future__ import annotations

import importlib
from pathlib import Path

from coverforge_trainer import config


def test_base_model_default():
    assert config.BASE_MODEL == "mlx-community/Qwen3.5-0.8B-MLX-4bit"


def test_schema_path_exists():
    assert isinstance(config.SCHEMA_PATH, Path)
    assert config.SCHEMA_PATH.exists()
    text = config.SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CoverSpec" in text


def test_lora_constants():
    assert isinstance(config.LORA_RANK, int)
    assert config.LORA_RANK == 8
    assert config.LORA_LAYERS == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert len(config.LORA_LAYERS) == 4
    assert config.LORA_ALPHA == 16
    assert config.LORA_DROPOUT == 0.05
    assert config.EPOCHS == 3
    assert config.LEARNING_RATE == 1e-4
    assert config.BATCH_SIZE == 4


def test_max_context_default():
    assert config.MAX_CONTEXT_TOKENS == 8192


def test_adapter_dir():
    assert Path("runs/cover-lora") == config.ADAPTER_DIR


def test_get_base_model_env_override(monkeypatch):
    monkeypatch.setenv("COVERFORGE_BASE_MODEL", "custom/model-4bit")
    assert config.get_base_model() == "custom/model-4bit"


def test_get_max_context_env_override(monkeypatch):
    monkeypatch.setenv("COVERFORGE_MAX_CONTEXT", "4096")
    assert config.get_max_context() == 4096


def test_config_module_reloads_with_env(monkeypatch):
    monkeypatch.setenv("COVERFORGE_BASE_MODEL", "env-model")
    reloaded = importlib.reload(importlib.import_module("coverforge_trainer.config"))
    try:
        assert reloaded.BASE_MODEL == "env-model"
    finally:
        importlib.reload(reloaded)
