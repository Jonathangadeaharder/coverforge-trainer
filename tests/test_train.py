from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import coverforge_trainer.train as train_mod
from coverforge_trainer import config
from coverforge_trainer.train import build_training_config, run_training


def test_build_training_config_shape():
    cfg = build_training_config()
    assert cfg["model"] == config.get_base_model()
    assert cfg["train"] is True
    assert cfg["fine_tune_type"] == "lora"
    assert cfg["data"] is not None
    assert cfg["adapter_path"] is not None
    lp = cfg["lora_parameters"]
    assert lp["rank"] == config.LORA_RANK
    assert lp["alpha"] == config.LORA_ALPHA
    assert lp["dropout"] == config.LORA_DROPOUT
    assert "scale" in lp
    assert cfg["lora_layers"] == config.LORA_LAYERS
    assert cfg["num_epochs"] == config.EPOCHS
    assert cfg["learning_rate"] == config.LEARNING_RATE
    assert cfg["batch_size"] == config.BATCH_SIZE
    assert cfg["iters"] > 0


def test_build_training_config_has_iters():
    cfg = build_training_config()
    assert isinstance(cfg["iters"], int)
    assert cfg["iters"] >= 1


def test_run_training_invokes_mlx_lm_lora(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "trained"
        result.stderr = ""
        return result

    monkeypatch.setattr(train_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(train_mod, "_yaml_dump", lambda cfg: "yaml-content")

    run_training(data_dir, out_dir)

    assert "lora" in captured["cmd"]
    assert any("lora" in str(c) for c in captured["cmd"])
    cfg_path = [c for c in captured["cmd"] if isinstance(c, str) and c.endswith(".yaml")][0]
    assert Path(cfg_path).exists()
    assert "--train" in captured["cmd"]


def test_run_training_failure_raises(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "boom"
        result.check_returncode.side_effect = subprocess.CalledProcessError(
            1, cmd, output="", stderr="boom"
        )
        return result

    monkeypatch.setattr(train_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(train_mod, "_yaml_dump", lambda cfg: "yaml-content")

    with pytest.raises(subprocess.CalledProcessError):
        run_training(data_dir, out_dir)
