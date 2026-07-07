from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from coverforge_trainer import config


def _yaml_dump(cfg: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _yaml_dump_fallback(cfg)
    return yaml.safe_dump(cfg, sort_keys=False)


def _yaml_dump_fallback(cfg: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                emit(f"{prefix}{k}", v)
        elif isinstance(value, list):
            for item in value:
                lines.append(f"{prefix}[]: {item}")
        else:
            lines.append(f"{prefix}: {value}")

    for k, v in cfg.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            emit("  ", v)
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            emit(k, v)
    return "\n".join(lines) + "\n"


def build_training_config() -> dict[str, Any]:
    scale = config.LORA_ALPHA / config.LORA_RANK if config.LORA_RANK else 2.0
    return {
        "model": config.get_base_model(),
        "train": True,
        "fine_tune_type": "lora",
        "data": str(Path("data").resolve()),
        "num_layers": -1,
        "batch_size": config.BATCH_SIZE,
        "num_epochs": config.EPOCHS,
        "iters": 100,
        "learning_rate": config.LEARNING_RATE,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "steps_per_save": 100,
        "val_batches": 25,
        "adapter_path": str(config.ADAPTER_DIR),
        "max_seq_length": config.get_max_context(),
        "grad_checkpoint": False,
        "seed": 42,
        "lora_parameters": {
            "rank": config.LORA_RANK,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "scale": scale,
        },
        "lora_layers": list(config.LORA_LAYERS),
    }


def run_training(data_dir: Path, out_dir: Path) -> None:
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_training_config()
    cfg["data"] = str(data_dir)
    cfg["adapter_path"] = str(out_dir)

    config_file = out_dir / "training_config.yaml"
    config_file.write_text(_yaml_dump(cfg), encoding="utf-8")

    # `python -m mlx_lm` triggers the broken AutoTokenizer.register() call in
    # mlx_lm/__init__.py. Run via -c to apply the shim first, then dispatch to
    # mlx_lm.cli.main() (argv set as if `python -m mlx_lm lora --config ...`).
    cmd = [
        sys.executable,
        "-c",
        "from coverforge_trainer import _mlx_compat; from mlx_lm.cli import main; main()",
        "mlx_lm",
        "lora",
        "--config",
        str(config_file),
        "--train",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        result.check_returncode()
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(f"Training complete. Adapter written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CoverSpec LoRA adapter via mlx-lm.")
    parser.add_argument("--data", type=str, default="data/")
    parser.add_argument("--out", type=str, default=str(config.ADAPTER_DIR))
    args = parser.parse_args()
    run_training(Path(args.data), Path(args.out))


__all__ = ["build_training_config", "main", "run_training"]
