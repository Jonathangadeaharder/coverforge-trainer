from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from coverforge_trainer.config import get_base_model, get_max_context
from coverforge_trainer.cover_spec import CoverSpec, parse_cover_spec
from coverforge_trainer.data.synthesize import SYSTEM_PROMPT

_mlx_lm: Any = None


def _get_mlx_lm():
    global _mlx_lm
    if _mlx_lm is None:
        # Apply the transformers-5 compat shim BEFORE importing mlx_lm.
        # mlx-lm 0.31.x breaks at import time on transformers 5.x due to a
        # changed AutoTokenizer.register() signature.
        import mlx_lm

        from coverforge_trainer import _mlx_compat as _  # noqa: F401

        _mlx_lm = mlx_lm
    return _mlx_lm


def _apply_chat_template(tokenizer: Any, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(m["content"] for m in messages)


_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}")


def load_subtitle_text(path: Path, max_chars: int) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            parts: list[str] = []
            for seg in data:
                if isinstance(seg, dict) and "text" in seg:
                    parts.append(str(seg["text"]).strip())
            text = " ".join(parts) if parts else json.dumps(data)
        elif isinstance(data, dict) and "text" in data:
            text = str(data["text"])
        else:
            text = json.dumps(data)
    else:
        raw_lines = p.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.isdigit():
                continue
            if _TIMESTAMP_RE.match(stripped):
                continue
            if stripped.upper() == "WEBVTT":
                continue
            kept.append(stripped)
        text = "\n".join(kept)

    if len(text) <= max_chars:
        return text

    head_len = int(max_chars * 0.4)
    tail_len = max_chars - head_len
    head = text[:head_len]
    tail = text[-tail_len:]
    return head + tail


def build_prompt(subtitle_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<subtitles>{subtitle_text}</subtitles>"},
    ]


def parse_model_output(text: str) -> CoverSpec:
    candidate = _extract_json(text)
    if candidate is None:
        raise ValueError(f"No JSON object found in model output: {text[:120]!r}")
    return parse_cover_spec(candidate)


def _extract_json(text: str) -> dict | None:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return _safe_json(fence.group(1))
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    return _safe_json(text[start : end + 1])


def _safe_json(s: str) -> dict | None:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def run_inference(
    subtitle_path: Path,
    model: str | None,
    adapter_path: Path | None,
    out_path: Path,
) -> CoverSpec:
    mlx_lm = _get_mlx_lm()
    base_model = model or get_base_model()
    max_tokens = get_max_context()

    load_kwargs: dict[str, Any] = {}
    if adapter_path is not None:
        load_kwargs["adapter_path"] = str(adapter_path)
    loaded = mlx_lm.load(base_model, **load_kwargs)
    if isinstance(loaded, tuple):
        mlx_model, tokenizer = loaded
    else:
        mlx_model, tokenizer = loaded.model, loaded.tokenizer

    subtitle_text = load_subtitle_text(Path(subtitle_path), max_chars=max_tokens)
    messages = build_prompt(subtitle_text)
    prompt_text = _apply_chat_template(tokenizer, messages)

    output = mlx_lm.generate(mlx_model, tokenizer, prompt=prompt_text, max_tokens=4096)
    spec = parse_model_output(output)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoverSpec inference on subtitles.")
    parser.add_argument("--subtitles", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    args = parser.parse_args()

    adapter = Path(args.adapter) if args.adapter else None
    spec = run_inference(
        Path(args.subtitles),
        model=args.model,
        adapter_path=adapter,
        out_path=Path(args.out),
    )
    print(spec.model_dump_json(indent=2))


__all__ = [
    "build_prompt",
    "load_subtitle_text",
    "main",
    "parse_model_output",
    "run_inference",
]
