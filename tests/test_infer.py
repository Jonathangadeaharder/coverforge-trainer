from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from coverforge_trainer.cover_spec import CoverSpec
from coverforge_trainer.data.synthesize import SYSTEM_PROMPT
from coverforge_trainer.infer import (
    build_prompt,
    load_subtitle_text,
    parse_model_output,
    run_inference,
)

VALID_SPEC_DICT = {
    "version": "1.0",
    "title": "Nicos Weg",
    "genre": "Drama",
    "mood": {"palette": "neon-cyber", "sceneElement": "skyline", "typography": "scifi"},
    "characters": [
        {
            "gender": "male",
            "age": "adult",
            "body": "normal",
            "hairColor": "black",
            "hairStyle": "short",
            "eyeColor": "brown",
            "skinTone": "light",
            "accessory": "none",
            "expression": "neutral",
            "clothingStyle": "tshirt",
            "clothingPattern": "solid",
            "prop": "none",
        }
    ],
}


def write_vtt(path: Path) -> None:
    path.write_text(
        "WEBVTT\n\n"
        "1\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "Nico: Hallo, ich heiße Nico.\n\n"
        "2\n"
        "00:00:04.000 --> 00:00:07.000\n"
        "Lisa: Wo ist die Apotheke?\n",
        encoding="utf-8",
    )


def test_load_subtitle_text_vtt(tmp_path: Path):
    p = tmp_path / "ep.vtt"
    write_vtt(p)
    text = load_subtitle_text(p, max_chars=10000)
    assert "Nico: Hallo, ich heiße Nico." in text
    assert "Lisa: Wo ist die Apotheke?" in text
    assert "-->" not in text
    assert "00:00" not in text
    assert "1\n" not in text.split("\n")


def test_load_subtitle_text_srt(tmp_path: Path):
    p = tmp_path / "ep.srt"
    p.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:04,000\n"
        "Hello world\n\n"
        "2\n"
        "00:00:04,000 --> 00:00:07,000\n"
        "Goodbye\n",
        encoding="utf-8",
    )
    text = load_subtitle_text(p, max_chars=10000)
    assert "Hello world" in text
    assert "Goodbye" in text
    assert "-->" not in text


def test_load_subtitle_text_json_segments(tmp_path: Path):
    p = tmp_path / "ep.json"
    segments = [
        {"start": 0, "end": 1, "text": "Hello "},
        {"start": 1, "end": 2, "text": "world"},
    ]
    p.write_text(json.dumps(segments), encoding="utf-8")
    text = load_subtitle_text(p, max_chars=10000)
    assert "Hello world" in text


def test_load_subtitle_text_truncation(tmp_path: Path):
    p = tmp_path / "big.txt"
    head = "HEAD" * 100
    middle = "X" * 18000
    tail = "TAIL" * 100
    p.write_text(head + middle + tail, encoding="utf-8")
    text = load_subtitle_text(p, max_chars=5000)
    assert len(text) <= 5000
    assert text.startswith(head[:20])
    assert text.endswith(tail[-20:])


def test_parse_model_output_clean_json():
    text = json.dumps(VALID_SPEC_DICT)
    spec = parse_model_output(text)
    assert isinstance(spec, CoverSpec)
    assert spec.title == "Nicos Weg"


def test_parse_model_output_markdown_fence():
    text = "```json\n" + json.dumps(VALID_SPEC_DICT) + "\n```"
    spec = parse_model_output(text)
    assert isinstance(spec, CoverSpec)


def test_parse_model_output_leading_prose():
    text = "Here is the spec:\n" + json.dumps(VALID_SPEC_DICT)
    spec = parse_model_output(text)
    assert isinstance(spec, CoverSpec)


def test_parse_model_output_garbage():
    with pytest.raises((ValidationError, ValueError)):
        parse_model_output("this is not json at all")


def test_build_prompt_structure():
    msgs = build_prompt("some subtitles here")
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "<subtitles>" in msgs[1]["content"]
    assert "some subtitles here" in msgs[1]["content"]
    assert "</subtitles>" in msgs[1]["content"]


def test_run_inference_mocked(tmp_path: Path, monkeypatch):
    sub = tmp_path / "ep.vtt"
    write_vtt(sub)
    out = tmp_path / "cover.json"

    fake_model = MagicMock(name="model")
    fake_tokenizer = MagicMock(name="tokenizer")

    def fake_load(model, adapter_path=None, **kwargs):
        assert adapter_path is not None or model is not None
        return fake_model, fake_tokenizer

    def fake_generate(model, tokenizer, prompt, max_tokens=4096, **kwargs):
        return json.dumps(VALID_SPEC_DICT)

    fake_apply = MagicMock()

    import coverforge_trainer.infer as infer_mod

    fake_mlx_lm = MagicMock()
    fake_mlx_lm.load = fake_load
    fake_mlx_lm.generate = fake_generate
    monkeypatch.setattr(infer_mod, "_mlx_lm", fake_mlx_lm)
    monkeypatch.setattr(infer_mod, "_apply_chat_template", fake_apply)

    spec = run_inference(sub, model="some/model", adapter_path=None, out_path=out)
    assert isinstance(spec, CoverSpec)
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["title"] == "Nicos Weg"
