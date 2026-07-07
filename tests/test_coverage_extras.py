from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import coverforge_trainer.infer as infer_mod
import coverforge_trainer.teacher as teacher_mod
import coverforge_trainer.train as train_mod
from coverforge_trainer.data import synthesize as synth_mod
from coverforge_trainer.infer import (
    _apply_chat_template,
    _extract_json,
    _get_mlx_lm,
    load_subtitle_text,
)
from coverforge_trainer.teacher import (
    SubprocessTeacherClient,
    _normalize_palette_name,
    _palette_from_name,
    label_description,
    recipe_to_cover_spec,
)


def base_recipe() -> dict:
    return {
        "gender": "maennlich",
        "age": "erwachsen",
        "body": "normal",
        "hairColor": "#1B1B22",
        "hairStyle": 1,
        "eyeColor": "#6B4226",
        "skinTone": "#F2C9A0",
        "accessory": "keine",
        "palette": {"name": "Neon Cyber"},
        "typography": "scifi",
        "expression": "neutral",
        "clothingStyle": "tshirt",
        "clothingPattern": "solid",
        "prop": "none",
        "sceneElement": "skyline",
    }


def test_normalize_palette_name_special_chars():
    assert _normalize_palette_name("Eis & Stahl") == "eis-stahl"
    assert _normalize_palette_name("Sonnen-Komödie") == "sonnen-komodie"
    assert _normalize_palette_name("Übergröße") == "ubergroße"


def test_palette_from_name_known():
    from coverforge_trainer.cover_spec import Palette

    assert _palette_from_name("Waldschatten") == Palette.WALDSCHATTEN


def test_palette_from_name_unknown_falls_back():
    from coverforge_trainer.cover_spec import Palette

    assert _palette_from_name("Bogus Name") in list(Palette)


def test_recipe_unknown_enum_values_default():
    r = base_recipe() | {
        "gender": "unknown",
        "age": "unknown",
        "body": "unknown",
        "accessory": "unknown",
        "expression": "unknown",
        "clothingStyle": "unknown",
        "clothingPattern": "unknown",
        "prop": "unknown",
        "sceneElement": "unknown",
        "typography": "unknown",
        "hairStyle": 99,
        "eyeColor": "#FFFFFF",
        "skinTone": "#FFFFFF",
    }
    spec = recipe_to_cover_spec("T", r)
    char = spec.characters[0]
    assert char.gender == "male"
    assert char.age == "adult"
    assert char.body == "normal"
    assert char.accessory == "none"
    assert char.expression == "neutral"
    assert char.clothingStyle == "tshirt"
    assert char.clothingPattern == "solid"
    assert char.prop == "none"
    assert char.hairStyle == "short"
    assert char.eyeColor == "brown"
    assert char.skinTone == "light"


def test_label_description_default_subprocess_client(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        result = MagicMock()
        result.stdout = json.dumps({"recipe": base_recipe()})
        result.stderr = ""
        result.returncode = 0
        return result

    monkeypatch.setattr(teacher_mod.subprocess, "run", fake_run)
    spec = label_description("T", "desc", "Drama")
    assert spec.title == "T"
    assert "tsx" in captured["cmd"]


def test_subprocess_teacher_client_custom_root(tmp_path: Path):
    client = SubprocessTeacherClient(project_root=tmp_path)
    assert client.project_root == tmp_path


def test_subprocess_teacher_client_runs(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        result = MagicMock()
        result.stdout = json.dumps({"recipe": base_recipe(), "hash": "abc"})
        result.stderr = ""
        result.returncode = 0
        return result

    monkeypatch.setattr(teacher_mod.subprocess, "run", fake_run)
    client = SubprocessTeacherClient()
    recipe = client.get_recipe("T", "d", "g")
    assert recipe["hairColor"] == "#1B1B22"
    assert captured["cwd"] is not None


def test_extract_json_no_brace():
    assert _extract_json("no braces here") is None


def test_extract_json_unbalanced():
    assert _extract_json("{ unbalanced") is None


def test_extract_json_array_returns_none():
    assert _extract_json("[1, 2, 3]") is None


def test_extract_json_in_string_braces():
    text = '{"k": "a { b } c"}'
    obj = _extract_json(text)
    assert obj == {"k": "a { b } c"}


def test_extract_json_fence_no_json_returns_none():
    assert _extract_json("```json\nnot json\n```") is None


def test_apply_chat_template_with_apply_method():
    tok = MagicMock()
    tok.apply_chat_template.return_value = "rendered"
    out = _apply_chat_template(tok, [{"role": "system", "content": "x"}])
    assert out == "rendered"


def test_apply_chat_template_fallback():
    class PlainTok:
        pass

    out = _apply_chat_template(
        PlainTok(), [{"role": "system", "content": "X"}, {"role": "user", "content": "Y"}]
    )
    assert "X" in out and "Y" in out


def test_get_mlx_lm_lazy(monkeypatch):
    infer_mod._mlx_lm = None
    fake = MagicMock(name="fake_mlx_lm")
    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", fake)
    assert _get_mlx_lm() is fake
    assert _get_mlx_lm() is fake


def test_load_subtitle_text_json_dict(tmp_path: Path):
    p = tmp_path / "ep.json"
    p.write_text(json.dumps({"text": "whole thing"}), encoding="utf-8")
    assert load_subtitle_text(p, max_chars=10000) == "whole thing"


def test_load_subtitle_text_json_other(tmp_path: Path):
    p = tmp_path / "ep.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    text = load_subtitle_text(p, max_chars=10000)
    assert "1" in text


def test_load_subtitle_text_short_text_no_truncation(tmp_path: Path):
    p = tmp_path / "ep.txt"
    p.write_text("short text", encoding="utf-8")
    assert load_subtitle_text(p, max_chars=10000) == "short text"


def test_load_subtitle_text_skips_webvtt_header(tmp_path: Path):
    p = tmp_path / "ep.vtt"
    p.write_text("WEBVTT\n\nHello line", encoding="utf-8")
    text = load_subtitle_text(p, max_chars=10000)
    assert "WEBVTT" not in text
    assert "Hello line" in text


def test_run_inference_with_adapter(tmp_path: Path, monkeypatch):
    sub = tmp_path / "ep.vtt"
    sub.write_text("WEBVTT\n\nNico: Hallo", encoding="utf-8")
    out = tmp_path / "cover.json"

    fake_model = MagicMock(name="model")
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_tokenizer.apply_chat_template.return_value = "prompt"

    valid = {
        "version": "1.0",
        "title": "T",
        "mood": {"palette": "neon-cyber"},
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

    fake_mlx_lm = MagicMock()
    fake_mlx_lm.load.return_value = (fake_model, fake_tokenizer)
    fake_mlx_lm.generate.return_value = json.dumps(valid)
    monkeypatch.setattr(infer_mod, "_mlx_lm", fake_mlx_lm)

    spec = infer_mod.run_inference(sub, model=None, adapter_path=tmp_path / "adapter", out_path=out)
    assert spec.title == "T"
    fake_mlx_lm.load.assert_called_once()
    assert "adapter_path" in fake_mlx_lm.load.call_args.kwargs


def test_run_inference_non_tuple_load(tmp_path: Path, monkeypatch):
    sub = tmp_path / "ep.vtt"
    sub.write_text("WEBVTT\n\nHello", encoding="utf-8")
    out = tmp_path / "cover.json"

    fake_model = MagicMock(name="model")
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_tokenizer.apply_chat_template.return_value = "prompt"

    loaded = MagicMock()
    loaded.model = fake_model
    loaded.tokenizer = fake_tokenizer

    valid = {
        "version": "1.0",
        "title": "T",
        "mood": {"palette": "neon-cyber"},
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

    fake_mlx_lm = MagicMock()
    fake_mlx_lm.load.return_value = loaded
    fake_mlx_lm.generate.return_value = json.dumps(valid)
    monkeypatch.setattr(infer_mod, "_mlx_lm", fake_mlx_lm)

    spec = infer_mod.run_inference(sub, model="m", adapter_path=None, out_path=out)
    assert spec.title == "T"


def test_infer_main_writes(monkeypatch, tmp_path: Path, capsys):
    sub = tmp_path / "ep.vtt"
    sub.write_text("WEBVTT\n\nHi", encoding="utf-8")
    out = tmp_path / "cover.json"

    valid = {
        "version": "1.0",
        "title": "T",
        "mood": {"palette": "neon-cyber"},
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
    fake_mlx_lm = MagicMock()
    fake_mlx_lm.load.return_value = (MagicMock(), MagicMock(apply_chat_template=lambda m, **k: "p"))
    fake_mlx_lm.generate.return_value = json.dumps(valid)
    monkeypatch.setattr(infer_mod, "_mlx_lm", fake_mlx_lm)

    monkeypatch.setattr(
        "sys.argv",
        ["infer", "--subtitles", str(sub), "--out", str(out), "--model", "m"],
    )
    infer_mod.main()
    assert out.exists()


def test_synthesize_main_writes(monkeypatch, tmp_path: Path):
    recipe = base_recipe()

    class FakeClient:
        def get_recipe(self, t, d, g):
            return recipe

    import coverforge_trainer.teacher as teacher_mod_for_synth

    monkeypatch.setattr(
        teacher_mod_for_synth,
        "SubprocessTeacherClient",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "sys.argv", ["synth", "--out", str(tmp_path), "--n-train", "2", "--n-eval", "1"]
    )
    synth_mod.main()
    assert (tmp_path / "train.jsonl").exists()


def test_yaml_dump_fallback(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "yaml", None)
    cfg = {"model": "m", "lora_parameters": {"rank": 8, "dropout": 0.0}, "lora_layers": ["q_proj"]}
    out = train_mod._yaml_dump(cfg)
    assert "model" in out and "rank" in out


def test_yaml_dump_with_yaml(monkeypatch):
    pytest.importorskip("yaml")
    cfg = {"model": "m", "lora_layers": ["q_proj"]}
    out = train_mod._yaml_dump(cfg)
    assert "model: m" in out


def test_train_main_invokes_run(monkeypatch, tmp_path: Path):
    called = {}

    def fake_run(data, out):
        called["data"] = data
        called["out"] = out

    monkeypatch.setattr(train_mod, "run_training", fake_run)
    monkeypatch.setattr(
        "sys.argv", ["train", "--data", str(tmp_path), "--out", str(tmp_path / "o")]
    )
    train_mod.main()
    assert called["data"] == tmp_path


def test_run_training_stdout_prints(monkeypatch, tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "TRAINLOG"
        result.stderr = "WARN"
        return result

    monkeypatch.setattr(train_mod.subprocess, "run", fake_run)
    train_mod.run_training(data_dir, out_dir)
    captured = capsys.readouterr()
    assert "TRAINLOG" in captured.out
    assert "WARN" in captured.err
    assert (out_dir / "training_config.yaml").exists()
