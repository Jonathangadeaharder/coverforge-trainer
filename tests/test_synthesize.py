from __future__ import annotations

import json
import random
from pathlib import Path

from coverforge_trainer.cover_spec import parse_cover_spec
from coverforge_trainer.data.synthesize import (
    SERIES_SEEDS,
    SUBTITLE_TEMPLATES,
    SYSTEM_PROMPT,
    generate_dataset,
    synthesize_example,
)
from coverforge_trainer.teacher import CallableTeacherClient


def sample_recipe() -> dict:
    return {
        "gender": "maennlich",
        "age": "erwachsen",
        "body": "normal",
        "hairColorName": "Schwarz",
        "hairColor": "#1B1B22",
        "hairStyle": 1,
        "eyeColorName": "Braun",
        "eyeColor": "#6B4226",
        "skinTone": "#F2C9A0",
        "accessory": "keine",
        "backgroundId": 0,
        "palette": {"name": "Neon Cyber"},
        "typography": "scifi",
        "hash": "deadbeef",
        "genre": "",
        "expression": "neutral",
        "clothingStyle": "tshirt",
        "clothingPattern": "solid",
        "prop": "none",
        "sceneElement": "skyline",
    }


def test_system_prompt_contains_keywords():
    assert "CoverSpec" in SYSTEM_PROMPT
    assert "json" in SYSTEM_PROMPT.lower()


def test_series_seeds_nonempty():
    assert len(SERIES_SEEDS) >= 10
    for seed in SERIES_SEEDS:
        assert len(seed) == 3
        title, desc, genre = seed
        assert isinstance(title, str) and title
        assert isinstance(desc, str) and desc
        assert isinstance(genre, str)


def test_subtitle_templates_cover_series():
    assert len(SUBTITLE_TEMPLATES) >= 5
    for key, lines in SUBTITLE_TEMPLATES.items():
        assert isinstance(key, str)
        assert len(lines) >= 8
        for line in lines:
            assert isinstance(line, str) and line


def test_synthesize_example_structure():
    recipe = sample_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    rng = random.Random(42)
    example = synthesize_example(SERIES_SEEDS[0], client, rng)
    assert "messages" in example
    msgs = example["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert "<subtitles>" in msgs[1]["content"]
    spec = parse_cover_spec(json.loads(msgs[2]["content"]))
    assert spec.title


def test_synthesize_example_deterministic():
    recipe = sample_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    e1 = synthesize_example(SERIES_SEEDS[0], client, random.Random(123))
    e2 = synthesize_example(SERIES_SEEDS[0], client, random.Random(123))
    assert e1 == e2


def test_generate_dataset_writes_jsonl(tmp_path: Path):
    recipe = sample_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    generate_dataset(tmp_path, client, n_train=5, n_eval=2, seed=42)
    train = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    eval_lines = (tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) == 5
    assert len(eval_lines) == 2
    for line in train:
        obj = json.loads(line)
        assert "messages" in obj
        assert len(obj["messages"]) == 3
        parse_cover_spec(json.loads(obj["messages"][2]["content"]))


def test_generate_dataset_deterministic(tmp_path: Path):
    recipe = sample_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    generate_dataset(tmp_path, client, n_train=5, n_eval=2, seed=7)
    t1 = (tmp_path / "train.jsonl").read_text(encoding="utf-8")
    e1 = (tmp_path / "eval.jsonl").read_text(encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    generate_dataset(other, client, n_train=5, n_eval=2, seed=7)
    t2 = (other / "train.jsonl").read_text(encoding="utf-8")
    e2 = (other / "eval.jsonl").read_text(encoding="utf-8")
    assert t1 == t2
    assert e1 == e2


def test_synthesize_example_seed_used_for_choice():
    recipe = sample_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    rng = random.Random(0)
    ex = synthesize_example(SERIES_SEEDS[0], client, rng)
    assert ex["messages"][0]["content"] == SYSTEM_PROMPT


def test_generate_dataset_creates_dir(tmp_path: Path):
    recipe = sample_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    nested = tmp_path / "nested" / "deep"
    generate_dataset(nested, client, n_train=1, n_eval=1, seed=1)
    assert (nested / "train.jsonl").exists()
