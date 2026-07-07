from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from coverforge_trainer import parse_cover_spec

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "cover-spec.schema.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def full_fixture() -> dict:
    return {
        "version": "1.0",
        "title": "Neon Requiem",
        "genre": "cyberpunk thriller",
        "mood": {
            "palette": "neon-cyber",
            "sceneElement": "skyline",
            "typography": "scifi",
        },
        "characters": [
            {
                "gender": "female",
                "age": "adult",
                "body": "thin",
                "hairColor": "black",
                "hairStyle": "long",
                "eyeColor": "blue",
                "skinTone": "light",
                "accessory": "sunglasses",
                "expression": "serious",
                "clothingStyle": "jacket",
                "clothingPattern": "stripes",
                "prop": "phone",
            }
        ],
        "source": {"subtitleEpisodeCount": 12, "modelId": "qwen-0.8b"},
    }


def test_schema_file_present():
    assert SCHEMA_PATH.exists()


def test_full_fixture_validates_against_json_schema():
    schema = load_schema()
    jsonschema.validate(full_fixture(), schema)


def test_pydantic_and_schema_agree_on_valid():
    schema = load_schema()
    fixture = full_fixture()
    jsonschema.validate(fixture, schema)
    spec = parse_cover_spec(fixture)
    assert spec.title == fixture["title"]


def test_schema_rejects_extra_root_property():
    schema = load_schema()
    fixture = full_fixture()
    fixture["unexpected"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(fixture, schema)


def test_schema_rejects_too_many_characters():
    schema = load_schema()
    fixture = full_fixture()
    fixture["characters"] = [{}, {}, {}, {}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(fixture, schema)
