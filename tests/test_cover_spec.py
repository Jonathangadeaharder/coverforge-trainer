from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coverforge_trainer import CoverSpec, load_cover_spec, parse_cover_spec


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
            },
            {
                "gender": "male",
                "age": "elderly",
                "body": "stout",
                "hairColor": "grey",
                "hairStyle": "short",
                "eyeColor": "brown",
                "skinTone": "tan",
                "accessory": "beard",
                "expression": "neutral",
                "clothingStyle": "collared",
                "clothingPattern": "solid",
                "prop": "none",
            },
        ],
        "source": {
            "subtitleEpisodeCount": 12,
            "modelId": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        },
    }


def test_valid_full_fixture_parses():
    spec = parse_cover_spec(full_fixture())
    assert spec.version == "1.0"
    assert spec.title == "Neon Requiem"
    assert spec.genre == "cyberpunk thriller"
    assert spec.mood is not None
    assert spec.mood.palette == "neon-cyber"
    assert spec.mood.sceneElement == "skyline"
    assert spec.mood.typography == "scifi"
    assert len(spec.characters) == 2
    assert spec.characters[0].gender == "female"
    assert spec.characters[1].prop == "none"
    assert spec.source is not None
    assert spec.source.subtitleEpisodeCount == 12
    assert spec.source.modelId == "mlx-community/Qwen3.5-0.8B-MLX-4bit"


def test_valid_minimal_spec():
    spec = parse_cover_spec({"version": "1.0", "title": "X", "characters": [{}]})
    assert spec.version == "1.0"
    assert spec.title == "X"
    assert spec.genre is None
    assert spec.mood is None
    assert spec.source is None
    assert len(spec.characters) == 1
    char = spec.characters[0]
    assert char.gender is None
    assert char.prop is None


def test_reject_wrong_version():
    with pytest.raises(ValidationError):
        parse_cover_spec({"version": "2.0", "title": "X", "characters": [{}]})


def test_reject_empty_title():
    with pytest.raises(ValidationError):
        parse_cover_spec({"version": "1.0", "title": "", "characters": [{}]})


def test_reject_title_too_long():
    with pytest.raises(ValidationError):
        parse_cover_spec({"version": "1.0", "title": "a" * 121, "characters": [{}]})


def test_reject_zero_characters():
    with pytest.raises(ValidationError):
        parse_cover_spec({"version": "1.0", "title": "X", "characters": []})


def test_reject_four_characters():
    with pytest.raises(ValidationError):
        parse_cover_spec(
            {
                "version": "1.0",
                "title": "X",
                "characters": [{}, {}, {}, {}],
            }
        )


def test_reject_unknown_palette():
    fixture = full_fixture()
    fixture["mood"]["palette"] = "bubblegum-pop"
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_reject_unknown_gender():
    fixture = full_fixture()
    fixture["characters"][0]["gender"] = "android"
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_reject_additional_property_root():
    fixture = full_fixture()
    fixture["surprise"] = True
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_reject_additional_property_character():
    fixture = full_fixture()
    fixture["characters"][0]["nickname"] = "Neo"
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_reject_additional_property_mood():
    fixture = full_fixture()
    fixture["mood"]["weather"] = "rain"
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_reject_negative_subtitle_episode_count():
    fixture = full_fixture()
    fixture["source"]["subtitleEpisodeCount"] = -1
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_reject_model_id_too_long():
    fixture = full_fixture()
    fixture["source"]["modelId"] = "x" * 81
    with pytest.raises(ValidationError):
        parse_cover_spec(fixture)


def test_optional_fields_default_none():
    spec = parse_cover_spec({"version": "1.0", "title": "X", "characters": [{}]})
    assert spec.genre is None
    assert spec.mood is None
    assert spec.source is None
    assert spec.characters[0].gender is None
    assert spec.characters[0].age is None


def test_load_cover_spec_from_file(tmp_path: Path):
    import json

    path = tmp_path / "cover.json"
    path.write_text(json.dumps(full_fixture()), encoding="utf-8")
    spec = load_cover_spec(path)
    assert isinstance(spec, CoverSpec)
    assert spec.title == "Neon Requiem"


def test_load_cover_spec_invalid_file(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": "9.9"}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_cover_spec(path)
