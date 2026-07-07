from __future__ import annotations

import pytest

from coverforge_trainer.cover_spec import CoverSpec
from coverforge_trainer.teacher import (
    CallableTeacherClient,
    SubprocessTeacherClient,
    TeacherClient,
    label_description,
    recipe_to_cover_spec,
)


def base_recipe() -> dict:
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


def test_recipe_to_cover_spec_basic_mapping():
    spec = recipe_to_cover_spec("Nicos Weg", base_recipe(), genre="Drama")
    assert isinstance(spec, CoverSpec)
    assert spec.title == "Nicos Weg"
    assert spec.genre == "Drama"
    assert spec.version == "1.0"
    assert spec.source is None
    char = spec.characters[0]
    assert char.gender == "male"
    assert char.age == "adult"
    assert char.body == "normal"
    assert char.hairColor == "black"
    assert char.hairStyle == "short"
    assert char.eyeColor == "brown"
    assert char.skinTone == "light"
    assert char.accessory == "none"
    assert char.expression == "neutral"
    assert char.clothingStyle == "tshirt"
    assert char.clothingPattern == "solid"
    assert char.prop == "none"
    assert spec.mood is not None
    assert spec.mood.palette == "neon-cyber"
    assert spec.mood.sceneElement == "skyline"
    assert spec.mood.typography == "scifi"
    assert len(spec.characters) == 1


@pytest.mark.parametrize(
    "german,neutral",
    [("maennlich", "male"), ("weiblich", "female")],
)
def test_gender_mapping(german, neutral):
    r = base_recipe() | {"gender": german}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].gender == neutral


@pytest.mark.parametrize(
    "german,neutral",
    [("kind", "child"), ("erwachsen", "adult"), ("alt", "elderly")],
)
def test_age_mapping(german, neutral):
    r = base_recipe() | {"age": german}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].age == neutral


@pytest.mark.parametrize(
    "german,neutral",
    [("duenn", "thin"), ("normal", "normal"), ("muskuloes", "muscular"), ("dick", "stout")],
)
def test_body_mapping(german, neutral):
    r = base_recipe() | {"body": german}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].body == neutral


@pytest.mark.parametrize(
    "hexcolor,expected",
    [
        ("#E8B84B", "blond"),
        ("#6B4226", "brown"),
        ("#C0392B", "red"),
        ("#1B1B22", "black"),
    ],
)
def test_hair_color_mapping(hexcolor, expected):
    r = base_recipe() | {"hairColor": hexcolor}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].hairColor == expected


def test_hair_color_grey_for_elderly():
    r = base_recipe() | {"age": "alt", "hairColor": "#C7CAD1"}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].hairColor == "grey"
    assert spec.characters[0].age == "elderly"


@pytest.mark.parametrize(
    "style,expected",
    [
        (0, "bald"),
        (1, "short"),
        (2, "mohawk"),
        (3, "long"),
        (4, "ponytail"),
        (5, "bun"),
        (6, "afro"),
        (7, "braids"),
    ],
)
def test_hair_style_mapping(style, expected):
    r = base_recipe() | {"hairStyle": style}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].hairStyle == expected


@pytest.mark.parametrize(
    "hexcolor,expected",
    [("#6B4226", "brown"), ("#3AA0E0", "blue"), ("#4CA46A", "green")],
)
def test_eye_color_mapping(hexcolor, expected):
    r = base_recipe() | {"eyeColor": hexcolor}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].eyeColor == expected


@pytest.mark.parametrize(
    "hexcolor,expected",
    [
        ("#F2C9A0", "light"),
        ("#E4B08A", "medium"),
        ("#C68642", "tan"),
        ("#8D5524", "deep"),
    ],
)
def test_skin_tone_mapping(hexcolor, expected):
    r = base_recipe() | {"skinTone": hexcolor}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].skinTone == expected


@pytest.mark.parametrize(
    "german,neutral",
    [
        ("keine", "none"),
        ("brille", "glasses"),
        ("sonnenbrille", "sunglasses"),
        ("hut", "hat"),
        ("muetze", "beanie"),
        ("bart", "beard"),
    ],
)
def test_accessory_mapping(german, neutral):
    r = base_recipe() | {"accessory": german}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].accessory == neutral


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Noir Thriller", "noir-thriller"),
        ("Neon Cyber", "neon-cyber"),
        ("Sonnen-Komödie", "sonnen-komoedie"),
        ("Pastell Indie", "pastell-indie"),
        ("Waldschatten", "waldschatten"),
        ("Blut-Sonnenuntergang", "blut-sonnenuntergang"),
        ("Royal Prestige", "royal-prestige"),
        ("Eis & Stahl", "eis-stahl"),
    ],
)
def test_palette_mapping(name, expected):
    r = base_recipe() | {"palette": {"name": name}}
    spec = recipe_to_cover_spec("T", r)
    assert spec.mood is not None
    assert spec.mood.palette == expected


def test_identity_fields_passthrough():
    r = base_recipe() | {
        "expression": "wink",
        "clothingStyle": "hoodie",
        "clothingPattern": "stripes",
        "prop": "guitar",
        "sceneElement": "mountains",
        "typography": "arthouse",
    }
    spec = recipe_to_cover_spec("T", r)
    char = spec.characters[0]
    assert char.expression == "wink"
    assert char.clothingStyle == "hoodie"
    assert char.clothingPattern == "stripes"
    assert char.prop == "guitar"
    assert spec.mood is not None
    assert spec.mood.sceneElement == "mountains"
    assert spec.mood.typography == "arthouse"


def test_label_description_with_callable_client():
    recipe = base_recipe()

    def fake(title, description, genre):
        return recipe

    client = CallableTeacherClient(fake)
    spec = label_description("Nicos Weg", "desc", "Drama", teacher=client)
    expected = recipe_to_cover_spec("Nicos Weg", recipe, genre="Drama")
    assert spec.model_dump() == expected.model_dump()


def test_label_description_default_genre():
    recipe = base_recipe()
    client = CallableTeacherClient(lambda t, d, g: recipe)
    spec = label_description("T", "d", teacher=client)
    assert spec.genre is None or spec.genre == ""


def test_label_description_bart_weiblich_coerced_by_engine():
    recipe = base_recipe() | {"accessory": "bart", "gender": "weiblich"}
    client = CallableTeacherClient(lambda t, d, g: recipe)
    spec = label_description("T", "d", teacher=client)
    # Engine coerces bart→brille for female; teacher just maps faithfully.
    assert spec.characters[0].accessory == "beard"
    assert spec.characters[0].gender == "female"


def test_teacher_client_protocol_subprocess_attribute():
    assert isinstance(SubprocessTeacherClient(), TeacherClient)
    assert isinstance(CallableTeacherClient(lambda t, d, g: {}), TeacherClient)


def test_recipe_to_cover_spec_unknown_hair_color_falls_back_to_black():
    r = base_recipe() | {"hairColor": "#FFFFFF"}
    spec = recipe_to_cover_spec("T", r)
    assert spec.characters[0].hairColor == "black"
