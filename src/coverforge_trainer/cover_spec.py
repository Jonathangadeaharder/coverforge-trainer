from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Palette(StrEnum):
    NOIR_THRILLER = "noir-thriller"
    NEON_CYBER = "neon-cyber"
    SONNEN_KOMOEDIE = "sonnen-komoedie"
    PASTELL_INDIE = "pastell-indie"
    WALDSCHATTEN = "waldschatten"
    BLUT_SONNENUNTERGANG = "blut-sonnenuntergang"
    ROYAL_PRESTIGE = "royal-prestige"
    EIS_STAHL = "eis-stahl"


class SceneElement(StrEnum):
    NONE = "none"
    SKYLINE = "skyline"
    MOUNTAINS = "mountains"
    STARS = "stars"
    CLOUDS = "clouds"
    CONFETTI = "confetti"


class Typography(StrEnum):
    BLOCKBUSTER = "blockbuster"
    PRESTIGE = "prestige"
    SCIFI = "scifi"
    ARTHOUSE = "arthouse"


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"


class Age(StrEnum):
    CHILD = "child"
    ADULT = "adult"
    ELDERLY = "elderly"


class Body(StrEnum):
    THIN = "thin"
    NORMAL = "normal"
    MUSCULAR = "muscular"
    STOUT = "stout"


class HairColor(StrEnum):
    BLOND = "blond"
    BROWN = "brown"
    RED = "red"
    BLACK = "black"
    GREY = "grey"


class HairStyle(StrEnum):
    BALD = "bald"
    SHORT = "short"
    MOHAWK = "mohawk"
    LONG = "long"
    PONYTAIL = "ponytail"
    BUN = "bun"
    AFRO = "afro"
    BRAIDS = "braids"


class EyeColor(StrEnum):
    BROWN = "brown"
    BLUE = "blue"
    GREEN = "green"


class SkinTone(StrEnum):
    LIGHT = "light"
    MEDIUM = "medium"
    TAN = "tan"
    DEEP = "deep"


class Accessory(StrEnum):
    NONE = "none"
    GLASSES = "glasses"
    SUNGLASSES = "sunglasses"
    HAT = "hat"
    BEANIE = "beanie"
    BEARD = "beard"


class Expression(StrEnum):
    NEUTRAL = "neutral"
    SMILE = "smile"
    SERIOUS = "serious"
    SURPRISED = "surprised"
    WINK = "wink"


class ClothingStyle(StrEnum):
    TSHIRT = "tshirt"
    HOODIE = "hoodie"
    JACKET = "jacket"
    DRESS = "dress"
    SWEATER = "sweater"
    COLLARED = "collared"


class ClothingPattern(StrEnum):
    SOLID = "solid"
    STRIPES = "stripes"
    DOTS = "dots"
    EMBLEM = "emblem"


class Prop(StrEnum):
    NONE = "none"
    BOOK = "book"
    PHONE = "phone"
    GUITAR = "guitar"
    COFFEE = "coffee"
    HEADPHONES = "headphones"
    PEN = "pen"
    UMBRELLA = "umbrella"
    CAMERA = "camera"
    SPEECH = "speech"


class Mood(BaseModel):
    model_config = ConfigDict(extra="forbid")

    palette: Palette | None = Field(default=None)
    sceneElement: SceneElement | None = Field(default=None)
    typography: Typography | None = Field(default=None)


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Gender | None = Field(default=None)
    age: Age | None = Field(default=None)
    body: Body | None = Field(default=None)
    hairColor: HairColor | None = Field(default=None)
    hairStyle: HairStyle | None = Field(default=None)
    eyeColor: EyeColor | None = Field(default=None)
    skinTone: SkinTone | None = Field(default=None)
    accessory: Accessory | None = Field(default=None)
    expression: Expression | None = Field(default=None)
    clothingStyle: ClothingStyle | None = Field(default=None)
    clothingPattern: ClothingPattern | None = Field(default=None)
    prop: Prop | None = Field(default=None)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtitleEpisodeCount: int | None = Field(default=None, ge=0)
    modelId: str | None = Field(default=None, max_length=80)


class CoverSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    title: str = Field(min_length=1, max_length=120)
    genre: str | None = Field(default=None, max_length=80)
    mood: Mood | None = Field(default=None)
    characters: list[Character] = Field(min_length=1, max_length=3)
    source: Source | None = Field(default=None)


def parse_cover_spec(data: dict) -> CoverSpec:
    return CoverSpec.model_validate(data)


def load_cover_spec(path: Path) -> CoverSpec:
    text = Path(path).read_text(encoding="utf-8")
    return parse_cover_spec(json.loads(text))


__all__ = [
    "Accessory",
    "Age",
    "Body",
    "Character",
    "ClothingPattern",
    "ClothingStyle",
    "CoverSpec",
    "EyeColor",
    "Expression",
    "Gender",
    "HairColor",
    "HairStyle",
    "Mood",
    "Palette",
    "Prop",
    "SceneElement",
    "Source",
    "Typography",
    "ValidationError",
    "load_cover_spec",
    "parse_cover_spec",
]
