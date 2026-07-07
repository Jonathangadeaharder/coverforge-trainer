from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from coverforge_trainer.cover_spec import (
    Accessory,
    Age,
    Body,
    Character,
    ClothingPattern,
    ClothingStyle,
    CoverSpec,
    Expression,
    EyeColor,
    Gender,
    HairColor,
    HairStyle,
    Mood,
    Palette,
    Prop,
    SceneElement,
    SkinTone,
    Typography,
)

COVERFORGE_ROOT = Path("/Users/jonathangadeaharder/projects/vidiomtm/coverforge")

GENDER_MAP = {"maennlich": Gender.MALE, "weiblich": Gender.FEMALE}
AGE_MAP = {"kind": Age.CHILD, "erwachsen": Age.ADULT, "alt": Age.ELDERLY}
BODY_MAP = {
    "duenn": Body.THIN,
    "normal": Body.NORMAL,
    "muskuloes": Body.MUSCULAR,
    "dick": Body.STOUT,
}
HAIR_COLOR_MAP = {
    "#E8B84B": HairColor.BLOND,
    "#6B4226": HairColor.BROWN,
    "#C0392B": HairColor.RED,
    "#1B1B22": HairColor.BLACK,
    "#C7CAD1": HairColor.GREY,
}
HAIR_STYLE_MAP = {
    0: HairStyle.BALD,
    1: HairStyle.SHORT,
    2: HairStyle.MOHAWK,
    3: HairStyle.LONG,
    4: HairStyle.PONYTAIL,
    5: HairStyle.BUN,
    6: HairStyle.AFRO,
    7: HairStyle.BRAIDS,
}
EYE_COLOR_MAP = {
    "#6B4226": EyeColor.BROWN,
    "#3AA0E0": EyeColor.BLUE,
    "#4CA46A": EyeColor.GREEN,
}
SKIN_TONE_MAP = {
    "#F2C9A0": SkinTone.LIGHT,
    "#E4B08A": SkinTone.MEDIUM,
    "#C68642": SkinTone.TAN,
    "#8D5524": SkinTone.DEEP,
}
ACCESSORY_MAP = {
    "keine": Accessory.NONE,
    "brille": Accessory.GLASSES,
    "sonnenbrille": Accessory.SUNGLASSES,
    "hut": Accessory.HAT,
    "muetze": Accessory.BEANIE,
    "bart": Accessory.BEARD,
}
PALETTE_NAME_MAP = {
    "Noir Thriller": Palette.NOIR_THRILLER,
    "Neon Cyber": Palette.NEON_CYBER,
    "Sonnen-Komödie": Palette.SONNEN_KOMOEDIE,
    "Pastell Indie": Palette.PASTELL_INDIE,
    "Waldschatten": Palette.WALDSCHATTEN,
    "Blut-Sonnenuntergang": Palette.BLUT_SONNENUNTERGANG,
    "Royal Prestige": Palette.ROYAL_PRESTIGE,
    "Eis & Stahl": Palette.EIS_STAHL,
}


def _map_enum(value, mapping, default, label):
    if value in mapping:
        return mapping[value]
    return default


def _palette_from_name(name: str) -> Palette:
    if name in PALETTE_NAME_MAP:
        return PALETTE_NAME_MAP[name]
    normalized = _normalize_palette_name(name)
    for member in Palette:
        if member.value == normalized:
            return member
    return Palette.NEON_CYBER


def _normalize_palette_name(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch == "ä":
            out.append("a")
        elif ch == "ö":
            out.append("o")
        elif ch == "ü":
            out.append("u")
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    raw = "".join(out)
    collapsed = []
    prev_dash = False
    for ch in raw:
        if ch == "-":
            if prev_dash:
                continue
            prev_dash = True
        else:
            prev_dash = False
        collapsed.append(ch)
    return "".join(collapsed)


def recipe_to_cover_spec(title: str, recipe: dict[str, Any], genre: str = "") -> CoverSpec:
    age = _map_enum(recipe.get("age"), AGE_MAP, Age.ADULT, "age")
    hair_color_raw = recipe.get("hairColor", "#1B1B22")
    if age == Age.ELDERLY:
        hair_color = HairColor.GREY
    else:
        hair_color = HAIR_COLOR_MAP.get(hair_color_raw, HairColor.BLACK)

    palette = _palette_from_name(recipe.get("palette", {}).get("name", ""))

    character = Character(
        gender=_map_enum(recipe.get("gender"), GENDER_MAP, Gender.MALE, "gender"),
        age=age,
        body=_map_enum(recipe.get("body"), BODY_MAP, Body.NORMAL, "body"),
        hairColor=hair_color,
        hairStyle=HAIR_STYLE_MAP.get(recipe.get("hairStyle", 1), HairStyle.SHORT),
        eyeColor=EYE_COLOR_MAP.get(str(recipe.get("eyeColor")), EyeColor.BROWN),
        skinTone=SKIN_TONE_MAP.get(str(recipe.get("skinTone")), SkinTone.LIGHT),
        accessory=_map_enum(recipe.get("accessory"), ACCESSORY_MAP, Accessory.NONE, "accessory"),
        expression=_coerce_enum(recipe.get("expression"), Expression, Expression.NEUTRAL),
        clothingStyle=_coerce_enum(
            recipe.get("clothingStyle"), ClothingStyle, ClothingStyle.TSHIRT
        ),
        clothingPattern=_coerce_enum(
            recipe.get("clothingPattern"), ClothingPattern, ClothingPattern.SOLID
        ),
        prop=_coerce_enum(recipe.get("prop"), Prop, Prop.NONE),
    )

    mood = Mood(
        palette=palette,
        sceneElement=_coerce_enum(recipe.get("sceneElement"), SceneElement, SceneElement.NONE),
        typography=_coerce_enum(recipe.get("typography"), Typography, Typography.BLOCKBUSTER),
    )

    return CoverSpec(
        version="1.0",
        title=title,
        genre=genre or None,
        mood=mood,
        characters=[character],
        source=None,
    )


def _coerce_enum(value, enum_cls, default):
    if value is None:
        return default
    for member in enum_cls:
        if member.value == value:
            return member
    return default


@runtime_checkable
class TeacherClient(Protocol):
    def get_recipe(self, title: str, description: str, genre: str) -> dict[str, Any]: ...


class CallableTeacherClient:
    def __init__(self, fn: Callable[[str, str, str], dict[str, Any]]):
        self._fn = fn

    def get_recipe(self, title: str, description: str, genre: str) -> dict[str, Any]:
        return self._fn(title, description, genre)


class SubprocessTeacherClient:
    def __init__(self, project_root: Path = COVERFORGE_ROOT):
        self.project_root = project_root

    def get_recipe(self, title: str, description: str, genre: str) -> dict[str, Any]:
        script = (
            "import {generateCoverRecipe} from './src/lib/coverEngine'; "
            "generateCoverRecipe(process.argv[2], process.argv[3], process.argv[4])"
            ".then(r => console.log(JSON.stringify(r)))"
        )
        result = subprocess.run(
            ["pnpm", "dlx", "tsx", "-e", script, title, description, genre],
            cwd=str(self.project_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        full = json.loads(result.stdout)
        return full.get("recipe", full)


def label_description(
    title: str,
    description: str,
    genre: str = "",
    teacher: TeacherClient | None = None,
) -> CoverSpec:
    if teacher is None:
        teacher = SubprocessTeacherClient()
    recipe = teacher.get_recipe(title, description, genre)
    return recipe_to_cover_spec(title, recipe, genre=genre)


__all__ = [
    "CallableTeacherClient",
    "SubprocessTeacherClient",
    "TeacherClient",
    "label_description",
    "recipe_to_cover_spec",
]
