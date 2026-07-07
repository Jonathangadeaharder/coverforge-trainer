"""End-to-end cross-language contract test.

Reads a recipe JSON produced by the TS engine (dump-recipe.ts), maps it to a
CoverSpec via the Python teacher, writes the CoverSpec JSON, then asserts the
TS consumer would accept it (the real TS validation runs separately via
apply-cover-spec.ts on the emitted JSON).

Run:
    cd coverforge-trainer
    uv run python e2e_cross_validate.py --recipe recipe.json --out spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coverforge_trainer.cover_spec import parse_cover_spec
from coverforge_trainer.teacher import recipe_to_cover_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="EXTRA ENGLISH")
    parser.add_argument("--genre", default="Comedy / Learning")
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text())
    spec = recipe_to_cover_spec(args.title, recipe, args.genre)

    # Round-trip: serialize -> re-parse to guarantee the JSON we hand to TS
    # is valid against the same pydantic model.
    spec_json = spec.model_dump_json(exclude_none=True, indent=2)
    args.out.write_text(spec_json)
    reparsed = parse_cover_spec(json.loads(spec_json))

    print(f"Wrote CoverSpec: {args.out}")
    print(f"  title={reparsed.title}")
    print(f"  mood.palette={reparsed.mood.palette if reparsed.mood else None}")
    print(f"  characters={len(reparsed.characters)}")
    for i, c in enumerate(reparsed.characters):
        print(f"    char{i + 1}: gender={c.gender} age={c.age} hair={c.hairColor}/{c.hairStyle}")


if __name__ == "__main__":
    main()
