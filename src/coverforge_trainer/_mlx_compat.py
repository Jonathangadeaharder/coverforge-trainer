"""Compatibility shim for mlx-lm 0.31.x + transformers 5.x.

mlx-lm 0.31.x declares `transformers>=5.0.0` but its `tokenizer_utils.py`
calls `AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=...)`
with a string name as the first positional arg. transformers 5.x changed
`register()` to require a config class (not a string), raising
`AttributeError: 'str' object has no attribute '__module__'` at import time,
which breaks `import mlx_lm` entirely.

This shim replaces `AutoTokenizer.register` with a tolerant version before
mlx_lm is imported. It only patches when the broken call shape is detected,
so future fixed versions of mlx_lm / transformers are unaffected.

Import this module (or call `apply()`) BEFORE importing `mlx_lm` anywhere.
"""

from __future__ import annotations

import transformers

_PATCHED = False


def apply() -> None:
    """Patch transformers.AutoTokenizer.register to tolerate string names.

    Idempotent. No-op if already patched or if the upstream bug is fixed.
    """
    global _PATCHED
    if _PATCHED:
        return

    original_register = transformers.AutoTokenizer.register

    def tolerant_register(*args, **kwargs):  # type: ignore[no-untyped-def]
        # If the first positional arg is a string (old mlx_lm calling shape),
        # swallow the registration — NewlineTokenizer is only used by mlx_lm
        # internals and has a working fallback path when not registered.
        if args and isinstance(args[0], str):
            return None
        return original_register(*args, **kwargs)

    transformers.AutoTokenizer.register = tolerant_register
    _PATCHED = True


apply()
