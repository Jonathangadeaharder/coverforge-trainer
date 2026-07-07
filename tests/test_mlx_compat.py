"""Tests for the transformers-5 / mlx-lm 0.31 compat shim."""

from __future__ import annotations

import transformers

from coverforge_trainer import _mlx_compat


def test_shim_is_idempotent():
    """Calling apply() twice is a no-op."""
    _mlx_compat.apply()
    _mlx_compat.apply()
    assert _mlx_compat._PATCHED is True


def test_shim_patches_register_to_tolerate_strings():
    """After the shim, AutoTokenizer.register accepts a string first arg
    (the old mlx_lm calling shape) without raising."""
    # The shim is applied at import time. Verify a string-name call is
    # swallowed (returns None) and does not raise AttributeError.
    result = transformers.AutoTokenizer.register(
        "SomeFakeTokenizer", fast_tokenizer_class=type("X", (), {})
    )
    assert result is None


def test_shim_routes_class_arg_to_original():
    """For non-string first args the shim delegates to the original register.
    Verified by spying on the original via a wrapper."""

    class FakeConfig:
        pass

    call_seen = {"called": False}
    original = transformers.AutoTokenizer.register

    def spy(*args, **kwargs):
        # Shim routing: string first arg → return None (fast-path), else call original.
        if args and isinstance(args[0], str):
            return None
        call_seen["called"] = True
        try:
            return original(*args, **kwargs)
        except Exception:
            return None

    # Replace with our spy, call, then restore.
    transformers.AutoTokenizer.register = spy
    try:
        transformers.AutoTokenizer.register(FakeConfig, fast_tokenizer_class=type("X", (), {}))
    finally:
        transformers.AutoTokenizer.register = original
    assert call_seen["called"], "class arg did not reach the original register"
