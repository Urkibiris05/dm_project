"""Smoke tests for the E4 experiment package."""

import importlib


def test_e4_public_modules_are_available() -> None:
    """The E4 package exposes its public modules as a stable entry point."""
    for module_name in ("E4.config", "E4.exp_minkowski", "E4.run_e4"):
        importlib.import_module(module_name)
