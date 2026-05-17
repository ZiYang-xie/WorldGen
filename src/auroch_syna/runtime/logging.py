"""Structured logging helpers.

Use ``get_logger(__name__)`` everywhere instead of ``print``. Callers
(daemon, CLI, viewer, embedders) configure verbosity once via
``configure(level=...)``.
"""
from __future__ import annotations

import logging
import os
import sys


_DEFAULT_FORMAT = "%(asctime)s %(levelname)-5s %(name)s — %(message)s"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure(level: str | int | None = None, *, stream=sys.stderr) -> None:
    """Initialize root logging for the auroch_syna namespace.

    Idempotent: subsequent calls update the level only.
    Honors ``AUROCH_SYNA_LOG_LEVEL`` if ``level`` is None.
    """
    global _CONFIGURED

    if level is None:
        level = os.environ.get("AUROCH_SYNA_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger("auroch_syna")
    root.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
