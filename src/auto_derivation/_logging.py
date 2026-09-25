"""Logging configuration for auto_derivation.

A single `setup_logging(verbose: int)` entrypoint, called from `cli.py`'s
top-level callback. The CLI exposes `-v` / `-vv`:

    -v   → INFO   (per-generation GP progress, LLM call boundaries)
    -vv  → DEBUG  (LLM retry attempts, panel I/O details)
    none → WARNING (only the things callers should actually notice)

Output goes to stderr so that `derive search ... > pareto.txt` keeps user-
facing summaries clean — logs are operational, the stdout payload is data.

`setup_logging` is idempotent: re-calling it replaces the handler instead of
stacking duplicates (Typer can invoke the callback once per process, but
tests sometimes drive the CLI multiple times).
"""
from __future__ import annotations

import logging
import sys

_PACKAGE_LOGGER_NAME = "auto_derivation"
_HANDLER_NAME = "auto_derivation.stderr"


def setup_logging(verbose: int = 0) -> None:
    """Configure the `auto_derivation` package logger.

    `verbose` is a count: 0 → WARNING, 1 → INFO, ≥2 → DEBUG.
    """
    if verbose <= 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    # Don't bubble up to the root logger — pytest's caplog grabs records
    # directly off the named logger, and we don't want our handler to also
    # print them.
    logger.propagate = False

    # Idempotency: drop any handler we previously installed.
    for h in list(logger.handlers):
        if getattr(h, "name", None) == _HANDLER_NAME:
            logger.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.name = _HANDLER_NAME
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
