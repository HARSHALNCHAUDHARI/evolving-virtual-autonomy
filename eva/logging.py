"""EVA structured logging.

Provides a consistent, centralized logging configuration.
All EVA components should obtain loggers through get_logger().
"""

from __future__ import annotations

import logging
import sys
from typing import ClassVar


_LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(level: str = "INFO", *, debug: bool = False) -> None:
    """Initialize the EVA logging subsystem.

    Should be called once during startup. Subsequent calls are no-ops
    to prevent duplicate handlers.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        debug: If True, forces DEBUG level regardless of `level`.
    """
    global _initialized
    if _initialized:
        return

    effective_level = "DEBUG" if debug else level.upper()

    root = logging.getLogger("eva")
    root.setLevel(effective_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(effective_level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(handler)
    root.propagate = False

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under the eva namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured Logger instance.
    """
    if name.startswith("eva."):
        return logging.getLogger(name)
    return logging.getLogger(f"eva.{name}")


def reset_logging() -> None:
    """Reset logging state. Intended for testing only."""
    global _initialized
    root = logging.getLogger("eva")
    root.handlers.clear()
    _initialized = False
