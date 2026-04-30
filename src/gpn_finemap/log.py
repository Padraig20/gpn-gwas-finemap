"""Logging helpers for CLI commands."""

from __future__ import annotations

import logging
import sys


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging for a CLI invocation."""

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
