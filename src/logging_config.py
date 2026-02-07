"""
Centralized logging configuration for the poker bot.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys
from pathlib import Path

from .config import get_log_level, get_log_file

_configured = False


def setup_logging() -> None:
    """Configure logging for the entire application. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    level_str = get_log_level()
    level = getattr(logging, level_str.upper(), logging.INFO)
    log_file = get_log_file()

    root = logging.getLogger("poker_bot")
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Always add stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # Optionally add file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the poker_bot namespace."""
    setup_logging()
    # Map module names: src.core.game_state -> poker_bot.core.game_state
    if name.startswith("src."):
        name = "poker_bot." + name[4:]
    elif not name.startswith("poker_bot."):
        name = "poker_bot." + name
    return logging.getLogger(name)
