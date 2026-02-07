"""
Configuration and environment variable loading.

Loads .env file from project root and provides access to configuration values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


# Load .env on import
_project_root = _find_project_root()
load_dotenv(_project_root / ".env")


def get_api_key() -> str | None:
    """Get the Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


def get_model(default: str = "claude-sonnet-4-5") -> str:
    """Get the model name from environment or default."""
    return os.environ.get("POKER_BOT_MODEL", default)


def get_log_level(default: str = "INFO") -> str:
    """Get log level from environment or default."""
    return os.environ.get("POKER_BOT_LOG_LEVEL", default)


def get_log_file() -> str | None:
    """Get log file path from environment, if set."""
    return os.environ.get("POKER_BOT_LOG_FILE")


def get_trace_dir(default: str = "traces") -> str:
    """Get trace output directory from environment or default."""
    return os.environ.get("POKER_BOT_TRACE_DIR", default)


def get_db_path(default: str = "poker_bot.db") -> str:
    """Get database path from environment or default."""
    return os.environ.get("POKER_BOT_DB_PATH", default)
