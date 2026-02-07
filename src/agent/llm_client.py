"""
LLM client for connecting the poker agent to Claude.

Provides a simple interface to call Claude API for agent reasoning.
"""

import os
from typing import Optional, Callable

from ..config import get_api_key, get_model
from ..logging_config import get_logger

logger = get_logger(__name__)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def create_claude_client(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5"
) -> Callable[[str], str]:
    """
    Create a Claude API client function.

    Args:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model: Model to use (default: claude-sonnet-4-5)

    Returns:
        Function that takes prompt and returns response
    """
    if not HAS_ANTHROPIC:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )

    key = api_key or get_api_key()
    if not key:
        raise ValueError(
            "No API key provided. Either pass api_key argument, "
            "set ANTHROPIC_API_KEY in .env file, or "
            "set ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=key)
    logger.info("Claude client created with model=%s", model)

    def call_claude(prompt: str) -> str:
        """Call Claude with the given prompt."""
        logger.debug("LLM request (%d chars)", len(prompt))
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response = message.content[0].text
        logger.debug("LLM response (%d chars)", len(response))
        return response

    return call_claude


def check_api_available() -> dict:
    """
    Check if Claude API is available.

    Returns:
        Dict with status info
    """
    result = {
        'anthropic_installed': HAS_ANTHROPIC,
        'api_key_set': bool(get_api_key()),
        'ready': False,
        'message': ''
    }

    if not HAS_ANTHROPIC:
        result['message'] = "Install anthropic: pip install anthropic"
        return result

    if not result['api_key_set']:
        result['message'] = "Set ANTHROPIC_API_KEY environment variable"
        return result

    try:
        client = anthropic.Anthropic()
        result['ready'] = True
        result['message'] = "Claude API ready"
    except Exception as e:
        result['message'] = f"API error: {e}"

    return result
