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
    Create a Claude API client for the agent.

    Args:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model: Model to use (default: claude-sonnet-4-5)
               Latest options: claude-sonnet-4-5, claude-opus-4-6, claude-haiku-4-5

    Returns:
        Function that takes prompt and returns response

    Example:
        llm_call = create_claude_client()
        agent = ReActAgent(llm_call=llm_call)
    """
    if not HAS_ANTHROPIC:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )

    # Get API key: argument > .env file > environment variable
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


def create_mock_client() -> Callable[[str], str]:
    """
    Create a mock LLM client for testing without API.

    Uses simple heuristics - not for real evaluation.
    """
    def mock_call(prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Simulate tool usage
        if "pot_odds" not in prompt_lower:
            return "Let me check pot odds.\n\nTOOL: pot_odds()"
        if "equity" not in prompt_lower:
            return "Calculating equity.\n\nTOOL: equity_calc()"
        if "gto" not in prompt_lower:
            return "Checking GTO.\n\nTOOL: gto_advisor(situation='open')"

        # Simple decision heuristics
        if "to_call: 0" in prompt_lower or "to call: 0" in prompt_lower:
            return "No bet to call.\n\nDECISION: check"

        if "strong" in prompt_lower and "hand" in prompt_lower:
            return "Strong hand, betting for value.\n\nDECISION: bet 0.67"

        return "Calling based on pot odds.\n\nDECISION: call"

    return mock_call


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

    # Try a minimal API call
    try:
        client = anthropic.Anthropic()
        # Just verify client creation works
        result['ready'] = True
        result['message'] = "Claude API ready"
    except Exception as e:
        result['message'] = f"API error: {e}"

    return result
