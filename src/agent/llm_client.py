"""
LLM client for connecting the poker agent to Claude.

Provides a simple interface to call Claude API for agent reasoning.
"""

import os
from typing import Optional, Callable

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def create_claude_client(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514"
) -> Callable[[str], str]:
    """
    Create a Claude API client for the agent.

    Args:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model: Model to use

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

    # Get API key from argument or environment
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Either pass api_key argument or "
            "set ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=key)

    def call_claude(prompt: str) -> str:
        """Call Claude with the given prompt."""
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text

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
        'api_key_set': bool(os.environ.get("ANTHROPIC_API_KEY")),
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
