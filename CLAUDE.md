# CLAUDE.md - Project Context for AI Agents

## Project Overview
This is a poker bot that uses Claude LLM as its decision-making engine for 6-max No-Limit Hold'em.
The core value proposition is that the LLM receives **rich contextual information** from multiple
analysis tools and a belief system to make informed poker decisions.

## Architecture: The Tool Pipeline is Sacred

The LLM agent's decision quality depends entirely on the information it receives. The following
tools and data sources MUST be included in every LLM prompt that makes a poker decision:

### Required in Every Decision Prompt
1. **Equity** - Monte Carlo hand equity calculation (`src/tools/equity.py`)
2. **Pot odds** - Mathematical pot odds for calling decisions
3. **GTO recommendation** - Whether to open/3bet/fold per position (`src/tools/gto.py`)
4. **Board texture** - Dry/wet/connected/flush draws analysis (`src/tools/board_texture.py`)
5. **Bet sizing** - Recommended sizing by hand strength and board (`src/tools/bet_sizing.py`)
6. **SL belief state** - Subjective Logic opinions about opponent types (`src/core/beliefs.py`)
7. **PHH hand history** - Actions taken this hand in PHH notation
8. **System prompt** - The SYSTEM_PROMPT from `experiments/tournament.py` explaining PHH/SL notation

### Critical Rules

**NEVER strip down, simplify, or bypass the prompt pipeline.**
- If a subclass overrides `decide()`, it MUST call `_build_prompt()` or include equivalent data
- If a new script creates LLM prompts, it MUST include all tool outputs listed above
- `max_tokens` for decision prompts must be >= 100 (120 for fast play, 200+ for traced/debug)
- Exceptions in LLM calls must ALWAYS be logged to stderr, never silently swallowed

**NEVER silently degrade behavior.**
- If falling back to a heuristic (mock LLM, forced decision, exception handler), log a warning
- If a tool call fails, log it - don't silently skip the data
- If using a mock/test client instead of real LLM, print a warning

## Key Files

| File | Purpose |
|------|---------|
| `experiments/tournament.py` | `SimpleLLMPlayer` - the core LLM player. `_build_prompt()` constructs the full prompt with all tools. |
| `run_llm_traced.py` | `TracedLLMPlayer` - extends SimpleLLMPlayer with full trace printing. Must call parent's `_build_prompt()`. |
| `src/agent/react.py` | `ReActAgent` - multi-step reasoning agent (slower, uses tool calls). |
| `src/tools/gto.py` | GTO advisor - preflop ranges, 3bet/4bet ranges, cbet recommendations |
| `src/tools/board_texture.py` | Board texture analysis - wetness, draws, connectedness |
| `src/tools/bet_sizing.py` | Bet sizing heuristics by hand strength, board, and SPR |
| `src/tools/equity.py` | Monte Carlo equity calculator |
| `src/core/beliefs.py` | Subjective Logic belief system for opponent modeling |

## Testing
```bash
pytest tests/ -v          # All unit tests (no API key needed)
python run_llm_traced.py  # Live LLM game with full trace (needs ANTHROPIC_API_KEY)
```

## Common Mistakes to Avoid
- Creating a new LLM player class that builds its own minimal prompt instead of using `_build_prompt()`
- Setting `max_tokens` too low (< 100) which truncates the LLM's ability to reason
- Catching exceptions with bare `except:` without logging
- Adding fallback behavior without logging that the fallback was triggered
- Removing tool outputs "for simplicity" or "to speed things up"
