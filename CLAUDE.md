# CLAUDE.md - Project Context for AI Agents

## Project Overview
Poker bot using Claude LLM for 6-max No-Limit Hold'em decisions.
The LLM receives rich context from analysis tools and a belief system.

## The One Agent: SimpleLLMPlayer

`SimpleLLMPlayer` in `experiments/tournament.py` is the **only** LLM agent.
`_build_prompt()` is the single source of truth for prompt construction.

Trace mode (`trace=True`) prints the full system prompt, user prompt,
and LLM response for debugging. It does NOT change behavior -- same
prompt, same max_tokens, same system prompt. Read-only visibility.

## Required Tool Outputs in Every Decision Prompt

1. **Equity** -- Monte Carlo calculation (`src/tools/equity.py`)
2. **Pot odds** -- calculated from game state
3. **GTO recommendation** -- open/3bet ranges (`src/tools/gto.py`)
4. **Board texture** -- dry/wet/draws (`src/tools/board_texture.py`)
5. **Bet sizing** -- by hand strength + board (`src/tools/bet_sizing.py`)
6. **SL beliefs** -- opponent type opinions (`src/core/beliefs.py`)
7. **PHH hand history** -- actions this hand
8. **System prompt** -- `SYSTEM_PROMPT` in `experiments/tournament.py`

## Critical Rules

**NEVER strip down, simplify, or bypass the prompt pipeline.**
- There are NO subclasses -- `SimpleLLMPlayer` is the only LLM agent
- New scripts must include all tool outputs listed above
- Do not artificially constrain `max_tokens` -- the LLM returns reasoning + action
- Exceptions in LLM calls must log to stderr, never silently swallowed

**NEVER silently degrade behavior.**
- Log warnings when falling back to heuristics
- Log when tool calls fail
- Log when using mock/test clients

## Strategy Bots (canonical names)

| Name     | Class              | Style                       |
|----------|--------------------|-----------------------------|
| `FISH`   | `CallingStation`   | Calls everything            |
| `NIT`    | `TightPassive`     | Only premiums               |
| `LAG`    | `LooseAggressive`  | Plays wide, bets often      |
| `TAG`    | `TagBot`           | Solid positional poker      |
| `RANDOM` | `RandomPlayer`     | Uniform random baseline     |

All defined in `src/game/opponents.py`.

## Ablation Support

`SimpleLLMPlayer` accepts `excluded_tools` parameter:
```python
player = SimpleLLMPlayer('hero', excluded_tools={'gto', 'beliefs'})
```
Valid: `equity`, `gto`, `board_texture`, `bet_sizing`, `beliefs`.

## Key Files

| File | Purpose |
|------|---------|
| `experiments/tournament.py` | `SimpleLLMPlayer`, `TournamentRunner`, `run_llm_tournament()` |
| `run_llm_traced.py` | Traced mode runner -- uses `SimpleLLMPlayer(trace=True)` |
| `src/tools/gto.py` | GTO advisor -- preflop ranges, 3bet, cbet |
| `src/tools/board_texture.py` | Board texture -- wetness, draws, connectedness |
| `src/tools/bet_sizing.py` | Bet sizing by hand strength, board, SPR |
| `src/tools/equity.py` | Monte Carlo equity calculator |
| `src/core/beliefs.py` | Subjective Logic belief system |
| `src/game/opponents.py` | Strategy bots (FISH, NIT, LAG, TAG, RANDOM) |

## Testing
```bash
pytest tests/ -v                                # Unit tests (no API key)
python run_llm_traced.py                        # Live LLM game with trace
python run_llm_traced.py --exclude gto          # Ablation run
```
