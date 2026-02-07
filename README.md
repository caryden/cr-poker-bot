# CR Poker Bot

LLM-based poker agent for 6-max No-Limit Hold'em. Combines Claude API reasoning with a ReAct agent loop, Subjective Logic belief tracking, and poker-specific analysis tools.

## Architecture

```
src/
  agent/         ReAct agent, LLM client, TRT self-verification
  core/          Primitives, game state, subjective logic, beliefs, memory
  evaluation/    Tracing, analysis, hand narratives
  formats/       PHH (Poker Hand History) export
  game/          Hand/session runner, rule-based opponents
  prompts/       LLM prompt templates
  tools/         Equity calc, pot odds, GTO advisor, board texture, bet sizing
```

The agent follows an **OBSERVE -> REASON -> VERIFY -> DECIDE** loop. At each decision point it can call poker tools (equity calculator, pot odds, GTO advisor, board texture analyzer, bet sizing) before committing to an action. Beliefs about opponents are maintained using Subjective Logic opinions `(belief, disbelief, uncertainty, base_rate)` and updated via observation-consistency fusion.

## Setup

### Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (for LLM-powered play)

### Install

```bash
# Clone the repo
git clone https://github.com/caryden/cr-poker-bot.git
cd cr-poker-bot

# Install with pip (includes python-dotenv)
pip install -e .

# Install dev dependencies (pytest)
pip install -e ".[dev]"

# Install LLM support (anthropic SDK)
pip install -e ".[llm]"
```

### Environment Variables

Copy the example and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required for LLM-powered play
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Optional configuration
POKER_BOT_MODEL=claude-sonnet-4-5    # claude-sonnet-4-5 | claude-opus-4-6 | claude-haiku-4-5
POKER_BOT_LOG_LEVEL=INFO             # DEBUG | INFO | WARNING | ERROR
POKER_BOT_LOG_FILE=poker_bot.log     # omit for stderr-only logging
POKER_BOT_TRACE_DIR=traces           # directory for hand narrative files
POKER_BOT_DB_PATH=poker_bot.db       # SQLite path for persistent memory
```

The `.env` file is loaded automatically via `python-dotenv` when any module is imported. You can also set these as regular environment variables.

## Usage

### Run Tests (no API key needed)

```bash
pytest tests/ -v
```

All tests use mock LLM clients and don't require an API key.

### Quick Evaluation (mock LLM)

```python
from src.evaluation import quick_eval

result = quick_eval(num_hands=100)
print(f"Win rate: {result.metrics.bb_per_100:.1f} BB/100")
```

### Full Session with LLM Agent

```python
from src.agent.llm_client import create_claude_client
from src.agent.react import ReActAgent
from src.game.runner import HandRunner, SessionRunner, SimplePokerEngine, AgentPlayer
from src.game.opponents import create_opponent
from src.evaluation.hand_narrative import NarrativeTracer

# Create LLM-powered agent
llm = create_claude_client()  # reads ANTHROPIC_API_KEY from .env
agent = ReActAgent(llm_call=llm)
hero = AgentPlayer(agent, player_id="hero")

# Set up opponents
villains = {
    f"villain_{i}": create_opponent(t, f"villain_{i}")
    for i, t in enumerate(["tag", "lag", "fish", "nit", "random"])
}

# Create engine and runner with narrative tracing
engine = SimplePokerEngine(big_blind=10.0, starting_stack=1000.0)
tracer = NarrativeTracer(output_dir="traces")
runner = HandRunner(engine, hero, villains, narrative_tracer=tracer)

# Run a session
session = SessionRunner(runner, num_hands=50)
stats = session.run()
print(stats.summary())

# Save interspersed narrative
tracer.save_session_summary("my_session")
```

### Hand Narrative Output

The narrative tracer produces human-readable files that interleave game actions with the hero's LLM reasoning:

```
============================================================
Hand abc123 | Hero: BTN with AhKd | 2026-02-06 14:30:00
============================================================

-- PREFLOP (pot 15) --
    UTG  folds
    HJ   folds
    CO   raises 25

>>> HERO TURN (BTN) | pot 40 | to_call 25 | stack 990 <<<
    [   Think] AKo is a premium hand on the BTN facing a CO open...
    [    Tool] equity_calc -> 62.3% equity vs CO range
    [    Tool] pot_odds -> need 38.5% equity, we have 62.3%
    [   Think] Clear 3-bet for value...
    [Decision] RAISE to 75 (confidence: 0.85, verified: True)

    SB   folds
    BB   folds
    CO   calls 75

-- FLOP: Ah 7c 2d (pot 165) --
    CO   checks

>>> HERO TURN (BTN) | pot 165 | to_call 0 | stack 925 | board Ah 7c 2d <<<
    [   Think] Top pair top kicker on a dry board...
    [    Tool] board_texture -> dry, static, favors aggressor
    [Decision] BET 110 (confidence: 0.90, verified: True)

=== Result: WON +18.5 BB (showdown) ===
```

### Check API Status

```python
from src.agent.llm_client import check_api_available

status = check_api_available()
print(status['message'])  # "Claude API ready" or error details
```

## Logging

Structured logging is available throughout the codebase. Control verbosity via the `POKER_BOT_LOG_LEVEL` environment variable:

| Level | What you see |
|-------|-------------|
| `ERROR` | Only errors |
| `WARNING` | Warnings + forced decisions |
| `INFO` | Hand summaries, street transitions, agent decisions |
| `DEBUG` | Full LLM prompts/responses, individual tool calls, belief updates |

```bash
# See everything (useful for debugging agent behavior)
POKER_BOT_LOG_LEVEL=DEBUG pytest tests/ -v

# Or set in .env
echo "POKER_BOT_LOG_LEVEL=DEBUG" >> .env
```

## Opponent Types

| Type | VPIP | Style | Use case |
|------|------|-------|----------|
| `random` | 100% | Random baseline | Sanity testing |
| `fish` / `calling_station` | 60-80% | Calls everything | Exploit passive players |
| `nit` / `tight_passive` | 10-15% | Only premiums | Test vs tight ranges |
| `lag` / `loose_aggressive` | 35-45% | Bets/raises often | Test vs aggression |
| `tag` / `tight_aggressive` | 20-25% | Solid strategy | Realistic opponent |

## Project Structure

| Module | Lines | Description |
|--------|-------|-------------|
| `src/core/` | ~3,000 | Game primitives, state, beliefs, memory, persistence |
| `src/agent/` | ~1,500 | ReAct agent, LLM client, TRT verification |
| `src/tools/` | ~2,500 | Equity, pot odds, GTO, board texture, bet sizing |
| `src/game/` | ~1,200 | Hand runner, session runner, opponents |
| `src/evaluation/` | ~1,200 | Tracing, analysis, hand narratives |
| `src/formats/` | ~300 | PHH hand history format |
| `src/prompts/` | ~300 | LLM prompt templates |

## License

MIT
