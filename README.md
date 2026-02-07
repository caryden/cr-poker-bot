# CR Poker Bot

LLM-based poker agent for 6-max No-Limit Hold'em. Claude makes decisions using equity calculations, GTO ranges, board texture analysis, bet sizing heuristics, and Subjective Logic opponent beliefs.

## Architecture

```
src/
  core/          Primitives, game state, subjective logic beliefs, memory
  tools/         Equity calc, pot odds, GTO advisor, board texture, bet sizing
  game/          Hand/session runner, rule-based strategy bots
  evaluation/    Tracing, analysis, hand narratives
  formats/       PHH (Poker Hand History) export
  prompts/       LLM prompt templates
  agent/         LLM client, tool interfaces
experiments/
  tournament.py  SimpleLLMPlayer (the hero), TournamentRunner
```

### Hero Agent: SimpleLLMPlayer

The single LLM agent (`SimpleLLMPlayer` in `experiments/tournament.py`) receives a prompt containing:

1. **Equity** -- Monte Carlo simulation (300 runs)
2. **Pot odds** -- mathematical calling threshold
3. **GTO recommendation** -- preflop open/3bet ranges
4. **Board texture** -- wet/dry, draws, connectedness
5. **Bet sizing** -- recommended size by hand strength + board
6. **SL beliefs** -- Subjective Logic opinions on opponent types
7. **PHH hand history** -- actions this hand in PHH notation

### Strategy Bots

| Name     | Class              | Style                       |
|----------|--------------------|-----------------------------|
| `FISH`   | `CallingStation`   | Calls everything (60-80% VPIP) |
| `NIT`    | `TightPassive`     | Only premiums (10-15% VPIP) |
| `LAG`    | `LooseAggressive`  | Plays wide, bets often      |
| `TAG`    | `TagBot`           | Solid positional poker      |
| `RANDOM` | `RandomPlayer`     | Uniform random (baseline)   |

## Setup

```bash
pip install -e .
pip install -e ".[dev]"       # pytest
pip install -e ".[llm]"       # anthropic SDK
cp .env.example .env          # add ANTHROPIC_API_KEY
```

## Usage

### Run Tests (no API key)

```bash
pytest tests/ -v
```

### Run LLM Hero (traced)

```bash
python run_llm_traced.py                        # 10 hands, full trace
python run_llm_traced.py --hands 30             # more hands
python run_llm_traced.py --exclude gto          # ablation: no GTO
python run_llm_traced.py --exclude beliefs gto  # ablation: no beliefs or GTO
```

### Run Tournament

```bash
python run_llm_vs_strategy.py     # LLM vs 5 random strategy bots, 200 hands
python run_llm_vs_random.py       # LLM vs 5 random players
```

### Run Ablation Experiments

```python
from experiments.tournament import run_llm_tournament

# Baseline (all tools)
baseline = run_llm_tournament(max_hands=200)

# Ablation: remove one tool at a time
no_gto     = run_llm_tournament(excluded_tools={'gto'}, max_hands=200)
no_beliefs = run_llm_tournament(excluded_tools={'beliefs'}, max_hands=200)
no_texture = run_llm_tournament(excluded_tools={'board_texture'}, max_hands=200)
no_sizing  = run_llm_tournament(excluded_tools={'bet_sizing'}, max_hands=200)
no_equity  = run_llm_tournament(excluded_tools={'equity'}, max_hands=200)
```

Valid ablation targets: `equity`, `gto`, `board_texture`, `bet_sizing`, `beliefs`.

## License

MIT
