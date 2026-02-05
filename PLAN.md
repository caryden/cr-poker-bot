# World-Class Poker Playing Agent: Implementation Plan

## Executive Summary

Building a world-class poker AI by combining an **LLM reasoning model** with **poker-specific tools** in a **ReAct-style agentic loop**. This hybrid approach leverages:
- LLM strategic reasoning and opponent modeling capabilities
- Pre-computed GTO strategies as consultable tools
- Memory/belief state for tracking opponent patterns
- Hand evaluation and equity calculation tools

Based on research into state-of-the-art agents (Libratus, Pluribus, DeepStack, ReBeL) and the latest LLM advances (PokerBench 2025 showing fine-tuned LLMs achieving 78% solver alignment, SpinGPT), this plan targets **Heads-Up No-Limit Texas Hold'em** with a GTO baseline + exploitation capabilities.

## Target Configuration
- **Variant**: Heads-Up No-Limit Texas Hold'em (2 players)
- **Architecture**: LLM reasoning agent with ReAct loop (API-based)
- **Strategy**: GTO baseline with adaptive opponent exploitation
- **Framework**: Leverage RLCard for game engine, custom agent layer

---

## Research Summary

### Key Breakthroughs in Poker AI

| Agent | Year | Innovation | Performance |
|-------|------|------------|-------------|
| **Libratus** | 2017 | CFR+ blueprint + nested subgame solving | Beat top pros heads-up |
| **DeepStack** | 2017 | Deep learning + continual re-solving | Expert-level HUNL |
| **Pluribus** | 2019 | Depth-limited search + 6-player support | Beat 6 pros simultaneously |
| **ReBeL** | 2020 | RL + search via public belief states | Nash equilibrium convergence |
| **AlphaHoldem** | 2022 | End-to-end deep RL, less compute | Beats DeepStack, Slumbot |
| **PokerBench/SpinGPT** | 2025 | LLM fine-tuning for poker | 78% solver alignment |

### Core Algorithms

1. **Counterfactual Regret Minimization (CFR)** - Foundation of all modern poker AI
   - Iteratively minimizes regret to converge on Nash equilibrium
   - Variants: CFR+, Linear CFR, MCCFR (Monte Carlo), Deep CFR

2. **Blueprint Strategy** - Pre-computed offline strategy
   - Created via millions of CFR iterations
   - Handles early game decisions cheaply

3. **Real-Time Subgame Solving** - Online refinement
   - Depth-limited search at decision points
   - Multiple continuation strategies (Pluribus innovation)

4. **Neural Network Approximation** - Scalability
   - Replace tabular CFR with neural nets (Deep CFR)
   - Enables handling of large game trees

### Critical Technical Components

1. **Game State Representation**
   - Card encoding (one-hot or learned embeddings)
   - Position, stack sizes, pot, betting history
   - Community cards and hand strength

2. **Abstraction Techniques**
   - **Hand abstraction**: Bucket similar hands (k-means clustering, Earth Mover's Distance)
   - **Action abstraction**: Limit bet sizes to discrete set
   - Reduces 10^161 game states to manageable size

3. **Opponent Modeling** (for exploitation)
   - Pattern recognition from betting history
   - Transformer-based action prediction
   - Balance between GTO (unexploitable) and exploitative play

---

## Recommended Architecture: LLM ReAct Agent

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM POKER AGENT SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     AGENT LOOP (ReAct Style)                       │ │
│  │  ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐  │ │
│  │  │ Observe │───►│   Reason    │───►│   Act    │───►│  Update  │  │ │
│  │  │  State  │    │   (LLM)     │    │          │    │  Memory  │  │ │
│  │  └─────────┘    └─────────────┘    └──────────┘    └──────────┘  │ │
│  │       ▲                │                │                │        │ │
│  │       └────────────────┴────────────────┴────────────────┘        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                   │                                     │
│                      ┌────────────┴────────────┐                       │
│                      ▼                         ▼                        │
│  ┌────────────────────────────┐  ┌────────────────────────────────┐   │
│  │      REASONING MODEL       │  │         MEMORY STORE           │   │
│  │    (Claude/GPT via API)    │  │                                │   │
│  │                            │  │  • Opponent betting patterns   │   │
│  │  • Strategic analysis      │  │  • VPIP, PFR, aggression stats │   │
│  │  • Opponent modeling       │  │  • Hand history summaries      │   │
│  │  • Range construction      │  │  • Exploitable tendencies      │   │
│  │  • Bet sizing reasoning    │  │  • Session win/loss tracking   │   │
│  └────────────────────────────┘  └────────────────────────────────┘   │
│                │                                                        │
│                │ Tool Calls                                             │
│                ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         TOOL LAYER                                │  │
│  │                                                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│  │
│  │  │   Hand      │  │   Equity    │  │    GTO      │  │ Opponent ││  │
│  │  │  Evaluator  │  │ Calculator  │  │  Advisor    │  │  Stats   ││  │
│  │  │             │  │             │  │             │  │          ││  │
│  │  │ "What hand  │  │ "Win % vs   │  │ "Optimal    │  │ "VPIP:   ││  │
│  │  │  do I have?"│  │  range X?"  │  │  action?"   │  │  45%..." ││  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘│  │
│  │                                                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │    Pot      │  │   Range     │  │  Position   │               │  │
│  │  │    Odds     │  │  Analysis   │  │   Context   │               │  │
│  │  │             │  │             │  │             │               │  │
│  │  │ "Need 25%   │  │ "Villain    │  │ "BTN vs BB, │               │  │
│  │  │  equity"    │  │  likely has"│  │  100bb deep"│               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      GAME ENGINE (RLCard)                         │  │
│  │           Texas Hold'em rules, deck, betting, hand eval           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### ReAct Loop Flow

```
1. OBSERVE: Receive game state (cards, pot, stack, actions)
           │
           ▼
2. THINK:  LLM analyzes situation using tools:
           - Check hand strength → "I have top pair"
           - Calculate equity → "62% vs villain's range"
           - Query GTO → "Optimal is bet 75% pot"
           - Check opponent stats → "Villain folds 70% to c-bets"
           │
           ▼
3. REASON: LLM synthesizes: "Given villain's high fold rate,
           I should bet larger to maximize fold equity"
           │
           ▼
4. ACT:    Execute decision (fold/call/raise amount)
           │
           ▼
5. UPDATE: Store outcome in memory for future exploitation
           │
           └──────► Loop back to OBSERVE
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Working poker environment and tool infrastructure

**Tasks**:
1. Set up Python project structure
   ```
   cr-poker-bot/
   ├── src/
   │   ├── agent/           # LLM agent core
   │   ├── tools/           # Poker tools for LLM
   │   ├── memory/          # Belief state storage
   │   ├── engine/          # Game interface
   │   └── utils/           # Helpers
   ├── tests/
   ├── configs/
   └── requirements.txt
   ```

2. Integrate RLCard for game engine
   - Heads-Up No-Limit Texas Hold'em environment
   - State observation handling
   - Action execution

3. Implement fast hand evaluation
   - Integrate `treys` library
   - Hand ranking and comparison

4. Create baseline agents for testing
   - Random agent
   - Rule-based (tight-aggressive) agent

**Deliverable**: Can run poker games programmatically, evaluate hands

---

### Phase 2: Tool Layer (Week 2)
**Goal**: Build poker-specific tools the LLM can invoke

**Tools to implement**:

```python
# 1. Hand Evaluator Tool
def evaluate_hand(hole_cards: list, board: list) -> dict:
    """Returns hand rank, name, and relative strength"""
    return {
        "rank": 1234,
        "hand_name": "Two Pair, Aces and Kings",
        "percentile": 0.85  # Top 15% of hands
    }

# 2. Equity Calculator Tool
def calculate_equity(hole_cards: list, board: list,
                     villain_range: str, num_simulations: int) -> dict:
    """Monte Carlo equity vs opponent range"""
    return {
        "equity": 0.62,
        "win": 0.58,
        "tie": 0.08,
        "lose": 0.34
    }

# 3. Pot Odds Calculator Tool
def calculate_pot_odds(pot: int, to_call: int) -> dict:
    """Calculate pot odds and required equity"""
    return {
        "pot_odds": 0.25,
        "required_equity": 0.20,
        "implied_odds_note": "Need 20% equity to call"
    }

# 4. GTO Advisor Tool (Pre-computed ranges/strategies)
def get_gto_advice(situation: str) -> dict:
    """Lookup GTO strategy for common situations"""
    return {
        "recommended_action": "raise",
        "bet_sizing": "75% pot",
        "frequency": {"raise": 0.6, "call": 0.3, "fold": 0.1},
        "reasoning": "Standard c-bet on dry board"
    }

# 5. Opponent Stats Tool
def get_opponent_stats(opponent_id: str) -> dict:
    """Retrieve tracked statistics for opponent"""
    return {
        "hands_played": 150,
        "vpip": 0.28,           # Voluntarily Put $ In Pot
        "pfr": 0.22,            # Pre-Flop Raise %
        "aggression_factor": 2.1,
        "fold_to_cbet": 0.65,
        "wtsd": 0.24,           # Went To ShowDown
        "tendencies": "Tight-aggressive, folds too much to aggression"
    }

# 6. Position Context Tool
def get_position_context(hero_position: str, villain_position: str,
                         stack_sizes: dict) -> dict:
    """Provide positional and stack depth context"""
    return {
        "position_advantage": "Hero has position",
        "effective_stack": 100,  # in big blinds
        "stack_depth": "deep",
        "spr": 8.5              # Stack-to-Pot Ratio
    }
```

**Deliverable**: All tools callable and tested independently

---

### Phase 3: LLM Agent Core (Week 3)
**Goal**: ReAct-style reasoning loop with LLM

**Tasks**:

1. **Design prompt template**
```
You are an expert poker player making decisions in Heads-Up No-Limit
Texas Hold'em. You have access to the following tools:

- evaluate_hand: Get hand strength and ranking
- calculate_equity: Monte Carlo equity vs opponent range
- calculate_pot_odds: Determine calling requirements
- get_gto_advice: Lookup game-theory optimal plays
- get_opponent_stats: Check opponent tendencies
- get_position_context: Understand positional dynamics

Current Game State:
{game_state}

Your Cards: {hole_cards}
Board: {board}
Pot: {pot} | To Call: {to_call}
Your Stack: {hero_stack} | Villain Stack: {villain_stack}

Opponent Memory Summary:
{opponent_summary}

Use ReAct format:
Thought: [Your analysis]
Action: [tool_name(args)] or DECISION: [fold/call/raise X]
Observation: [tool result]
... (repeat as needed)
Final Decision: [your action with reasoning]
```

2. **Implement agent loop**
```python
class PokerAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools = PokerToolkit()
        self.memory = OpponentMemory()

    async def decide(self, game_state: GameState) -> Action:
        """ReAct loop for decision making"""
        prompt = self.build_prompt(game_state)

        while True:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=self.tools.get_tool_definitions()
            )

            if response.stop_reason == "tool_use":
                # Execute tool and continue
                tool_result = self.execute_tool(response.tool_use)
                prompt = self.add_observation(prompt, tool_result)
            else:
                # Parse final decision
                return self.parse_action(response.content)
```

3. **API integration** (Anthropic Claude or OpenAI)
   - Async calls for performance
   - Retry logic and error handling
   - Token usage tracking

**Deliverable**: Agent can play hands using LLM reasoning with tool calls

---

### Phase 4: Memory & Belief System (Week 4)
**Goal**: Persistent opponent modeling and pattern tracking

**Memory Components**:

```python
class OpponentMemory:
    """Tracks opponent patterns across hands"""

    def __init__(self, opponent_id: str):
        self.opponent_id = opponent_id
        self.hands_played = 0

        # Aggregate statistics
        self.stats = {
            "vpip": RunningAverage(),
            "pfr": RunningAverage(),
            "aggression_factor": RunningAverage(),
            "fold_to_cbet": RunningAverage(),
            "fold_to_3bet": RunningAverage(),
            "wtsd": RunningAverage(),  # Went to showdown
            "wsd": RunningAverage(),   # Won at showdown
        }

        # Situational patterns
        self.patterns = {
            "preflop_open_range": RangeEstimator(),
            "cbet_frequency_by_board": {},
            "check_raise_frequency": RunningAverage(),
            "river_bluff_frequency": RunningAverage(),
        }

        # Recent hand history (for LLM context)
        self.recent_hands = deque(maxlen=10)

        # Notable exploitable tendencies
        self.tendencies = []

    def update(self, hand_history: HandHistory):
        """Update stats after each hand"""
        ...

    def get_summary(self) -> str:
        """Generate natural language summary for LLM"""
        return f"""
        Opponent Profile ({self.hands_played} hands):
        - Style: {self.classify_style()}
        - VPIP: {self.stats['vpip'].value:.0%} (plays {self.classify_vpip()})
        - PFR: {self.stats['pfr'].value:.0%}
        - Aggression: {self.stats['aggression_factor'].value:.1f}
        - Folds to C-bet: {self.stats['fold_to_cbet'].value:.0%}

        Key Tendencies:
        {self.format_tendencies()}

        Recent Notable Hands:
        {self.format_recent_hands()}
        """
```

**Belief State Storage**:
- SQLite for persistent storage across sessions
- JSON for serialization
- In-memory cache for fast access

**Deliverable**: Agent tracks and uses opponent patterns for exploitation

---

### Phase 5: GTO Knowledge Base (Week 5)
**Goal**: Pre-computed GTO strategies for LLM reference

**Tasks**:

1. **Preflop ranges** (from solver outputs)
   - Opening ranges by position
   - 3-bet/4-bet ranges
   - Calling ranges vs different positions

2. **Postflop heuristics**
   - C-bet frequencies by board texture
   - Check-raise frequencies
   - Bet sizing guidelines

3. **Common spot lookup**
   - Build database of common situations
   - Store optimal frequencies and sizing

```python
class GTOAdvisor:
    def __init__(self):
        self.preflop_ranges = load_preflop_ranges()
        self.postflop_heuristics = load_postflop_heuristics()

    def get_preflop_advice(self, position: str,
                           action_to_us: str) -> dict:
        """Get GTO preflop strategy"""
        ...

    def get_postflop_advice(self, board_texture: str,
                            position: str,
                            pot_type: str) -> dict:
        """Get GTO postflop guidance"""
        ...
```

**Sources for GTO data**:
- Public solver outputs (GTO Wizard data where available)
- Academic papers on preflop equilibria
- Community-sourced range charts

**Deliverable**: LLM can query GTO baseline for any common situation

---

### Phase 6: Integration & Testing (Week 6)
**Goal**: End-to-end working agent with evaluation

**Tasks**:

1. **Full integration**
   - Connect all components
   - Handle edge cases (all-in, side pots, timeouts)

2. **Evaluation framework**
   ```python
   def evaluate_agent(agent, opponent, num_hands=10000):
       """Run agent vs opponent and measure performance"""
       results = play_session(agent, opponent, num_hands)
       return {
           "bb_per_100": calculate_winrate(results),
           "vpip": calculate_vpip(results),
           "aggression": calculate_aggression(results),
           "showdown_winrate": calculate_sd_winrate(results)
       }
   ```

3. **Baseline comparisons**
   - vs Random: Should win massively (>100 bb/100)
   - vs Rule-based: Should be profitable
   - vs Slumbot API (if available): Measure vs real bot

4. **Ablation studies**
   - Agent with vs without memory
   - Agent with vs without GTO tools
   - Different LLM models

**Deliverable**: Quantified agent performance against baselines

---

### Phase 7: Optimization & Polish (Week 7+)
**Goal**: Production-ready agent

**Tasks**:

1. **Latency optimization**
   - Parallel tool calls where possible
   - Cache common queries
   - Optimize prompt length

2. **Cost optimization**
   - Use smaller model for simple decisions
   - Batch similar computations
   - Token usage monitoring

3. **Prompt engineering refinement**
   - A/B test different prompt structures
   - Add few-shot examples for tricky spots
   - Improve exploitation reasoning

4. **Extended features**
   - Multiple opponent tracking
   - Session-based adaptation
   - Tilt detection and adjustment

5. **Optional: Fine-tuning**
   - Collect agent decisions and outcomes
   - Fine-tune smaller model on successful plays
   - Similar to SpinGPT approach

---

## Technology Stack

| Component | Recommendation | Alternative |
|-----------|---------------|-------------|
| Language | Python 3.11+ | - |
| LLM Provider | Anthropic (Claude) | OpenAI (GPT-4) |
| Game Environment | RLCard | Custom engine |
| Hand Evaluation | `treys` | `pokereval` |
| Memory Storage | SQLite | PostgreSQL |
| Async | asyncio + httpx | aiohttp |
| Testing | pytest + pytest-asyncio | - |
| Experiment Tracking | WandB | Custom logging |

### Key Dependencies
```
# Core
anthropic>=0.18.0        # Claude API
rlcard>=1.1.0            # Poker game engine
treys>=0.1.8             # Hand evaluation

# Utilities
numpy>=1.24.0
pydantic>=2.0            # Data validation
sqlalchemy>=2.0          # Database ORM
httpx>=0.25.0            # Async HTTP client

# Development
pytest>=7.0
pytest-asyncio>=0.21
wandb>=0.15              # Experiment tracking
rich>=13.0               # CLI output
```

---

## Verification Strategy

### Unit Tests
- Hand evaluation correctness (known hand rankings)
- Equity calculator accuracy (compare to online calculators)
- Pot odds calculations
- Memory persistence and retrieval

### Integration Tests
- Complete game simulation with agent
- Tool invocation during ReAct loop
- Memory updates after hands

### Performance Benchmarks
1. **vs Random**: Should win >100 bb/100 hands
2. **vs Rule-based TAG**: Should be profitable (>10 bb/100)
3. **vs Self (ablation)**: Memory-enabled should beat memory-disabled
4. **Optional - vs Slumbot API**: Measure against established bot

### Metrics to Track
- **Win rate**: bb/100 hands (primary metric)
- **API costs**: $ per 1000 hands
- **Decision latency**: seconds per decision
- **Tool usage**: which tools most valuable
- **Exploitation success**: win rate vs specific player types

### Sample Size Requirements
- Minimum 10,000 hands for statistical significance
- Track confidence intervals on win rate
- Use duplicate boards where possible

---

## Project Structure

```
cr-poker-bot/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── poker_agent.py      # Main ReAct agent loop
│   │   ├── prompts.py          # Prompt templates
│   │   └── action_parser.py    # Parse LLM output to actions
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── hand_evaluator.py   # Hand strength tool
│   │   ├── equity_calculator.py # Monte Carlo equity
│   │   ├── pot_odds.py         # Pot odds calculation
│   │   ├── gto_advisor.py      # GTO lookup tool
│   │   ├── opponent_stats.py   # Opponent stats tool
│   │   └── position_context.py # Position/stack tool
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── opponent_memory.py  # Per-opponent tracking
│   │   ├── stats_tracker.py    # Running statistics
│   │   ├── hand_history.py     # Hand history storage
│   │   └── database.py         # SQLite persistence
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_interface.py   # RLCard wrapper
│   │   ├── state.py            # Game state classes
│   │   └── actions.py          # Action definitions
│   │
│   ├── gto/
│   │   ├── __init__.py
│   │   ├── preflop_ranges.py   # Preflop range charts
│   │   ├── postflop_heuristics.py
│   │   └── data/               # Range files (JSON)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cards.py            # Card utilities
│       └── evaluation.py       # Agent evaluation
│
├── tests/
│   ├── test_tools/
│   ├── test_memory/
│   ├── test_agent/
│   └── test_integration/
│
├── scripts/
│   ├── play.py                 # Run agent interactively
│   ├── evaluate.py             # Run evaluation session
│   └── analyze.py              # Analyze results
│
├── data/
│   ├── gto_ranges/             # Precomputed ranges
│   └── sessions/               # Saved game sessions
│
├── configs/
│   └── agent_config.yaml       # Agent configuration
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Key Resources

### Papers
- [Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400)
- [Superhuman AI for heads-up no-limit poker (Libratus)](https://www.science.org/doi/10.1126/science.aao1733)
- [Combining Deep RL and Search (ReBeL)](https://arxiv.org/abs/2007.13544)
- [PokerBench: Training LLMs for poker](https://arxiv.org/abs/2501.08328)

### Code Repositories
- [PokerRL Framework](https://github.com/EricSteinberger/PokerRL)
- [RLCard Toolkit](https://github.com/datamllab/rlcard)
- [Pluribus Open Source](https://github.com/keithlee96/pluribus-poker-AI)
- [ReBeL (Facebook)](https://github.com/facebookresearch/rebel)

### Tutorials
- [CFR Explained](https://int8.io/counterfactual-regret-minimization-for-poker-ai/)
- [MIT Poker Theory Course](https://ocw.mit.edu/courses/15-s50-poker-theory-and-analytics-january-iap-2015/)

---

## Summary

This plan builds an LLM-based poker agent incrementally:

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| 1. Foundation | Game environment | RLCard integration, hand eval |
| 2. Tools | LLM-callable poker tools | 6 specialized tools |
| 3. Agent Core | ReAct reasoning loop | Working LLM agent |
| 4. Memory | Opponent modeling | Persistent stats tracking |
| 5. GTO | Strategy baseline | Pre-computed ranges |
| 6. Testing | Validation | Performance benchmarks |
| 7. Polish | Production ready | Optimized latency/cost |

### Why This Approach?

**Advantages of LLM-based agent over traditional CFR:**
- **Interpretable reasoning**: Can explain decisions in natural language
- **Flexible exploitation**: LLMs excel at pattern recognition and adaptation
- **Rapid iteration**: No training compute needed, just prompt engineering
- **Leverages existing knowledge**: LLMs already know poker strategy
- **Modern research direction**: PokerBench (2025) shows 78% solver alignment possible

**Trade-offs:**
- **Latency**: API calls slower than local inference (mitigated by caching)
- **Cost**: Per-token pricing (mitigated by smaller models for simple spots)
- **GTO precision**: May not match pure solver output (mitigated by GTO tools)

### Key Success Factors

1. **Well-designed tools**: The LLM is only as good as the information available
2. **Quality GTO data**: Baseline strategy must be sound
3. **Effective memory**: Exploitation requires accurate opponent modeling
4. **Prompt engineering**: ReAct format must guide reasoning effectively

---

## Next Steps

Ready to begin implementation. Phase 1 deliverables:
1. Project structure setup
2. RLCard integration for game engine
3. `treys` integration for hand evaluation
4. Random and rule-based baseline agents
5. Basic test suite
