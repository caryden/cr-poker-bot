# World-Class Poker Playing Agent: Implementation Plan

## Executive Summary

Building a world-class poker AI by combining an **LLM (Large Language Model) reasoning model** with **poker-specific tools** in a **ReAct-style agentic loop** (Reasoning + Acting). This hybrid approach leverages:
- LLM strategic reasoning and opponent modeling capabilities
- Pre-computed GTO (Game Theory Optimal) strategies as consultable tools
- **Belief state conditioning** using subjective logic for uncertainty-aware reasoning
- Memory/belief state for tracking opponent patterns
- Hand evaluation and equity calculation tools

Based on research into state-of-the-art agents (Libratus, Pluribus, DeepStack, ReBeL) and the latest LLM advances (PokerBench 2025 showing fine-tuned LLMs achieving 78% solver alignment, SpinGPT), this plan targets **6-Max No-Limit Texas Hold'em** (6 players) with full table position simulation, GTO baseline, and exploitation capabilities.

---

## Glossary of Terms

### Poker Terms
| Term | Definition |
|------|------------|
| **NLHE** | No-Limit Hold'em - Texas Hold'em variant where players can bet any amount up to their stack |
| **6-Max** | 6-handed table format (6 players maximum) |
| **HUNL** | Heads-Up No-Limit - 2-player variant of No-Limit Hold'em |
| **Hero** | The player controlled by our agent |
| **Villain** | Any opponent player at the table |
| **Hole Cards** | The two private cards dealt to each player |
| **Board** | Community cards shared by all players (flop + turn + river) |
| **Flop** | First three community cards dealt together |
| **Turn** | Fourth community card |
| **River** | Fifth and final community card |
| **Pot** | Total chips wagered in the current hand |
| **Stack** | A player's total chips |
| **Effective Stack** | The smaller stack between two players (maximum at risk) |
| **SPR** | Stack-to-Pot Ratio - effective stack divided by pot size |
| **Position** | Seat relative to the dealer button determining action order |
| **IP** | In Position - acting last in a betting round (information advantage) |
| **OOP** | Out of Position - acting first in a betting round |
| **C-bet** | Continuation Bet - betting the flop after raising preflop |
| **3-bet** | Re-raising a raise (the third bet in a sequence) |
| **4-bet** | Re-raising a 3-bet |
| **All-in** | Betting all remaining chips |
| **Equity** | Probability of winning the hand at showdown |
| **Pot Odds** | Ratio of current pot to the cost of calling |
| **Implied Odds** | Expected future winnings factored into calling decisions |
| **Range** | Set of possible hands a player could hold |
| **Showdown** | Revealing cards at the end to determine winner |

### Table Positions (6-Max)
| Position | Abbreviation | Description |
|----------|--------------|-------------|
| **Under the Gun** | UTG | First to act preflop (seat 1 after BB) |
| **Hijack** | HJ | Two seats before button |
| **Cutoff** | CO | One seat before button |
| **Button/Dealer** | BTN | Last to act postflop (best position) |
| **Small Blind** | SB | Posts forced half-bet, first to act postflop |
| **Big Blind** | BB | Posts forced full bet, last to act preflop |

### Player Statistics
| Stat | Full Name | Definition |
|------|-----------|------------|
| **VPIP** | Voluntarily Put $ In Pot | % of hands where player puts money in (not counting blinds) |
| **PFR** | Pre-Flop Raise | % of hands where player raises preflop |
| **AF** | Aggression Factor | (bets + raises) / calls - measures aggression |
| **WTSD** | Went To ShowDown | % of hands that reach showdown |
| **W$SD** | Won $ at ShowDown | % of showdowns won |
| **Fold to C-bet** | - | % of time player folds to continuation bets |
| **3-bet %** | - | % of time player re-raises a raise |

### Player Archetypes
| Type | Abbreviation | Description |
|------|--------------|-------------|
| **Tight-Aggressive** | TAG | Plays few hands but aggressively |
| **Loose-Aggressive** | LAG | Plays many hands aggressively |
| **Tight-Passive** | Nit/Rock | Plays few hands passively |
| **Loose-Passive** | Fish/Calling Station | Plays many hands passively (calls too much) |
| **Maniac** | - | Extremely aggressive, bets/raises constantly |

### Technical/AI Terms
| Term | Definition |
|------|------------|
| **GTO** | Game Theory Optimal - unexploitable Nash equilibrium strategy |
| **CFR** | Counterfactual Regret Minimization - algorithm for finding Nash equilibria |
| **CFR+** | Improved CFR variant with faster convergence |
| **MCCFR** | Monte Carlo CFR - sampling-based CFR for large games |
| **Deep CFR** | CFR with neural network function approximation |
| **ReAct** | Reasoning + Acting - agentic loop pattern for LLMs |
| **Nash Equilibrium** | Strategy where no player benefits from unilateral deviation |
| **EV** | Expected Value - average outcome weighted by probability |
| **Abstraction** | Reducing game complexity by grouping similar states |
| **Blueprint Strategy** | Pre-computed offline strategy for common situations |
| **Subgame Solving** | Real-time strategy refinement for specific situations |

### Subjective Logic Terms
| Term | Definition |
|------|------------|
| **Opinion** | Tuple (b, d, u, a) representing belief state |
| **Belief (b)** | Degree of belief that proposition is true |
| **Disbelief (d)** | Degree of belief that proposition is false |
| **Uncertainty (u)** | Degree of uncommitted belief (lack of evidence) |
| **Base Rate (a)** | Prior probability when uncertainty is maximal |
| **Vacuous Opinion** | Maximum uncertainty (0, 0, 1, a) - no evidence |
| **Dogmatic Opinion** | Zero uncertainty - complete certainty |
| **Projected Probability** | P = b + a·u - probability accounting for uncertainty |

---

## Target Configuration

### Game Format
- **Variant**: No-Limit Texas Hold'em (NLHE)
- **Table Size**: 6-Max (6 players) - standard online format
- **Blinds**: Configurable (default 5/10 with 1000 chip stacks = 100 BB deep)
- **Positions**: Full rotation through UTG, HJ, CO, BTN, SB, BB

### Architecture
- **Core**: LLM reasoning agent with belief-conditioned ReAct loop
- **Belief System**: Subjective logic framework for uncertainty-aware decisions
- **API**: Anthropic Claude (primary) or OpenAI GPT-4 (alternative)
- **Strategy**: GTO baseline with adaptive opponent exploitation

### Game Assumptions
1. **Multi-player focus**: Primary target is 6-Max tables (6 players)
2. **Position rotation**: Hero occupies all positions across hands
3. **Multiple villains**: Track beliefs about each opponent independently
4. **Variable stack depths**: Handle 20-200 BB effective stacks
5. **Heads-up capability**: Degenerate case when table is 2-handed
6. **No side pots initially**: Simplify to single pot (Phase 1)
7. **Rake-free**: No rake calculation (can be added later)

### Framework
- **Game Engine**: RLCard (extended for 6-max support)
- **Hand Evaluation**: `treys` library
- **Belief Tracking**: Custom subjective logic implementation
- **Persistence**: SQLite for opponent models

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

## Recommended Architecture: Belief-Conditioned LLM ReAct Agent

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     LLM POKER AGENT SYSTEM (6-MAX)                            │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │              BELIEF-CONDITIONED AGENT LOOP (ReAct Style)                 │ │
│  │                                                                          │ │
│  │  ┌─────────┐   ┌──────────┐   ┌─────────────┐   ┌──────┐   ┌────────┐  │ │
│  │  │ Observe │──►│  Update  │──►│   Reason    │──►│ Act  │──►│ Update │  │ │
│  │  │  State  │   │ Beliefs  │   │(LLM|Beliefs)│   │      │   │ Memory │  │ │
│  │  └─────────┘   └──────────┘   └─────────────┘   └──────┘   └────────┘  │ │
│  │       ▲              │               │               │           │      │ │
│  │       │              ▼               │               │           │      │ │
│  │       │     ┌────────────────┐       │               │           │      │ │
│  │       │     │  BELIEF STATE  │◄──────┘               │           │      │ │
│  │       │     │ (Subjective    │                       │           │      │ │
│  │       │     │    Logic)      │                       │           │      │ │
│  │       │     └────────────────┘                       │           │      │ │
│  │       └──────────────────────────────────────────────┴───────────┘      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                   │                                           │
│          ┌────────────────────────┼────────────────────────┐                 │
│          ▼                        ▼                        ▼                  │
│  ┌────────────────┐    ┌───────────────────┐    ┌──────────────────────┐    │
│  │   REASONING    │    │   BELIEF STORE    │    │    MEMORY STORE      │    │
│  │     MODEL      │    │ (Per-Opponent)    │    │                      │    │
│  │ (Claude/GPT)   │    │                   │    │ • Hand histories     │    │
│  │                │    │ • Range beliefs   │    │ • Session stats      │    │
│  │ Conditioned on:│    │   (b,d,u,a)       │    │ • Aggregate stats    │    │
│  │ • Game state   │    │ • Type beliefs    │    │ • Notable hands      │    │
│  │ • All beliefs  │    │ • Tendency beliefs│    │                      │    │
│  │ • Uncertainty  │    │ • Skill beliefs   │    │ Villains 1-5:        │    │
│  │                │    │                   │    │ • VPIP, PFR, AF      │    │
│  └────────────────┘    │ Villains 1-5:     │    │ • Fold to C-bet      │    │
│          │             │ ω_range[v]        │    │ • 3-bet %, WTSD      │    │
│          │             │ ω_type[v]         │    │                      │    │
│          │             │ ω_bluff[v]        │    └──────────────────────┘    │
│          │             └───────────────────┘                                 │
│          │ Tool Calls                                                        │
│          ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                            TOOL LAYER                                    ││
│  │                                                                          ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ││
│  │  │   Hand    │ │  Equity   │ │   GTO     │ │ Opponent  │ │  Belief   │ ││
│  │  │ Evaluator │ │Calculator │ │  Advisor  │ │   Stats   │ │  Query    │ ││
│  │  │           │ │           │ │           │ │           │ │           │ ││
│  │  │ Strength, │ │ Win % vs  │ │ Optimal   │ │ Per-villain│ │ "How sure │ ││
│  │  │ draws     │ │ ranges    │ │ action    │ │ stats     │ │ is V1 LAG?"│ ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ ││
│  │                                                                          ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐               ││
│  │  │   Pot     │ │  Range    │ │ Position  │ │  Table    │               ││
│  │  │   Odds    │ │ Analysis  │ │  Context  │ │ Dynamics  │               ││
│  │  │           │ │           │ │           │ │           │               ││
│  │  │ Required  │ │ Villain's │ │ 6 seats,  │ │ Stack     │               ││
│  │  │ equity    │ │ likely has│ │ action    │ │ sizes,    │               ││
│  │  │           │ │           │ │ order     │ │ dynamics  │               ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    GAME ENGINE (Extended RLCard)                         ││
│  │     6-Max NLHE: positions, multi-way pots, side pots, all-in equity      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Belief-Conditioned ReAct Loop Flow

The key innovation is that the LLM's reasoning is **conditioned on explicit belief states** represented using subjective logic. This allows the agent to reason about uncertainty and adjust confidence appropriately.

```
1. OBSERVE: Receive game state from 6-max table
           - Hero's cards and position (UTG/HJ/CO/BTN/SB/BB)
           - Board cards (community)
           - Pot size, stack sizes for all 6 players
           - Action history this hand
           - Active villains (1-5 opponents)
           │
           ▼
2. UPDATE BELIEFS: Revise beliefs based on observed actions
           For each villain V who acted:
           │
           │  ω_range[V] = update_range_belief(action, position, history)
           │  ω_type[V]  = update_type_belief(action, ω_type[V])
           │  ω_bluff[V] = update_bluff_belief(action, street, ω_bluff[V])
           │
           │  Each belief is a subjective logic opinion: (b, d, u, a)
           │  - High uncertainty (u) → insufficient data, stay closer to GTO
           │  - Low uncertainty → can exploit with confidence
           │
           ▼
3. REASON (LLM | Beliefs): LLM analyzes WITH belief context
           │
           │  Prompt includes:
           │  - Game state (cards, pot, stacks, positions)
           │  - Belief summaries for each active villain:
           │      "V1 (CO): ω_type = (0.7, 0.1, 0.2, 0.5) → likely LAG (70% belief)"
           │      "V2 (BTN): ω_type = (0.3, 0.2, 0.5, 0.5) → uncertain (50% uncertainty)"
           │  - Projected probabilities accounting for uncertainty
           │
           │  Tool calls gather information:
           │  - evaluate_hand() → "Top pair, good kicker"
           │  - calculate_equity(vs_range=belief_weighted_range) → "58% vs V1, 62% vs V2"
           │  - get_gto_advice() → "Standard c-bet 33% pot"
           │  - query_belief("V1", "folds_to_cbet") → (0.65, 0.15, 0.20, 0.50)
           │
           │  LLM synthesis considers uncertainty:
           │  "V1 is likely LAG (70% belief, 20% uncertain), folds to c-bet 65%
           │   with reasonable confidence. V2 is unknown (50% uncertain).
           │   Against V1, exploit with larger c-bet. Against V2, stay balanced."
           │
           ▼
4. ACT:    Execute decision (fold/check/call/bet X/raise X)
           - Action accounts for multi-way dynamics
           - Size adjusted for number of opponents
           - Position in action order matters
           │
           ▼
5. UPDATE MEMORY: Store outcomes for future belief updates
           - Record action taken and result
           - Update opponent statistics
           - Flag notable hands for LLM context
           │
           └──────► Loop back to OBSERVE (next action or next hand)
```

---

## Belief State Framework (Subjective Logic)

### Why Subjective Logic?

Traditional probabilistic models collapse uncertainty into point estimates. In poker, distinguishing between:
- "I'm 60% sure villain is bluffing" (based on 100 hands)
- "I'm 60% sure villain is bluffing" (based on 3 hands)

is critical. **Subjective logic** explicitly represents this uncertainty.

### Opinion Representation

A **subjective logic opinion** about proposition X is a tuple ω = (b, d, u, a) where:
- **b** (belief): Evidence-based belief that X is true
- **d** (disbelief): Evidence-based belief that X is false
- **u** (uncertainty): Lack of evidence (uncommitted belief)
- **a** (base rate): Prior probability when u=1 (no evidence)

Constraint: b + d + u = 1

**Projected probability**: P(X) = b + a·u

### Example: Villain Type Belief

```python
# After 5 hands, villain has shown aggression
ω_is_lag = Opinion(
    belief=0.4,      # Some evidence of LAG behavior
    disbelief=0.1,   # Little evidence against
    uncertainty=0.5, # Still very uncertain (few hands)
    base_rate=0.25   # Prior: 25% of players are LAG
)
# Projected probability: P(LAG) = 0.4 + 0.25 * 0.5 = 0.525

# After 100 hands, pattern is clear
ω_is_lag = Opinion(
    belief=0.75,     # Strong evidence of LAG behavior
    disbelief=0.15,  # Some hands were passive
    uncertainty=0.1, # Low uncertainty (many hands)
    base_rate=0.25   # Prior unchanged
)
# Projected probability: P(LAG) = 0.75 + 0.25 * 0.1 = 0.775
```

### Belief Categories

For each villain, we maintain beliefs about:

```python
@dataclass
class VillainBeliefs:
    """Subjective logic beliefs about a single opponent"""

    villain_id: str
    hands_observed: int

    # Player type beliefs (mutually exclusive)
    type_beliefs: dict[str, Opinion]  # TAG, LAG, Nit, Fish, Maniac, Unknown

    # Tendency beliefs (independent)
    ω_folds_to_cbet: Opinion      # Folds to continuation bets
    ω_folds_to_3bet: Opinion      # Folds to 3-bets
    ω_bluffs_river: Opinion       # Bluffs on river
    ω_overvalues_tp: Opinion      # Overvalues top pair
    ω_slowplays_monsters: Opinion # Slowplays big hands

    # Range beliefs by position and action
    range_beliefs: dict[tuple[Position, ActionType], RangeOpinion]

    # Skill level belief
    ω_skill_level: Opinion  # High skill vs low skill

    def get_projected_type(self) -> tuple[str, float, float]:
        """Return most likely type with probability and uncertainty"""
        best_type = None
        best_proj = 0
        best_uncert = 1

        for player_type, opinion in self.type_beliefs.items():
            proj = opinion.projected_probability()
            if proj > best_proj:
                best_type = player_type
                best_proj = proj
                best_uncert = opinion.uncertainty

        return (best_type, best_proj, best_uncert)
```

### Belief Update Rules

Beliefs are updated using subjective logic **trust discounting** and **consensus operators**:

```python
class BeliefUpdater:
    """Update beliefs based on observed actions"""

    def update_on_action(self, villain: VillainBeliefs,
                         action: Action, context: GameContext) -> VillainBeliefs:
        """
        Update beliefs after observing villain's action.

        Uses subjective logic cumulative fusion:
        ω_new = ω_old ⊕ ω_evidence
        """
        # Generate evidence opinion from action
        evidence = self._action_to_evidence(action, context)

        # Fuse with existing belief
        for belief_name, evidence_opinion in evidence.items():
            old_opinion = getattr(villain, belief_name)
            new_opinion = self._cumulative_fusion(old_opinion, evidence_opinion)
            setattr(villain, belief_name, new_opinion)

        villain.hands_observed += 1
        return villain

    def _cumulative_fusion(self, ω1: Opinion, ω2: Opinion) -> Opinion:
        """
        Combine two opinions from same source (cumulative evidence).

        As evidence accumulates, uncertainty decreases.
        """
        # Cumulative fusion formula from subjective logic
        k = ω1.uncertainty + ω2.uncertainty - ω1.uncertainty * ω2.uncertainty

        if k == 0:
            return ω1  # Both dogmatic, no change

        b = (ω1.belief * ω2.uncertainty + ω2.belief * ω1.uncertainty) / k
        d = (ω1.disbelief * ω2.uncertainty + ω2.disbelief * ω1.uncertainty) / k
        u = (ω1.uncertainty * ω2.uncertainty) / k
        a = (ω1.base_rate + ω2.base_rate) / 2  # Average base rates

        return Opinion(b, d, u, a)

    def _action_to_evidence(self, action: Action,
                            context: GameContext) -> dict[str, Opinion]:
        """
        Convert observed action to evidence opinions.

        Each action provides weak evidence about various beliefs.
        """
        evidence = {}

        # Example: Large raise on river
        if action.action_type == ActionType.RAISE and context.street == Street.RIVER:
            if action.amount > context.pot:
                # Over-pot raise: evidence of polarized play (strong or bluff)
                evidence['ω_bluffs_river'] = Opinion(
                    belief=0.15,      # Slight evidence of bluffing tendency
                    disbelief=0.05,
                    uncertainty=0.80, # Single action is weak evidence
                    base_rate=0.25
                )

        # Example: Folding to c-bet
        if action.action_type == ActionType.FOLD and context.facing_cbet:
            evidence['ω_folds_to_cbet'] = Opinion(
                belief=0.2,
                disbelief=0.0,
                uncertainty=0.8,
                base_rate=0.45  # Average fold-to-cbet is ~45%
            )

        return evidence
```

### Belief-to-Strategy Translation

High uncertainty beliefs → GTO play (unexploitable)
Low uncertainty beliefs → Exploitative adjustments

```python
def get_exploitation_weight(uncertainty: float) -> float:
    """
    How much to deviate from GTO based on belief uncertainty.

    High uncertainty → stay GTO (weight = 0)
    Low uncertainty → exploit maximally (weight = 1)
    """
    # Sigmoid function centered at u=0.3
    # Below 0.2 uncertainty: exploit heavily
    # Above 0.5 uncertainty: mostly GTO
    return 1 / (1 + math.exp(10 * (uncertainty - 0.3)))

def blend_strategies(gto_action: Action, exploit_action: Action,
                     belief_uncertainty: float) -> Action:
    """
    Blend GTO and exploitative strategies based on certainty.
    """
    weight = get_exploitation_weight(belief_uncertainty)

    if weight < 0.3:
        return gto_action  # Too uncertain, play GTO
    elif weight > 0.7:
        return exploit_action  # Confident, exploit
    else:
        # Mixed strategy: sometimes GTO, sometimes exploit
        if random.random() < weight:
            return exploit_action
        return gto_action
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

### Phase 4: Belief System & Memory (Week 4)
**Goal**: Subjective logic belief framework and persistent opponent modeling

This phase implements the core innovation: **belief-conditioned reasoning** using subjective logic to handle uncertainty in opponent modeling.

**Subjective Logic Implementation**:

```python
@dataclass
class Opinion:
    """
    Subjective logic opinion: (belief, disbelief, uncertainty, base_rate)

    Represents uncertain knowledge about a proposition.
    - b + d + u = 1 (additivity constraint)
    - Projected probability: P = b + a*u
    """
    belief: float       # Evidence FOR the proposition
    disbelief: float    # Evidence AGAINST the proposition
    uncertainty: float  # Lack of evidence (uncommitted)
    base_rate: float    # Prior probability (used when u > 0)

    def __post_init__(self):
        # Validate additivity
        assert abs(self.belief + self.disbelief + self.uncertainty - 1.0) < 0.001
        assert 0 <= self.base_rate <= 1

    @property
    def projected_probability(self) -> float:
        """Expected probability accounting for uncertainty"""
        return self.belief + self.base_rate * self.uncertainty

    @classmethod
    def vacuous(cls, base_rate: float = 0.5) -> "Opinion":
        """Maximum uncertainty - no evidence"""
        return cls(0.0, 0.0, 1.0, base_rate)

    @classmethod
    def from_frequency(cls, successes: int, failures: int,
                       base_rate: float = 0.5) -> "Opinion":
        """
        Create opinion from observed frequencies.

        Uses the standard mapping from beta distribution:
        b = successes / (total + 2)
        d = failures / (total + 2)
        u = 2 / (total + 2)
        """
        total = successes + failures
        if total == 0:
            return cls.vacuous(base_rate)

        W = 2  # Weight of prior (non-informative)
        b = successes / (total + W)
        d = failures / (total + W)
        u = W / (total + W)

        return cls(b, d, u, base_rate)

    def fuse_cumulative(self, other: "Opinion") -> "Opinion":
        """
        Cumulative fusion: combine evidence from same source over time.

        As evidence accumulates, uncertainty decreases.
        """
        k = self.uncertainty + other.uncertainty - self.uncertainty * other.uncertainty

        if k == 0:
            return self  # Both dogmatic

        b = (self.belief * other.uncertainty + other.belief * self.uncertainty) / k
        d = (self.disbelief * other.uncertainty + other.disbelief * self.uncertainty) / k
        u = (self.uncertainty * other.uncertainty) / k
        a = (self.base_rate + other.base_rate) / 2

        return Opinion(b, d, u, a)
```

**Per-Villain Belief Tracking**:

```python
@dataclass
class VillainBeliefs:
    """Subjective logic beliefs about a single opponent"""

    villain_id: str
    hands_observed: int = 0

    # Player type beliefs (multinomial - one must be true)
    type_opinions: dict[str, Opinion] = field(default_factory=lambda: {
        "TAG": Opinion.vacuous(0.25),   # Tight-Aggressive
        "LAG": Opinion.vacuous(0.15),   # Loose-Aggressive
        "Nit": Opinion.vacuous(0.15),   # Tight-Passive
        "Fish": Opinion.vacuous(0.30),  # Loose-Passive (most common)
        "Maniac": Opinion.vacuous(0.05),
        "Unknown": Opinion.vacuous(0.10),
    })

    # Tendency beliefs (independent binomial)
    ω_folds_to_cbet: Opinion = field(default_factory=lambda: Opinion.vacuous(0.45))
    ω_folds_to_3bet: Opinion = field(default_factory=lambda: Opinion.vacuous(0.55))
    ω_bluffs_river: Opinion = field(default_factory=lambda: Opinion.vacuous(0.25))
    ω_continuation_bets: Opinion = field(default_factory=lambda: Opinion.vacuous(0.65))
    ω_check_raises: Opinion = field(default_factory=lambda: Opinion.vacuous(0.08))

    # Skill assessment
    ω_is_skilled: Opinion = field(default_factory=lambda: Opinion.vacuous(0.30))

    def get_projected_type(self) -> tuple[str, float, float]:
        """Return (type, probability, uncertainty) for most likely type"""
        best = max(self.type_opinions.items(),
                   key=lambda x: x[1].projected_probability)
        return (best[0], best[1].projected_probability, best[1].uncertainty)

    def get_exploitation_confidence(self) -> float:
        """
        How confident are we in exploiting this villain?

        Returns 0-1 where:
        - 0 = stay GTO (high uncertainty)
        - 1 = exploit maximally (low uncertainty, clear patterns)
        """
        # Average uncertainty across key beliefs
        beliefs = [
            self.ω_folds_to_cbet,
            self.ω_folds_to_3bet,
            self.ω_bluffs_river,
        ]
        avg_uncertainty = sum(b.uncertainty for b in beliefs) / len(beliefs)

        # Convert to confidence (sigmoid centered at 0.3 uncertainty)
        import math
        return 1 / (1 + math.exp(10 * (avg_uncertainty - 0.3)))


@dataclass
class TableBeliefs:
    """Beliefs about all villains at the table"""

    beliefs: dict[str, VillainBeliefs] = field(default_factory=dict)

    def get_villain_beliefs(self, villain_id: str) -> VillainBeliefs:
        if villain_id not in self.beliefs:
            self.beliefs[villain_id] = VillainBeliefs(villain_id=villain_id)
        return self.beliefs[villain_id]

    def update_from_action(self, villain_id: str, action: Action,
                           context: "ActionContext") -> None:
        """Update beliefs based on observed action"""
        beliefs = self.get_villain_beliefs(villain_id)
        updater = BeliefUpdater()
        updater.update_on_action(beliefs, action, context)
```

**Memory Components** (separate from beliefs):

```python
class OpponentMemory:
    """Tracks raw statistics and hand histories (feeds into beliefs)"""

    def __init__(self, opponent_id: str):
        self.opponent_id = opponent_id
        self.hands_played = 0

        # Raw aggregate statistics (frequentist)
        self.stats = {
            "vpip_hands": 0, "vpip_opportunities": 0,
            "pfr_hands": 0, "pfr_opportunities": 0,
            "cbet_made": 0, "cbet_opportunities": 0,
            "fold_to_cbet": 0, "faced_cbet": 0,
            "went_to_showdown": 0, "could_showdown": 0,
            "won_at_showdown": 0, "showdowns": 0,
        }

        # Recent hand history (for LLM context)
        self.recent_hands: deque[HandHistory] = deque(maxlen=10)

        # Notable hands (big pots, unusual plays)
        self.notable_hands: list[HandHistory] = []

    def update(self, hand_history: HandHistory):
        """Update raw stats after each hand"""
        self.hands_played += 1
        # ... update individual stats ...
        self.recent_hands.append(hand_history)

    def to_beliefs(self) -> VillainBeliefs:
        """Convert raw stats to subjective logic beliefs"""
        beliefs = VillainBeliefs(villain_id=self.opponent_id,
                                  hands_observed=self.hands_played)

        # Convert fold-to-cbet frequency to opinion
        if self.stats["faced_cbet"] > 0:
            beliefs.ω_folds_to_cbet = Opinion.from_frequency(
                successes=self.stats["fold_to_cbet"],
                failures=self.stats["faced_cbet"] - self.stats["fold_to_cbet"],
                base_rate=0.45  # Population average
            )

        # ... convert other stats ...
        return beliefs

    def get_summary(self) -> str:
        """Generate natural language summary for LLM"""
        vpip = self.stats["vpip_hands"] / max(1, self.stats["vpip_opportunities"])
        pfr = self.stats["pfr_hands"] / max(1, self.stats["pfr_opportunities"])

        return f"""
        Opponent: {self.opponent_id} ({self.hands_played} hands)
        - VPIP: {vpip:.0%} | PFR: {pfr:.0%}
        - Sample size: {'small' if self.hands_played < 30 else 'medium' if self.hands_played < 100 else 'large'}

        Recent notable hands:
        {self._format_notable_hands()}
        """
```

**Storage**:
- SQLite for persistent storage across sessions
- Beliefs serialized as JSON
- In-memory cache for active session

**Deliverable**:
- Subjective logic Opinion class with fusion operators
- Per-villain belief tracking with uncertainty
- Belief-to-exploitation confidence translation
- Integration with ReAct loop for belief-conditioned prompts

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
│   │   ├── poker_agent.py      # Main belief-conditioned ReAct agent loop
│   │   ├── prompts.py          # Prompt templates with belief formatting
│   │   └── action_parser.py    # Parse LLM output to actions
│   │
│   ├── beliefs/                # NEW: Subjective logic framework
│   │   ├── __init__.py
│   │   ├── opinion.py          # Opinion class (b, d, u, a)
│   │   ├── operators.py        # Fusion, discount, consensus operators
│   │   ├── villain_beliefs.py  # Per-villain belief state
│   │   ├── table_beliefs.py    # Aggregate beliefs for all villains
│   │   └── belief_updater.py   # Action → belief update logic
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── hand_evaluator.py   # Hand strength tool
│   │   ├── equity_calculator.py # Monte Carlo equity
│   │   ├── pot_odds.py         # Pot odds calculation
│   │   ├── gto_advisor.py      # GTO lookup tool
│   │   ├── opponent_stats.py   # Opponent stats tool
│   │   ├── belief_query.py     # NEW: Query belief state
│   │   ├── position_context.py # Position/stack tool
│   │   └── table_dynamics.py   # NEW: Multi-way pot dynamics
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── opponent_memory.py  # Per-opponent raw stats tracking
│   │   ├── stats_tracker.py    # Running statistics (frequentist)
│   │   ├── hand_history.py     # Hand history storage
│   │   └── database.py         # SQLite persistence
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_interface.py   # Extended RLCard wrapper (6-max)
│   │   ├── state.py            # GameState, TableState, PlayerState
│   │   ├── actions.py          # Action definitions
│   │   ├── positions.py        # NEW: 6-max position management
│   │   └── side_pots.py        # NEW: Side pot calculation
│   │
│   ├── gto/
│   │   ├── __init__.py
│   │   ├── preflop_ranges.py   # Preflop range charts (by position)
│   │   ├── postflop_heuristics.py
│   │   └── data/               # Range files (JSON)
│   │       ├── ranges_utg.json
│   │       ├── ranges_hj.json
│   │       ├── ranges_co.json
│   │       ├── ranges_btn.json
│   │       ├── ranges_sb.json
│   │       └── ranges_bb.json
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cards.py            # Card utilities
│       └── evaluation.py       # Agent evaluation
│
├── tests/
│   ├── test_beliefs/           # NEW: Belief system tests
│   │   ├── test_opinion.py
│   │   ├── test_fusion.py
│   │   └── test_updater.py
│   ├── test_tools/
│   ├── test_memory/
│   ├── test_engine/            # NEW: 6-max engine tests
│   ├── test_agent/
│   └── test_integration/
│
├── scripts/
│   ├── play.py                 # Run agent interactively
│   ├── evaluate.py             # Run evaluation session
│   ├── analyze.py              # Analyze results
│   └── simulate_table.py       # NEW: Simulate 6-max games
│
├── data/
│   ├── gto_ranges/             # Precomputed ranges
│   ├── sessions/               # Saved game sessions
│   └── beliefs/                # NEW: Serialized belief states
│
├── configs/
│   ├── agent_config.yaml       # Agent configuration
│   └── belief_priors.yaml      # NEW: Base rates for beliefs
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

---

## Detailed Technical Specifications

### Game State Representation (6-Max)

The `GameState` class is the central data structure passed to the LLM agent. It captures all information needed for decision-making in a multi-player context.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class Position(Enum):
    """6-Max table positions in action order (preflop)"""
    UTG = "UTG"           # Under the Gun - first to act preflop
    HJ = "HJ"             # Hijack - two before button
    CO = "CO"             # Cutoff - one before button
    BTN = "BTN"           # Button/Dealer - last postflop (best position)
    SB = "SB"             # Small Blind - posts half blind
    BB = "BB"             # Big Blind - posts full blind, last preflop

    @property
    def is_blind(self) -> bool:
        return self in (Position.SB, Position.BB)

    @property
    def is_late_position(self) -> bool:
        """Late position = more information advantage"""
        return self in (Position.CO, Position.BTN)

    @property
    def is_early_position(self) -> bool:
        return self in (Position.UTG, Position.HJ)

class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

@dataclass
class Card:
    rank: str  # '2'-'9', 'T', 'J', 'Q', 'K', 'A'
    suit: str  # 'h', 'd', 'c', 's'

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_string(cls, s: str) -> "Card":
        return cls(rank=s[0], suit=s[1])

@dataclass
class Action:
    player_id: str                     # Who took the action
    action_type: ActionType
    amount: Optional[int] = None       # In chips, for bet/raise

    def __str__(self) -> str:
        if self.amount:
            return f"{self.player_id}:{self.action_type.value} {self.amount}"
        return f"{self.player_id}:{self.action_type.value}"

@dataclass
class BettingRound:
    street: Street
    actions: list[Action]
    pot_at_start: int

@dataclass
class PlayerState:
    """State of a single player at the table"""
    player_id: str
    stack: int                # Current stack in chips
    position: Position
    is_active: bool           # Still in hand (hasn't folded)
    is_all_in: bool           # Has put all chips in
    amount_invested: int      # Chips invested this street
    total_invested: int       # Chips invested entire hand

    @property
    def can_act(self) -> bool:
        """Can this player still take actions?"""
        return self.is_active and not self.is_all_in

@dataclass
class TableState:
    """State of all players at a 6-max table"""
    players: dict[str, PlayerState]  # player_id -> state
    dealer_position: int             # Seat number with button (0-5)

    def get_active_players(self) -> list[PlayerState]:
        """Players still in the hand"""
        return [p for p in self.players.values() if p.is_active]

    def get_players_can_act(self) -> list[PlayerState]:
        """Players who can still make decisions"""
        return [p for p in self.players.values() if p.can_act]

    def get_player_by_position(self, position: Position) -> Optional[PlayerState]:
        """Get player in specific position"""
        for p in self.players.values():
            if p.position == position:
                return p
        return None

@dataclass
class GameState:
    """Complete game state for 6-max NLHE decision making"""

    # Hand identification
    hand_id: str

    # Game configuration
    small_blind: int
    big_blind: int
    max_players: int = 6

    # Current street
    street: Street

    # Cards
    hero_cards: tuple[Card, Card]
    board: list[Card]  # 0-5 cards depending on street

    # Players (6-max table)
    hero_id: str
    table: TableState

    # Pot and betting
    pot: int                    # Main pot
    side_pots: list[int] = field(default_factory=list)  # Side pots if all-ins
    current_bet: int            # Current bet to call
    min_raise: int              # Minimum raise amount

    # Action tracking
    whose_turn: str             # player_id of current actor
    betting_history: list[BettingRound] = field(default_factory=list)

    # Derived properties
    @property
    def hero(self) -> PlayerState:
        return self.table.players[self.hero_id]

    @property
    def villains(self) -> list[PlayerState]:
        """All opponents (active or not)"""
        return [p for p in self.table.players.values() if p.player_id != self.hero_id]

    @property
    def active_villains(self) -> list[PlayerState]:
        """Opponents still in the hand"""
        return [p for p in self.villains if p.is_active]

    @property
    def num_active_players(self) -> int:
        return len(self.table.get_active_players())

    @property
    def is_heads_up(self) -> bool:
        """Only 2 players remain"""
        return self.num_active_players == 2

    @property
    def is_multiway(self) -> bool:
        """More than 2 players in pot"""
        return self.num_active_players > 2

    @property
    def to_call(self) -> int:
        return self.current_bet - self.hero.amount_invested

    @property
    def pot_odds(self) -> float:
        if self.to_call == 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    @property
    def effective_stack(self) -> int:
        """Smallest stack among active players (max chips at risk)"""
        active = self.table.get_active_players()
        return min(p.stack + p.amount_invested for p in active)

    @property
    def stack_to_pot_ratio(self) -> float:
        """SPR (Stack-to-Pot Ratio) - important for postflop decisions"""
        if self.pot == 0:
            return float('inf')
        return self.effective_stack / self.pot

    @property
    def hero_has_position(self) -> bool:
        """Does hero act last among remaining players postflop?"""
        if self.street == Street.PREFLOP:
            return False  # Position is complex preflop

        # Postflop: BTN acts last, then positions in reverse order
        position_order = [Position.SB, Position.BB, Position.UTG,
                         Position.HJ, Position.CO, Position.BTN]
        active_positions = [p.position for p in self.table.get_active_players()]

        hero_pos = self.hero.position
        hero_order = position_order.index(hero_pos)

        for pos in position_order[hero_order + 1:]:
            if pos in active_positions:
                return False  # Someone acts after hero

        return True

    @property
    def players_to_act_after_hero(self) -> int:
        """How many players act after hero this round?"""
        # Implementation depends on action order tracking
        return sum(1 for v in self.active_villains
                   if self._acts_after(self.hero.position, v.position))

    def _acts_after(self, pos1: Position, pos2: Position) -> bool:
        """Does pos2 act after pos1?"""
        if self.street == Street.PREFLOP:
            preflop_order = [Position.UTG, Position.HJ, Position.CO,
                           Position.BTN, Position.SB, Position.BB]
            return preflop_order.index(pos2) > preflop_order.index(pos1)
        else:
            postflop_order = [Position.SB, Position.BB, Position.UTG,
                            Position.HJ, Position.CO, Position.BTN]
            return postflop_order.index(pos2) > postflop_order.index(pos1)

    def get_legal_actions(self) -> list[Action]:
        """Return all legal actions for hero in current state"""
        actions = []

        if self.to_call > 0:
            actions.append(Action(self.hero_id, ActionType.FOLD))
            actions.append(Action(self.hero_id, ActionType.CALL, self.to_call))
        else:
            actions.append(Action(self.hero_id, ActionType.CHECK))

        # Can bet/raise if have chips beyond call amount
        chips_after_call = self.hero.stack - self.to_call
        if chips_after_call > 0:
            if self.to_call > 0:
                # Raise
                min_raise_total = self.current_bet + self.min_raise
                if chips_after_call >= self.min_raise:
                    actions.append(Action(self.hero_id, ActionType.RAISE, min_raise_total))
                    # Standard sizings (as % of pot)
                    for sizing in [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]:
                        raise_amount = int(self.pot * sizing)
                        total = self.current_bet + raise_amount
                        if raise_amount >= self.min_raise and total <= self.hero.stack:
                            actions.append(Action(self.hero_id, ActionType.RAISE, total))
            else:
                # Bet (no prior bet to call)
                for sizing in [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5]:
                    bet_amount = int(self.pot * sizing)
                    if bet_amount >= self.big_blind and bet_amount <= self.hero.stack:
                        actions.append(Action(self.hero_id, ActionType.BET, bet_amount))

            # All-in always available
            actions.append(Action(self.hero_id, ActionType.ALL_IN, self.hero.stack))

        return actions

    def to_prompt_string(self, beliefs: Optional["TableBeliefs"] = None) -> str:
        """Format state for LLM prompt with belief context"""
        board_str = " ".join(str(c) for c in self.board) if self.board else "none"
        history_str = self._format_history()
        table_str = self._format_table()
        belief_str = self._format_beliefs(beliefs) if beliefs else ""

        return f"""
=== GAME STATE ===
Street: {self.street.value.upper()}
Your Cards: {self.hero_cards[0]} {self.hero_cards[1]}
Board: {board_str}

=== TABLE (6-Max) ===
Your Position: {self.hero.position.value}
{table_str}

=== POT & BETTING ===
Pot: {self.pot} chips
To Call: {self.to_call} chips
SPR: {self.stack_to_pot_ratio:.1f}
Players in Hand: {self.num_active_players} {'(Heads-Up)' if self.is_heads_up else '(Multi-way)'}

=== ACTION HISTORY ===
{history_str}

=== VILLAIN BELIEFS ===
{belief_str if belief_str else "(No belief data available)"}

=== LEGAL ACTIONS ===
{', '.join(a.action_type.value + (f' {a.amount}' if a.amount else '') for a in self.get_legal_actions())}
"""

    def _format_table(self) -> str:
        """Format table state showing all players"""
        lines = []
        position_order = [Position.UTG, Position.HJ, Position.CO,
                         Position.BTN, Position.SB, Position.BB]

        for pos in position_order:
            player = self.table.get_player_by_position(pos)
            if player:
                status = "HERO" if player.player_id == self.hero_id else ""
                if not player.is_active:
                    status = "folded"
                elif player.is_all_in:
                    status = "ALL-IN"

                stack_bb = player.stack / self.big_blind
                lines.append(
                    f"  {pos.value:4} | {player.player_id:10} | "
                    f"{player.stack:6} chips ({stack_bb:5.0f} BB) | {status}"
                )

        return "\n".join(lines)

    def _format_history(self) -> str:
        lines = []
        for round in self.betting_history:
            actions_str = " → ".join(str(a) for a in round.actions)
            lines.append(f"  {round.street.value}: {actions_str}")
        return "\n".join(lines) if lines else "  (no actions yet)"

    def _format_beliefs(self, beliefs: "TableBeliefs") -> str:
        """Format belief summaries for each active villain"""
        lines = []
        for villain in self.active_villains:
            v_beliefs = beliefs.get_villain_beliefs(villain.player_id)
            if v_beliefs:
                player_type, prob, uncert = v_beliefs.get_projected_type()
                lines.append(
                    f"  {villain.player_id} ({villain.position.value}): "
                    f"likely {player_type} ({prob:.0%} belief, {uncert:.0%} uncertainty)"
                )
                # Add key tendencies if low uncertainty
                if uncert < 0.4:
                    fold_cbet = v_beliefs.ω_folds_to_cbet.projected_probability()
                    lines.append(f"    - Folds to C-bet: {fold_cbet:.0%}")
        return "\n".join(lines) if lines else "No beliefs available"
```

### Hand History Format

For memory persistence and analysis, hands are stored in a structured format:

```python
@dataclass
class HandHistory:
    """Complete record of a played hand"""

    hand_id: str
    timestamp: datetime

    # Game setup
    small_blind: int
    big_blind: int
    hero_position: Position
    hero_starting_stack: int
    villain_starting_stack: int

    # Cards (revealed at end or if shown)
    hero_cards: tuple[Card, Card]
    villain_cards: Optional[tuple[Card, Card]]  # If shown down
    board: list[Card]

    # Actions by street
    preflop_actions: list[tuple[str, Action]]  # (player_id, action)
    flop_actions: list[tuple[str, Action]]
    turn_actions: list[tuple[str, Action]]
    river_actions: list[tuple[str, Action]]

    # Result
    winner: str  # player_id
    pot_won: int
    showdown: bool

    # Computed stats for memory
    def get_villain_stats_update(self) -> dict:
        """Extract stats to update opponent model"""
        stats = {}

        # VPIP: Did villain voluntarily put money in pot preflop?
        stats["vpip"] = self._villain_vpip()

        # PFR: Did villain raise preflop?
        stats["pfr"] = self._villain_pfr()

        # ... more stats
        return stats
```

---

## Tool API Specifications

Each tool has a well-defined schema for the LLM to invoke:

### 1. Hand Evaluator Tool

```python
# Schema for LLM tool definition
HAND_EVALUATOR_SCHEMA = {
    "name": "evaluate_hand",
    "description": "Evaluate the strength of your current hand",
    "input_schema": {
        "type": "object",
        "properties": {
            "hole_cards": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Your two hole cards, e.g. ['Ah', 'Kd']"
            },
            "board": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Community cards, e.g. ['Qs', 'Jh', '2c']"
            }
        },
        "required": ["hole_cards", "board"]
    }
}

# Implementation
class HandEvaluatorTool:
    def __init__(self):
        from treys import Evaluator, Card
        self.evaluator = Evaluator()
        self.Card = Card

    def execute(self, hole_cards: list[str], board: list[str]) -> dict:
        """
        Returns:
        {
            "hand_rank": 1234,        # Lower is better (1 = royal flush)
            "hand_class": 2,          # 1-9 hand class
            "hand_name": "Two Pair",
            "hand_description": "Two Pair, Aces and Kings with Queen kicker",
            "percentile": 0.85,       # Hand is better than 85% of possible hands
            "made_hand": true,        # Has at least pair
            "draws": {
                "flush_draw": false,
                "oesd": false,         # Open-ended straight draw
                "gutshot": true,
                "overcards": 0
            }
        }
        """
        try:
            treys_hole = [self.Card.new(c) for c in hole_cards]
            treys_board = [self.Card.new(c) for c in board]

            rank = self.evaluator.evaluate(treys_board, treys_hole)
            hand_class = self.evaluator.get_rank_class(rank)

            return {
                "hand_rank": rank,
                "hand_class": hand_class,
                "hand_name": self.evaluator.class_to_string(hand_class),
                "hand_description": self._get_detailed_description(
                    hole_cards, board, hand_class
                ),
                "percentile": 1 - (rank / 7462),  # 7462 is worst hand
                "made_hand": hand_class <= 8,  # Pair or better
                "draws": self._analyze_draws(hole_cards, board)
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_draws(self, hole_cards: list[str], board: list[str]) -> dict:
        """Analyze drawing potential"""
        all_cards = hole_cards + board
        suits = [c[1] for c in all_cards]
        ranks = [c[0] for c in all_cards]

        # Flush draw: 4 cards of same suit
        suit_counts = {s: suits.count(s) for s in 'hdcs'}
        flush_draw = any(c == 4 for c in suit_counts.values())

        # Straight draws - simplified
        rank_values = self._ranks_to_values(ranks)
        oesd, gutshot = self._analyze_straight_draws(rank_values)

        return {
            "flush_draw": flush_draw,
            "oesd": oesd,
            "gutshot": gutshot,
            "overcards": self._count_overcards(hole_cards, board)
        }
```

### 2. Equity Calculator Tool

```python
EQUITY_CALCULATOR_SCHEMA = {
    "name": "calculate_equity",
    "description": "Calculate win probability against an opponent range using Monte Carlo simulation",
    "input_schema": {
        "type": "object",
        "properties": {
            "hole_cards": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Your hole cards"
            },
            "board": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current board cards"
            },
            "villain_range": {
                "type": "string",
                "description": "Opponent's estimated range in standard notation, e.g. 'AA-TT,AKs-ATs,AKo-AQo' or 'top20%' or 'random'"
            },
            "num_simulations": {
                "type": "integer",
                "default": 10000,
                "description": "Number of Monte Carlo simulations"
            }
        },
        "required": ["hole_cards", "board", "villain_range"]
    }
}

class EquityCalculatorTool:
    def __init__(self):
        self.range_parser = RangeParser()

    def execute(self, hole_cards: list[str], board: list[str],
                villain_range: str, num_simulations: int = 10000) -> dict:
        """
        Returns:
        {
            "equity": 0.62,           # Overall equity (win + tie/2)
            "win_pct": 0.58,
            "tie_pct": 0.08,
            "lose_pct": 0.34,
            "simulations_run": 10000,
            "villain_combos": 150,    # How many hands in villain's range
            "best_villain_hands": ["AA", "KK", "AK"],  # Top of range
            "equity_vs_top": 0.25,    # Equity vs top of range
            "equity_vs_bottom": 0.85  # Equity vs bottom of range
        }
        """
        # Parse villain range to list of possible hands
        villain_hands = self.range_parser.parse(villain_range)

        # Remove hands that conflict with known cards
        dead_cards = set(hole_cards + board)
        villain_hands = [h for h in villain_hands
                        if not any(c in dead_cards for c in h)]

        wins, ties, losses = 0, 0, 0

        for _ in range(num_simulations):
            # Sample villain hand from range
            v_hand = random.choice(villain_hands)

            # Run out remaining board
            result = self._simulate_runout(hole_cards, v_hand, board)

            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            else:
                losses += 1

        total = wins + ties + losses
        return {
            "equity": (wins + ties/2) / total,
            "win_pct": wins / total,
            "tie_pct": ties / total,
            "lose_pct": losses / total,
            "simulations_run": total,
            "villain_combos": len(villain_hands),
            "best_villain_hands": self._get_top_hands(villain_hands),
            "equity_vs_top": self._equity_vs_subset(
                hole_cards, board, villain_hands[:10]
            ),
            "equity_vs_bottom": self._equity_vs_subset(
                hole_cards, board, villain_hands[-10:]
            )
        }


class RangeParser:
    """Parse poker range notation to list of hand combos"""

    RANKS = "AKQJT98765432"
    SUITS = "hdcs"

    def parse(self, range_str: str) -> list[tuple[str, str]]:
        """
        Parse range notation:
        - "AA" -> all AA combos (6 combos)
        - "AKs" -> all suited AK (4 combos)
        - "AKo" -> all offsuit AK (12 combos)
        - "AA-TT" -> AA, KK, QQ, JJ, TT
        - "AKs-ATs" -> AKs, AQs, AJs, ATs
        - "top20%" -> top 20% of hands
        - "random" -> all hands
        """
        if range_str.lower() == "random":
            return self._all_hands()

        if "%" in range_str:
            return self._top_percent(range_str)

        hands = []
        for part in range_str.split(","):
            part = part.strip()
            if "-" in part:
                hands.extend(self._parse_range(part))
            else:
                hands.extend(self._parse_hand(part))

        return hands

    def _parse_hand(self, hand: str) -> list[tuple[str, str]]:
        """Parse single hand like 'AA', 'AKs', 'AKo'"""
        combos = []

        if len(hand) == 2:
            # Pair like "AA"
            r = hand[0]
            for s1, s2 in [('h','d'),('h','c'),('h','s'),
                           ('d','c'),('d','s'),('c','s')]:
                combos.append((f"{r}{s1}", f"{r}{s2}"))

        elif hand.endswith('s'):
            # Suited like "AKs"
            r1, r2 = hand[0], hand[1]
            for s in self.SUITS:
                combos.append((f"{r1}{s}", f"{r2}{s}"))

        elif hand.endswith('o'):
            # Offsuit like "AKo"
            r1, r2 = hand[0], hand[1]
            for s1 in self.SUITS:
                for s2 in self.SUITS:
                    if s1 != s2:
                        combos.append((f"{r1}{s1}", f"{r2}{s2}"))

        return combos
```

### 3. GTO Advisor Tool

```python
GTO_ADVISOR_SCHEMA = {
    "name": "get_gto_advice",
    "description": "Get game-theory optimal strategy recommendation for current situation",
    "input_schema": {
        "type": "object",
        "properties": {
            "situation_type": {
                "type": "string",
                "enum": ["preflop_open", "preflop_vs_raise", "preflop_vs_3bet",
                        "cbet", "vs_cbet", "turn_barrel", "river_decision"],
                "description": "Type of situation"
            },
            "position": {
                "type": "string",
                "enum": ["BTN", "BB"],
                "description": "Hero's position"
            },
            "hand": {
                "type": "string",
                "description": "Hand in standard notation, e.g. 'AKs', 'JJ'"
            },
            "board_texture": {
                "type": "string",
                "description": "Board texture description for postflop, e.g. 'Ah7c2d' or 'dry' or 'wet'"
            },
            "pot_type": {
                "type": "string",
                "enum": ["limped", "srp", "3bet", "4bet"],
                "description": "Type of pot (single raised, 3bet, etc.)"
            }
        },
        "required": ["situation_type", "position", "hand"]
    }
}

class GTOAdvisorTool:
    def __init__(self, data_path: str = "data/gto_ranges"):
        self.preflop_ranges = self._load_preflop_ranges(data_path)
        self.cbet_strategies = self._load_cbet_strategies(data_path)

    def execute(self, situation_type: str, position: str, hand: str,
                board_texture: str = None, pot_type: str = "srp") -> dict:
        """
        Returns:
        {
            "recommended_action": "raise",
            "bet_sizing": "2.5x BB" or "75% pot",
            "frequency": {
                "raise": 0.6,
                "call": 0.3,
                "fold": 0.1
            },
            "reasoning": "AKs is a premium hand that should raise for value...",
            "alternative_lines": [
                {"action": "call", "ev_difference": -0.5, "when": "vs very tight 3bettor"}
            ],
            "range_context": "This hand is in the top 5% of opening range"
        }
        """
        if situation_type == "preflop_open":
            return self._get_preflop_open_advice(position, hand)
        elif situation_type == "cbet":
            return self._get_cbet_advice(position, hand, board_texture, pot_type)
        # ... other situations

    def _get_preflop_open_advice(self, position: str, hand: str) -> dict:
        """Lookup preflop opening strategy"""
        range_data = self.preflop_ranges.get(position, {})
        hand_normalized = self._normalize_hand(hand)

        if hand_normalized in range_data.get("raise", []):
            return {
                "recommended_action": "raise",
                "bet_sizing": "2.5x BB",
                "frequency": {"raise": 1.0, "fold": 0.0},
                "reasoning": f"{hand} is a standard open from {position}",
                "range_context": f"Part of ~{len(range_data.get('raise', []))/169:.0%} opening range"
            }
        else:
            return {
                "recommended_action": "fold",
                "frequency": {"fold": 1.0},
                "reasoning": f"{hand} is not in standard {position} opening range"
            }
```

### 4. Opponent Stats Tool

```python
OPPONENT_STATS_SCHEMA = {
    "name": "get_opponent_stats",
    "description": "Get tracked statistics and tendencies for the current opponent",
    "input_schema": {
        "type": "object",
        "properties": {
            "opponent_id": {
                "type": "string",
                "description": "Opponent identifier"
            },
            "stat_type": {
                "type": "string",
                "enum": ["summary", "preflop", "postflop", "tendencies", "recent"],
                "default": "summary",
                "description": "Type of stats to retrieve"
            }
        },
        "required": ["opponent_id"]
    }
}

class OpponentStatsTool:
    def __init__(self, memory_store: "OpponentMemory"):
        self.memory = memory_store

    def execute(self, opponent_id: str, stat_type: str = "summary") -> dict:
        """
        Returns for stat_type="summary":
        {
            "hands_played": 150,
            "player_type": "TAG",  # Tight-Aggressive
            "stats": {
                "vpip": 0.28,
                "pfr": 0.22,
                "aggression_factor": 2.1,
                "wtsd": 0.24,
                "wsd": 0.52
            },
            "preflop_tendencies": {
                "open_raise_pct": 0.22,
                "fold_to_3bet": 0.65,
                "cold_call_pct": 0.08
            },
            "postflop_tendencies": {
                "cbet_pct": 0.68,
                "fold_to_cbet": 0.45,
                "check_raise_pct": 0.08,
                "river_bluff_freq": 0.25
            },
            "exploits": [
                "Folds too much to 3-bets (65%) - can 3-bet wider for value",
                "Low river bluff frequency - can hero fold more rivers",
                "High c-bet frequency but folds to raises - consider floating"
            ],
            "reliability": "high"  # Based on sample size
        }
        """
        profile = self.memory.get_opponent_profile(opponent_id)

        if stat_type == "summary":
            return self._format_summary(profile)
        elif stat_type == "tendencies":
            return self._format_tendencies(profile)
        elif stat_type == "recent":
            return self._format_recent_hands(profile)
        # ...
```

---

## Prompt Engineering Specification

### System Prompt Template

```python
SYSTEM_PROMPT = """You are an expert heads-up no-limit Texas Hold'em poker player.
Your goal is to maximize expected value while balancing GTO play with opponent exploitation.

## Your Decision Process

1. **ASSESS** the situation: position, stack depths, pot size, betting action
2. **EVALUATE** your hand strength using available tools
3. **CONSIDER** opponent tendencies from memory
4. **COMPARE** to GTO baseline
5. **DECIDE** on the optimal action with reasoning

## Available Tools

{tool_descriptions}

## Key Principles

- **Pot Odds**: Only call if equity exceeds pot odds requirement
- **Position**: Value position highly - it provides information advantage
- **Stack Depth**: Adjust play based on SPR (stack-to-pot ratio)
- **Balance**: Mix strategies to remain unexploitable when unsure
- **Exploitation**: Deviate from GTO when opponent has clear leaks

## Output Format

Think through your decision step by step:
1. State your hand and the situation
2. Use tools to gather information
3. Analyze the key factors
4. Make your decision with clear reasoning

Final action format: ACTION: <fold|check|call|bet X|raise X>
"""

USER_PROMPT_TEMPLATE = """
## Current Hand

{game_state_formatted}

## Opponent Profile

{opponent_summary}

What is your action? Use tools as needed, then decide.
"""
```

### Few-Shot Examples

```python
FEW_SHOT_EXAMPLES = [
    {
        "situation": "Preflop open with premium hand",
        "game_state": """
Street: PREFLOP
Your Cards: Ah Kd
Board: none
Position: Button
Pot: 15 chips (blinds)
Your Stack: 1000 chips (100 BB)
Villain Stack: 1000 chips
To Call: 0 chips
Action History: (first to act)
""",
        "reasoning": """Let me assess this spot.

I have AKo on the button with 100bb effective stacks. This is a premium hand.

[Tool: get_gto_advice(situation_type="preflop_open", position="BTN", hand="AKo")]

The GTO advisor confirms AKo is a clear raise from the button. Standard sizing is 2.5x BB.

Given:
- Premium hand (top 3% of hands)
- Position advantage
- Deep stacks allow for postflop play
- Raising builds pot with a hand that plays well postflop

ACTION: raise 25"""
    },
    {
        "situation": "Facing c-bet with middle pair",
        "game_state": """
Street: FLOP
Your Cards: 9h 9c
Board: Kh 7d 2s
Position: Big Blind (out of position)
Pot: 60 chips
Your Stack: 970 chips
Villain Stack: 970 chips
To Call: 40 chips (villain bet 40)
SPR: 16.2
Action History:
  preflop: Villain raises 25 → Hero calls 15
  flop: Villain bets 40
""",
        "reasoning": """Let me work through this spot.

I have 99 on K72 rainbow facing a c-bet.

[Tool: evaluate_hand(hole_cards=["9h", "9c"], board=["Kh", "7d", "2s"])]

Result: Middle pair, 9s. Below top pair but ahead of most unpaired hands.

[Tool: get_opponent_stats(opponent_id="villain", stat_type="summary")]

Opponent c-bets 68% of the time but folds to raises 55% of the time.

[Tool: calculate_pot_odds(pot=60, to_call=40)]

Need 40% equity to call.

[Tool: calculate_equity(hole_cards=["9h", "9c"], board=["Kh", "7d", "2s"],
       villain_range="AA-TT,AKs-ATs,AKo-AJo,KQs-KTs,QJs")]

Equity: 38% against a standard c-betting range.

Analysis:
- We're slightly below the equity needed to call (38% vs 40% required)
- However, villain has high fold-to-raise frequency (55%)
- This board favors villain's range (has more Kx)
- But villain c-bets too frequently, meaning he has many weak hands

Options:
1. Fold: Mathematically correct vs pure value range
2. Call: Viable given implied odds and villain's wide c-bet range
3. Raise: Exploitative given villain's 55% fold-to-raise

Given villain's exploitable tendencies, a check-raise is profitable. We can fold out
better hands (AT, QJ) and get value/protection from worse.

ACTION: raise 120"""
    }
]
```

---

## Opponent Modeling Algorithms

### Player Classification

```python
class PlayerClassifier:
    """Classify opponents into archetypes based on stats"""

    ARCHETYPES = {
        "rock": {
            "description": "Very tight, only plays premium hands",
            "vpip_range": (0, 0.15),
            "pfr_range": (0, 0.12),
            "af_range": (0, 1.5),
            "exploits": [
                "Fold all but premium hands to their aggression",
                "Steal blinds relentlessly",
                "Don't bluff - they only have it"
            ]
        },
        "nit": {
            "description": "Tight and passive",
            "vpip_range": (0.15, 0.22),
            "pfr_range": (0.10, 0.18),
            "af_range": (0, 1.2),
            "exploits": [
                "Steal blinds frequently",
                "Value bet thinner - they call too much",
                "Avoid big bluffs"
            ]
        },
        "tag": {
            "description": "Tight-aggressive, solid player",
            "vpip_range": (0.20, 0.28),
            "pfr_range": (0.18, 0.25),
            "af_range": (1.5, 3.0),
            "exploits": [
                "Most balanced - focus on GTO play",
                "Look for situational leaks",
                "Position is crucial"
            ]
        },
        "lag": {
            "description": "Loose-aggressive, plays many hands aggressively",
            "vpip_range": (0.28, 0.45),
            "pfr_range": (0.22, 0.35),
            "af_range": (2.0, 4.0),
            "exploits": [
                "Widen calling range",
                "Let them bluff into you",
                "Trap with strong hands"
            ]
        },
        "fish": {
            "description": "Loose-passive, calls too much",
            "vpip_range": (0.35, 1.0),
            "pfr_range": (0, 0.15),
            "af_range": (0, 1.0),
            "exploits": [
                "Value bet relentlessly",
                "Don't bluff - they call everything",
                "Bet bigger for value"
            ]
        },
        "maniac": {
            "description": "Extremely aggressive, bets and raises constantly",
            "vpip_range": (0.40, 1.0),
            "pfr_range": (0.30, 1.0),
            "af_range": (3.0, 10.0),
            "exploits": [
                "Tighten up and let them spew",
                "Call down lighter",
                "Set traps with monsters"
            ]
        }
    }

    def classify(self, stats: dict) -> tuple[str, float]:
        """
        Returns (archetype, confidence) based on stats.
        Requires minimum 30 hands for reliable classification.
        """
        if stats.get("hands_played", 0) < 30:
            return ("unknown", 0.0)

        vpip = stats.get("vpip", 0.25)
        pfr = stats.get("pfr", 0.20)
        af = stats.get("aggression_factor", 1.5)

        best_match = None
        best_score = 0

        for archetype, criteria in self.ARCHETYPES.items():
            score = self._calculate_match_score(vpip, pfr, af, criteria)
            if score > best_score:
                best_score = score
                best_match = archetype

        return (best_match, best_score)
```

### Bayesian Range Estimation

```python
class BayesianRangeEstimator:
    """
    Maintain probability distribution over opponent's range
    based on observed actions.
    """

    def __init__(self):
        # Start with uniform prior over all hands
        self.range_probs = {hand: 1/1326 for hand in ALL_HAND_COMBOS}

    def update_on_action(self, action: Action, game_state: GameState,
                         player_tendencies: dict):
        """
        Bayesian update: P(hand|action) ∝ P(action|hand) × P(hand)

        P(action|hand) comes from combining:
        - GTO frequencies for this spot
        - Player's historical tendencies
        """
        action_probs = self._estimate_action_likelihoods(
            action, game_state, player_tendencies
        )

        # Update each hand's probability
        new_probs = {}
        for hand, prior in self.range_probs.items():
            likelihood = action_probs.get(hand, 0.001)  # Small floor
            new_probs[hand] = likelihood * prior

        # Normalize
        total = sum(new_probs.values())
        self.range_probs = {h: p/total for h, p in new_probs.items()}

    def get_current_range(self, threshold: float = 0.001) -> list[str]:
        """Return hands currently in villain's range"""
        return [h for h, p in self.range_probs.items() if p >= threshold]

    def get_range_string(self) -> str:
        """Format range for equity calculator"""
        # Convert to standard notation
        hands = self.get_current_range()
        return self._hands_to_range_notation(hands)
```

---

## GTO Data Format Specification

### Preflop Range Files

```json
// data/gto_ranges/preflop_btn_open.json
{
    "position": "BTN",
    "situation": "open",
    "effective_stack_bb": 100,
    "range": {
        "raise": [
            "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
            "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
            "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
            "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s",
            "KQo", "KJo", "KTo", "K9o",
            "QJs", "QTs", "Q9s", "Q8s",
            "QJo", "QTo",
            "JTs", "J9s", "J8s",
            "JTo",
            "T9s", "T8s", "T7s",
            "98s", "97s", "96s",
            "87s", "86s",
            "76s", "75s",
            "65s", "64s",
            "54s", "53s",
            "43s"
        ],
        "fold": ["remaining"]
    },
    "sizing": {
        "default": "2.5x",
        "vs_fish": "3x"
    }
}
```

### C-bet Strategy Files

```json
// data/gto_ranges/cbet_strategy.json
{
    "board_categories": {
        "dry_high": {
            "examples": ["Ah7c2d", "Kh8s3c"],
            "texture": "High card, disconnected, rainbow",
            "cbet_frequency": 0.75,
            "sizing": "33% pot",
            "hands_to_cbet": {
                "always": ["top_pair+", "overpair"],
                "frequently": ["middle_pair", "Ax_high", "gutshot"],
                "sometimes": ["underpair", "backdoor_flush"],
                "never": ["nothing"]
            }
        },
        "wet_connected": {
            "examples": ["Jh Th 8d", "9s 8c 7h"],
            "texture": "Connected, two-tone or monotone",
            "cbet_frequency": 0.45,
            "sizing": "66-75% pot",
            "hands_to_cbet": {
                "always": ["two_pair+", "strong_draw"],
                "frequently": ["overpair", "top_pair_good_kicker"],
                "sometimes": ["top_pair_weak", "combo_draw"],
                "never": ["weak_pair", "air"]
            }
        }
    }
}
```

---

## Evaluation Metrics and Benchmarking Protocol

### Primary Metrics

```python
@dataclass
class EvaluationResults:
    """Complete evaluation metrics for an agent"""

    # Sample info
    num_hands: int
    opponent_type: str

    # Primary metric: Win rate in big blinds per 100 hands
    bb_per_100: float
    bb_per_100_ci95: tuple[float, float]  # 95% confidence interval

    # Profit/Loss
    total_profit_bb: float
    biggest_pot_won_bb: float
    biggest_pot_lost_bb: float

    # Playing style metrics
    vpip: float           # Voluntarily Put $ In Pot
    pfr: float            # Pre-Flop Raise %
    aggression_factor: float  # (bet + raise) / call
    wtsd: float           # Went To ShowDown %
    wsd: float            # Won at ShowDown %

    # Positional performance
    bb_per_100_btn: float  # Win rate in position
    bb_per_100_bb: float   # Win rate out of position

    # Decision quality
    correct_folds: float   # Folded when behind (estimated)
    thin_value_success: float  # Successful thin value bets
    bluff_success: float   # Successful bluffs

    # Tool usage stats
    avg_tool_calls_per_hand: float
    most_used_tools: list[str]

    # Cost metrics
    avg_tokens_per_hand: int
    avg_latency_ms: int
    cost_per_1000_hands: float


class EvaluationProtocol:
    """Standard protocol for agent evaluation"""

    MINIMUM_HANDS = 10000
    CONFIDENCE_LEVEL = 0.95

    @staticmethod
    def run_evaluation(agent: "PokerAgent",
                       opponent: "BaseAgent",
                       num_hands: int = 10000,
                       duplicate_boards: bool = True) -> EvaluationResults:
        """
        Run statistically significant evaluation.

        Args:
            agent: Agent being evaluated
            opponent: Opponent agent (random, rule-based, or another AI)
            num_hands: Minimum hands to play
            duplicate_boards: If True, play each board from both positions

        Returns:
            EvaluationResults with confidence intervals
        """
        results = []

        for hand_num in range(num_hands):
            if duplicate_boards:
                # Play same cards from both positions
                board, hero_cards, villain_cards = deal_hand()

                # Hero as BTN
                r1 = play_hand(agent, opponent, hero_cards, villain_cards,
                              hero_position=Position.BUTTON)
                results.append(r1)

                # Hero as BB (swap cards)
                r2 = play_hand(agent, opponent, villain_cards, hero_cards,
                              hero_position=Position.BIG_BLIND)
                results.append(r2)
            else:
                r = play_hand(agent, opponent)
                results.append(r)

        return EvaluationResults.from_hand_results(results)

    @staticmethod
    def calculate_confidence_interval(win_rates: list[float],
                                      confidence: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval for win rate"""
        import scipy.stats as stats

        mean = np.mean(win_rates)
        std_err = stats.sem(win_rates)
        ci = stats.t.interval(confidence, len(win_rates)-1,
                             loc=mean, scale=std_err)
        return ci
```

### Benchmark Suite

```python
class BenchmarkSuite:
    """Standard benchmarks for poker agents"""

    BENCHMARKS = [
        {
            "name": "vs_random",
            "description": "Baseline against random player",
            "opponent": RandomAgent,
            "expected_bb100": ">100",
            "hands": 10000
        },
        {
            "name": "vs_call_station",
            "description": "Player who calls too much",
            "opponent": CallStationAgent,
            "expected_bb100": ">50",
            "hands": 10000
        },
        {
            "name": "vs_tight_passive",
            "description": "Rock/nit player type",
            "opponent": TightPassiveAgent,
            "expected_bb100": ">20",
            "hands": 10000
        },
        {
            "name": "vs_tag_bot",
            "description": "Competent tight-aggressive bot",
            "opponent": TAGRuleBasedAgent,
            "expected_bb100": ">5",
            "hands": 20000
        },
        {
            "name": "vs_self",
            "description": "Self-play for balance testing",
            "opponent": "self",
            "expected_bb100": "~0 (within CI)",
            "hands": 20000
        }
    ]

    def run_all(self, agent: "PokerAgent") -> dict:
        """Run complete benchmark suite"""
        results = {}

        for benchmark in self.BENCHMARKS:
            print(f"Running benchmark: {benchmark['name']}")

            if benchmark["opponent"] == "self":
                opponent = agent  # Self-play
            else:
                opponent = benchmark["opponent"]()

            result = EvaluationProtocol.run_evaluation(
                agent, opponent, benchmark["hands"]
            )

            results[benchmark["name"]] = {
                "bb_per_100": result.bb_per_100,
                "ci95": result.bb_per_100_ci95,
                "meets_expected": self._check_expectation(
                    result.bb_per_100, benchmark["expected_bb100"]
                )
            }

        return results
```

---

## Next Steps

Ready to begin implementation. Phase 1 deliverables:
1. Project structure setup
2. RLCard integration for game engine
3. `treys` integration for hand evaluation
4. Random and rule-based baseline agents
5. Basic test suite
