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
| **Projected Probability** | P = b + a·u - probability accounting for uncertainty |
| **Vacuous Opinion** | Maximum uncertainty (0, 0, 1, a) - no evidence |
| **Dogmatic Opinion** | Zero uncertainty - complete certainty |

### Test-Time Reasoning Terms
| Term | Definition |
|------|------------|
| **TRT** | Test-time Recursive Thinking - iterative self-improvement at inference |
| **Rollout** | A candidate solution path explored during reasoning |
| **Strategy-Conditioned Generation** | Generating solutions based on specific strategic approach |
| **Self-Verification** | Validating correctness without external ground truth |
| **Back-Verification** | Working backward from answer to check validity |
| **Accumulated Knowledge** | Information gathered across reasoning iterations |
| **Convergence** | When further iterations no longer improve the solution |
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

### The Power of Disbelief: Raw Belief Exposure

A key insight from Test-time Recursive Thinking ([arXiv:2602.03094](https://arxiv.org/abs/2602.03094)) is that **negative constraints ("don't do X") are more valuable than positive guidance ("do Y")**. This maps directly to subjective logic's disbelief component.

Rather than converting beliefs to natural language (losing information), we **expose the raw (b, d, u) tuples** directly to the agent and let it reason about them.

#### Why Raw Tuples > Natural Language

| Approach | Example | Problem |
|----------|---------|---------|
| Natural language | "Villain doesn't bluff rivers" | Sounds dogmatic, hides uncertainty |
| Projected probability | "15% chance villain bluffs" | Collapses (b=0.08, d=0.72, u=0.20) into single number |
| **Raw tuple** | **(b=0.08, d=0.72, u=0.20)** | Agent sees full picture, reasons appropriately |

The agent can see:
- **High disbelief (d=0.72)** → Strong evidence AGAINST bluffing
- **Low uncertainty (u=0.20)** → This is reliable information
- **Low belief (b=0.08)** → Very little evidence FOR bluffing

#### Sorting by Knowledge (b + d)

Beliefs are sorted **descending by knowledge**, where knowledge = b + d = 1 - u.

Higher knowledge means more evidence (either for or against), thus more useful for decision-making.

```python
@dataclass
class Opinion:
    """Subjective logic opinion exposing raw (b, d, u, a) to agent."""
    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float

    @property
    def knowledge(self) -> float:
        """Total evidence accumulated (b + d). Higher = more informative."""
        return self.belief + self.disbelief

    def to_tuple_str(self) -> str:
        """Raw tuple format for agent consumption."""
        return f"(b={self.belief:.2f}, d={self.disbelief:.2f}, u={self.uncertainty:.2f})"


@dataclass
class LabeledBelief:
    """A belief with its label, for sorting and presentation."""
    label: str           # e.g., "folds_to_cbet"
    opinion: Opinion
    samples: int         # Number of observations

    @property
    def knowledge(self) -> float:
        return self.opinion.knowledge


class BeliefStatePresenter:
    """
    Present beliefs to agent as raw (b, d, u) tuples.

    Key principles:
    1. Expose raw tuples, don't convert to natural language
    2. Sort by knowledge (b+d) descending - most informative first
    3. Let agent reason about disbelief's value for narrowing search
    """

    def format_for_prompt(self, beliefs: VillainBeliefs) -> str:
        """
        Format beliefs for LLM prompt with raw tuples.

        Sorted by knowledge (b+d) descending.
        """
        # Collect all beliefs with labels
        labeled = [
            LabeledBelief("folds_to_cbet", beliefs.ω_folds_to_cbet,
                         beliefs.samples_cbet),
            LabeledBelief("folds_to_3bet", beliefs.ω_folds_to_3bet,
                         beliefs.samples_3bet),
            LabeledBelief("bluffs_river", beliefs.ω_bluffs_river,
                         beliefs.samples_river),
            LabeledBelief("slowplays_monsters", beliefs.ω_slowplays_monsters,
                         beliefs.samples_slowplay),
            LabeledBelief("is_aggressive", beliefs.ω_is_aggressive,
                         beliefs.samples_aggression),
            # ... other beliefs
        ]

        # Sort by knowledge (b+d) descending
        labeled.sort(key=lambda x: x.knowledge, reverse=True)

        lines = [
            f"Villain: {beliefs.villain_id} ({beliefs.hands_observed} hands)",
            "",
            "Beliefs (sorted by knowledge, highest first):",
            "  Format: label: (b=belief, d=disbelief, u=uncertainty) [n samples]",
            ""
        ]

        for lb in labeled:
            o = lb.opinion
            lines.append(
                f"  {lb.label}: (b={o.belief:.2f}, d={o.disbelief:.2f}, "
                f"u={o.uncertainty:.2f}) [{lb.samples} samples]"
            )

        return "\n".join(lines)
```

#### Example: Raw Belief Presentation

```
Villain: player_42 (150 hands)

Beliefs (sorted by knowledge, highest first):
  Format: label: (b=belief, d=disbelief, u=uncertainty) [n samples]

  folds_to_cbet:      (b=0.12, d=0.68, u=0.20) [45 samples]
  bluffs_river:       (b=0.08, d=0.72, u=0.20) [25 samples]
  is_aggressive:      (b=0.65, d=0.15, u=0.20) [150 samples]
  folds_to_3bet:      (b=0.25, d=0.35, u=0.40) [12 samples]
  slowplays_monsters: (b=0.10, d=0.30, u=0.60) [5 samples]
```

The agent sees:
- `folds_to_cbet` has **high disbelief (0.68)** and low uncertainty → reliable negative
- `bluffs_river` has **high disbelief (0.72)** → villain does NOT bluff rivers
- `is_aggressive` has **high belief (0.65)** → villain IS aggressive
- `slowplays_monsters` has **high uncertainty (0.60)** → not enough data

#### System Prompt: Explaining Disbelief Utility

The system prompt explains how to use disbelief for narrowing exploration:

```python
SYSTEM_PROMPT_BELIEF_SECTION = """
## Understanding Belief Tuples

Beliefs are presented as subjective logic opinions: (b, d, u)
- b (belief): Evidence FOR the proposition
- d (disbelief): Evidence AGAINST the proposition
- u (uncertainty): Lack of evidence (b + d + u = 1)

Beliefs are sorted by knowledge (b + d), highest first. Higher knowledge
means more evidence has been accumulated, making the belief more reliable.

## The Value of Disbelief for Decision Making

HIGH DISBELIEF IS ESPECIALLY VALUABLE. Here's why:

1. **Disbelief narrows the search space**: If d(bluffs_river) = 0.72,
   you can largely eliminate "villain is bluffing" from consideration.
   This prunes entire branches of your reasoning tree.

2. **Disbelief is more actionable than belief**: Knowing villain DOES NOT
   do something eliminates options. Knowing villain MIGHT do something
   only adjusts probabilities.

3. **Use disbelief as constraints**: When d > 0.6 and u < 0.3, treat it
   as a near-certain negative. Don't bluff someone who doesn't fold.
   Don't fold to someone who doesn't bluff.

4. **Belief requires more caution**: Even b = 0.7 means 30% chance you're
   wrong. But d = 0.7 means the action is reliably NOT taken.

## Decision Heuristics

- High d, low u → Strong constraint (villain does NOT do X)
- High b, low u → Strong tendency (villain DOES do X)
- High u → Insufficient data, weight toward GTO/default play
- Low knowledge (b+d) → Ignore this belief, not enough signal
"""
```

#### TRT Integration: Disbelief Narrows Rollouts

In TRT strategy rollouts, high disbelief prunes exploration:

```python
class TRTRolloutPruner:
    """
    Use high-disbelief beliefs to prune TRT rollout space.

    Key insight: If we're confident villain DOESN'T do X,
    don't waste compute exploring scenarios where villain does X.
    """

    def get_villain_response_distribution(
        self,
        action: Action,
        beliefs: VillainBeliefs,
        game_state: GameState
    ) -> dict[str, float]:
        """
        Get likely villain responses, using disbelief to narrow options.
        """
        responses = {}

        if action.action_type in (ActionType.BET, ActionType.RAISE):
            # Check if villain folds
            fold_belief = self._get_fold_belief(beliefs, game_state)

            if fold_belief.disbelief > 0.6 and fold_belief.uncertainty < 0.3:
                # HIGH DISBELIEF: Villain does NOT fold
                # Prune fold from consideration entirely
                responses["fold"] = 0.0
                responses["call"] = 0.7
                responses["raise"] = 0.3
            else:
                # Normal distribution based on projected probability
                fold_prob = fold_belief.belief + fold_belief.base_rate * fold_belief.uncertainty
                responses["fold"] = fold_prob
                responses["call"] = (1 - fold_prob) * 0.7
                responses["raise"] = (1 - fold_prob) * 0.3

        return responses

    def should_explore_bluff(self, beliefs: VillainBeliefs,
                             game_state: GameState) -> bool:
        """
        Should we even consider bluffing in TRT rollouts?

        If villain has high disbelief for folding, don't waste
        compute on bluff rollouts.
        """
        fold_belief = self._get_fold_belief(beliefs, game_state)

        # High disbelief + low uncertainty = don't explore bluffs
        if fold_belief.disbelief > 0.6 and fold_belief.uncertainty < 0.3:
            return False  # Prune bluff exploration entirely

        return True
```

#### Practical Example: Agent Reasoning with Raw Tuples

```
[BELIEF STATE]
Villain: player_42 (150 hands)

Beliefs (sorted by knowledge, highest first):
  bluffs_river:  (b=0.08, d=0.72, u=0.20) [25 samples]
  folds_to_cbet: (b=0.12, d=0.68, u=0.20) [45 samples]

[AGENT REASONING]
Looking at the belief tuples:

1. bluffs_river: d=0.72 with u=0.20 means strong evidence villain
   does NOT bluff rivers. With 25 samples and only 20% uncertainty,
   this is reliable. I should not fold with showdown value.

2. folds_to_cbet: d=0.68 with u=0.20 means villain does NOT fold
   to c-bets. Bluffing c-bet is -EV. Only c-bet for value.

The high disbelief values narrow my decision space:
- Eliminate: bluffing river, bluffing c-bet
- Remaining: value bet or check

This is simpler than reasoning about "maybe villain folds 20%..."
The disbelief tells me what NOT to do, which is more actionable.
```

This approach:
1. **Preserves information** - No rounding to natural language
2. **Sorts by usefulness** - Most informative beliefs first
3. **Explains disbelief's value** - System prompt teaches agent
4. **Enables search pruning** - TRT uses disbelief to narrow rollouts

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

### Belief Revision via Observation-Consistency Mapping

Observations don't directly update beliefs. Instead, each observation O_i is mapped to each belief B_j through a **consistency opinion** — an SL opinion about the proposition "O_i is consistent with B_j".

```
Observation O_i  ──────►  ω_consistency(O_i, B_j)  ──────►  SL Fusion  ──────►  B_j (revised)
                              │
                              ├─ High b: O_i strongly supports B_j
                              ├─ High d: O_i strongly contradicts B_j
                              └─ High u: O_i is ambiguous w.r.t. B_j
```

#### The Consistency Opinion

For each (observation, belief) pair, we compute an opinion about their consistency:

```python
@dataclass
class Observation:
    """A single observed event in the game."""
    obs_id: str
    obs_type: str           # "action", "showdown", "timing", etc.
    action: Optional[Action]
    revealed_cards: Optional[tuple[Card, Card]]  # If showdown
    context: GameContext    # Street, position, pot size, etc.
    timestamp: datetime


@dataclass
class ConsistencyOpinion:
    """
    Opinion about: "Observation O is consistent with Belief B"

    This is the bridge between what we see and what we believe.
    """
    observation: Observation
    belief_label: str       # Which belief this relates to
    opinion: Opinion        # The consistency opinion (b, d, u, a)
    reasoning: str          # Why this consistency rating


class ObservationBeliefMapper:
    """
    Map observations to consistency opinions for each belief.

    Key insight: An observation may be:
    - Strongly consistent (high b) with some beliefs
    - Strongly inconsistent (high d) with others
    - Ambiguous (high u) for most

    This is more principled than direct belief updates.
    """

    def compute_consistency(self, obs: Observation,
                           belief_label: str,
                           current_belief: Opinion) -> ConsistencyOpinion:
        """
        Compute how consistent observation is with a belief.

        Returns opinion about "obs is consistent with belief_label"
        """
        # Dispatch based on observation type
        if obs.obs_type == "action":
            return self._action_consistency(obs, belief_label)
        elif obs.obs_type == "showdown":
            return self._showdown_consistency(obs, belief_label)
        elif obs.obs_type == "timing":
            return self._timing_consistency(obs, belief_label)
        else:
            # Unknown observation type - maximally uncertain
            return ConsistencyOpinion(
                observation=obs,
                belief_label=belief_label,
                opinion=Opinion.vacuous(),
                reasoning="Unknown observation type"
            )

    def _action_consistency(self, obs: Observation,
                           belief_label: str) -> ConsistencyOpinion:
        """
        How consistent is this action with the belief?
        """
        action = obs.action
        ctx = obs.context

        # Example: Villain 3-bets preflop
        if action.action_type == ActionType.RAISE and ctx.street == Street.PREFLOP:
            if ctx.facing_raise:  # This is a 3-bet

                if belief_label == "is_TAG":
                    # 3-betting is consistent with TAG play
                    return ConsistencyOpinion(
                        observation=obs,
                        belief_label=belief_label,
                        opinion=Opinion(b=0.6, d=0.1, u=0.3, a=0.25),
                        reasoning="3-betting is consistent with TAG style"
                    )

                elif belief_label == "is_passive":
                    # 3-betting is inconsistent with passive play
                    return ConsistencyOpinion(
                        observation=obs,
                        belief_label=belief_label,
                        opinion=Opinion(b=0.05, d=0.7, u=0.25, a=0.3),
                        reasoning="3-betting contradicts passive style"
                    )

                elif belief_label == "is_LAG":
                    # 3-betting is somewhat consistent with LAG
                    return ConsistencyOpinion(
                        observation=obs,
                        belief_label=belief_label,
                        opinion=Opinion(b=0.5, d=0.15, u=0.35, a=0.15),
                        reasoning="3-betting is consistent with LAG"
                    )

        # Example: Villain folds to c-bet
        if action.action_type == ActionType.FOLD and ctx.facing_cbet:

            if belief_label == "folds_to_cbet":
                # Direct evidence FOR this belief
                return ConsistencyOpinion(
                    observation=obs,
                    belief_label=belief_label,
                    opinion=Opinion(b=0.8, d=0.05, u=0.15, a=0.45),
                    reasoning="Folding to c-bet directly supports this belief"
                )

            elif belief_label == "is_calling_station":
                # Folding contradicts calling station
                return ConsistencyOpinion(
                    observation=obs,
                    belief_label=belief_label,
                    opinion=Opinion(b=0.05, d=0.75, u=0.20, a=0.20),
                    reasoning="Folding contradicts calling station tendency"
                )

        # Default: observation is ambiguous for this belief
        return ConsistencyOpinion(
            observation=obs,
            belief_label=belief_label,
            opinion=Opinion(b=0.1, d=0.1, u=0.8, a=0.5),
            reasoning="Observation is ambiguous for this belief"
        )

    def _showdown_consistency(self, obs: Observation,
                              belief_label: str) -> ConsistencyOpinion:
        """
        Showdown reveals actual hand - strong evidence.

        If villain shows 72o after 3-betting, that's STRONG disbelief
        for "is_TAG" and STRONG belief for "is_maniac".
        """
        revealed = obs.revealed_cards
        ctx = obs.context

        # Get hand strength category
        hand_strength = self._categorize_hand(revealed)

        if belief_label == "is_TAG":
            if ctx.villain_3bet and hand_strength == "trash":
                # 3-bet with trash → strongly inconsistent with TAG
                return ConsistencyOpinion(
                    observation=obs,
                    belief_label=belief_label,
                    opinion=Opinion(b=0.02, d=0.88, u=0.10, a=0.25),
                    reasoning="3-betting trash hand strongly contradicts TAG"
                )
            elif ctx.villain_3bet and hand_strength == "premium":
                # 3-bet with premium → consistent with TAG
                return ConsistencyOpinion(
                    observation=obs,
                    belief_label=belief_label,
                    opinion=Opinion(b=0.75, d=0.05, u=0.20, a=0.25),
                    reasoning="3-betting premium hand supports TAG"
                )

        # ... similar logic for other beliefs

        return ConsistencyOpinion(
            observation=obs,
            belief_label=belief_label,
            opinion=Opinion.vacuous(),
            reasoning="No specific consistency inference"
        )
```

#### SL Algebra for Belief Revision

Once we have consistency opinions, we use SL operators to revise beliefs:

```python
class SLBeliefReviser:
    """
    Revise beliefs using subjective logic algebra.

    Key operators:
    - Cumulative Fusion (⊕): Combine evidence from same source over time
    - Averaging Fusion (⊙): Combine independent opinions
    - Deduction (⊛): Derive belief from consistency + prior
    """

    def revise_belief(self,
                      current_belief: Opinion,
                      consistency: ConsistencyOpinion) -> Opinion:
        """
        Revise a belief based on observation consistency.

        Uses SL deduction: If we believe "O is consistent with B",
        and we observe O, what should we believe about B?

        ω_B_new = ω_B_old ⊕ deduce(ω_consistency)
        """
        # Convert consistency opinion to evidence about the belief
        evidence = self._consistency_to_evidence(consistency.opinion)

        # Cumulative fusion with existing belief
        revised = self.cumulative_fusion(current_belief, evidence)

        return revised

    def _consistency_to_evidence(self, consistency: Opinion) -> Opinion:
        """
        Convert "O is consistent with B" opinion to evidence about B.

        High b (consistent) → evidence FOR B
        High d (inconsistent) → evidence AGAINST B
        High u (ambiguous) → weak evidence
        """
        # The consistency opinion's belief becomes evidence for the belief
        # The consistency opinion's disbelief becomes evidence against
        # Scale by (1 - uncertainty) to weight by confidence

        weight = 1 - consistency.uncertainty

        return Opinion(
            belief=consistency.belief * weight,
            disbelief=consistency.disbelief * weight,
            uncertainty=1 - weight,  # Remaining goes to uncertainty
            base_rate=consistency.base_rate
        )

    def cumulative_fusion(self, ω1: Opinion, ω2: Opinion) -> Opinion:
        """
        Cumulative fusion: ω1 ⊕ ω2

        Combines evidence from same source over time.
        As evidence accumulates, uncertainty decreases.
        """
        k = ω1.uncertainty + ω2.uncertainty - ω1.uncertainty * ω2.uncertainty

        if k == 0:
            return ω1  # Both dogmatic

        b = (ω1.belief * ω2.uncertainty + ω2.belief * ω1.uncertainty) / k
        d = (ω1.disbelief * ω2.uncertainty + ω2.disbelief * ω1.uncertainty) / k
        u = (ω1.uncertainty * ω2.uncertainty) / k
        a = (ω1.base_rate + ω2.base_rate) / 2

        return Opinion(b, d, u, a)

    def trust_discount(self, ω_belief: Opinion,
                       ω_trust: Opinion) -> Opinion:
        """
        Trust discounting: Adjust belief by trust in source.

        If we don't fully trust the evidence source, discount it.
        """
        # Discounted belief is scaled by trust
        trust_factor = ω_trust.belief + ω_trust.base_rate * ω_trust.uncertainty

        return Opinion(
            belief=ω_belief.belief * trust_factor,
            disbelief=ω_belief.disbelief * trust_factor,
            uncertainty=1 - trust_factor + ω_belief.uncertainty * trust_factor,
            base_rate=ω_belief.base_rate
        )


class BeliefRevisionEngine:
    """
    Full belief revision pipeline.

    Observation → Consistency Opinions → SL Fusion → Revised Beliefs
    """

    def __init__(self):
        self.mapper = ObservationBeliefMapper()
        self.reviser = SLBeliefReviser()

    def process_observation(self, obs: Observation,
                           beliefs: VillainBeliefs) -> VillainBeliefs:
        """
        Process a single observation and revise all relevant beliefs.
        """
        # Get all belief labels we track
        belief_labels = beliefs.get_all_belief_labels()

        for label in belief_labels:
            current = beliefs.get_belief(label)

            # Compute consistency of observation with this belief
            consistency = self.mapper.compute_consistency(obs, label, current)

            # Only revise if consistency is informative (not vacuous)
            if consistency.opinion.uncertainty < 0.9:
                revised = self.reviser.revise_belief(current, consistency)
                beliefs.set_belief(label, revised)

                # Log the revision for transparency
                self._log_revision(obs, label, current, consistency, revised)

        beliefs.observations_processed += 1
        return beliefs

    def _log_revision(self, obs: Observation, label: str,
                      old: Opinion, consistency: ConsistencyOpinion,
                      new: Opinion) -> None:
        """Log belief revision for debugging and transparency."""
        print(f"  {label}: {old.to_tuple_str()} → {new.to_tuple_str()}")
        print(f"    via consistency: {consistency.opinion.to_tuple_str()}")
        print(f"    reasoning: {consistency.reasoning}")
```

#### Example: Observation-Belief Revision Flow

```python
# Observation: Villain 3-bets preflop, later shows 72o at showdown

obs_action = Observation(
    obs_type="action",
    action=Action(ActionType.RAISE, amount=30),  # 3-bet
    context=GameContext(street=Street.PREFLOP, facing_raise=True)
)

obs_showdown = Observation(
    obs_type="showdown",
    revealed_cards=(Card("7", "h"), Card("2", "s")),
    context=GameContext(villain_3bet=True)
)

# Initial beliefs
beliefs = VillainBeliefs(
    is_TAG=Opinion(b=0.3, d=0.2, u=0.5, a=0.25),      # Uncertain
    is_LAG=Opinion(b=0.2, d=0.3, u=0.5, a=0.15),      # Uncertain
    is_maniac=Opinion(b=0.1, d=0.4, u=0.5, a=0.05),   # Lean no
)

# Process 3-bet action
engine.process_observation(obs_action, beliefs)
# is_TAG: consistency (b=0.6, d=0.1, u=0.3) → "3-bet supports TAG"
# is_LAG: consistency (b=0.5, d=0.15, u=0.35) → "3-bet supports LAG"
# is_maniac: consistency (b=0.4, d=0.2, u=0.4) → "3-bet somewhat supports maniac"

# After action observation:
# is_TAG: (b=0.3, d=0.2, u=0.5) → (b=0.42, d=0.17, u=0.41)  # More likely TAG
# is_LAG: (b=0.2, d=0.3, u=0.5) → (b=0.33, d=0.24, u=0.43)  # More likely LAG

# Process showdown - villain had 72o!
engine.process_observation(obs_showdown, beliefs)
# is_TAG: consistency (b=0.02, d=0.88, u=0.10) → "72o strongly contradicts TAG"
# is_maniac: consistency (b=0.85, d=0.05, u=0.10) → "72o strongly supports maniac"

# After showdown observation:
# is_TAG: (b=0.42, d=0.17, u=0.41) → (b=0.15, d=0.58, u=0.27)  # Now high disbelief!
# is_maniac: (b=0.25, d=0.32, u=0.43) → (b=0.52, d=0.18, u=0.30) # Now likely maniac

# Final belief state:
# is_TAG:    (b=0.15, d=0.58, u=0.27) → "Villain is NOT TAG" (high d, low u)
# is_maniac: (b=0.52, d=0.18, u=0.30) → "Villain likely IS maniac" (high b)
```

#### Consistency Matrix

For common observations, we define expected consistency with each belief type:

| Observation | is_TAG | is_LAG | is_Nit | is_Fish | is_Maniac |
|-------------|--------|--------|--------|---------|-----------|
| 3-bet preflop | b=0.6 | b=0.5 | d=0.6 | u=0.8 | b=0.4 |
| Fold to c-bet | u=0.6 | d=0.4 | b=0.5 | d=0.6 | d=0.5 |
| Overbet river | b=0.3 | b=0.5 | d=0.7 | u=0.7 | b=0.7 |
| Call 3 streets | d=0.3 | b=0.3 | d=0.6 | b=0.8 | b=0.4 |
| Check-raise bluff | b=0.4 | b=0.6 | d=0.8 | d=0.7 | b=0.7 |
| Show trash hand | d=0.8 | b=0.4 | d=0.9 | b=0.6 | b=0.85 |
| Show premium | b=0.5 | u=0.6 | b=0.7 | u=0.7 | d=0.4 |

This matrix encodes domain knowledge about poker player types.

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

## Test-Time Recursive Thinking (TRT) for Poker Decisions

Inspired by [Test-time Recursive Thinking (arXiv:2602.03094)](https://arxiv.org/abs/2602.03094), the agent employs **iterative self-improvement at inference time** rather than single-pass reasoning. This is critical for poker where:

1. Decisions have high EV variance - one mistake can cost the entire stack
2. Multiple viable strategies exist (GTO vs exploit)
3. Self-verification against GTO baseline is possible without external feedback
4. Accumulated game knowledge should inform reasoning

### TRT-Enhanced Decision Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRT-ENHANCED POKER REASONING                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     RECURSIVE THINKING LOOP                           │   │
│  │                                                                       │   │
│  │   ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │   │
│  │   │ Initial │───►│  Strategy   │───►│   Verify    │───►│ Refined  │ │   │
│  │   │ Analysis│    │  Rollouts   │    │  Candidate  │    │ Decision │ │   │
│  │   └─────────┘    └─────────────┘    └─────────────┘    └──────────┘ │   │
│  │        │               │                   │                 │       │   │
│  │        │               ▼                   ▼                 │       │   │
│  │        │        ┌─────────────┐    ┌─────────────┐          │       │   │
│  │        │        │ GTO Rollout │    │ GTO Check   │          │       │   │
│  │        │        │ Exploit Rol │    │ EV Compare  │          │       │   │
│  │        │        │ Defensive R │    │ Back-verify │          │       │   │
│  │        │        └─────────────┘    └─────────────┘          │       │   │
│  │        │                                                     │       │   │
│  │        └──────── ITERATE IF NOT CONVERGED ◄──────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    ACCUMULATED KNOWLEDGE BASE                         │   │
│  │   • Successful exploitation patterns vs opponent types                │   │
│  │   • Situations where GTO outperformed exploitation (and vice versa)   │   │
│  │   • Back-verification failures → adjust future reasoning              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Strategy-Conditioned Rollouts

Instead of generating a single action, TRT explores multiple strategic approaches in parallel:

```python
class StrategyRollout:
    """A candidate decision path under a specific strategic approach"""

    strategy: str           # "GTO", "Exploit", "Defensive", "Trapping"
    action: Action          # The action this strategy recommends
    reasoning: str          # LLM's reasoning for this action
    ev_estimate: float      # Estimated expected value
    confidence: float       # Confidence in this estimate
    verification: dict      # Self-verification results


class TRTDecisionMaker:
    """
    Test-time Recursive Thinking for poker decisions.

    Key insight from TRT paper: LLMs can self-improve at inference time
    by conditioning on rollout-specific strategies and self-generated
    verification signals.
    """

    STRATEGIES = {
        "GTO": "Play game-theory optimal, unexploitable baseline",
        "Exploit": "Deviate from GTO to exploit opponent tendencies",
        "Defensive": "Minimize losses against unknown/skilled opponents",
        "Trapping": "Slowplay strong hands to induce bluffs",
        "Bluff-Heavy": "Increase bluff frequency against tight opponents",
    }

    def __init__(self, agent: "PokerAgent", max_iterations: int = 3):
        self.agent = agent
        self.max_iterations = max_iterations
        self.knowledge_base = AccumulatedKnowledge()

    async def decide(self, game_state: GameState,
                     beliefs: TableBeliefs) -> Action:
        """
        TRT-enhanced decision making with recursive refinement.
        """
        iteration = 0
        best_rollout = None

        while iteration < self.max_iterations:
            # 1. Generate strategy-conditioned rollouts
            rollouts = await self._generate_rollouts(game_state, beliefs)

            # 2. Self-verify each rollout
            verified_rollouts = await self._verify_rollouts(
                rollouts, game_state, beliefs
            )

            # 3. Select best rollout based on verification
            current_best = self._select_best_rollout(verified_rollouts)

            # 4. Check for convergence
            if self._has_converged(best_rollout, current_best):
                break

            best_rollout = current_best
            iteration += 1

            # 5. Accumulate knowledge for next iteration
            self._accumulate_knowledge(verified_rollouts)

        # 6. Update knowledge base with final decision
        self.knowledge_base.record_decision(game_state, best_rollout)

        return best_rollout.action

    async def _generate_rollouts(self, game_state: GameState,
                                  beliefs: TableBeliefs) -> list[StrategyRollout]:
        """
        Generate candidate actions under different strategies.

        Each strategy conditions the LLM's reasoning differently.
        """
        rollouts = []

        for strategy_name, strategy_desc in self.STRATEGIES.items():
            # Skip strategies that don't apply
            if not self._strategy_applies(strategy_name, game_state, beliefs):
                continue

            # Generate rollout with strategy-conditioned prompt
            rollout = await self.agent.generate_strategy_rollout(
                game_state=game_state,
                beliefs=beliefs,
                strategy=strategy_name,
                strategy_description=strategy_desc,
                accumulated_knowledge=self.knowledge_base.get_relevant(
                    game_state, strategy_name
                )
            )
            rollouts.append(rollout)

        return rollouts

    async def _verify_rollouts(self, rollouts: list[StrategyRollout],
                                game_state: GameState,
                                beliefs: TableBeliefs) -> list[StrategyRollout]:
        """
        Self-verify each rollout without external feedback.

        Verification signals (inspired by TRT):
        1. GTO consistency - does action align with GTO baseline?
        2. EV estimation - calculate expected value vs opponent range
        3. Back-verification - if we take this action, what happens next?
        4. Comparative evaluation - how does this compare to alternatives?
        """
        for rollout in rollouts:
            verification = {}

            # 1. GTO Consistency Check
            gto_advice = self.agent.tools.get_gto_advice(game_state)
            verification["gto_alignment"] = self._compute_gto_alignment(
                rollout.action, gto_advice
            )
            verification["gto_action"] = gto_advice.recommended_action

            # 2. EV Estimation
            ev_result = await self._estimate_ev(
                rollout.action, game_state, beliefs
            )
            verification["ev_estimate"] = ev_result["ev"]
            verification["ev_confidence"] = ev_result["confidence"]

            # 3. Back-Verification: What happens if villain calls/raises/folds?
            back_verify = await self._back_verify(
                rollout.action, game_state, beliefs
            )
            verification["villain_responses"] = back_verify
            verification["back_verify_score"] = back_verify["overall_score"]

            # 4. Comparative: Is this better than the GTO play?
            if rollout.strategy != "GTO":
                verification["ev_vs_gto"] = (
                    verification["ev_estimate"] -
                    self._get_gto_ev(game_state, beliefs)
                )

            rollout.verification = verification

        return rollouts

    async def _back_verify(self, action: Action, game_state: GameState,
                           beliefs: TableBeliefs) -> dict:
        """
        Back-verification: Simulate villain responses and evaluate outcomes.

        "If I bet 75% pot, what happens when villain:
         - Folds? (I win pot)
         - Calls? (We see turn/river with X equity)
         - Raises? (I face a tough decision with Y hand)"
        """
        results = {}

        # Get villain's likely response frequencies
        for villain in game_state.active_villains:
            v_beliefs = beliefs.get_villain_beliefs(villain.player_id)

            # Estimate response probabilities based on beliefs
            if action.action_type in (ActionType.BET, ActionType.RAISE):
                fold_prob = v_beliefs.ω_folds_to_aggression.projected_probability
                call_prob = 0.7 * (1 - fold_prob)  # Simplified
                raise_prob = 0.3 * (1 - fold_prob)

                results[villain.player_id] = {
                    "fold": {
                        "probability": fold_prob,
                        "outcome_ev": game_state.pot  # Win current pot
                    },
                    "call": {
                        "probability": call_prob,
                        "outcome_ev": self._ev_if_called(action, game_state, v_beliefs)
                    },
                    "raise": {
                        "probability": raise_prob,
                        "outcome_ev": self._ev_if_raised(action, game_state, v_beliefs)
                    }
                }

        # Compute weighted EV across all villain responses
        total_ev = 0
        for v_id, responses in results.items():
            for response, data in responses.items():
                total_ev += data["probability"] * data["outcome_ev"]

        results["overall_score"] = total_ev
        return results

    def _select_best_rollout(self,
                             rollouts: list[StrategyRollout]) -> StrategyRollout:
        """
        Select best rollout considering verification signals.

        Scoring combines:
        - EV estimate (primary)
        - Back-verification score
        - Confidence in estimates
        - GTO alignment (as safety factor)
        """
        def score_rollout(r: StrategyRollout) -> float:
            v = r.verification

            # Base score is EV estimate
            score = v.get("ev_estimate", 0)

            # Weight by confidence
            score *= v.get("ev_confidence", 0.5)

            # Bonus for strong back-verification
            score += 0.1 * v.get("back_verify_score", 0)

            # Small bonus for GTO alignment (safety)
            score += 0.05 * v.get("gto_alignment", 0)

            return score

        return max(rollouts, key=score_rollout)

    def _has_converged(self, prev: StrategyRollout,
                       curr: StrategyRollout) -> bool:
        """Check if reasoning has converged (same action, similar EV)"""
        if prev is None:
            return False

        if prev.action.action_type != curr.action.action_type:
            return False

        # Check if amounts are close (within 10%)
        if prev.action.amount and curr.action.amount:
            diff = abs(prev.action.amount - curr.action.amount)
            if diff / max(prev.action.amount, 1) > 0.1:
                return False

        return True
```

### Accumulated Knowledge Base

TRT emphasizes learning from reasoning iterations. For poker:

```python
class AccumulatedKnowledge:
    """
    Knowledge accumulated across hands and sessions.

    Stores patterns of successful/unsuccessful strategic choices
    to inform future reasoning iterations.
    """

    def __init__(self):
        # Strategy effectiveness by situation
        self.strategy_outcomes: dict[str, list[StrategyOutcome]] = defaultdict(list)

        # Exploitation patterns that worked
        self.successful_exploits: list[ExploitPattern] = []

        # Situations where GTO outperformed exploitation
        self.gto_better_situations: list[SituationPattern] = []

        # Back-verification failures (predicted X, actual Y)
        self.prediction_errors: list[PredictionError] = []

    def get_relevant(self, game_state: GameState,
                     strategy: str) -> str:
        """
        Retrieve relevant accumulated knowledge for current decision.

        Returns natural language summary for LLM context.
        """
        relevant = []

        # Find similar past situations
        similar = self._find_similar_situations(game_state)

        for sit in similar[:3]:  # Top 3 most similar
            if strategy in sit.strategy_outcomes:
                outcome = sit.strategy_outcomes[strategy]
                relevant.append(
                    f"Similar spot ({sit.description}): {strategy} strategy "
                    f"resulted in {outcome.result} ({outcome.ev_diff:+.1f} BB vs GTO)"
                )

        # Add relevant exploitation patterns
        if strategy == "Exploit":
            patterns = self._get_relevant_exploits(game_state)
            for p in patterns[:2]:
                relevant.append(
                    f"Exploit pattern: {p.description} worked {p.success_rate:.0%} "
                    f"of the time against {p.opponent_type}"
                )

        return "\n".join(relevant) if relevant else "No relevant prior knowledge."

    def record_decision(self, game_state: GameState,
                        rollout: StrategyRollout) -> None:
        """Record this decision for future reference (updated after hand completes)"""
        # Store for later update when we know the outcome
        self._pending_decisions.append({
            "game_state": game_state,
            "rollout": rollout,
            "timestamp": datetime.now()
        })

    def update_outcome(self, hand_result: HandResult) -> None:
        """Update knowledge base with actual hand outcome"""
        # Match pending decision to result
        # Calculate actual EV vs predicted
        # Update strategy effectiveness stats
        # Flag any prediction errors for learning
        ...
```

### Self-Verification Without External Feedback

The key TRT insight is that verification doesn't require ground truth. For poker:

| Verification Type | How It Works in Poker |
|-------------------|----------------------|
| **GTO Consistency** | Compare action to pre-computed GTO baseline |
| **EV Calculation** | Monte Carlo equity × pot math gives expected value |
| **Back-Verification** | "If I bet, villain folds X%, calls Y%, raises Z% → weighted EV" |
| **Internal Consistency** | Does the action match the stated reasoning? |
| **Comparative** | Is exploit EV > GTO EV given our belief confidence? |

```python
class SelfVerifier:
    """Verify poker decisions without external ground truth"""

    def verify_action(self, action: Action, game_state: GameState,
                      beliefs: TableBeliefs, reasoning: str) -> VerificationResult:
        """
        Multi-signal verification inspired by TRT.
        """
        signals = {}

        # 1. GTO alignment (0-1, 1 = exact match)
        gto = self.gto_advisor.get_advice(game_state)
        signals["gto_alignment"] = self._action_similarity(action, gto.action)

        # 2. Mathematical EV verification
        ev_calc = self._calculate_ev(action, game_state, beliefs)
        signals["ev"] = ev_calc["expected_value"]
        signals["ev_confidence"] = ev_calc["confidence"]

        # 3. Back-verification (simulate responses)
        back_verify = self._simulate_villain_responses(action, game_state, beliefs)
        signals["back_verify_ev"] = back_verify["weighted_ev"]
        signals["worst_case_ev"] = back_verify["worst_case"]

        # 4. Reasoning consistency
        signals["reasoning_consistent"] = self._check_reasoning_consistency(
            action, reasoning, game_state
        )

        # 5. Belief-action coherence
        # Does the action make sense given our beliefs?
        signals["belief_coherent"] = self._check_belief_coherence(
            action, beliefs, game_state
        )

        return VerificationResult(
            signals=signals,
            overall_confidence=self._aggregate_confidence(signals),
            warnings=self._generate_warnings(signals)
        )
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

### Phase 3: LLM Agent Core with TRT (Week 3)
**Goal**: ReAct-style reasoning loop with LLM, enhanced by Test-time Recursive Thinking

This phase implements the core agent with TRT-inspired iterative self-improvement from [arXiv:2602.03094](https://arxiv.org/abs/2602.03094).

**Key TRT Integration Points**:
1. **Strategy-conditioned rollouts** - Generate candidate actions under GTO/Exploit/Defensive strategies
2. **Self-verification** - Verify actions against GTO baseline, EV calculations, back-verification
3. **Iterative refinement** - Refine decision until convergence
4. **Accumulated knowledge** - Learn from past strategy outcomes

**Tasks**:

1. **Design prompt template** (with TRT strategy conditioning)
```
You are an expert poker player making decisions in 6-Max No-Limit
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
│   │   ├── action_parser.py    # Parse LLM output to actions
│   │   └── trt/                # NEW: Test-time Recursive Thinking
│   │       ├── __init__.py
│   │       ├── trt_decision.py     # TRT-enhanced decision maker
│   │       ├── strategy_rollouts.py # Strategy-conditioned generation
│   │       ├── self_verifier.py    # Self-verification without ground truth
│   │       ├── back_verifier.py    # Back-verification of actions
│   │       └── accumulated_knowledge.py # Knowledge base across iterations
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
│   ├── test_beliefs/           # Belief system tests
│   │   ├── test_opinion.py
│   │   ├── test_fusion.py
│   │   └── test_updater.py
│   ├── test_trt/               # NEW: TRT reasoning tests
│   │   ├── test_strategy_rollouts.py
│   │   ├── test_self_verifier.py
│   │   ├── test_back_verifier.py
│   │   └── test_convergence.py
│   ├── test_tools/
│   ├── test_memory/
│   ├── test_engine/            # 6-max engine tests
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
│   ├── beliefs/                # Serialized belief states
│   └── knowledge/              # NEW: TRT accumulated knowledge
│       ├── strategy_outcomes.json
│       ├── exploit_patterns.json
│       └── prediction_errors.json
│
├── configs/
│   ├── agent_config.yaml       # Agent configuration
│   ├── belief_priors.yaml      # Base rates for beliefs
│   └── trt_config.yaml         # NEW: TRT parameters (iterations, strategies)
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
