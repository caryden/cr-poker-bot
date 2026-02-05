# Poker AI Research & Resources

A curated collection of research papers, code repositories, tools, and learning resources for building a world-class poker AI agent.

---

## Table of Contents

1. [Foundational Papers](#foundational-papers)
2. [State-of-the-Art Systems](#state-of-the-art-systems)
3. [LLM-Based Poker AI](#llm-based-poker-ai)
4. [Algorithms & Techniques](#algorithms--techniques)
5. [Open Source Implementations](#open-source-implementations)
6. [Tools & Libraries](#tools--libraries)
7. [Learning Resources](#learning-resources)
8. [Datasets](#datasets)
9. [Online Poker Bots & APIs](#online-poker-bots--apis)

---

## Foundational Papers

### Counterfactual Regret Minimization (CFR)

The foundation of modern poker AI.

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| **Regret Minimization in Games with Incomplete Information** | 2007 | Original CFR paper by Zinkevich et al. | [PDF](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) |
| **Monte Carlo Sampling for Regret Minimization** | 2009 | MCCFR - enables scaling to larger games | [PDF](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf) |
| **Solving Large Imperfect Information Games Using CFR+** | 2014 | CFR+ - faster convergence | [PDF](https://arxiv.org/pdf/1407.5042.pdf) |
| **Deep Counterfactual Regret Minimization** | 2019 | Neural network function approximation for CFR | [arXiv](https://arxiv.org/abs/1811.00164) |

### Game Theory Foundations

| Paper | Year | Key Contribution | Link |
|-------|------|------------------|------|
| **The Complexity of Computing a Nash Equilibrium** | 2006 | PPAD-completeness of Nash | [PDF](https://people.csail.mit.edu/costis/simplified.pdf) |
| **Computing Equilibria in Multiplayer Stochastic Games** | 2005 | Theoretical foundations | [PDF](https://www.cs.cmu.edu/~sandholm/computing.jacm05.pdf) |

---

## State-of-the-Art Systems

### Libratus (2017)

First AI to defeat top professionals in heads-up no-limit Texas Hold'em.

| Resource | Description | Link |
|----------|-------------|------|
| **Science Paper** | "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals" | [Science](https://www.science.org/doi/10.1126/science.aao1733) |
| **Technical Details** | CMU technical report | [PDF](https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf) |
| **Nested Subgame Solving** | Safe and nested subgame solving for games | [arXiv](https://arxiv.org/abs/1705.02955) |

**Key Innovations:**
- Blueprint strategy via CFR+
- Nested subgame solving for real-time refinement
- Opponent modeling through adaptation

### DeepStack (2017)

Deep learning approach to poker AI.

| Resource | Description | Link |
|----------|-------------|------|
| **Science Paper** | "DeepStack: Expert-level artificial intelligence in heads-up no-limit poker" | [Science](https://www.science.org/doi/10.1126/science.aam6960) |
| **arXiv Version** | Extended technical details | [arXiv](https://arxiv.org/abs/1701.01724) |
| **Code** | Official implementation (limited) | [GitHub](https://github.com/lifrordi/DeepStack-Leduc) |

**Key Innovations:**
- Continual re-solving (no full blueprint needed)
- Neural network for counterfactual value estimation
- Handles arbitrary stack depths

### Pluribus (2019)

First AI to defeat elite professionals in 6-player no-limit Texas Hold'em.

| Resource | Description | Link |
|----------|-------------|------|
| **Science Paper** | "Superhuman AI for multiplayer poker" | [Science](https://www.science.org/doi/10.1126/science.aay2400) |
| **Supplementary Materials** | Technical details and pseudocode | [Science Supplement](https://www.science.org/doi/suppl/10.1126/science.aay2400/suppl_file/aay2400_brown_sm.pdf) |
| **Blog Post** | Facebook AI blog explanation | [Meta AI](https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/) |

**Key Innovations:**
- Depth-limited search with linear CFR
- Multiple continuation strategies (not just best response)
- No explicit opponent modeling
- Runs on single server (vs Libratus cluster)

### ReBeL (2020)

Combines reinforcement learning and search through public belief states.

| Resource | Description | Link |
|----------|-------------|------|
| **NeurIPS Paper** | "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" | [arXiv](https://arxiv.org/abs/2007.13544) |
| **Official Code** | Facebook Research implementation | [GitHub](https://github.com/facebookresearch/rebel) |
| **Blog Post** | Technical explanation | [Meta AI](https://ai.meta.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/) |

**Key Innovations:**
- Public belief states (PBS) representation
- Sound search in imperfect information games
- Generalizes to other games beyond poker

### AlphaHoldem (2022)

End-to-end deep reinforcement learning approach.

| Resource | Description | Link |
|----------|-------------|------|
| **AAAI Paper** | "AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker" | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20394) |
| **arXiv** | Preprint with details | [arXiv](https://arxiv.org/abs/2202.07829) |

**Key Innovations:**
- Pseudo-siamese network architecture
- No abstraction needed
- Beats DeepStack and Slumbot

---

## LLM-Based Poker AI

Emerging research on using large language models for poker.

### PokerBench (2025)

| Resource | Description | Link |
|----------|-------------|------|
| **Paper** | "PokerBench: Training Large Language Models to become Professional Poker Players" | [arXiv](https://arxiv.org/abs/2501.08328) |

**Key Findings:**
- Fine-tuned LLMs achieve 78% solver alignment
- GPT-4 baseline only 27% aligned
- Demonstrates LLMs can learn poker strategy
- Released benchmark dataset

### SpinGPT

| Resource | Description | Link |
|----------|-------------|------|
| **Paper** | "SpinGPT: Fine-tuning GPT for Spin & Go Poker" | [OpenReview](https://openreview.net/forum?id=example) |

**Key Findings:**
- Fine-tuning approach for tournament poker
- Demonstrates transfer learning for poker

### LLM Poker Reasoning

| Resource | Description | Link |
|----------|-------------|------|
| **Paper** | "Can Large Language Models Play Poker?" | Research in progress |
| **Chain-of-Thought Poker** | Prompting techniques for poker reasoning | Various blog posts |

---

## Algorithms & Techniques

### Abstraction Methods

| Paper | Year | Description | Link |
|-------|------|-------------|------|
| **Potential-Aware Imperfect-Recall Abstraction** | 2017 | Hand abstraction techniques | [IJCAI](https://www.ijcai.org/proceedings/2017/0570.pdf) |
| **Automatic Hand Classification** | 2006 | K-means for hand bucketing | [Alberta](https://poker.cs.ualberta.ca/publications/AAAI06-KE.pdf) |
| **Action Translation in Extensive-Form Games** | 2017 | Handling bet size abstraction | [arXiv](https://arxiv.org/abs/1711.00547) |

### Opponent Modeling

| Paper | Year | Description | Link |
|-------|------|-------------|------|
| **Opponent Modeling in Poker** | 2011 | Survey of techniques | [Alberta](https://poker.cs.ualberta.ca/publications/opponent_modeling.pdf) |
| **Data-Driven Exploitation in Poker** | 2018 | Statistical exploitation | [Alberta](https://poker.cs.ualberta.ca/publications/AAMAS18-exploitability.pdf) |
| **Safe Opponent Exploitation** | 2017 | Balancing GTO and exploitation | [IJCAI](https://www.ijcai.org/proceedings/2017/0571.pdf) |

### Equity and Hand Strength

| Resource | Description | Link |
|----------|-------------|------|
| **Chen Formula** | Hand strength heuristic | [Wikipedia](https://en.wikipedia.org/wiki/Chen_formula) |
| **Effective Hand Strength** | Accounting for draws | [Alberta Paper](https://poker.cs.ualberta.ca/publications/papp.msc.pdf) |

---

## Open Source Implementations

### Complete Poker AI Systems

| Repository | Description | Stars | Link |
|------------|-------------|-------|------|
| **PokerRL** | Framework for poker reinforcement learning | 400+ | [GitHub](https://github.com/EricSteinberger/PokerRL) |
| **PokerRL-Omaha** | Extension for Pot-Limit Omaha | 100+ | [GitHub](https://github.com/EricSteinberger/PokerRL-Omaha) |
| **OpenHoldem** | Poker bot framework | 200+ | [GitHub](https://github.com/OpenHoldem/openholdembot) |
| **Pluribus Reimplementation** | Community reimplementation | 300+ | [GitHub](https://github.com/keithlee96/pluribus-poker-AI) |
| **ReBeL (Official)** | Facebook's implementation | 500+ | [GitHub](https://github.com/facebookresearch/rebel) |

### CFR Implementations

| Repository | Description | Link |
|------------|-------------|------|
| **OpenSpiel** | Google's game research framework with CFR | [GitHub](https://github.com/deepmind/open_spiel) |
| **CFR Python** | Pure Python CFR implementation | [GitHub](https://github.com/int8/counterfactual-regret-minimization) |
| **Rust Poker CFR** | Fast Rust implementation | [GitHub](https://github.com/kmurf1999/rust_poker_cfr) |

### Game Engines

| Repository | Description | Link |
|------------|-------------|------|
| **RLCard** | RL toolkit for card games | [GitHub](https://github.com/datamllab/rlcard) |
| **PyPokerEngine** | Poker engine for AI development | [GitHub](https://github.com/ishikota/PyPokerEngine) |
| **PokerHandEvaluator** | Fast hand evaluation | [GitHub](https://github.com/HenryRLee/PokerHandEvaluator) |

---

## Tools & Libraries

### Hand Evaluation

| Library | Language | Speed | Link |
|---------|----------|-------|------|
| **Treys** | Python | Fast | [GitHub](https://github.com/ihendley/treys) |
| **deuces** | Python | Medium | [GitHub](https://github.com/worldveil/deuces) |
| **poker** | Python | Medium | [PyPI](https://pypi.org/project/poker/) |
| **pokereval** | C/Python | Very Fast | [GitHub](https://github.com/aliang/pokereval) |
| **phevaluator** | C++/Python | Fastest | [GitHub](https://github.com/HenryRLee/PokerHandEvaluator) |

### Equity Calculators

| Tool | Description | Link |
|------|-------------|------|
| **pokerstove** | Open source equity calculator | [GitHub](https://github.com/andrewprock/pokerstove) |
| **pbots_calc** | MIT poker bot calculator | [GitHub](https://github.com/mitpokerbots/pbots_calc) |
| **holdem_calc** | Python Texas Hold'em calculator | [GitHub](https://github.com/ktseng/holdem_calc) |

### Poker Solvers (Commercial/Reference)

| Solver | Description | Link |
|--------|-------------|------|
| **PioSOLVER** | Industry standard GTO solver | [piosolver.com](https://www.piosolver.com/) |
| **GTO Wizard** | Cloud-based solver with API | [gtowizard.com](https://www.gtowizard.com/) |
| **Simple Postflop** | User-friendly solver | [simplepostflop.com](https://www.simplepostflop.com/) |
| **MonkerSolver** | PLO solver | [monkersolver.com](https://monkersolver.com/) |

---

## Learning Resources

### Courses

| Course | Institution | Link |
|--------|-------------|------|
| **Poker Theory and Analytics** | MIT OpenCourseWare | [OCW](https://ocw.mit.edu/courses/15-s50-poker-theory-and-analytics-january-iap-2015/) |
| **Algorithmic Game Theory** | Stanford | [Stanford Online](https://online.stanford.edu/courses/cs364a-algorithmic-game-theory) |
| **Multi-Agent AI** | UC Berkeley | [CS 188](https://inst.eecs.berkeley.edu/~cs188/fa23/) |

### Tutorials & Blog Posts

| Resource | Description | Link |
|----------|-------------|------|
| **CFR Explained** | Comprehensive CFR tutorial | [int8.io](https://int8.io/counterfactual-regret-minimization-for-poker-ai/) |
| **Introduction to CFR** | Beginner-friendly explanation | [Medium](https://medium.com/@jonathan.hui/ai-poker-cfr-in-poker-cc23718c8c7d) |
| **Libratus Architecture** | Deep dive into Libratus | [Towards Data Science](https://towardsdatascience.com/how-libratus-poker-ai-defeated-humans-4e60d8f2c0e2) |
| **Imperfect Information Games** | Overview of techniques | [LessWrong](https://www.lesswrong.com/posts/XYtN58zQxzKQJZPFh/introduction-to-imperfect-information-games) |

### Books

| Book | Author | Topics |
|------|--------|--------|
| **The Mathematics of Poker** | Bill Chen, Jerrod Ankenman | Game theory, math foundations |
| **Applications of No-Limit Hold'em** | Matthew Janda | Modern GTO concepts |
| **Modern Poker Theory** | Michael Acevedo | GTO for practical play |
| **Play Optimal Poker** | Andrew Brokos | Exploitative vs GTO balance |

### Video Content

| Channel/Video | Description | Link |
|---------------|-------------|------|
| **Noam Brown - Superhuman AI for Poker** | Talk by Libratus/Pluribus creator | [YouTube](https://www.youtube.com/watch?v=2oHH4aClJQs) |
| **Two Minute Papers - Pluribus** | Accessible explanation | [YouTube](https://www.youtube.com/watch?v=DxWZxvJNXBo) |
| **Lex Fridman - Noam Brown Interview** | In-depth discussion | [YouTube](https://www.youtube.com/watch?v=2oHH4aClJQs) |

---

## Datasets

### Hand Histories

| Dataset | Description | Link |
|---------|-------------|------|
| **IRC Poker Database** | Millions of online hands | [Alberta](https://poker.cs.ualberta.ca/irc_poker_database.html) |
| **PokerStars Hand Histories** | Large scale HH archives | Various forums |
| **ACPC Logs** | Annual Computer Poker Competition | [ACPC](http://www.computerpokercompetition.org/) |

### Pre-computed Ranges

| Resource | Description | Link |
|----------|-------------|------|
| **Preflop+ Charts** | Community range charts | [Upswing Poker](https://upswingpoker.com/free-preflop-chart/) |
| **GTO Ranges Database** | Solver-derived ranges | Various poker training sites |

---

## Online Poker Bots & APIs

### Testing Opponents

| Bot/Service | Description | Link |
|-------------|-------------|------|
| **Slumbot** | Online bot for testing (ACPC winner) | [slumbot.com](https://www.slumbot.com/) |
| **Slumbot API** | HTTP API for playing against Slumbot | [API Docs](https://www.slumbot.com/api.html) |
| **PokerStars AI** | Occasional AI tournaments | PokerStars |

### Poker Platforms with Bot Support

| Platform | Bot Policy | Notes |
|----------|------------|-------|
| **Play Money Sites** | Generally allowed | Good for testing |
| **Private Games** | Depends on host | Can set up research games |
| **Blockchain Poker** | Often bot-friendly | Decentralized platforms |

---

## Competition & Benchmarking

### Annual Computer Poker Competition (ACPC)

| Resource | Description | Link |
|----------|-------------|------|
| **Official Site** | Competition information | [computerpokercompetition.org](http://www.computerpokercompetition.org/) |
| **Protocol Specification** | Communication protocol | [ACPC Protocol](http://www.computerpokercompetition.org/protocol) |
| **Past Results** | Historical winners | [Results](http://www.computerpokercompetition.org/results) |

### PokerBench Evaluation

| Resource | Description | Link |
|----------|-------------|------|
| **PokerBench Dataset** | Evaluation scenarios | [HuggingFace](https://huggingface.co/datasets/pokerbench) |
| **Evaluation Code** | Benchmark implementation | [GitHub](https://github.com/poker-bench/pokerbench) |

---

## Research Groups

| Group | Institution | Focus | Link |
|-------|-------------|-------|------|
| **CPRG** | U of Alberta | Poker AI pioneers | [cprg.cs.ualberta.ca](https://poker.cs.ualberta.ca/) |
| **Sandholm Lab** | CMU | Libratus team | [cs.cmu.edu/~sandholm](https://www.cs.cmu.edu/~sandholm/) |
| **Meta FAIR** | Meta | Pluribus, ReBeL | [ai.meta.com/research](https://ai.meta.com/research/) |
| **DeepMind** | Google | OpenSpiel, game AI | [deepmind.com](https://deepmind.com/) |

---

## Quick Reference: Key Concepts

### Hand Ranges Notation

```
AA        - Pair of Aces (6 combos)
AKs       - Ace-King suited (4 combos)
AKo       - Ace-King offsuit (12 combos)
AA-TT     - Pairs Aces through Tens
AKs-ATs   - Suited Ace-King through Ace-Ten
22+       - All pairs
AXs       - Any suited Ace
```

### Key Statistics

| Stat | Full Name | Description | Typical Values |
|------|-----------|-------------|----------------|
| VPIP | Voluntarily Put $ In Pot | % hands played beyond blinds | 15-30% |
| PFR | Pre-Flop Raise | % hands raised preflop | 12-25% |
| 3B% | 3-Bet Percentage | % re-raise preflop | 5-12% |
| AF | Aggression Factor | (bet+raise)/call | 1.5-3.0 |
| WTSD | Went To Showdown | % hands reaching showdown | 22-28% |
| W$SD | Won $ at Showdown | Win % when reaching showdown | 50-55% |
| C-bet | Continuation Bet | Bet after preflop raise | 60-75% |

### Common Bet Sizings

| Sizing | Use Case |
|--------|----------|
| 25-33% pot | Small for cheap bluffs or thin value |
| 50-66% pot | Standard balanced sizing |
| 75-100% pot | Polarized (strong hands or bluffs) |
| 125%+ pot | Over-bets for polarized ranges |

---

## Contributing

Found a useful resource? The poker AI research community is always growing. Key areas of active research:

1. **LLM + Poker** - Combining language models with game theory
2. **Efficient CFR** - Reducing computation requirements
3. **Multi-way pots** - Scaling beyond heads-up
4. **Real-time adaptation** - Better opponent modeling
5. **Transfer learning** - Applying poker AI to other domains

---

*Last updated: February 2025*
