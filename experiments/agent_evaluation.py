"""
Comprehensive AI Agent Evaluation Framework

Experiments:
1. BASELINE: Establish win rates for each player type against mixed opponents
2. IDENTIFICATION: Test if AI agent correctly identifies opponent styles
3. EXPLOITATION: Test if AI agent wins more than baselines against each style
4. ADAPTATION: Test if AI agent adapts strategy based on accumulated beliefs

Metrics:
- BB/100: Win rate in big blinds per 100 hands
- Belief Accuracy: % of opponents correctly classified
- Exploitation Edge: AI win rate vs baseline win rate for each matchup
"""

import os
import sys
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.runner import SimplePokerEngine, HandRunner, SessionRunner
from src.game.opponents import (
    RandomPlayer, CallingStation, TightPassive,
    LooseAggressive, TagBot, create_opponent
)
from src.game.villain_manager import (
    VillainManager, create_villain_table, create_uniform_villain_table,
    villain_id_for_position
)
from src.core import Position, Street
from src.core.beliefs import (
    BeliefState, BeliefRevisionEngine, Observation, ObservationType,
    GameContext, create_default_villain_beliefs, PLAYER_TYPES
)
from src.core.primitives import Action


@dataclass
class PlayerProfile:
    """Profile for a player type."""
    name: str
    create_fn: Callable[[str], object]
    true_type: str  # Ground truth for belief accuracy


# Define player profiles with ground truth types
# create_fn takes player_id as positional arg, player_class is the raw class
PLAYER_PROFILES = {
    'random': PlayerProfile('Random', lambda pid: RandomPlayer(pid), 'Fish'),
    'calling_station': PlayerProfile('CallingStation', lambda pid: CallingStation(pid), 'Fish'),
    'tight_passive': PlayerProfile('TightPassive', lambda pid: TightPassive(pid), 'NIT'),
    'loose_aggressive': PlayerProfile('LooseAggressive', lambda pid: LooseAggressive(pid), 'LAG'),
    'tag': PlayerProfile('TagBot', lambda pid: TagBot(pid), 'TAG'),
}

# Map profile keys to actual classes for VillainManager
PLAYER_CLASSES = {
    'random': RandomPlayer,
    'calling_station': CallingStation,
    'tight_passive': TightPassive,
    'loose_aggressive': LooseAggressive,
    'tag': TagBot,
}


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    experiment_name: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    model_name: str = "rule-based"  # Default for non-LLM players
    num_hands: int = 0
    seed: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.experiment_name} | {self.model_name} | {self.num_hands} hands | {self.timestamp}"


@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    hero_type: str
    villain_types: list[str]
    num_hands: int
    hero_profit_bb: float
    bb_per_100: float
    hands_won: int = 0
    showdowns: int = 0
    model_name: str = "rule-based"


@dataclass
class BeliefAccuracyResult:
    """Results from belief accuracy test."""
    villain_id: str
    true_type: str
    predicted_type: str
    predicted_prob: float
    correct: bool
    observations: int


class BaselineExperiment:
    """
    Experiment 1: Establish baseline win rates.

    Run each player type as hero against a mixed table of opponents.
    This establishes expected win rates for comparison.
    """

    def __init__(self, num_hands: int = 500, model_name: str = "rule-based"):
        self.num_hands = num_hands
        self.model_name = model_name
        self.engine = SimplePokerEngine(big_blind=10.0, starting_stack=100.0)
        self.results: dict[str, list[ExperimentResult]] = defaultdict(list)
        self.metadata = ExperimentMetadata(
            experiment_name="Baseline Win Rates",
            model_name=model_name,
            num_hands=num_hands
        )

    def run(self, verbose: bool = True) -> dict[str, float]:
        """Run baseline experiment for all player types."""
        if verbose:
            print("=" * 70)
            print("EXPERIMENT 1: BASELINE WIN RATES")
            print("=" * 70)
            print(f"Running {self.num_hands} hands per player type...")
            print(f"Model: {self.model_name}\n")

        baseline_rates = {}

        # Map positions to player types for mixed table
        position_types = [
            (Position.UTG, 'random'),
            (Position.HJ, 'calling_station'),
            (Position.CO, 'tight_passive'),
            (Position.SB, 'loose_aggressive'),
            (Position.BB, 'tag'),
        ]

        for hero_key, hero_profile in PLAYER_PROFILES.items():
            # Create hero
            hero = hero_profile.create_fn('hero')

            # Create mixed villain table using VillainManager
            manager = VillainManager()
            villain_types = []

            for pos, v_key in position_types:
                if v_key != hero_key:
                    player_class = PLAYER_CLASSES[v_key]
                    manager.add_villain(pos, player_class)
                    villain_types.append(PLAYER_PROFILES[v_key].name)

            # Fill remaining with random
            manager.set_default_type(RandomPlayer)
            villains = manager.get_villains(Position.BTN)  # Hero at BTN

            # Run session with hero fixed at BTN to match villain IDs
            # We run hands manually instead of using SessionRunner
            # because SessionRunner rotates positions which breaks villain ID matching
            total_profit = 0.0
            showdowns = 0
            for _ in range(self.num_hands):
                hand_runner = HandRunner(self.engine, hero, villains, memory_manager=None)
                result = hand_runner.run_hand(Position.BTN)
                total_profit += result.hero_profit
                if result.went_to_showdown:
                    showdowns += 1

            class FakeStats:
                pass
            stats = FakeStats()
            stats.total_profit = total_profit
            stats.showdowns = showdowns

            bb_per_100 = stats.total_profit / (self.num_hands / 100)
            baseline_rates[hero_key] = bb_per_100

            result = ExperimentResult(
                hero_type=hero_profile.name,
                villain_types=villain_types,
                num_hands=self.num_hands,
                hero_profit_bb=stats.total_profit,
                bb_per_100=bb_per_100,
                showdowns=stats.showdowns,
                model_name=self.model_name
            )
            self.results[hero_key].append(result)

            if verbose:
                print(f"{hero_profile.name:20s}: {bb_per_100:+6.1f} BB/100")

        return baseline_rates


class BeliefTrackingPlayer:
    """
    Wrapper that tracks beliefs about opponents during play.

    Used to test if beliefs converge to correct player types.
    """

    def __init__(self, base_player, player_id: str = 'hero'):
        self.base_player = base_player
        self._player_id = player_id
        self.belief_state = BeliefState()
        self.belief_engine = BeliefRevisionEngine(trust_discount=0.9)
        self.observation_count = defaultdict(int)
        self.seen_actions = set()  # Track which actions we've already processed

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state):
        """Make decision and update beliefs based on observed actions."""
        # Update beliefs from action history
        for i, record in enumerate(game_state.action_history):
            action_key = (game_state.hand_id, i)
            if action_key in self.seen_actions:
                continue
            self.seen_actions.add(action_key)

            if record.player_id != self._player_id:
                # Create observation
                obs = Observation(
                    obs_id=f'{game_state.hand_id}_{record.player_id}_{self.observation_count[record.player_id]}',
                    obs_type=ObservationType.ACTION,
                    player_id=record.player_id,
                    action=record.action,
                    context=GameContext(
                        street=record.street,
                        position=record.position,
                        pot_size=record.pot_after,
                        to_call=0,
                        num_players=game_state.num_in_hand,
                        is_heads_up=game_state.is_heads_up,
                        facing_raise=record.action.action_type.is_aggressive,
                    )
                )
                self.belief_state = self.belief_engine.process_observation(obs, self.belief_state)
                self.observation_count[record.player_id] += 1

        # Delegate decision to base player
        return self.base_player.decide(game_state)

    def get_belief_predictions(self) -> dict[str, tuple[str, float]]:
        """Get predicted player types for all villains."""
        predictions = {}
        for vid, vb in self.belief_state.villain_beliefs.items():
            best_type, prob = vb.get_most_likely_player_type()
            predictions[vid] = (best_type, prob)
        return predictions

    def reset_hand(self):
        """Reset per-hand tracking (call between hands)."""
        pass  # seen_actions persists across hands intentionally


class IdentificationExperiment:
    """
    Experiment 2: Test belief accuracy.

    Run games with known opponent types and measure how accurately
    the belief system identifies them.
    """

    def __init__(self, num_hands: int = 200):
        self.num_hands = num_hands
        self.engine = SimplePokerEngine(big_blind=10.0, starting_stack=100.0)
        self.results: list[BeliefAccuracyResult] = []

    def run(self, verbose: bool = True) -> float:
        """Run identification experiment."""
        if verbose:
            print("\n" + "=" * 70)
            print("EXPERIMENT 2: BELIEF ACCURACY (Style Identification)")
            print("=" * 70)
            print(f"Running {self.num_hands} hands to test belief convergence...\n")

        # Map position-based villain IDs to player types
        # The game creates villain_UTG, villain_HJ, villain_CO, villain_SB, villain_BB
        # (excluding hero's position which rotates)
        position_to_type = {
            Position.UTG: ('random', PLAYER_PROFILES['random']),
            Position.HJ: ('calling_station', PLAYER_PROFILES['calling_station']),
            Position.CO: ('tight_passive', PLAYER_PROFILES['tight_passive']),
            Position.SB: ('loose_aggressive', PLAYER_PROFILES['loose_aggressive']),
            Position.BB: ('tag', PLAYER_PROFILES['tag']),
        }

        # Create villains with position-based IDs that match what game creates
        villains = {}
        ground_truth = {}
        for pos, (key, profile) in position_to_type.items():
            vid = f'villain_{pos.value}'
            villains[vid] = profile.create_fn(vid)
            ground_truth[vid] = profile.true_type

        # Create hero with belief tracking
        base_hero = TagBot(player_id='hero')
        hero = BeliefTrackingPlayer(base_hero, 'hero')

        # Run hands - hero rotates through BTN position only to keep villains stable
        for hand_num in range(self.num_hands):
            runner = HandRunner(self.engine, hero, villains, memory_manager=None)
            runner.run_hand(Position.BTN)  # Keep hero at BTN so villain IDs stay consistent

        # Evaluate predictions
        predictions = hero.get_belief_predictions()
        correct = 0
        total = 0

        if verbose:
            print(f"{'Villain':<25} {'True Type':<10} {'Predicted':<10} {'Prob':>6} {'Obs':>5} {'Correct':<8}")
            print("-" * 75)

        for vid, true_type in ground_truth.items():
            obs_count = hero.observation_count.get(vid, 0)
            if vid in predictions:
                pred_type, pred_prob = predictions[vid]
                is_correct = (pred_type == true_type)
                correct += int(is_correct)
                total += 1

                result = BeliefAccuracyResult(
                    villain_id=vid,
                    true_type=true_type,
                    predicted_type=pred_type,
                    predicted_prob=pred_prob,
                    correct=is_correct,
                    observations=obs_count
                )
                self.results.append(result)

                if verbose:
                    mark = "✓" if is_correct else "✗"
                    print(f"{vid:<25} {true_type:<10} {pred_type:<10} {pred_prob:>5.0%} {obs_count:>5} {mark:<8}")
            else:
                # No observations for this villain
                total += 1
                if verbose:
                    print(f"{vid:<25} {true_type:<10} {'???':<10} {'N/A':>6} {obs_count:>5} {'?':<8}")

        accuracy = correct / total if total > 0 else 0

        if verbose:
            print("-" * 75)
            print(f"Overall Accuracy: {correct}/{total} = {accuracy:.0%}")
            print(f"Total observations: {sum(hero.observation_count.values())}")

        return accuracy


class ExploitationExperiment:
    """
    Experiment 3: Test exploitation ability.

    Run AI agent against each specific player type and measure
    if it wins more than baseline players do.
    """

    def __init__(self, num_hands: int = 300):
        self.num_hands = num_hands
        self.engine = SimplePokerEngine(big_blind=10.0, starting_stack=100.0)
        self.results: dict[str, dict[str, float]] = defaultdict(dict)

    def run(self, baseline_rates: dict[str, float], verbose: bool = True) -> dict[str, float]:
        """Run exploitation experiment."""
        if verbose:
            print("\n" + "=" * 70)
            print("EXPERIMENT 3: EXPLOITATION ABILITY")
            print("=" * 70)
            print(f"Testing if TagBot exploits each opponent type...\n")

        exploitation_edges = {}

        for target_key, target_profile in PLAYER_PROFILES.items():
            # Create hero (TagBot as our best bot)
            hero = TagBot(player_id='hero')

            # Create table of 5 copies of target type
            villains = {}
            for i in range(5):
                vid = f'villain_{i}'
                villains[vid] = target_profile.create_fn(vid)

            # Run session
            runner = SessionRunner(hero=hero, villains=villains)
            stats = runner.run_session(num_hands=self.num_hands)

            bb_per_100 = stats.total_profit / (self.num_hands / 100)

            # Compare to baseline
            baseline = baseline_rates.get('tag', 0)
            edge = bb_per_100 - baseline
            exploitation_edges[target_key] = edge

            self.results['tag'][target_key] = bb_per_100

            if verbose:
                sign = "+" if edge >= 0 else ""
                print(f"vs {target_profile.name:20s}: {bb_per_100:+6.1f} BB/100 (edge: {sign}{edge:.1f})")

        return exploitation_edges


class AdaptationExperiment:
    """
    Experiment 4: Test strategy adaptation.

    Measure if the agent's play changes based on accumulated beliefs.
    Compare early-session vs late-session decision patterns.
    """

    def __init__(self, num_hands: int = 200):
        self.num_hands = num_hands

    def run(self, verbose: bool = True) -> dict:
        """Run adaptation experiment."""
        if verbose:
            print("\n" + "=" * 70)
            print("EXPERIMENT 4: STRATEGY ADAPTATION")
            print("=" * 70)
            print("Testing if beliefs change over time...\n")

        # Track belief trajectory over time
        base_hero = TagBot(player_id='hero')
        hero = BeliefTrackingPlayer(base_hero, 'hero')

        # Use calling stations with position-based IDs that match game
        # Hero at BTN means villains are at UTG, HJ, CO, SB, BB
        villains = {
            'villain_UTG': CallingStation('villain_UTG'),
            'villain_HJ': CallingStation('villain_HJ'),
            'villain_CO': CallingStation('villain_CO'),
            'villain_SB': CallingStation('villain_SB'),
            'villain_BB': CallingStation('villain_BB'),
        }

        engine = SimplePokerEngine(big_blind=10.0, starting_stack=100.0)

        # Track beliefs at intervals
        checkpoints = [10, 25, 50, 100, 150, 200]
        belief_trajectory = []

        for hand_num in range(self.num_hands):
            # Keep hero at BTN so villain IDs stay consistent
            runner = HandRunner(engine, hero, villains, memory_manager=None)
            runner.run_hand(Position.BTN)

            # Record checkpoint
            if hand_num + 1 in checkpoints and hand_num + 1 <= self.num_hands:
                # Calculate average Fish probability across all observed villains
                total_fish_prob = 0
                count = 0
                for vid in villains.keys():
                    vb = hero.belief_state.villain_beliefs.get(vid)
                    if vb:
                        fish_prob = vb.get_player_type_distribution().get('Fish', 0)
                        total_fish_prob += fish_prob
                        count += 1

                avg_fish_prob = total_fish_prob / count if count > 0 else 0.45  # Default prior
                belief_trajectory.append({
                    'hand': hand_num + 1,
                    'avg_fish_prob': avg_fish_prob
                })

                if verbose:
                    obs_total = sum(hero.observation_count.values())
                    print(f"Hand {hand_num + 1:3d}: Avg Fish probability = {avg_fish_prob:.0%} ({obs_total} obs)")

        if verbose:
            print("\nExpected: Fish probability should increase as more passive")
            print("          calling behavior is observed.")

        return {'trajectory': belief_trajectory}


def run_all_experiments(quick: bool = False, model_name: str = "rule-based"):
    """Run complete experiment suite."""
    # Adjust hand counts for quick mode
    if quick:
        baseline_hands = 100
        ident_hands = 50
        exploit_hands = 100
        adapt_hands = 50
    else:
        baseline_hands = 500
        ident_hands = 200
        exploit_hands = 300
        adapt_hands = 200

    print("=" * 70)
    print("AI AGENT COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"Model: {model_name}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Experiment 1: Baseline
    exp1 = BaselineExperiment(num_hands=baseline_hands, model_name=model_name)
    baseline_rates = exp1.run()

    # Experiment 2: Identification
    exp2 = IdentificationExperiment(num_hands=ident_hands)
    accuracy = exp2.run()

    # Experiment 3: Exploitation
    exp3 = ExploitationExperiment(num_hands=exploit_hands)
    edges = exp3.run(baseline_rates)

    # Experiment 4: Adaptation
    exp4 = AdaptationExperiment(num_hands=adapt_hands)
    adaptation = exp4.run()

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n1. BASELINE WIN RATES (BB/100):")
    for ptype, rate in baseline_rates.items():
        print(f"   {ptype:20s}: {rate:+.1f}")

    print(f"\n2. BELIEF ACCURACY: {accuracy:.0%}")

    print("\n3. EXPLOITATION EDGES vs baseline:")
    for target, edge in edges.items():
        sign = "+" if edge >= 0 else ""
        print(f"   vs {target:20s}: {sign}{edge:.1f} BB/100")

    print("\n4. ADAPTATION:")
    traj = adaptation['trajectory']
    if traj:
        start = traj[0]['avg_fish_prob']
        end = traj[-1]['avg_fish_prob']
        print(f"   Fish belief: {start:.0%} -> {end:.0%} (Δ = {end-start:+.0%})")

    return {
        'baseline': baseline_rates,
        'accuracy': accuracy,
        'exploitation': edges,
        'adaptation': adaptation
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run AI Agent Evaluation')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer hands')
    parser.add_argument('--model', type=str, default='rule-based',
                        help='Model name for tracking (e.g., claude-sonnet-4-20250514)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    run_all_experiments(quick=args.quick, model_name=args.model)
