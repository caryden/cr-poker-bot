"""Run LLM hero with full reasoning trace visible.

Usage:
    python run_llm_traced.py                    # Full tools
    python run_llm_traced.py --exclude gto      # Ablation: no GTO
    python run_llm_traced.py --hands 30         # More hands
"""

import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.tournament import TournamentPlayer, TournamentRunner, SimpleLLMPlayer
from src.game.opponents import CallingStation, TightPassive, LooseAggressive, TagBot, RandomPlayer

# Bot menu — canonical names
STRATEGY_BOTS = {
    "FISH": CallingStation,
    "NIT": TightPassive,
    "LAG": LooseAggressive,
    "TAG": TagBot,
    "RANDOM": RandomPlayer,
}


def main():
    parser = argparse.ArgumentParser(description="Run LLM hero with trace")
    parser.add_argument('--hands', type=int, default=10, help='Number of hands')
    parser.add_argument('--exclude', nargs='*', default=[], help='Tools to exclude (ablation)')
    args = parser.parse_args()

    excluded_tools = set(args.exclude) if args.exclude else None

    starting_stack = 1000.0
    starting_blind = 10.0

    # Pick 5 bots randomly from the menu — use neutral IDs so LLM can't cheat
    bot_names = list(STRATEGY_BOTS.keys())
    selected = random.choices(bot_names, k=5)
    villains = []  # (neutral_id, strategy_name, cls)
    for i, name in enumerate(selected, 1):
        villains.append((f"Player_{i}", name, STRATEGY_BOTS[name]))

    print("=" * 70)
    print("TRACED LLM TABLE")
    print("=" * 70)
    print(f"Hero:     Hero (SimpleLLMPlayer trace=True, claude-sonnet-4-5)")
    if excluded_tools:
        print(f"Ablation: excluding {excluded_tools}")
    for pid, strat, cls in villains:
        print(f"Villain:  {pid} = {strat} ({cls.__name__})")
    print(f"Stack:    {starting_stack}")
    print(f"Blinds:   {starting_blind/2:.0f}/{starting_blind:.0f}")
    print(f"Max hands: {args.hands}")
    print("=" * 70)

    llm_player = SimpleLLMPlayer("Hero", model="claude-sonnet-4-5",
                                  excluded_tools=excluded_tools, trace=True)

    players = [TournamentPlayer("Hero", llm_player, starting_stack)]
    for pid, strat, cls in villains:
        players.append(TournamentPlayer(pid, cls(pid), starting_stack))

    runner = TournamentRunner(
        players=players,
        starting_stack=starting_stack,
        starting_blind=starting_blind,
        blind_increase_hands=15,
        max_hands=args.hands,
        verbose=True,
    )

    result = runner.run()

    print(f"\n{'=' * 70}")
    print(f"Hero made {llm_player.decision_count} decisions across {result.hands_played} hands")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
