"""Run LLM hero vs 5 randomly-selected strategy bots (no RandomPlayer)."""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.tournament import TournamentPlayer, TournamentRunner, SimpleLLMPlayer
from src.game.opponents import CallingStation, TightPassive, LooseAggressive, TagBot

STRATEGY_BOTS = [
    ("Fish", CallingStation),
    ("Nit", TightPassive),
    ("LAG", LooseAggressive),
    ("TAG", TagBot),
]


def main():
    starting_stack = 1000.0
    starting_blind = 10.0

    # Randomly select 5 strategy bots (with replacement)
    selected = random.choices(STRATEGY_BOTS, k=5)

    # Deduplicate names with counters
    name_counts = {}
    villains = []
    for label, cls in selected:
        name_counts[label] = name_counts.get(label, 0) + 1
        pid = f"{label}_{name_counts[label]}"
        villains.append((pid, cls))

    print("=" * 70)
    print("TABLE SETUP")
    print("=" * 70)
    print(f"Hero:     LLM_Hero (SimpleLLMPlayer, claude-sonnet-4-5)")
    for pid, cls in villains:
        print(f"Villain:  {pid} ({cls.__name__})")
    print(f"Stack:    {starting_stack}")
    print(f"Blinds:   {starting_blind/2:.0f}/{starting_blind:.0f}")
    print("=" * 70)
    print()

    llm_player = SimpleLLMPlayer("LLM_Hero", model="claude-sonnet-4-5")

    players = [TournamentPlayer("LLM_Hero", llm_player, starting_stack)]
    for pid, cls in villains:
        players.append(TournamentPlayer(pid, cls(pid), starting_stack))

    runner = TournamentRunner(
        players=players,
        starting_stack=starting_stack,
        starting_blind=starting_blind,
        blind_increase_hands=15,
        max_hands=200,
        verbose=True,
    )

    result = runner.run()

    # Extra summary
    print("\n" + "=" * 70)
    print("DETAILED SUMMARY")
    print("=" * 70)
    print(f"Winner:           {result.winner}")
    print(f"Total hands:      {result.hands_played}")
    print(f"Duration:         {result.duration_seconds:.1f}s")
    print(f"Avg time/hand:    {result.duration_seconds / max(result.hands_played, 1):.2f}s")
    print(f"Final blind lvl:  {result.final_blind_level}")
    print()
    print("Finish order:")
    for i, pid in enumerate(result.finish_order):
        tp = runner.players[pid]
        marker = " <-- HERO" if pid == "LLM_Hero" else ""
        print(f"  {i+1}. {pid:20s}  {tp.stack:>7.0f} chips  ({tp.hands_played} hands){marker}")
    print()
    hero_finish = result.finish_order.index("LLM_Hero") + 1
    print(f"LLM Hero finished: #{hero_finish} of {len(result.finish_order)}")


if __name__ == "__main__":
    main()
