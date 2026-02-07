"""Run LLM hero vs 5 random-style opponents in a tournament."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.tournament import (
    TournamentPlayer, TournamentRunner, SimpleLLMPlayer
)
from src.game.opponents import RandomPlayer


def main():
    starting_stack = 1000.0
    starting_blind = 10.0

    print("Initializing SimpleLLM agent (claude-sonnet-4-5)...", flush=True)
    llm_player = SimpleLLMPlayer('Hero', model='claude-sonnet-4-5')

    # Create 5 random opponents with neutral IDs
    players = [
        TournamentPlayer('Hero', llm_player, starting_stack),
        TournamentPlayer('Player_1', RandomPlayer('Player_1'), starting_stack),
        TournamentPlayer('Player_2', RandomPlayer('Player_2'), starting_stack),
        TournamentPlayer('Player_3', RandomPlayer('Player_3'), starting_stack),
        TournamentPlayer('Player_4', RandomPlayer('Player_4'), starting_stack),
        TournamentPlayer('Player_5', RandomPlayer('Player_5'), starting_stack),
    ]

    runner = TournamentRunner(
        players=players,
        starting_stack=starting_stack,
        starting_blind=starting_blind,
        blind_increase_hands=15,
        max_hands=200,
        verbose=True,
    )

    result = runner.run()

    # Extra summary stats
    print("\n" + "=" * 70)
    print("SUMMARY STATS")
    print("=" * 70)
    print(f"Winner: {result.winner}")
    print(f"Total hands: {result.hands_played}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Avg time/hand: {result.duration_seconds / max(result.hands_played, 1):.2f}s")
    print(f"Final blind level: {result.final_blind_level}")
    print(f"\nFinish order:")
    for i, pid in enumerate(result.finish_order):
        tp = runner.players[pid]
        print(f"  {i+1}. {pid}: {tp.stack:.0f} chips ({tp.hands_played} hands played)")
    print(f"\nLLM Hero finished: #{result.finish_order.index('Hero') + 1} of 6")


if __name__ == '__main__':
    main()
