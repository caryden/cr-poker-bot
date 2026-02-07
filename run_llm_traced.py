"""Run LLM hero with full reasoning trace visible."""

import os
import sys
import random
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.tournament import TournamentPlayer, TournamentRunner, SimpleLLMPlayer
from src.game.opponents import CallingStation, TightPassive, LooseAggressive, TagBot
from src.core.game_state import GameState, PlayerStatus
from src.core.primitives import Action

STRATEGY_BOTS = [
    ("Fish", CallingStation),
    ("Nit", TightPassive),
    ("LAG", LooseAggressive),
    ("TAG", TagBot),
]


class TracedLLMPlayer(SimpleLLMPlayer):
    """SimpleLLMPlayer with full reasoning trace printed."""

    def __init__(self, player_id: str = 'LLM_Hero', model: str = 'claude-sonnet-4-5'):
        super().__init__(player_id, model)
        self.decision_count = 0

    def decide(self, game_state: GameState) -> Action:
        from src.tools.equity import calculate_equity

        hero = game_state.players.get(self._player_id)
        if not hero:
            for pid, p in game_state.players.items():
                if pid == self._player_id or 'LLM' in pid:
                    hero = p
                    break
        if not hero:
            return Action.fold()

        hero_cards = hero.hole_cards
        if not hero_cards:
            return Action.check() if game_state.to_call == 0 else Action.fold()

        num_opponents = len([p for p in game_state.players.values()
                           if p.status == PlayerStatus.ACTIVE and p.player_id != self._player_id])
        num_opponents = max(1, num_opponents)

        eq = calculate_equity(hero_cards, game_state.board, num_opponents=num_opponents, simulations=300)
        pot_odds = game_state.to_call / (game_state.pot.total + game_state.to_call) if game_state.to_call > 0 else 0

        board_str = str(game_state.board) if game_state.board and game_state.board.cards else 'none'

        prompt = f'''Poker. ONLY respond: fold/check/call/bet N/raise N/all-in
Board: {board_str}, Hand: {hero_cards}
Pot: {game_state.pot.total:.0f}, To call: {game_state.to_call:.0f}, Stack: {hero.stack:.0f}
Equity: {eq.equity:.0%}, Pot odds: {pot_odds:.0%}'''

        self.decision_count += 1
        print(f"\n  {'─' * 60}")
        print(f"  LLM Decision #{self.decision_count}")
        print(f"  {'─' * 60}")
        print(f"  Street:    {game_state.street.value}")
        print(f"  Position:  {hero.position.value}")
        print(f"  Hand:      {hero_cards}")
        print(f"  Board:     {board_str}")
        print(f"  Pot:       {game_state.pot.total:.0f}  |  To call: {game_state.to_call:.0f}  |  Stack: {hero.stack:.0f}")
        print(f"  Equity:    {eq.equity:.0%}  |  Pot odds: {pot_odds:.0%}")
        print(f"  ┌── Prompt ──")
        for line in prompt.strip().splitlines():
            print(f"  │ {line}")
        print(f"  └──────────")

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                messages=[{'role': 'user', 'content': prompt}]
            )
            response = resp.content[0].text.strip().lower()
        except Exception as e:
            print(f"  LLM Error: {e}")
            action = Action.check() if game_state.to_call == 0 else Action.fold()
            print(f"  Action:    {action} (fallback)")
            print(f"  {'─' * 60}")
            return action

        action = self._parse_action(response, game_state.to_call, hero.stack)
        print(f"  LLM says:  \"{response}\"")
        print(f"  Action:    {action}")
        print(f"  {'─' * 60}")
        return action


def main():
    starting_stack = 1000.0
    starting_blind = 10.0

    # Pick 5 strategy bots randomly
    selected = random.choices(STRATEGY_BOTS, k=5)
    name_counts = {}
    villains = []
    for label, cls in selected:
        name_counts[label] = name_counts.get(label, 0) + 1
        pid = f"{label}_{name_counts[label]}"
        villains.append((pid, cls))

    print("=" * 70)
    print("TRACED LLM TABLE")
    print("=" * 70)
    print(f"Hero:     LLM_Hero (TracedLLMPlayer, claude-sonnet-4-5)")
    for pid, cls in villains:
        print(f"Villain:  {pid} ({cls.__name__})")
    print(f"Stack:    {starting_stack}")
    print(f"Blinds:   {starting_blind/2:.0f}/{starting_blind:.0f}")
    print(f"Max hands: 30 (short game for readable trace)")
    print("=" * 70)

    llm_player = TracedLLMPlayer("LLM_Hero", model="claude-sonnet-4-5")

    players = [TournamentPlayer("LLM_Hero", llm_player, starting_stack)]
    for pid, cls in villains:
        players.append(TournamentPlayer(pid, cls(pid), starting_stack))

    runner = TournamentRunner(
        players=players,
        starting_stack=starting_stack,
        starting_blind=starting_blind,
        blind_increase_hands=15,
        max_hands=30,
        verbose=True,
    )

    result = runner.run()

    print(f"\n{'=' * 70}")
    print(f"LLM made {llm_player.decision_count} decisions across {result.hands_played} hands")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
