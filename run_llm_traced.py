"""Run LLM hero with full reasoning trace visible."""

import os
import sys
import random
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.tournament import (
    TournamentPlayer, TournamentRunner, SimpleLLMPlayer, SYSTEM_PROMPT,
)
from src.game.opponents import CallingStation, TightPassive, LooseAggressive, TagBot
from src.core.game_state import GameState, PlayerStatus
from src.core.primitives import Action, Street

STRATEGY_BOTS = [
    ("Fish", CallingStation),
    ("Nit", TightPassive),
    ("LAG", LooseAggressive),
    ("TAG", TagBot),
]

# Enhanced system prompt: same PHH/SL docs but asks for reasoning
TRACED_SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip() + """

## Response Format
First, provide 2-3 sentences explaining your reasoning (board texture, equity vs pot odds,
GTO recommendation, beliefs about opponents, bet sizing rationale).
Then on a FINAL line by itself, give your action: fold / check / call / bet N / raise N / all-in"""


class TracedLLMPlayer(SimpleLLMPlayer):
    """SimpleLLMPlayer that prints the full prompt/response trace.

    Delegates prompt-building to the parent (beliefs + PHH history)
    and enriches it with GTO, board texture, and bet sizing tool outputs.
    """

    def __init__(self, player_id: str = 'LLM_Hero', model: str = 'claude-sonnet-4-5'):
        super().__init__(player_id, model)
        self.decision_count = 0

    def decide(self, game_state: GameState) -> Action:
        from src.tools.equity import calculate_equity
        from src.tools.gto import should_open, should_3bet
        from src.tools.board_texture import analyze_board, get_texture_description
        from src.tools.bet_sizing import BetSizingAdvisor, HandStrengthCategory

        # --- resolve hero (same as parent) ---
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

        num_opponents = max(1, len([
            p for p in game_state.players.values()
            if p.status == PlayerStatus.ACTIVE and p.player_id != self._player_id
        ]))

        # --- compute tool outputs ---
        eq = calculate_equity(hero_cards, game_state.board,
                              num_opponents=num_opponents, simulations=300)
        pot_odds = (game_state.to_call / (game_state.pot.total + game_state.to_call)
                    if game_state.to_call > 0 else 0)

        # GTO recommendation
        gto_rec = should_open(hero_cards, hero.position)
        if game_state.to_call > 0 and game_state.street == Street.PREFLOP:
            # Facing a raise — use 3-bet logic vs earliest raiser position
            from src.core.primitives import ActionType
            raiser_pos = None
            for pid, p in game_state.players.items():
                if pid != self._player_id and p.status == PlayerStatus.ACTIVE:
                    raiser_pos = p.position
                    break
            if raiser_pos:
                gto_rec = should_3bet(hero_cards, hero.position, raiser_pos)

        # Board texture (postflop only)
        board_texture_str = None
        if game_state.board and game_state.board.cards:
            board_texture_str = get_texture_description(game_state.board)
            texture = analyze_board(game_state.board)

        # Bet sizing advisor
        sizing_advisor = BetSizingAdvisor()
        spr = hero.stack / game_state.pot.total if game_state.pot.total > 0 else 10
        # Rough hand strength from equity
        if eq.equity >= 0.85:
            hand_cat = HandStrengthCategory.MONSTER
        elif eq.equity >= 0.70:
            hand_cat = HandStrengthCategory.VERY_STRONG
        elif eq.equity >= 0.55:
            hand_cat = HandStrengthCategory.STRONG
        elif eq.equity >= 0.40:
            hand_cat = HandStrengthCategory.MEDIUM
        elif eq.equity >= 0.25:
            hand_cat = HandStrengthCategory.MARGINAL
        else:
            hand_cat = HandStrengthCategory.TRASH

        sizing_rec = None
        if game_state.street == Street.PREFLOP:
            sizing_rec = sizing_advisor.get_preflop_sizing(
                hero.position,
                facing_raise=game_state.to_call > 0,
                raise_amount=game_state.to_call
            )
        elif game_state.board and game_state.board.cards:
            texture = analyze_board(game_state.board)
            if game_state.street == Street.FLOP:
                sizing_rec = sizing_advisor.get_flop_sizing(texture, hand_cat, False, spr)
            elif game_state.street == Street.TURN:
                sizing_rec = sizing_advisor.get_turn_sizing(texture, hand_cat, False, spr)
            elif game_state.street == Street.RIVER:
                sizing_rec = sizing_advisor.get_river_sizing(
                    texture, hand_cat, game_state.pot.total, hero.stack)

        # --- build prompt: start with parent's _build_prompt (beliefs + PHH) ---
        base_prompt = self._build_prompt(game_state, hero, eq.equity, pot_odds)

        # Append tool outputs
        tool_parts = ["\n## Tool Analysis"]
        tool_parts.append(f"GTO: {gto_rec}")
        if board_texture_str:
            tool_parts.append(f"Board texture: {board_texture_str}")
        tool_parts.append(f"Hand strength: {hand_cat.name} (equity {eq.equity:.0%})")
        if sizing_rec:
            tool_parts.append(f"Bet sizing suggestion: {sizing_rec}")

        prompt = base_prompt.rstrip("\nYour action:") + "\n".join(tool_parts) + "\n\nYour action (with reasoning):"

        self.decision_count += 1

        # --- print trace ---
        print(f"\n  {'─' * 70}")
        print(f"  LLM Decision #{self.decision_count}")
        print(f"  {'─' * 70}")

        print(f"  ┌── System Prompt ──")
        for line in TRACED_SYSTEM_PROMPT.strip().splitlines():
            print(f"  │ {line}")
        print(f"  └──────────")

        print(f"  ┌── User Prompt ──")
        for line in prompt.strip().splitlines():
            print(f"  │ {line}")
        print(f"  └──────────")

        # --- call LLM ---
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system=TRACED_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}]
            )
            response = resp.content[0].text.strip()
        except Exception as e:
            print(f"  LLM Error: {e}")
            action = Action.check() if game_state.to_call == 0 else Action.fold()
            print(f"  Action:    {action} (fallback)")
            print(f"  {'─' * 70}")
            return action

        print(f"  ┌── LLM Response ──")
        for line in response.splitlines():
            print(f"  │ {line}")
        print(f"  └──────────")

        # Parse action from the last line
        action_line = response.strip().splitlines()[-1].lower()
        action = self._parse_action(action_line, game_state.to_call, hero.stack)
        print(f"  → Parsed action: {action}")
        print(f"  {'─' * 70}")
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
    print(f"Max hands: 10 (short game for readable trace)")
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
        max_hands=10,
        verbose=True,
    )

    result = runner.run()

    print(f"\n{'=' * 70}")
    print(f"LLM made {llm_player.decision_count} decisions across {result.hands_played} hands")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
