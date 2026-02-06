"""
Tournament runner for AI agent evaluation.

Runs a full sit-and-go style tournament with:
- 6 players (1 LLM agent + 5 bots)
- Escalating blinds
- Elimination when busted
- Continues until one player wins
"""

import os
import sys
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, Protocol
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.primitives import Action, ActionType, Position, HoleCards, Card, Board
from src.core.game_state import GameState, PlayerState, PlayerStatus, create_6max_game
from src.core import Street
from src.game.runner import SimplePokerEngine, HandRunner, Deck
from src.game.opponents import RandomPlayer, CallingStation, TightPassive, LooseAggressive, TagBot
from src.agent.react import ReActAgent
from src.agent.tools import ToolRegistry


class PlayerProtocol(Protocol):
    @property
    def player_id(self) -> str: ...
    def decide(self, game_state: GameState) -> Action: ...


@dataclass
class TournamentPlayer:
    """A player in the tournament with their stack."""
    player_id: str
    player: PlayerProtocol
    stack: float
    position: Optional[Position] = None
    is_eliminated: bool = False
    finish_position: int = 0
    hands_played: int = 0

    def __str__(self):
        status = "OUT" if self.is_eliminated else f"${self.stack:.0f}"
        return f"{self.player_id}: {status}"


@dataclass
class TournamentResult:
    """Results from a tournament."""
    winner: str
    finish_order: list[str]  # First = winner, last = first eliminated
    hands_played: int
    final_blind_level: int
    duration_seconds: float
    hand_history: list[dict] = field(default_factory=list)


class LLMAgentPlayer:
    """Wrapper to make ReActAgent compatible with PlayerProtocol."""

    def __init__(self, agent: ReActAgent, player_id: str = 'hero'):
        self.agent = agent
        self._player_id = player_id

    @property
    def player_id(self) -> str:
        return self._player_id

    def decide(self, game_state: GameState) -> Action:
        decision = self.agent.decide(game_state)
        return decision.action


class TournamentRunner:
    """
    Runs a sit-and-go style tournament.

    Features:
    - Escalating blinds (double every N hands)
    - Player elimination
    - Position rotation
    - Tracks finish order
    """

    def __init__(
        self,
        players: list[TournamentPlayer],
        starting_stack: float = 1000.0,
        starting_blind: float = 10.0,
        blind_increase_hands: int = 20,
        max_hands: int = 500,
        verbose: bool = True
    ):
        self.players = {p.player_id: p for p in players}
        self.starting_stack = starting_stack
        self.starting_blind = starting_blind
        self.blind_increase_hands = blind_increase_hands
        self.max_hands = max_hands
        self.verbose = verbose

        # Initialize stacks
        for p in self.players.values():
            p.stack = starting_stack

        # Tournament state
        self.hands_played = 0
        self.blind_level = 1
        self.current_blind = starting_blind
        self.finish_order: list[str] = []
        self.hand_history: list[dict] = []

    def get_active_players(self) -> list[TournamentPlayer]:
        """Get players still in tournament."""
        return [p for p in self.players.values() if not p.is_eliminated]

    def get_current_blind(self) -> float:
        """Get current big blind based on level."""
        return self.starting_blind * (2 ** (self.blind_level - 1))

    def assign_positions(self, active: list[TournamentPlayer], button_idx: int) -> dict[Position, TournamentPlayer]:
        """Assign positions to active players."""
        n = len(active)

        if n == 2:
            # Heads-up: BTN posts SB and acts first preflop, other player is BB
            positions = [Position.BTN, Position.BB]
        elif n == 3:
            positions = [Position.BTN, Position.SB, Position.BB]
        else:
            positions = [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO][:n]

        # Rotate based on button
        assigned = {}
        for i, pos in enumerate(positions):
            player_idx = (button_idx + i) % n
            active[player_idx].position = pos
            assigned[pos] = active[player_idx]

        return assigned

    def run_hand(self, position_map: dict[Position, TournamentPlayer]) -> dict[str, float]:
        """Run a single hand and return profit/loss for each player."""
        active = list(position_map.values())
        n = len(active)

        if n < 2:
            return {}

        # Create stacks dict
        stacks = {p.position: p.stack for p in active}

        # Determine hero (first player in list for game state purposes)
        hero = active[0]
        hero_pos = hero.position

        # Create game state
        deck = Deck()
        deck.shuffle()

        # Deal cards to hero
        hero_cards_list = deck.draw(2)
        hero_cards = HoleCards(hero_cards_list[0], hero_cards_list[1])

        # Deal cards to all players (for showdown)
        all_hole_cards = {hero.player_id: hero_cards}
        for p in active[1:]:
            cards = deck.draw(2)
            all_hole_cards[p.player_id] = HoleCards(cards[0], cards[1])

        # Create game
        game = create_6max_game(
            hand_id=f"hand_{self.hands_played}",
            hero_id=hero.player_id,
            hero_position=hero_pos,
            hero_cards=hero_cards,
            stacks=stacks,
            small_blind=self.current_blind / 2,
            big_blind=self.current_blind
        )

        # Fix player IDs and hole cards in game state
        new_players = {}
        for pos, tp in position_map.items():
            old_pid = f"villain_{pos.value}" if pos != hero_pos else hero.player_id
            if old_pid in game.players:
                player = game.players[old_pid]
                player.player_id = tp.player_id
                player.hole_cards = all_hole_cards.get(tp.player_id)
                new_players[tp.player_id] = player
            else:
                # Hero case
                player = game.players[hero.player_id]
                player.hole_cards = hero_cards
                new_players[tp.player_id] = player
        game.players = new_players
        game.hero_id = hero.player_id

        # Track initial stacks
        initial_stacks = {pid: p.stack for pid, p in game.players.items()}

        # Create villain dict for runner
        villains = {tp.player_id: tp.player for tp in active if tp.player_id != hero.player_id}

        # Play the hand
        acted_this_street = set()
        last_aggressor = None

        # Get action order
        def get_next_to_act():
            active_players = [
                p for p in game.players.values()
                if p.status == PlayerStatus.ACTIVE
            ]
            if not active_players:
                return None

            # Order by position
            pos_order = [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]
            if game.street != Street.PREFLOP:
                pos_order = [Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO, Position.BTN]

            for pos in pos_order:
                for p in active_players:
                    if p.position == pos and p.player_id not in acted_this_street:
                        if last_aggressor is None or p.player_id != last_aggressor or len(acted_this_street) == len(active_players) - 1:
                            return p.player_id
            return None

        def betting_complete():
            active_players = [p for p in game.players.values() if p.status == PlayerStatus.ACTIVE]
            if not active_players:
                return True
            return all(p.bet_this_street == game.current_bet for p in active_players)

        def advance_street():
            if game.street == Street.RIVER:
                return False

            # Deal community cards
            if game.street == Street.PREFLOP:
                game.street = Street.FLOP
                flop = deck.draw(3)
                game.board = Board(cards=flop)
            elif game.street == Street.FLOP:
                game.street = Street.TURN
                game.board.cards.append(deck.draw(1)[0])
            elif game.street == Street.TURN:
                game.street = Street.RIVER
                game.board.cards.append(deck.draw(1)[0])

            # Reset betting
            game.current_bet = 0
            game.min_raise = self.current_blind
            for p in game.players.values():
                p.bet_this_street = 0

            return True

        def apply_action(player_id: str, action: Action):
            nonlocal last_aggressor
            player = game.players[player_id]
            old_bet = game.current_bet

            if action.action_type == ActionType.FOLD:
                player.status = PlayerStatus.FOLDED
            elif action.action_type == ActionType.CHECK:
                pass
            elif action.action_type == ActionType.CALL:
                call_amount = min(game.current_bet - player.bet_this_street, player.stack)
                player.stack -= call_amount
                player.bet_this_street += call_amount
                game.pot.add(call_amount)
                # If player put in all chips but couldn't fully call, mark as all-in
                if player.stack == 0:
                    player.status = PlayerStatus.ALL_IN
            elif action.action_type in (ActionType.BET, ActionType.RAISE):
                target = action.amount
                additional = target - player.bet_this_street
                actual = min(additional, player.stack)
                player.stack -= actual
                player.bet_this_street += actual
                game.pot.add(actual)
                if player.bet_this_street > old_bet:
                    game.min_raise = max(game.min_raise, player.bet_this_street - old_bet)
                game.current_bet = player.bet_this_street
                # If player put in all chips, mark as all-in
                if player.stack == 0:
                    player.status = PlayerStatus.ALL_IN
            elif action.action_type == ActionType.ALL_IN:
                all_in = player.stack
                player.stack = 0
                player.bet_this_street += all_in
                game.pot.add(all_in)
                if player.bet_this_street > game.current_bet:
                    game.current_bet = player.bet_this_street
                player.status = PlayerStatus.ALL_IN

            # Track aggression
            if action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                if player.bet_this_street > old_bet:
                    last_aggressor = player_id
                    acted_this_street.clear()
                    acted_this_street.add(player_id)
                    return

            acted_this_street.add(player_id)

        # Main game loop
        max_actions = 100
        action_count = 0

        while action_count < max_actions:
            # Check if hand is over
            active_count = sum(1 for p in game.players.values()
                             if p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN))
            if active_count <= 1:
                break

            # Get next player
            next_player = get_next_to_act()
            if next_player is None:
                # If betting is complete or stuck, advance street
                if not advance_street():
                    break  # Showdown
                acted_this_street.clear()
                last_aggressor = None
                continue

            game.acting_player = next_player

            # Get decision
            if next_player == hero.player_id:
                action = hero.player.decide(game)
            else:
                villain = villains.get(next_player)
                if villain:
                    action = villain.decide(game)
                else:
                    to_call = max(0, game.current_bet - game.players[next_player].bet_this_street)
                    action = Action.check() if to_call == 0 else Action.fold()

            apply_action(next_player, action)
            action_count += 1

        # Distribute pot
        in_hand = [p for p in game.players.values()
                   if p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)]

        if len(in_hand) == 1:
            in_hand[0].stack += game.pot.total
        elif len(in_hand) > 1:
            # Showdown - evaluate hands
            from src.tools.hand_eval import evaluate_hand
            best_rank = None
            winners = []
            for p in in_hand:
                if p.hole_cards and game.board and len(game.board.cards) >= 3:
                    rank = evaluate_hand(p.hole_cards, game.board)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        winners = [p]
                    elif rank == best_rank:
                        winners.append(p)

            if winners:
                share = game.pot.total / len(winners)
                for w in winners:
                    w.stack += share

        # Calculate profits
        profits = {}
        for pid, player in game.players.items():
            initial = initial_stacks.get(pid, self.starting_stack)
            profits[pid] = player.stack - initial

        return profits

    def run(self) -> TournamentResult:
        """Run the tournament to completion."""
        start_time = time.time()
        button_idx = 0

        if self.verbose:
            print("=" * 70, flush=True)
            print("TOURNAMENT START", flush=True)
            print("=" * 70, flush=True)
            print(f"Players: {len(self.players)}", flush=True)
            print(f"Starting stack: {self.starting_stack}", flush=True)
            print(f"Starting blind: {self.starting_blind}", flush=True)
            print(f"Blinds increase every {self.blind_increase_hands} hands", flush=True)
            print(flush=True)

        while self.hands_played < self.max_hands:
            active = self.get_active_players()

            # Check for winner
            if len(active) == 1:
                winner = active[0]
                winner.finish_position = 1
                self.finish_order.insert(0, winner.player_id)
                break

            if len(active) < 2:
                break

            # Check for blind increase
            if self.hands_played > 0 and self.hands_played % self.blind_increase_hands == 0:
                self.blind_level += 1
                self.current_blind = self.get_current_blind()
                if self.verbose:
                    print(f"\n*** BLINDS INCREASE: {self.current_blind/2:.0f}/{self.current_blind:.0f} ***\n", flush=True)

            # Assign positions
            position_map = self.assign_positions(active, button_idx)

            # Run hand
            self.hands_played += 1

            if self.verbose:
                stacks = " | ".join(f"{p.player_id[:3]}: {p.stack:.0f}" for p in active)
                print(f"Hand {self.hands_played} | Blinds {self.current_blind/2:.0f}/{self.current_blind:.0f} | {stacks}", flush=True)

            try:
                profits = self.run_hand(position_map)

                # Update stacks and check for eliminations
                for pid, profit in profits.items():
                    if pid in self.players:
                        tp = self.players[pid]
                        tp.stack += profit
                        tp.hands_played += 1

                        if tp.stack <= 0:
                            tp.is_eliminated = True
                            tp.finish_position = len(self.get_active_players()) + 1
                            self.finish_order.append(pid)
                            if self.verbose:
                                print(f"*** {pid} ELIMINATED (#{tp.finish_position}) ***", flush=True)

            except Exception as e:
                if self.verbose:
                    print(f"Hand error: {e}")
                continue

            # Rotate button
            button_idx = (button_idx + 1) % len(active)

        # Finalize results
        duration = time.time() - start_time

        # Set finish positions for remaining players (sorted by chip count)
        remaining = self.get_active_players()
        remaining.sort(key=lambda p: -p.stack)
        for i, p in enumerate(remaining):
            p.finish_position = i + 1

        # Build final standings: winners first (by chip count), then eliminated in reverse order
        final_standings = [p.player_id for p in remaining]
        for pid in reversed(self.finish_order):
            if pid not in final_standings:
                final_standings.append(pid)
        self.finish_order = final_standings

        winner = self.finish_order[0] if self.finish_order else "None"

        if self.verbose:
            print("\n" + "=" * 70, flush=True)
            print("TOURNAMENT COMPLETE", flush=True)
            print("=" * 70, flush=True)
            print(f"Winner: {winner}", flush=True)
            print(f"Hands played: {self.hands_played}", flush=True)
            print(f"Final blind level: {self.blind_level} ({self.current_blind/2:.0f}/{self.current_blind:.0f})", flush=True)
            print(f"Duration: {duration:.1f}s", flush=True)
            print("\nFinal standings:", flush=True)
            for i, pid in enumerate(self.finish_order):
                p = self.players[pid]
                print(f"  {i+1}. {pid}: {p.stack:.0f} chips ({p.hands_played} hands)", flush=True)

        return TournamentResult(
            winner=winner,
            finish_order=self.finish_order,
            hands_played=self.hands_played,
            final_blind_level=self.blind_level,
            duration_seconds=duration,
            hand_history=self.hand_history
        )


def run_bot_only_tournament(
    starting_stack: float = 1000.0,
    starting_blind: float = 10.0,
    blind_increase_hands: int = 10,
    max_hands: int = 200,
    verbose: bool = True
) -> TournamentResult:
    """Run a tournament with only bots (no LLM) for testing."""

    print("Running bot-only tournament...", flush=True)

    # Create bot players
    players = [
        TournamentPlayer('CallingStation1', CallingStation('CallingStation1'), starting_stack),
        TournamentPlayer('CallingStation2', CallingStation('CallingStation2'), starting_stack),
        TournamentPlayer('TightPassive', TightPassive('TightPassive'), starting_stack),
        TournamentPlayer('LooseAggressive', LooseAggressive('LooseAggressive'), starting_stack),
        TournamentPlayer('TagBot', TagBot('TagBot'), starting_stack),
        TournamentPlayer('Random', RandomPlayer('Random'), starting_stack),
    ]

    runner = TournamentRunner(
        players=players,
        starting_stack=starting_stack,
        starting_blind=starting_blind,
        blind_increase_hands=blind_increase_hands,
        max_hands=max_hands,
        verbose=verbose
    )

    return runner.run()


def run_llm_tournament(
    model: str = "claude-sonnet-4-5",
    starting_stack: float = 1000.0,
    starting_blind: float = 10.0,
    blind_increase_hands: int = 15,
    max_hands: int = 200,
    verbose: bool = True
) -> TournamentResult:
    """Run a tournament with LLM agent vs bots."""

    # Create LLM agent
    from src.agent.llm_client import create_claude_client

    print(f"Initializing LLM agent ({model})...")
    llm_call = create_claude_client(model=model)
    tools = ToolRegistry()
    react_agent = ReActAgent(tool_registry=tools, max_steps=4, llm_call=llm_call)
    llm_player = LLMAgentPlayer(react_agent, 'LLM_Agent')

    # Create bot opponents
    players = [
        TournamentPlayer('LLM_Agent', llm_player, starting_stack),
        TournamentPlayer('CallingStation', CallingStation('CallingStation'), starting_stack),
        TournamentPlayer('TightPassive', TightPassive('TightPassive'), starting_stack),
        TournamentPlayer('LooseAggressive', LooseAggressive('LooseAggressive'), starting_stack),
        TournamentPlayer('TagBot', TagBot('TagBot'), starting_stack),
        TournamentPlayer('Random', RandomPlayer('Random'), starting_stack),
    ]

    # Run tournament
    runner = TournamentRunner(
        players=players,
        starting_stack=starting_stack,
        starting_blind=starting_blind,
        blind_increase_hands=blind_increase_hands,
        max_hands=max_hands,
        verbose=verbose
    )

    return runner.run()


def run_multiple_tournaments(
    num_tournaments: int = 5,
    model: str = "claude-sonnet-4-5",
    **kwargs
) -> dict:
    """Run multiple tournaments and aggregate results."""

    print("=" * 70)
    print(f"RUNNING {num_tournaments} TOURNAMENTS")
    print(f"Model: {model}")
    print("=" * 70)

    results = []
    wins = defaultdict(int)
    finishes = defaultdict(list)

    for i in range(num_tournaments):
        print(f"\n{'='*70}")
        print(f"TOURNAMENT {i+1}/{num_tournaments}")
        print(f"{'='*70}")

        result = run_llm_tournament(model=model, **kwargs)
        results.append(result)

        # Track stats
        wins[result.winner] += 1
        for pos, pid in enumerate(result.finish_order):
            finishes[pid].append(pos + 1)

    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(f"\nWins ({num_tournaments} tournaments):")
    for pid, win_count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = win_count / num_tournaments * 100
        print(f"  {pid}: {win_count} wins ({pct:.0f}%)")

    print(f"\nAverage finish position:")
    for pid, positions in sorted(finishes.items(), key=lambda x: sum(x[1])/len(x[1])):
        avg = sum(positions) / len(positions)
        print(f"  {pid}: {avg:.2f}")

    return {
        'results': results,
        'wins': dict(wins),
        'finishes': dict(finishes),
        'num_tournaments': num_tournaments
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run poker tournament')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-5', help='LLM model to use')
    parser.add_argument('--num', type=int, default=1, help='Number of tournaments')
    parser.add_argument('--stack', type=float, default=1000, help='Starting stack')
    parser.add_argument('--blind', type=float, default=10, help='Starting blind')
    parser.add_argument('--blind-increase', type=int, default=15, help='Hands between blind increases')
    parser.add_argument('--max-hands', type=int, default=200, help='Maximum hands per tournament')
    args = parser.parse_args()

    if args.num == 1:
        run_llm_tournament(
            model=args.model,
            starting_stack=args.stack,
            starting_blind=args.blind,
            blind_increase_hands=args.blind_increase,
            max_hands=args.max_hands
        )
    else:
        run_multiple_tournaments(
            num_tournaments=args.num,
            model=args.model,
            starting_stack=args.stack,
            starting_blind=args.blind,
            blind_increase_hands=args.blind_increase,
            max_hands=args.max_hands
        )
