"""
Game runner that integrates all components for playing poker hands.

Provides:
- HandRunner: Plays a single hand with the agent
- SessionRunner: Plays multiple hands and tracks results
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Protocol
from enum import Enum, auto
import random

from ..core import (
    Card, HoleCards, Board, Position, Street, ActionType, Action,
    GameState, PlayerState, PlayerStatus, ActionRecord, create_6max_game,
    BeliefState, VillainBeliefs, create_default_villain_beliefs
)
from ..core.memory import OpponentMemory, TableMemory
from ..core.persistence import PersistentMemoryManager
from ..logging_config import get_logger

# Lazy import to avoid circular dependency: evaluation.__init__ -> evaluation.runner -> game.runner
NarrativeTracer = None
HandNarrative = None

def _ensure_narrative_imports():
    global NarrativeTracer, HandNarrative
    if NarrativeTracer is None:
        from ..evaluation.hand_narrative import NarrativeTracer as _NT, HandNarrative as _HN
        NarrativeTracer = _NT
        HandNarrative = _HN

logger = get_logger(__name__)


class PlayerProtocol(Protocol):
    """Protocol for any player (agent or opponent)."""

    def decide(self, game_state: GameState) -> Action:
        """Make a decision given the current game state."""
        ...

    @property
    def player_id(self) -> str:
        """Return player identifier."""
        ...


@dataclass
class HandResult:
    """Result of a single hand."""
    hand_id: str
    hero_profit: float  # In big blinds
    hero_cards: Optional[HoleCards]
    went_to_showdown: bool
    hero_position: Position
    actions_taken: int
    final_pot: float

    @property
    def won(self) -> bool:
        return self.hero_profit > 0


@dataclass
class SessionStats:
    """Statistics for a session of hands."""
    hands_played: int = 0
    total_profit: float = 0.0  # In big blinds
    showdowns: int = 0
    showdowns_won: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0

    @property
    def bb_per_100(self) -> float:
        """Win rate in big blinds per 100 hands."""
        if self.hands_played == 0:
            return 0.0
        return (self.total_profit / self.hands_played) * 100

    @property
    def vpip(self) -> float:
        """Voluntarily put money in pot percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.vpip_hands / self.hands_played

    @property
    def pfr(self) -> float:
        """Pre-flop raise percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.pfr_hands / self.hands_played

    @property
    def wtsd(self) -> float:
        """Went to showdown percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.showdowns / self.hands_played

    @property
    def won_at_sd(self) -> float:
        """Won at showdown percentage."""
        if self.showdowns == 0:
            return 0.0
        return self.showdowns_won / self.showdowns

    def __str__(self) -> str:
        return (
            f"Hands: {self.hands_played} | "
            f"Profit: {self.total_profit:+.1f} BB | "
            f"BB/100: {self.bb_per_100:+.1f} | "
            f"VPIP: {self.vpip:.0%} | PFR: {self.pfr:.0%}"
        )


class Deck:
    """A standard 52-card deck."""

    def __init__(self):
        self.cards: list[Card] = []
        self.reset()

    def reset(self):
        """Reset deck to full 52 cards."""
        self.cards = [
            Card.from_str(f"{r}{s}")
            for r in "23456789TJQKA"
            for s in "shdc"
        ]

    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def draw(self, n: int = 1) -> list[Card]:
        """Draw n cards from the deck."""
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def remove(self, cards: list[Card]):
        """Remove specific cards from deck."""
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)


class SimplePokerEngine:
    """
    Simple poker engine for running hands.

    Handles:
    - Dealing cards
    - Managing betting rounds
    - Pot management
    - Showdown evaluation
    """

    def __init__(self, big_blind: float = 10.0, starting_stack: float = 1000.0):
        self.big_blind = big_blind
        self.small_blind = big_blind / 2
        self.starting_stack = starting_stack
        self.deck = Deck()

    def create_hand(self, hero_position: Position, num_players: int = 6) -> GameState:
        """Create a new hand with hero at specified position."""
        import uuid

        self.deck.reset()
        self.deck.shuffle()

        # Deal hole cards to hero first
        hero_cards_list = self.deck.draw(2)
        hero_cards = HoleCards(hero_cards_list[0], hero_cards_list[1])

        # Create stacks for all positions
        all_positions = [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]
        stacks = {pos: self.starting_stack for pos in all_positions[:num_players]}

        # Create game state
        game = create_6max_game(
            hand_id=str(uuid.uuid4())[:8],
            hero_id="hero",
            hero_position=hero_position,
            hero_cards=hero_cards,
            stacks=stacks,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )

        # Deal to villains (hidden)
        for player in game.players.values():
            if player.player_id != game.hero_id:
                villain_cards = self.deck.draw(2)
                player.hole_cards = HoleCards(villain_cards[0], villain_cards[1])

        return game

    def deal_flop(self, game: GameState) -> None:
        """Deal the flop."""
        flop_cards = self.deck.draw(3)
        game.board.deal_flop(flop_cards[0], flop_cards[1], flop_cards[2])
        game.street = Street.FLOP

    def deal_turn(self, game: GameState) -> None:
        """Deal the turn."""
        turn_card = self.deck.draw(1)[0]
        game.board.deal_turn(turn_card)
        game.street = Street.TURN

    def deal_river(self, game: GameState) -> None:
        """Deal the river."""
        river_card = self.deck.draw(1)[0]
        game.board.deal_river(river_card)
        game.street = Street.RIVER

    def apply_action(self, game: GameState, player_id: str, action: Action) -> None:
        """Apply an action to the game state."""
        player = game.players[player_id]
        pot_before = game.pot.total
        old_current_bet = game.current_bet

        if action.action_type == ActionType.FOLD:
            player.status = PlayerStatus.FOLDED

        elif action.action_type == ActionType.CHECK:
            pass  # No change needed

        elif action.action_type == ActionType.CALL:
            call_amount = min(action.amount, player.stack)
            player.stack -= call_amount
            player.bet_this_street += call_amount
            game.pot.add(call_amount)

        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            # action.amount is the target total bet (raise TO amount)
            # Calculate how much more we need to add
            additional_amount = action.amount - player.bet_this_street
            actual_add = min(additional_amount, player.stack)
            player.stack -= actual_add
            player.bet_this_street += actual_add
            game.pot.add(actual_add)
            # Update current bet and min raise
            if player.bet_this_street > old_current_bet:
                raise_size = player.bet_this_street - old_current_bet
                game.min_raise = max(game.min_raise, raise_size)
            game.current_bet = player.bet_this_street

        elif action.action_type == ActionType.ALL_IN:
            all_in_amount = player.stack
            player.stack = 0
            player.bet_this_street += all_in_amount
            game.pot.add(all_in_amount)
            if player.bet_this_street > game.current_bet:
                raise_size = player.bet_this_street - old_current_bet
                if raise_size >= game.min_raise:
                    game.min_raise = raise_size
                game.current_bet = player.bet_this_street
            player.status = PlayerStatus.ALL_IN

        # Record action
        pot_after = game.pot.total
        record = ActionRecord(
            player_id=player_id,
            position=player.position,
            street=game.street,
            action=action,
            pot_before=pot_before,
            pot_after=pot_after
        )
        game.action_history.append(record)

    def get_active_players(self, game: GameState) -> list[str]:
        """Get list of active player IDs."""
        return [
            pid for pid, p in game.players.items()
            if p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        ]

    def is_hand_over(self, game: GameState) -> bool:
        """Check if hand is over."""
        active = self.get_active_players(game)
        # Hand over if only one player left or we're past river
        return len(active) <= 1 or (
            game.street == Street.RIVER and
            self._betting_complete(game)
        )

    def _betting_complete(self, game: GameState) -> bool:
        """Check if betting round is complete."""
        active_players = [
            p for p in game.players.values()
            if p.status == PlayerStatus.ACTIVE
        ]
        if not active_players:
            return True

        # All active players must have matched the current bet
        current_bet = game.current_bet
        return all(p.bet_this_street == current_bet for p in active_players)


class HandRunner:
    """
    Runs a single poker hand with the agent.

    Coordinates:
    - Game state management
    - Agent decisions
    - Opponent actions
    - Memory updates
    """

    def __init__(
        self,
        engine: SimplePokerEngine,
        hero: PlayerProtocol,
        villains: dict[str, PlayerProtocol],
        memory_manager: Optional[PersistentMemoryManager] = None,
        narrative_tracer: Optional[NarrativeTracer] = None
    ):
        self.engine = engine
        self.hero = hero
        self.villains = villains
        self.memory_manager = memory_manager
        self.narrative_tracer = narrative_tracer

    def run_hand(self, hero_position: Position) -> HandResult:
        """Run a single hand and return the result."""
        game = self.engine.create_hand(hero_position)
        hand_id = game.hand_id
        hero_cards = game.hero.hole_cards
        # Use starting_stack (before blinds) for proper profit calculation
        initial_stack = self.engine.starting_stack
        actions_taken = 0

        logger.info("Hand %s: Hero at %s with %s", hand_id, hero_position.value, hero_cards)

        # Set up narrative tracing if enabled
        narrative: Optional[HandNarrative] = None
        if self.narrative_tracer:
            narrative = self.narrative_tracer.start_hand(
                hand_id, "hero", hero_position, hero_cards
            )

        # Track who has acted this street
        acted_this_street: set[str] = set()
        last_aggressor: Optional[str] = None
        current_street = game.street

        # Play through streets
        while not self.engine.is_hand_over(game):
            # Get next player to act
            next_player = self._get_next_to_act(game, acted_this_street, last_aggressor)
            if next_player is None:
                # Move to next street
                if not self._advance_street(game):
                    break
                acted_this_street.clear()
                last_aggressor = None
                # Record the new street in narrative
                if narrative:
                    narrative.add_street(game.street, str(game.board), game.pot.total)
                continue

            # Set acting player so game state knows who's deciding
            game.acting_player = next_player

            # Get action
            if next_player == self.hero.player_id:
                # Capture hero reasoning in narrative
                hero_narrative = None
                if narrative:
                    hero_narrative = narrative.begin_hero_decision(game)

                action = self.hero.decide(game)
                if hero_narrative:
                    hero_narrative.set_decision(action, 0.0, False)

                if narrative:
                    narrative.end_hero_decision()
            else:
                villain = self.villains.get(next_player)
                if villain:
                    action = villain.decide(game)
                else:
                    # Default to check/fold for unregistered villains
                    player = game.players[next_player]
                    to_call_for_player = max(0, game.current_bet - player.bet_this_street)
                    action = Action.check() if to_call_for_player == 0 else Action.fold()

            # Track aggression (BET, RAISE, or ALL_IN that raises)
            player = game.players[next_player]
            is_aggressive = action.action_type in (ActionType.BET, ActionType.RAISE)
            # ALL_IN is aggressive if it raises the current bet
            if action.action_type == ActionType.ALL_IN:
                # After apply_action, check if player's bet exceeds previous current_bet
                # We need to check before apply, so compute what bet would be
                potential_bet = player.bet_this_street + player.stack  # all-in amount
                is_aggressive = potential_bet > game.current_bet

            if is_aggressive:
                last_aggressor = next_player
                # Others need to respond to raise
                acted_this_street = {next_player}
            else:
                acted_this_street.add(next_player)

            # Record action in narrative (for non-hero players, hero already recorded above)
            if narrative and next_player != self.hero.player_id:
                narrative.add_action(next_player, player.position, action, game.pot.total)

            # Apply action
            logger.debug("%s %s: %s (pot: %.0f)", game.street.value, next_player, action, game.pot.total)
            self.engine.apply_action(game, next_player, action)
            actions_taken += 1

            # Update memory for villain actions
            if next_player != self.hero.player_id and self.memory_manager:
                self._record_villain_action(next_player, action, game)

        # Distribute pot to winner(s)
        self._distribute_pot(game)

        # Calculate result
        hero_profit = (game.hero.stack - initial_stack) / self.engine.big_blind
        went_to_showdown = game.street == Street.RIVER and len(self.engine.get_active_players(game)) > 1

        # Finalize narrative
        if self.narrative_tracer:
            self.narrative_tracer.end_hand(hero_profit, went_to_showdown)

        return HandResult(
            hand_id=hand_id,
            hero_profit=hero_profit,
            hero_cards=hero_cards,
            went_to_showdown=went_to_showdown,
            hero_position=hero_position,
            actions_taken=actions_taken,
            final_pot=game.pot.total
        )

    def _distribute_pot(self, game: GameState) -> None:
        """Distribute pot to winner(s) at end of hand."""
        from ..tools.hand_eval import evaluate_hand

        # Get players still in hand (not folded)
        in_hand = [
            p for p in game.players.values()
            if p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        ]

        if not in_hand:
            return

        pot_amount = game.pot.total

        if len(in_hand) == 1:
            # Everyone else folded - award pot to remaining player
            winner = in_hand[0]
            winner.stack += pot_amount
        else:
            # Showdown - compare hands
            best_rank = None
            winners = []

            for player in in_hand:
                if player.hole_cards and game.board:
                    hand_rank = evaluate_hand(player.hole_cards, game.board)
                    if best_rank is None or hand_rank > best_rank:
                        best_rank = hand_rank
                        winners = [player]
                    elif hand_rank == best_rank:
                        winners.append(player)

            # Split pot among winners
            if winners:
                share = pot_amount / len(winners)
                for winner in winners:
                    winner.stack += share

    def _get_next_to_act(
        self, game: GameState, acted_this_street: set[str], last_aggressor: Optional[str]
    ) -> Optional[str]:
        """Determine which player acts next."""
        active_players = [
            p for p in game.players.values()
            if p.status == PlayerStatus.ACTIVE
        ]

        if not active_players:
            return None

        # Sort by position for action order
        position_order = [
            Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB
        ]
        # Postflop: SB acts first
        if game.street != Street.PREFLOP:
            position_order = [
                Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO, Position.BTN
            ]

        sorted_players = sorted(
            active_players,
            key=lambda p: position_order.index(p.position) if p.position in position_order else 99
        )

        # Find first player who hasn't acted or needs to respond to a raise
        for player in sorted_players:
            if player.player_id not in acted_this_street:
                return player.player_id
            # If there was a raise, players need to respond (except the raiser)
            if (last_aggressor and
                player.player_id != last_aggressor and
                player.bet_this_street < game.current_bet):
                return player.player_id

        return None

    def _advance_street(self, game: GameState) -> bool:
        """Advance to the next street. Returns False if hand is over."""
        # Reset bets for new street
        for player in game.players.values():
            player.bet_this_street = 0
        game.current_bet = 0

        if game.street == Street.PREFLOP:
            self.engine.deal_flop(game)
            logger.info("Flop: %s (pot: %.0f)", game.board, game.pot.total)
            return True
        elif game.street == Street.FLOP:
            self.engine.deal_turn(game)
            logger.info("Turn: %s (pot: %.0f)", game.board, game.pot.total)
            return True
        elif game.street == Street.TURN:
            self.engine.deal_river(game)
            logger.info("River: %s (pot: %.0f)", game.board, game.pot.total)
            return True
        return False

    def _record_villain_action(self, villain_id: str, action: Action, game: GameState) -> None:
        """Record villain action to memory."""
        if self.memory_manager:
            memory = self.memory_manager.get_opponent_memory(villain_id)
            if game.street == Street.PREFLOP:
                memory.record_preflop_action(
                    action.action_type,
                    game.players[villain_id].position,
                    facing_raise=game.current_bet > self.engine.big_blind
                )
            else:
                memory.record_postflop_action(action.action_type)


class SessionRunner:
    """
    Runs a session of multiple hands and tracks statistics.
    """

    def __init__(
        self,
        hero: PlayerProtocol,
        villains: dict[str, PlayerProtocol],
        big_blind: float = 10.0,
        starting_stack: float = 1000.0,
        memory_manager: Optional[PersistentMemoryManager] = None
    ):
        self.engine = SimplePokerEngine(big_blind, starting_stack)
        self.hero = hero
        self.villains = villains
        self.memory_manager = memory_manager
        self.stats = SessionStats()
        self.hand_history: list[HandResult] = []

    def run_session(self, num_hands: int, verbose: bool = False) -> SessionStats:
        """Run a session of hands."""
        positions = list(Position)

        for i in range(num_hands):
            # Rotate hero position
            hero_pos = positions[i % len(positions)]

            # Run hand
            runner = HandRunner(
                self.engine, self.hero, self.villains, self.memory_manager
            )
            result = runner.run_hand(hero_pos)

            # Update stats
            self._update_stats(result)
            self.hand_history.append(result)

            if verbose and (i + 1) % 100 == 0:
                print(f"Hand {i + 1}: {self.stats}")

        return self.stats

    def _update_stats(self, result: HandResult) -> None:
        """Update session statistics."""
        self.stats.hands_played += 1
        self.stats.total_profit += result.hero_profit

        if result.went_to_showdown:
            self.stats.showdowns += 1
            if result.won:
                self.stats.showdowns_won += 1

    def get_summary(self) -> str:
        """Get session summary."""
        lines = [
            "=== Session Summary ===",
            f"Hands played: {self.stats.hands_played}",
            f"Total profit: {self.stats.total_profit:+.1f} BB",
            f"Win rate: {self.stats.bb_per_100:+.1f} BB/100",
            f"VPIP: {self.stats.vpip:.0%}",
            f"PFR: {self.stats.pfr:.0%}",
            f"WTSD: {self.stats.wtsd:.0%}",
            f"W$SD: {self.stats.won_at_sd:.0%}",
        ]
        return "\n".join(lines)
