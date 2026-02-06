"""
Tests for bug fixes identified in code review.

Validates:
1. HoleCards swap bug fix
2. expected_value_call signature fix
3. process_game_state dedup fix
4. _parse_params quoted string fix
5. Hand narrative tracing
"""

import pytest
from src.core.primitives import Card, Rank, Suit, HoleCards, Action, ActionType, Position, Street, Board
from src.core.game_state import GameState, create_6max_game
from src.core.beliefs import (
    BeliefRevisionEngine, BeliefState, Observation, ObservationType, GameContext,
    create_default_villain_beliefs
)
from src.tools.pot_odds import expected_value_call
from src.evaluation.hand_narrative import (
    HandNarrative, NarrativeTracer, HeroDecisionNarrative, HeroThought
)


# =============================================================================
# 1. HoleCards swap bug fix
# =============================================================================

class TestHoleCardsSwapFix:
    """HoleCards.__post_init__ should swap cards so higher rank is card1."""

    def test_swap_preserves_both_cards(self):
        """The critical bug: after swap, both cards must be distinct."""
        low = Card(Rank.TWO, Suit.HEARTS)
        high = Card(Rank.ACE, Suit.SPADES)
        # card1=2h, card2=As -> should swap to card1=As, card2=2h
        hc = HoleCards(low, high)
        assert hc.card1 == high, f"card1 should be Ace, got {hc.card1}"
        assert hc.card2 == low, f"card2 should be Two, got {hc.card2}"

    def test_no_swap_when_already_ordered(self):
        """When card1 is already higher, no swap needed."""
        high = Card(Rank.KING, Suit.DIAMONDS)
        low = Card(Rank.SEVEN, Suit.CLUBS)
        hc = HoleCards(high, low)
        assert hc.card1 == high
        assert hc.card2 == low

    def test_swap_with_same_rank_different_suit(self):
        """Same rank cards should use suit for ordering."""
        c1 = Card(Rank.TEN, Suit.CLUBS)    # clubs = lower suit value
        c2 = Card(Rank.TEN, Suit.SPADES)   # spades = higher suit value
        hc = HoleCards(c1, c2)
        # Should have higher suit value first
        assert hc.card1.suit.value >= hc.card2.suit.value

    def test_hole_cards_notation_after_swap(self):
        """Notation should reflect canonical ordering."""
        low = Card(Rank.FIVE, Suit.HEARTS)
        high = Card(Rank.ACE, Suit.HEARTS)
        hc = HoleCards(low, high)
        # Should start with ace
        assert hc.card1.rank == Rank.ACE
        assert hc.card2.rank == Rank.FIVE

    def test_iteration_after_swap(self):
        """Iterating over HoleCards should yield both distinct cards."""
        c1 = Card(Rank.THREE, Suit.DIAMONDS)
        c2 = Card(Rank.QUEEN, Suit.HEARTS)
        hc = HoleCards(c1, c2)
        cards = list(hc)
        assert len(cards) == 2
        assert cards[0] != cards[1], "Both cards should be distinct after swap"
        assert set(cards) == {c1, c2}


# =============================================================================
# 2. expected_value_call signature fix
# =============================================================================

class TestExpectedValueCallSignature:
    """The convenience function should accept bb_size parameter."""

    def test_with_3_args(self):
        """Original 3-arg call should still work."""
        result = expected_value_call(100, 50, 0.5)
        assert result.ev == pytest.approx(25.0)
        assert result.is_profitable is True

    def test_with_4_args(self):
        """New 4-arg call (with bb_size) should work."""
        result = expected_value_call(100, 50, 0.5, bb_size=10.0)
        assert result.ev == pytest.approx(25.0)
        assert result.ev_bb == pytest.approx(2.5)

    def test_bb_size_affects_ev_bb(self):
        """Different bb_size should change ev_bb but not ev."""
        r1 = expected_value_call(100, 50, 0.5, bb_size=1.0)
        r2 = expected_value_call(100, 50, 0.5, bb_size=10.0)
        assert r1.ev == r2.ev
        assert r1.ev_bb != r2.ev_bb


# =============================================================================
# 3. process_game_state dedup fix
# =============================================================================

class TestProcessGameStateDedup:
    """BeliefRevisionEngine should not reprocess already-seen actions."""

    def test_no_duplicate_observations(self):
        """Calling process_game_state twice should not double-process."""
        engine = BeliefRevisionEngine()
        belief_state = BeliefState()

        # Create a game state with some actions
        stacks = {pos: 1000.0 for pos in Position}
        game = create_6max_game(
            hand_id="test1", hero_id="hero", hero_position=Position.BTN,
            hero_cards=HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)),
            stacks=stacks, small_blind=5.0, big_blind=10.0
        )

        # Simulate a villain action in the action history
        from src.core.game_state import ActionRecord
        game.action_history.append(ActionRecord(
            player_id="villain_utg", position=Position.UTG,
            street=Street.PREFLOP, action=Action.raise_to(30),
            pot_before=15, pot_after=45
        ))

        # Process once
        engine.process_game_state(game, belief_state)
        obs_count_after_first = len(engine.observation_history)

        # Process again - should NOT add duplicate observations
        engine.process_game_state(game, belief_state)
        obs_count_after_second = len(engine.observation_history)

        assert obs_count_after_second == obs_count_after_first, \
            "Should not reprocess already-seen actions"

    def test_new_actions_are_processed(self):
        """New actions added between calls should be processed."""
        engine = BeliefRevisionEngine()
        belief_state = BeliefState()

        stacks = {pos: 1000.0 for pos in Position}
        game = create_6max_game(
            hand_id="test2", hero_id="hero", hero_position=Position.BTN,
            hero_cards=HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)),
            stacks=stacks, small_blind=5.0, big_blind=10.0
        )

        from src.core.game_state import ActionRecord
        game.action_history.append(ActionRecord(
            player_id="villain_utg", position=Position.UTG,
            street=Street.PREFLOP, action=Action.raise_to(30),
            pot_before=15, pot_after=45
        ))

        engine.process_game_state(game, belief_state)
        count_first = len(engine.observation_history)

        # Add another action
        game.action_history.append(ActionRecord(
            player_id="villain_hj", position=Position.HJ,
            street=Street.PREFLOP, action=Action.call(30),
            pot_before=45, pot_after=75
        ))

        engine.process_game_state(game, belief_state)
        count_second = len(engine.observation_history)

        assert count_second == count_first + 1, \
            "New action should be processed"

    def test_reset_action_tracking(self):
        """reset_action_tracking should allow reprocessing for new hands."""
        engine = BeliefRevisionEngine()
        belief_state = BeliefState()

        stacks = {pos: 1000.0 for pos in Position}
        game = create_6max_game(
            hand_id="test3", hero_id="hero", hero_position=Position.BTN,
            hero_cards=HoleCards(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)),
            stacks=stacks, small_blind=5.0, big_blind=10.0
        )

        from src.core.game_state import ActionRecord
        game.action_history.append(ActionRecord(
            player_id="villain_utg", position=Position.UTG,
            street=Street.PREFLOP, action=Action.raise_to(30),
            pot_before=15, pot_after=45
        ))

        engine.process_game_state(game, belief_state)
        engine.reset_action_tracking()
        assert engine._processed_action_count == 0


# =============================================================================
# 4. _parse_params quoted string fix
# =============================================================================

class TestParseParamsQuotedStrings:
    """The param parser should handle commas inside quoted strings."""

    def _make_agent(self):
        """Create a minimal ReActAgent for testing _parse_params."""
        from src.agent.react import ReActAgent
        from src.agent.llm_client import create_mock_client
        mock = create_mock_client()
        return ReActAgent(llm_call=mock)

    def test_simple_params(self):
        agent = self._make_agent()
        result = agent._parse_params("x=1, y=2")
        assert result["x"] == 1.0
        assert result["y"] == 2.0

    def test_quoted_string_with_commas(self):
        """Key fix: commas inside quotes should not split."""
        agent = self._make_agent()
        result = agent._parse_params('vs_range="AA,KK,QQ", equity=0.65')
        assert result["vs_range"] == "AA,KK,QQ"
        assert result["equity"] == 0.65

    def test_single_quoted_string_with_commas(self):
        agent = self._make_agent()
        result = agent._parse_params("range='JJ+,AKs,AKo', num=1000")
        assert result["range"] == "JJ+,AKs,AKo"
        assert result["num"] == 1000.0

    def test_empty_params(self):
        agent = self._make_agent()
        result = agent._parse_params("")
        assert result == {}

    def test_string_without_quotes(self):
        agent = self._make_agent()
        result = agent._parse_params("situation=open")
        assert result["situation"] == "open"


# =============================================================================
# 5. Hand narrative tracing
# =============================================================================

class TestHandNarrative:
    """Test the interspersed game+LLM narrative system."""

    def test_basic_narrative_rendering(self):
        cards = HoleCards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS)
        )
        narrative = HandNarrative("test1", "hero", Position.BTN, cards)
        narrative.add_street(Street.PREFLOP, "", 15)
        narrative.add_action("villain_utg", Position.UTG, Action.fold(), 15)
        narrative.add_action("villain_co", Position.CO, Action.raise_to(25), 40)

        rendered = narrative.render()
        assert "Hand test1" in rendered
        assert "BTN" in rendered
        assert "PREFLOP" in rendered
        assert "UTG" in rendered
        assert "CO" in rendered

    def test_hero_decision_in_narrative(self):
        cards = HoleCards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS)
        )
        narrative = HandNarrative("test2", "hero", Position.BTN, cards)
        narrative.add_street(Street.PREFLOP, "", 15)

        # Create a game state for hero decision
        stacks = {pos: 1000.0 for pos in Position}
        game = create_6max_game(
            hand_id="test2", hero_id="hero", hero_position=Position.BTN,
            hero_cards=cards, stacks=stacks, small_blind=5.0, big_blind=10.0
        )

        hero_dec = narrative.begin_hero_decision(game)
        hero_dec.add_thought("AKo is a premium hand")
        hero_dec.add_tool_call("equity_calc", "62.3% equity")
        hero_dec.set_decision(Action.raise_to(75), 0.85, True)
        narrative.end_hero_decision()

        rendered = narrative.render()
        assert "HERO TURN" in rendered
        assert "AKo is a premium hand" in rendered
        assert "equity_calc" in rendered
        assert "62.3% equity" in rendered
        assert "raise" in rendered.lower()

    def test_narrative_result(self):
        cards = HoleCards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS)
        )
        narrative = HandNarrative("test3", "hero", Position.BTN, cards)
        narrative.set_result(15.5, went_to_showdown=True)
        rendered = narrative.render()
        assert "WON" in rendered
        assert "+15.5" in rendered
        assert "showdown" in rendered

    def test_narrative_tracer_session(self, tmp_path):
        tracer = NarrativeTracer(output_dir=str(tmp_path / "traces"))
        cards = HoleCards(
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS)
        )
        narrative = tracer.start_hand("h1", "hero", Position.BTN, cards)
        narrative.add_street(Street.PREFLOP, "", 15)
        narrative.add_action("v1", Position.UTG, Action.fold(), 15)
        tracer.end_hand(5.0, False)

        assert len(tracer.narratives) == 1
        filepath = tracer.save_session_summary("sess1")
        assert "sess1" in filepath

    def test_hero_decision_narrative_render(self):
        dec = HeroDecisionNarrative(
            street=Street.FLOP, position=Position.BTN,
            pot=100, to_call=50, stack=950,
            hole_cards="AhKd", board="Ah 7c 2d"
        )
        dec.add_thought("Top pair top kicker")
        dec.add_tool_call("pot_odds", "need 33% equity")
        dec.set_decision(Action.call(50), 0.75, True)

        rendered = dec.render()
        assert "HERO TURN (BTN)" in rendered
        assert "pot 100" in rendered
        assert "board Ah 7c 2d" in rendered
        assert "Top pair top kicker" in rendered
        assert "call" in rendered.lower()
