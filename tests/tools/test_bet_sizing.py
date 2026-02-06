"""Tests for bet sizing heuristics."""

import pytest
from src.core import Position, Street
from src.tools.bet_sizing import (
    HandStrengthCategory, BetPurpose, BetSizing, BetSizingAdvisor,
    get_preflop_sizing, get_value_sizing, get_bluff_sizing
)
from src.tools.board_texture import BoardWetness, BoardPairedness, BoardHighness, BoardTexture


class TestBetSizing:
    """Tests for BetSizing dataclass."""

    def test_get_amount_pot_fraction(self):
        """Test amount calculation from pot fraction."""
        sizing = BetSizing(size_pot_fraction=0.5, purpose=BetPurpose.VALUE)
        amount = sizing.get_amount(pot=100)
        assert amount == 50.0

    def test_get_amount_bb(self):
        """Test amount calculation from BB sizing."""
        sizing = BetSizing(size_pot_fraction=0, size_bb=3.0, purpose=BetPurpose.VALUE)
        amount = sizing.get_amount(pot=100, big_blind=2.0)
        assert amount == 6.0

    def test_string_representation(self):
        """Test string formatting."""
        sizing = BetSizing(
            size_pot_fraction=0.67,
            purpose=BetPurpose.VALUE,
            reasoning="Test sizing"
        )
        s = str(sizing)
        assert "67%" in s
        assert "VALUE" in s

    def test_bb_string_representation(self):
        """Test string formatting for BB sizing."""
        sizing = BetSizing(
            size_pot_fraction=0,
            size_bb=2.5,
            purpose=BetPurpose.VALUE,
            reasoning="Open raise"
        )
        s = str(sizing)
        assert "2.5BB" in s


class TestBetSizingAdvisor:
    """Tests for BetSizingAdvisor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advisor = BetSizingAdvisor()

    def test_preflop_open_utg(self):
        """Test preflop opening sizing from UTG."""
        sizing = self.advisor.get_preflop_sizing(Position.UTG)
        assert sizing.size_bb == 2.5
        assert sizing.purpose == BetPurpose.VALUE

    def test_preflop_open_sb(self):
        """Test preflop opening sizing from SB (larger)."""
        sizing = self.advisor.get_preflop_sizing(Position.SB)
        assert sizing.size_bb == 3.0

    def test_preflop_3bet(self):
        """Test 3-bet sizing."""
        sizing = self.advisor.get_preflop_sizing(
            Position.BTN, facing_raise=True, raise_amount=2.5
        )
        # 3x the open
        assert sizing.size_bb == 7.5

    def test_preflop_4bet(self):
        """Test 4-bet sizing."""
        sizing = self.advisor.get_preflop_sizing(
            Position.CO, facing_raise=True, raise_amount=7.5
        )
        # 2.5x the 3bet
        assert sizing.size_bb == 18.75

    def test_flop_sizing_dry_board_value(self):
        """Test flop sizing on dry board with value hand."""
        texture = BoardTexture(
            wetness=BoardWetness.DRY,
            pairedness=BoardPairedness.UNPAIRED,
            highness=BoardHighness.HIGH,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            monotone=False,
            rainbow=True,
            connected=False,
            high_card=None,
            num_cards=3
        )

        sizing = self.advisor.get_flop_sizing(
            texture,
            hand_strength=HandStrengthCategory.STRONG,
            is_aggressor=True,
            spr=10
        )

        # Small bet on dry board
        assert sizing.size_pot_fraction <= 0.5
        assert sizing.purpose == BetPurpose.VALUE

    def test_flop_sizing_wet_board(self):
        """Test flop sizing on wet board."""
        texture = BoardTexture(
            wetness=BoardWetness.WET,
            pairedness=BoardPairedness.UNPAIRED,
            highness=BoardHighness.MEDIUM,
            flush_possible=False,
            flush_draw_possible=True,
            straight_possible=False,
            straight_draw_possible=True,
            monotone=False,
            rainbow=False,
            connected=True,
            high_card=None,
            num_cards=3
        )

        sizing = self.advisor.get_flop_sizing(
            texture,
            hand_strength=HandStrengthCategory.VERY_STRONG,
            is_aggressor=True,
            spr=10
        )

        # Larger bet on wet board
        assert sizing.size_pot_fraction >= 0.67

    def test_low_spr_sizing(self):
        """Test sizing with low stack-to-pot ratio."""
        texture = BoardTexture(
            wetness=BoardWetness.NEUTRAL,
            pairedness=BoardPairedness.UNPAIRED,
            highness=BoardHighness.MEDIUM,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            monotone=False,
            rainbow=True,
            connected=False,
            high_card=None,
            num_cards=3
        )

        sizing = self.advisor.get_flop_sizing(
            texture,
            hand_strength=HandStrengthCategory.STRONG,
            is_aggressor=True,
            spr=2  # Low SPR
        )

        # Commit with strong hands at low SPR
        assert sizing.size_pot_fraction >= 0.67

    def test_river_monster_sizing(self):
        """Test river sizing with monster hand."""
        texture = BoardTexture(
            wetness=BoardWetness.NEUTRAL,
            pairedness=BoardPairedness.UNPAIRED,
            highness=BoardHighness.MEDIUM,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            monotone=False,
            rainbow=True,
            connected=False,
            high_card=None,
            num_cards=5
        )

        sizing = self.advisor.get_river_sizing(
            texture,
            hand_strength=HandStrengthCategory.MONSTER,
            pot=100,
            stack=80  # SPR < 1
        )

        # Shove with monster at low SPR
        assert sizing.size_pot_fraction >= 0.75
        assert sizing.purpose == BetPurpose.VALUE

    def test_river_bluff_sizing(self):
        """Test river sizing for bluffs."""
        texture = BoardTexture(
            wetness=BoardWetness.NEUTRAL,
            pairedness=BoardPairedness.UNPAIRED,
            highness=BoardHighness.MEDIUM,
            flush_possible=True,
            flush_draw_possible=False,
            straight_possible=True,
            straight_draw_possible=False,
            monotone=False,
            rainbow=False,
            connected=False,
            high_card=None,
            num_cards=5
        )

        sizing = self.advisor.get_river_sizing(
            texture,
            hand_strength=HandStrengthCategory.TRASH,
            pot=100,
            stack=200
        )

        # Polarized bluff sizing
        assert sizing.purpose == BetPurpose.BLUFF
        assert sizing.size_pot_fraction >= 0.67

    def test_raise_sizing(self):
        """Test raise sizing when facing a bet."""
        sizing = self.advisor.get_raise_sizing(
            facing_bet=30,
            pot=50,
            street=Street.FLOP,
            hand_strength=HandStrengthCategory.VERY_STRONG
        )

        # Should be approximately 3x on flop
        assert sizing.purpose == BetPurpose.VALUE


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_preflop_sizing(self):
        """Test preflop sizing convenience function."""
        sizing = get_preflop_sizing(Position.BTN)
        assert sizing.size_bb is not None
        assert sizing.size_bb > 0

    def test_get_value_sizing(self):
        """Test value sizing by street."""
        flop_size = get_value_sizing(100, Street.FLOP)
        turn_size = get_value_sizing(100, Street.TURN)
        river_size = get_value_sizing(100, Street.RIVER)

        # Sizes should increase by street
        assert flop_size < turn_size < river_size

    def test_get_bluff_sizing(self):
        """Test bluff sizing matches value sizing."""
        value = get_value_sizing(100, Street.RIVER)
        bluff = get_bluff_sizing(100, Street.RIVER)

        # Should be balanced
        assert value == bluff
