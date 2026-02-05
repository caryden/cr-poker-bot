"""Tests for board texture analyzer."""

import pytest
from src.core import Card, Board, Rank, Suit
from src.tools.board_texture import (
    BoardWetness, BoardPairedness, BoardHighness, BoardTexture,
    BoardAnalyzer, analyze_board, is_favorable_cbet_board
)


class TestBoardAnalyzer:
    """Tests for BoardAnalyzer class."""

    def test_dry_rainbow_board(self):
        """Test analysis of a dry rainbow board."""
        board = Board(cards=[
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)

        assert texture.wetness in (BoardWetness.VERY_DRY, BoardWetness.DRY)
        assert texture.pairedness == BoardPairedness.UNPAIRED
        assert texture.rainbow is True
        assert texture.monotone is False
        assert texture.flush_possible is False

    def test_wet_connected_board(self):
        """Test analysis of a wet connected board."""
        board = Board(cards=[
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)

        assert texture.wetness in (BoardWetness.WET, BoardWetness.VERY_WET, BoardWetness.NEUTRAL)
        assert texture.connected is True
        assert texture.flush_draw_possible is True

    def test_monotone_board(self):
        """Test analysis of a monotone (all one suit) board."""
        board = Board(cards=[
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.TEN, Suit.CLUBS),
            Card(Rank.SIX, Suit.CLUBS),
        ])
        texture = analyze_board(board)

        assert texture.monotone is True
        assert texture.flush_possible is True
        assert texture.wetness in (BoardWetness.WET, BoardWetness.VERY_WET)

    def test_paired_board(self):
        """Test analysis of a paired board."""
        board = Board(cards=[
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.FOUR, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)

        assert texture.pairedness == BoardPairedness.PAIRED

    def test_trips_board(self):
        """Test analysis of a board with trips."""
        board = Board(cards=[
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.JACK, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)

        assert texture.pairedness == BoardPairedness.TRIPS

    def test_high_board(self):
        """Test classification of high boards."""
        board = Board(cards=[
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)

        assert texture.highness == BoardHighness.HIGH
        assert texture.high_card == Rank.ACE

    def test_low_board(self):
        """Test classification of low boards."""
        board = Board(cards=[
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)

        assert texture.highness == BoardHighness.LOW

    def test_turn_board(self):
        """Test analysis of turn (4 cards)."""
        board = Board(cards=[
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.DIAMONDS),
            Card(Rank.TWO, Suit.CLUBS),
        ])
        texture = analyze_board(board)

        assert texture.num_cards == 4
        assert texture.rainbow is True

    def test_river_board(self):
        """Test analysis of river (5 cards)."""
        board = Board(cards=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.TWO, Suit.HEARTS),
        ])
        texture = analyze_board(board)

        assert texture.num_cards == 5
        assert texture.monotone is True
        assert texture.flush_possible is True

    def test_empty_board(self):
        """Test analysis of empty board."""
        board = Board()
        texture = analyze_board(board)

        assert texture.num_cards == 0
        assert texture.wetness == BoardWetness.NEUTRAL


class TestBoardTextureProperties:
    """Tests for BoardTexture properties."""

    def test_is_draw_heavy(self):
        """Test draw-heavy board detection."""
        # Wet board
        wet_board = Board(cards=[
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
        ])
        texture = analyze_board(wet_board)
        assert texture.is_draw_heavy is True

    def test_is_static(self):
        """Test static board detection."""
        dry_board = Board(cards=[
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
        ])
        texture = analyze_board(dry_board)
        assert texture.is_static is True

    def test_favors_aggressor(self):
        """Test aggressor-favorable board detection."""
        # High, dry board - favors aggressor
        board = Board(cards=[
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.THREE, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)
        assert texture.favors_aggressor is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_favorable_cbet_board(self):
        """Test c-bet favorability check."""
        # Dry, high board - favorable
        board = Board(cards=[
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
        ])
        assert is_favorable_cbet_board(board) is True

    def test_texture_string_representation(self):
        """Test texture string formatting."""
        board = Board(cards=[
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
        ])
        texture = analyze_board(board)
        description = str(texture)

        # Should contain relevant info
        assert "Rainbow" in description or "Two-Tone" in description
