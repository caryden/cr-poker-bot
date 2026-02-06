"""
Standardized villain ID management.

This module provides a consistent way to create and manage villain IDs
throughout the codebase. The game state uses position-based IDs like
'villain_UTG', 'villain_BB', etc. This module ensures all components
use the same ID format.

Usage:
    from src.game.villain_manager import VillainManager, create_villain_table

    # Create a table of villains (excluding hero position)
    villains = create_villain_table(
        hero_position=Position.BTN,
        villain_types={
            Position.UTG: CallingStation,
            Position.HJ: TightPassive,
            ...
        }
    )

    # Or use VillainManager for more control
    manager = VillainManager()
    manager.add_villain(Position.UTG, CallingStation)
    villains = manager.get_villains(hero_position=Position.BTN)
"""

from typing import Optional, Callable, Type, Union
from dataclasses import dataclass, field

from ..core.primitives import Position


def villain_id_for_position(position: Position) -> str:
    """
    Get the standard villain ID for a position.

    This is the canonical way to generate villain IDs.
    The game state uses this format, so all code should too.

    Args:
        position: The position of the villain

    Returns:
        Standard villain ID like 'villain_UTG'
    """
    return f"villain_{position.value}"


def position_from_villain_id(villain_id: str) -> Optional[Position]:
    """
    Extract position from a villain ID.

    Args:
        villain_id: A villain ID like 'villain_UTG'

    Returns:
        Position enum or None if not a valid villain ID
    """
    if not villain_id.startswith("villain_"):
        return None

    pos_str = villain_id.replace("villain_", "")
    try:
        return Position(pos_str)
    except ValueError:
        return None


def is_villain_id(player_id: str) -> bool:
    """Check if a player ID is a villain ID (not hero)."""
    return player_id.startswith("villain_")


@dataclass
class VillainConfig:
    """Configuration for a villain at a position."""
    position: Position
    player_class: type
    player_id: str = field(init=False)

    def __post_init__(self):
        self.player_id = villain_id_for_position(self.position)

    def create_player(self):
        """Create the player instance with proper ID."""
        return self.player_class(player_id=self.player_id)


class VillainManager:
    """
    Manages villain creation and ID mapping.

    Ensures consistent villain IDs across game state, runner, and belief tracking.
    """

    # All 6-max positions
    ALL_POSITIONS = [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]

    def __init__(self):
        self._configs: dict[Position, VillainConfig] = {}
        self._default_class: Optional[type] = None

    def set_default_type(self, player_class: type) -> 'VillainManager':
        """Set default player class for positions without explicit config."""
        self._default_class = player_class
        return self

    def add_villain(self, position: Position, player_class: type) -> 'VillainManager':
        """Add a villain at a specific position."""
        self._configs[position] = VillainConfig(position, player_class)
        return self

    def add_villains(self, villain_types: dict[Position, type]) -> 'VillainManager':
        """Add multiple villains."""
        for pos, cls in villain_types.items():
            self.add_villain(pos, cls)
        return self

    def get_villain_id(self, position: Position) -> str:
        """Get the standard villain ID for a position."""
        return villain_id_for_position(position)

    def get_villains(self, hero_position: Position) -> dict[str, object]:
        """
        Create villain dict for use with HandRunner/SessionRunner.

        Args:
            hero_position: Hero's position (villains at this position excluded)

        Returns:
            Dict mapping villain IDs to player instances
        """
        villains = {}
        for pos in self.ALL_POSITIONS:
            if pos == hero_position:
                continue

            if pos in self._configs:
                config = self._configs[pos]
                villains[config.player_id] = config.create_player()
            elif self._default_class:
                vid = villain_id_for_position(pos)
                villains[vid] = self._default_class(player_id=vid)

        return villains

    def get_all_villain_ids(self, hero_position: Position) -> list[str]:
        """Get list of all villain IDs for a given hero position."""
        return [
            villain_id_for_position(pos)
            for pos in self.ALL_POSITIONS
            if pos != hero_position
        ]

    def get_ground_truth(self, type_mapping: dict[type, str]) -> dict[str, str]:
        """
        Get ground truth mapping from villain IDs to type names.

        Args:
            type_mapping: Maps player classes to type names
                          e.g., {CallingStation: 'Fish', TagBot: 'TAG'}

        Returns:
            Dict mapping villain IDs to type names
        """
        ground_truth = {}
        for pos, config in self._configs.items():
            if config.player_class in type_mapping:
                ground_truth[config.player_id] = type_mapping[config.player_class]
        return ground_truth


def create_villain_table(
    hero_position: Position,
    villain_types: Optional[dict[Position, type]] = None,
    default_type: Optional[type] = None,
    fill_all: bool = True
) -> dict[str, object]:
    """
    Create a complete villain table with proper IDs.

    Args:
        hero_position: Hero's position (excluded from villains)
        villain_types: Optional mapping of positions to player classes
        default_type: Default class for unspecified positions
        fill_all: If True, fill all positions with default_type

    Returns:
        Dict mapping villain IDs to player instances

    Example:
        from src.game.opponents import CallingStation, TagBot

        villains = create_villain_table(
            hero_position=Position.BTN,
            villain_types={
                Position.UTG: CallingStation,
                Position.BB: TagBot,
            },
            default_type=CallingStation,
            fill_all=True
        )
        # Returns villains for UTG, HJ, CO, SB, BB (BTN is hero)
    """
    manager = VillainManager()

    if villain_types:
        manager.add_villains(villain_types)

    if default_type and fill_all:
        manager.set_default_type(default_type)

    return manager.get_villains(hero_position)


def create_uniform_villain_table(
    hero_position: Position,
    player_class: type
) -> dict[str, object]:
    """
    Create a table where all villains are the same type.

    Args:
        hero_position: Hero's position
        player_class: Class for all villains

    Returns:
        Dict mapping villain IDs to player instances
    """
    return create_villain_table(
        hero_position=hero_position,
        default_type=player_class,
        fill_all=True
    )
