"""Tests for belief state management and revision."""

import pytest
import math
from src.core.beliefs import (
    ObservationType, GameContext, Observation,
    ConsistencyOpinion, BeliefCategory,
    PLAYER_TYPES, DEFAULT_PLAYER_TYPE_BASE_RATES,
    ObservationBeliefMapper, SLBeliefReviser,
    VillainBeliefs, BeliefState, BeliefRevisionEngine,
    create_default_villain_beliefs
)
from src.core.subjective_logic import Opinion, MultinomialOpinion
from src.core.primitives import Action, ActionType, Position, Street


class TestPlayerTypeConstants:
    """Tests for player type constants."""

    def test_player_types_defined(self):
        assert "TAG" in PLAYER_TYPES
        assert "LAG" in PLAYER_TYPES
        assert "NIT" in PLAYER_TYPES
        assert "Fish" in PLAYER_TYPES
        assert "Maniac" in PLAYER_TYPES

    def test_base_rates_sum_to_one(self):
        total = sum(DEFAULT_PLAYER_TYPE_BASE_RATES.values())
        assert math.isclose(total, 1.0)

    def test_base_rates_match_player_types(self):
        assert set(DEFAULT_PLAYER_TYPE_BASE_RATES.keys()) == set(PLAYER_TYPES)


class TestVillainBeliefs:
    """Tests for VillainBeliefs class."""

    def test_villain_beliefs_creation(self):
        beliefs = VillainBeliefs("villain1")
        assert beliefs.player_id == "villain1"
        assert beliefs.player_type is not None

    def test_villain_has_multinomial_player_type(self):
        beliefs = VillainBeliefs("villain1")

        assert isinstance(beliefs.player_type, MultinomialOpinion)
        assert set(beliefs.player_type.categories) == set(PLAYER_TYPES)

    def test_player_type_starts_vacuous(self):
        beliefs = VillainBeliefs("villain1")

        assert beliefs.player_type.is_vacuous
        assert beliefs.player_type.uncertainty > 0.99

    def test_update_player_type(self):
        beliefs = VillainBeliefs("villain1")

        beliefs.update_player_type("Fish", strength=0.3)

        # Uncertainty should decrease
        assert beliefs.player_type.uncertainty < 1.0
        # Fish belief should increase
        assert beliefs.player_type.beliefs["Fish"] > 0

    def test_update_player_type_invalid_raises(self):
        beliefs = VillainBeliefs("villain1")

        with pytest.raises(ValueError, match="Unknown player type"):
            beliefs.update_player_type("InvalidType", strength=0.3)

    def test_get_player_type_distribution(self):
        beliefs = VillainBeliefs("villain1")
        beliefs.update_player_type("TAG", strength=0.4)

        dist = beliefs.get_player_type_distribution()

        assert "TAG" in dist
        assert "Fish" in dist
        assert sum(dist.values()) > 0.99  # Should sum to ~1

    def test_get_most_likely_player_type(self):
        beliefs = VillainBeliefs("villain1")

        # Update with TAG evidence
        for _ in range(3):
            beliefs.update_player_type("TAG", strength=0.2)

        best_type, prob = beliefs.get_most_likely_player_type()

        assert best_type == "TAG"
        assert prob > 0.3  # Should be significant

    def test_add_binomial_belief(self):
        beliefs = VillainBeliefs("villain1")

        beliefs.add_belief("is aggressive", Opinion(0.5, 0.2, 0.3, 0.4))

        assert "is aggressive" in beliefs.beliefs
        assert beliefs.get_belief("is aggressive") is not None

    def test_to_agent_format_includes_player_type(self):
        beliefs = VillainBeliefs("villain1")
        beliefs.update_player_type("LAG", strength=0.4)
        beliefs.add_belief("bluffs often", Opinion(0.4, 0.2, 0.4, 0.3))

        formatted = beliefs.to_agent_format()

        assert "villain1" in formatted
        assert "Type:" in formatted
        assert "LAG" in formatted
        assert "bluffs often" in formatted


class TestCreateDefaultVillainBeliefs:
    """Tests for create_default_villain_beliefs function."""

    def test_creates_beliefs_with_player_id(self):
        beliefs = create_default_villain_beliefs("player123")
        assert beliefs.player_id == "player123"

    def test_creates_vacuous_player_type(self):
        beliefs = create_default_villain_beliefs("player123")
        assert beliefs.player_type.is_vacuous

    def test_creates_tendency_beliefs(self):
        beliefs = create_default_villain_beliefs("player123")

        # Should have aggression beliefs
        assert beliefs.get_belief("is aggressive") is not None
        assert beliefs.get_belief("is passive") is not None

        # Should have bluff frequency beliefs
        assert beliefs.get_belief("bluffs often (high frequency)") is not None

    def test_tendency_beliefs_are_vacuous(self):
        beliefs = create_default_villain_beliefs("player123")

        for belief in beliefs.all_beliefs():
            assert belief.opinion.is_vacuous


class TestBeliefState:
    """Tests for BeliefState class."""

    def test_get_or_create_villain(self):
        state = BeliefState()

        villain_beliefs = state.get_or_create_villain("villain1")

        assert villain_beliefs is not None
        assert villain_beliefs.player_id == "villain1"

    def test_get_or_create_returns_same(self):
        state = BeliefState()

        beliefs1 = state.get_or_create_villain("villain1")
        beliefs2 = state.get_or_create_villain("villain1")

        assert beliefs1 is beliefs2

    def test_to_agent_format(self):
        state = BeliefState()
        state.get_or_create_villain("villain1")

        formatted = state.to_agent_format()

        assert "OPPONENT READS" in formatted
        assert "villain1" in formatted

    def test_default_beliefs_include_tendencies(self):
        """Regression: tournament runner must use create_default_villain_beliefs
        so that binomial tendencies (aggression, bluff freq, etc.) are present."""
        state = BeliefState()
        state.villain_beliefs["v1"] = create_default_villain_beliefs("v1")

        formatted = state.to_agent_format()

        assert "is aggressive" in formatted
        assert "is passive" in formatted
        assert "bluffs often" in formatted
        assert "positionally aware" in formatted

    def test_bare_get_or_create_lacks_tendencies(self):
        """Documents that get_or_create_villain does NOT add tendencies —
        you must use create_default_villain_beliefs."""
        state = BeliefState()
        state.get_or_create_villain("v1")

        assert len(state.villain_beliefs["v1"].beliefs) == 0


class TestObservationBeliefMapper:
    """Tests for ObservationBeliefMapper class."""

    @pytest.fixture
    def mapper(self):
        return ObservationBeliefMapper()

    @pytest.fixture
    def preflop_context(self):
        return GameContext(
            street=Street.PREFLOP,
            position=Position.BTN,
            pot_size=15.0,
            to_call=10.0,
            num_players=4,
            is_heads_up=False,
            facing_raise=True
        )

    def test_compute_player_type_evidence_fold_preflop(self, mapper):
        ctx = GameContext(
            street=Street.PREFLOP,
            position=Position.UTG,
            pot_size=10.0,
            to_call=5.0,
            num_players=6,
            is_heads_up=False
        )
        obs = Observation(
            obs_id="1",
            obs_type=ObservationType.ACTION,
            player_id="villain1",
            action=Action.fold(),
            context=ctx
        )

        evidence = mapper.compute_player_type_evidence(obs)

        assert evidence is not None
        player_type, strength, _ = evidence
        assert player_type == "NIT"  # Folding preflop suggests tight

    def test_compute_player_type_evidence_call_preflop(self, mapper):
        ctx = GameContext(
            street=Street.PREFLOP,
            position=Position.BTN,
            pot_size=10.0,
            to_call=5.0,
            num_players=4,
            is_heads_up=False
        )
        obs = Observation(
            obs_id="1",
            obs_type=ObservationType.ACTION,
            player_id="villain1",
            action=Action.call(5.0),
            context=ctx
        )

        evidence = mapper.compute_player_type_evidence(obs)

        assert evidence is not None
        player_type, _, _ = evidence
        assert player_type == "Fish"  # Calling/limping suggests recreational

    def test_compute_player_type_evidence_4bet(self, mapper):
        ctx = GameContext(
            street=Street.PREFLOP,
            position=Position.CO,
            pot_size=50.0,
            to_call=30.0,
            num_players=3,
            is_heads_up=False,
            facing_raise=True,
            raise_count=3
        )
        obs = Observation(
            obs_id="1",
            obs_type=ObservationType.ACTION,
            player_id="villain1",
            action=Action.raise_to(90.0),
            context=ctx
        )

        evidence = mapper.compute_player_type_evidence(obs)

        assert evidence is not None
        player_type, _, _ = evidence
        assert player_type == "LAG"  # 4-bet+ suggests LAG


class TestBeliefRevisionEngine:
    """Tests for BeliefRevisionEngine class."""

    @pytest.fixture
    def engine(self):
        return BeliefRevisionEngine(trust_discount=0.9)

    def test_process_observation_updates_player_type(self, engine):
        state = BeliefState()
        villain_beliefs = state.get_or_create_villain("villain1")
        initial_uncertainty = villain_beliefs.player_type.uncertainty

        # Observe a fold (suggests NIT)
        ctx = GameContext(
            street=Street.PREFLOP,
            position=Position.UTG,
            pot_size=10.0,
            to_call=5.0,
            num_players=6,
            is_heads_up=False
        )
        obs = Observation(
            obs_id="1",
            obs_type=ObservationType.ACTION,
            player_id="villain1",
            action=Action.fold(),
            context=ctx
        )

        state = engine.process_observation(obs, state)

        # Uncertainty should decrease
        assert state.villain_beliefs["villain1"].player_type.uncertainty < initial_uncertainty
        # NIT belief should increase
        assert state.villain_beliefs["villain1"].player_type.beliefs["NIT"] > 0

    def test_multiple_observations_accumulate(self, engine):
        state = BeliefState()

        # Multiple Fish-like observations (calling)
        ctx = GameContext(
            street=Street.PREFLOP,
            position=Position.BTN,
            pot_size=15.0,
            to_call=10.0,
            num_players=4,
            is_heads_up=False
        )

        for i in range(5):
            obs = Observation(
                obs_id=str(i),
                obs_type=ObservationType.ACTION,
                player_id="villain1",
                action=Action.call(10.0),
                context=ctx
            )
            state = engine.process_observation(obs, state)

        # Should strongly believe Fish
        best_type, prob = state.villain_beliefs["villain1"].get_most_likely_player_type()
        assert best_type == "Fish"
        assert prob > 0.4
