"""Tests for Subjective Logic implementation."""

import pytest
import math
from src.core.subjective_logic import (
    Opinion, Belief, sort_beliefs_by_knowledge, format_beliefs_for_agent
)


class TestOpinion:
    """Tests for Opinion class."""

    def test_opinion_creation(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        assert op.belief == 0.5
        assert op.disbelief == 0.3
        assert op.uncertainty == 0.2
        assert op.base_rate == 0.5

    def test_opinion_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            Opinion(0.5, 0.5, 0.5, 0.5)

    def test_opinion_bounds(self):
        with pytest.raises(ValueError):
            Opinion(-0.1, 0.5, 0.6, 0.5)

        with pytest.raises(ValueError):
            Opinion(0.5, 1.5, -1.0, 0.5)

    def test_knowledge_property(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        assert op.knowledge == 0.8  # b + d = 0.5 + 0.3

        vacuous = Opinion.vacuous()
        assert vacuous.knowledge == 0.0

    def test_projected_probability(self):
        # P = b + a*u
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        expected = 0.5 + 0.5 * 0.2  # 0.6
        assert math.isclose(op.projected_probability, expected)

    def test_vacuous_opinion(self):
        vacuous = Opinion.vacuous()
        assert vacuous.belief == 0.0
        assert vacuous.disbelief == 0.0
        assert vacuous.uncertainty == 1.0
        assert vacuous.is_vacuous

    def test_dogmatic_belief(self):
        dogmatic = Opinion.dogmatic_belief()
        assert dogmatic.belief == 1.0
        assert dogmatic.disbelief == 0.0
        assert dogmatic.uncertainty == 0.0
        assert dogmatic.is_dogmatic

    def test_dogmatic_disbelief(self):
        dogmatic = Opinion.dogmatic_disbelief()
        assert dogmatic.belief == 0.0
        assert dogmatic.disbelief == 1.0
        assert dogmatic.uncertainty == 0.0
        assert dogmatic.is_dogmatic

    def test_from_probability(self):
        # 70% probability, 80% confidence
        op = Opinion.from_probability(0.7, confidence=0.8)
        assert math.isclose(op.uncertainty, 0.2)
        assert math.isclose(op.belief, 0.8 * 0.7)  # 0.56
        assert math.isclose(op.disbelief, 0.8 * 0.3)  # 0.24

    def test_to_tuple_str(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        assert op.to_tuple_str() == "(b=0.50, d=0.30, u=0.20)"

    def test_str_format(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        assert str(op) == "(b=0.50, d=0.30, u=0.20)"


class TestCumulativeFusion:
    """Tests for cumulative fusion operator."""

    def test_fusion_reduces_uncertainty(self):
        op1 = Opinion(0.4, 0.2, 0.4, 0.5)
        op2 = Opinion(0.5, 0.1, 0.4, 0.5)

        fused = op1.cumulative_fusion(op2)

        # Uncertainty should decrease after fusion
        assert fused.uncertainty < op1.uncertainty
        assert fused.uncertainty < op2.uncertainty

    def test_fusion_with_vacuous(self):
        # Fusing with vacuous should return similar to original
        op = Opinion(0.6, 0.2, 0.2, 0.5)
        vacuous = Opinion.vacuous()

        fused = op.cumulative_fusion(vacuous)

        # Should be close to original (vacuous adds no info)
        assert math.isclose(fused.belief, op.belief, rel_tol=0.01)
        assert math.isclose(fused.disbelief, op.disbelief, rel_tol=0.01)

    def test_fusion_operator_overload(self):
        op1 = Opinion(0.4, 0.2, 0.4, 0.5)
        op2 = Opinion(0.5, 0.1, 0.4, 0.5)

        fused1 = op1.cumulative_fusion(op2)
        fused2 = op1 + op2

        assert fused1.belief == fused2.belief
        assert fused1.disbelief == fused2.disbelief
        assert fused1.uncertainty == fused2.uncertainty

    def test_fusion_preserves_constraint(self):
        op1 = Opinion(0.4, 0.2, 0.4, 0.5)
        op2 = Opinion(0.3, 0.3, 0.4, 0.5)

        fused = op1 + op2

        # Must still sum to 1
        total = fused.belief + fused.disbelief + fused.uncertainty
        assert math.isclose(total, 1.0)

    def test_fusion_of_agreeing_opinions(self):
        # Two opinions that both believe
        op1 = Opinion(0.7, 0.1, 0.2, 0.5)
        op2 = Opinion(0.6, 0.1, 0.3, 0.5)

        fused = op1 + op2

        # Belief should be reinforced
        assert fused.belief > 0.5

    def test_fusion_of_conflicting_opinions(self):
        # One believes, one disbelieves
        op1 = Opinion(0.7, 0.1, 0.2, 0.5)
        op2 = Opinion(0.1, 0.7, 0.2, 0.5)

        fused = op1 + op2

        # Result should be more uncertain or middle ground
        assert fused.uncertainty < 0.2  # Still decreases


class TestTrustDiscount:
    """Tests for trust discounting operator."""

    def test_full_trust_preserves_opinion(self):
        op = Opinion(0.6, 0.2, 0.2, 0.5)
        full_trust = Opinion.dogmatic_belief()

        discounted = op.trust_discount(full_trust)

        assert math.isclose(discounted.belief, op.belief, rel_tol=0.01)
        assert math.isclose(discounted.disbelief, op.disbelief, rel_tol=0.01)

    def test_no_trust_gives_vacuous(self):
        op = Opinion(0.6, 0.2, 0.2, 0.5)
        no_trust = Opinion.dogmatic_disbelief()

        discounted = op.trust_discount(no_trust)

        # Should be close to vacuous
        assert discounted.uncertainty > 0.99

    def test_discount_by_factor(self):
        op = Opinion(0.6, 0.2, 0.2, 0.5)

        # 50% discount
        discounted = op.discount_by_factor(0.5)

        assert math.isclose(discounted.belief, 0.3)  # 0.6 * 0.5
        assert math.isclose(discounted.disbelief, 0.1)  # 0.2 * 0.5
        assert math.isclose(discounted.uncertainty, 0.6)  # 1 - 0.4


class TestNegation:
    """Tests for negation operator."""

    def test_negation_swaps_belief_disbelief(self):
        op = Opinion(0.6, 0.2, 0.2, 0.4)
        neg = op.negation()

        assert neg.belief == op.disbelief
        assert neg.disbelief == op.belief
        assert neg.uncertainty == op.uncertainty

    def test_negation_inverts_base_rate(self):
        op = Opinion(0.6, 0.2, 0.2, 0.3)
        neg = op.negation()

        assert neg.base_rate == 0.7  # 1 - 0.3

    def test_negation_operator_overload(self):
        op = Opinion(0.6, 0.2, 0.2, 0.4)
        neg1 = op.negation()
        neg2 = -op

        assert neg1.belief == neg2.belief
        assert neg1.disbelief == neg2.disbelief

    def test_double_negation(self):
        op = Opinion(0.6, 0.2, 0.2, 0.4)
        double_neg = -(-op)

        assert math.isclose(double_neg.belief, op.belief)
        assert math.isclose(double_neg.disbelief, op.disbelief)


class TestBelief:
    """Tests for Belief wrapper class."""

    def test_belief_creation(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        belief = Belief("is aggressive", op, "player_type")

        assert belief.label == "is aggressive"
        assert belief.opinion == op
        assert belief.category == "player_type"

    def test_belief_knowledge(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        belief = Belief("test", op)

        assert belief.knowledge == op.knowledge

    def test_belief_to_agent_format(self):
        op = Opinion(0.5, 0.3, 0.2, 0.5)
        belief = Belief("is aggressive", op)

        formatted = belief.to_agent_format()
        assert "is aggressive" in formatted
        assert "(b=0.50, d=0.30, u=0.20)" in formatted


class TestBeliefSorting:
    """Tests for belief sorting and formatting."""

    def test_sort_by_knowledge(self):
        b1 = Belief("high knowledge", Opinion(0.6, 0.3, 0.1, 0.5))  # k=0.9
        b2 = Belief("low knowledge", Opinion(0.2, 0.1, 0.7, 0.5))   # k=0.3
        b3 = Belief("mid knowledge", Opinion(0.4, 0.2, 0.4, 0.5))   # k=0.6

        sorted_beliefs = sort_beliefs_by_knowledge([b1, b2, b3])

        assert sorted_beliefs[0].label == "high knowledge"
        assert sorted_beliefs[1].label == "mid knowledge"
        assert sorted_beliefs[2].label == "low knowledge"

    def test_format_beliefs_for_agent(self):
        b1 = Belief("belief1", Opinion(0.6, 0.3, 0.1, 0.5))
        b2 = Belief("belief2", Opinion(0.4, 0.2, 0.4, 0.5))

        formatted = format_beliefs_for_agent([b1, b2])

        assert "belief1" in formatted
        assert "belief2" in formatted


class TestDeduction:
    """Tests for deduction operator."""

    def test_deduction_basic(self):
        # If X then likely Y, if not X then unlikely Y
        x_opinion = Opinion(0.7, 0.1, 0.2, 0.5)
        y_given_x = Opinion(0.8, 0.1, 0.1, 0.5)
        y_given_not_x = Opinion(0.1, 0.7, 0.2, 0.5)

        y_opinion = x_opinion.deduction(y_given_x, y_given_not_x)

        # Y should be likely since X is likely and Y given X is likely
        assert y_opinion.projected_probability > 0.5


class TestConfidenceWeightedValue:
    """Tests for confidence weighted value."""

    def test_high_belief_high_knowledge(self):
        op = Opinion(0.8, 0.1, 0.1, 0.5)
        # k=0.9, b-d=0.7, value = 0.9 * 0.7 = 0.63
        assert op.confidence_weighted_value() > 0.5

    def test_high_disbelief_high_knowledge(self):
        op = Opinion(0.1, 0.8, 0.1, 0.5)
        # k=0.9, b-d=-0.7, value = 0.9 * -0.7 = -0.63
        assert op.confidence_weighted_value() < -0.5

    def test_vacuous_has_zero_value(self):
        vacuous = Opinion.vacuous()
        assert vacuous.confidence_weighted_value() == 0.0
