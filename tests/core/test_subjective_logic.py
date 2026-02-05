"""Tests for Subjective Logic implementation."""

import pytest
import math
from src.core.subjective_logic import (
    Opinion, Belief, sort_beliefs_by_knowledge, format_beliefs_for_agent,
    MultinomialOpinion, MultinomialBelief
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


# ============== Multinomial Opinion Tests ==============

class TestMultinomialOpinion:
    """Tests for MultinomialOpinion class."""

    def test_multinomial_creation(self):
        beliefs = {"A": 0.3, "B": 0.2, "C": 0.0}
        base_rates = {"A": 0.4, "B": 0.3, "C": 0.3}
        op = MultinomialOpinion(beliefs, 0.5, base_rates)

        assert op.beliefs["A"] == 0.3
        assert op.uncertainty == 0.5
        assert op.base_rates["A"] == 0.4

    def test_multinomial_constraint_beliefs_plus_uncertainty(self):
        # beliefs + uncertainty must equal 1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MultinomialOpinion(
                {"A": 0.5, "B": 0.5},  # sum = 1.0
                0.5,  # u = 0.5, total = 1.5
                {"A": 0.5, "B": 0.5}
            )

    def test_multinomial_constraint_base_rates(self):
        # base_rates must sum to 1
        with pytest.raises(ValueError, match="Base rates must sum to 1.0"):
            MultinomialOpinion(
                {"A": 0.2, "B": 0.2},
                0.6,
                {"A": 0.3, "B": 0.3}  # sum = 0.6
            )

    def test_multinomial_categories_must_match(self):
        # beliefs and base_rates must have same categories
        with pytest.raises(ValueError, match="same categories"):
            MultinomialOpinion(
                {"A": 0.2, "B": 0.2},
                0.6,
                {"A": 0.5, "C": 0.5}  # different category
            )

    def test_multinomial_vacuous(self):
        categories = ["TAG", "LAG", "NIT", "Fish"]
        op = MultinomialOpinion.vacuous(categories)

        assert op.uncertainty == 1.0
        assert all(b == 0.0 for b in op.beliefs.values())
        assert op.is_vacuous
        # Uniform base rates
        assert all(math.isclose(a, 0.25) for a in op.base_rates.values())

    def test_multinomial_vacuous_with_custom_base_rates(self):
        categories = ["A", "B", "C"]
        base_rates = {"A": 0.5, "B": 0.3, "C": 0.2}
        op = MultinomialOpinion.vacuous(categories, base_rates)

        assert op.base_rates["A"] == 0.5
        assert op.uncertainty == 1.0

    def test_multinomial_projected_probability(self):
        # P(x) = b_x + a_x * u
        beliefs = {"A": 0.3, "B": 0.1}
        base_rates = {"A": 0.6, "B": 0.4}
        op = MultinomialOpinion(beliefs, 0.6, base_rates)

        # P(A) = 0.3 + 0.6 * 0.6 = 0.3 + 0.36 = 0.66
        assert math.isclose(op.projected_probability("A"), 0.66)
        # P(B) = 0.1 + 0.4 * 0.6 = 0.1 + 0.24 = 0.34
        assert math.isclose(op.projected_probability("B"), 0.34)

    def test_multinomial_all_projected_probabilities(self):
        categories = ["A", "B", "C"]
        op = MultinomialOpinion.vacuous(categories)

        probs = op.all_projected_probabilities()

        # With vacuous opinion and uniform priors, all equal
        assert all(math.isclose(p, 1/3) for p in probs.values())

    def test_multinomial_most_likely_category(self):
        beliefs = {"TAG": 0.4, "LAG": 0.1, "Fish": 0.1}
        base_rates = {"TAG": 0.3, "LAG": 0.3, "Fish": 0.4}
        op = MultinomialOpinion(beliefs, 0.4, base_rates)

        best_cat, prob = op.most_likely_category()

        assert best_cat == "TAG"  # Highest projected probability

    def test_multinomial_knowledge(self):
        beliefs = {"A": 0.3, "B": 0.2}
        op = MultinomialOpinion(beliefs, 0.5, {"A": 0.5, "B": 0.5})

        assert op.knowledge == 0.5  # 1 - uncertainty

    def test_multinomial_dogmatic(self):
        categories = ["A", "B", "C"]
        op = MultinomialOpinion.dogmatic("B", categories)

        assert op.uncertainty == 0.0
        assert op.beliefs["B"] == 1.0
        assert op.beliefs["A"] == 0.0
        assert op.is_dogmatic

    def test_multinomial_from_observation(self):
        categories = ["TAG", "LAG", "NIT", "Fish"]
        op = MultinomialOpinion.from_observation("Fish", categories, confidence=0.3)

        assert op.beliefs["Fish"] == 0.3
        assert op.beliefs["TAG"] == 0.0
        assert op.uncertainty == 0.7


class TestMultinomialFusion:
    """Tests for multinomial cumulative fusion."""

    def test_fusion_reduces_uncertainty(self):
        categories = ["A", "B", "C"]
        op1 = MultinomialOpinion.from_observation("A", categories, 0.3)
        op2 = MultinomialOpinion.from_observation("A", categories, 0.3)

        fused = op1.cumulative_fusion(op2)

        # Uncertainty should decrease
        assert fused.uncertainty < op1.uncertainty
        # Belief in A should increase
        assert fused.beliefs["A"] > op1.beliefs["A"]

    def test_fusion_with_vacuous(self):
        categories = ["A", "B"]
        op = MultinomialOpinion.from_observation("A", categories, 0.4)
        vacuous = MultinomialOpinion.vacuous(categories)

        fused = op.cumulative_fusion(vacuous)

        # Should be close to original
        assert math.isclose(fused.beliefs["A"], op.beliefs["A"], rel_tol=0.01)

    def test_fusion_operator_overload(self):
        categories = ["X", "Y"]
        op1 = MultinomialOpinion.from_observation("X", categories, 0.3)
        op2 = MultinomialOpinion.from_observation("X", categories, 0.2)

        fused1 = op1.cumulative_fusion(op2)
        fused2 = op1 + op2

        assert fused1.uncertainty == fused2.uncertainty
        assert fused1.beliefs["X"] == fused2.beliefs["X"]

    def test_fusion_preserves_constraint(self):
        categories = ["A", "B", "C"]
        op1 = MultinomialOpinion.from_observation("A", categories, 0.4)
        op2 = MultinomialOpinion.from_observation("B", categories, 0.3)

        fused = op1 + op2

        # beliefs + uncertainty must equal 1
        total = sum(fused.beliefs.values()) + fused.uncertainty
        assert math.isclose(total, 1.0)

    def test_fusion_different_categories_raises(self):
        op1 = MultinomialOpinion.vacuous(["A", "B"])
        op2 = MultinomialOpinion.vacuous(["X", "Y"])

        with pytest.raises(ValueError, match="different categories"):
            op1.cumulative_fusion(op2)


class TestMultinomialDiscount:
    """Tests for multinomial discount operator."""

    def test_discount_increases_uncertainty(self):
        categories = ["A", "B"]
        op = MultinomialOpinion.from_observation("A", categories, 0.5)

        discounted = op.discount_by_factor(0.5)

        assert discounted.uncertainty > op.uncertainty
        assert discounted.beliefs["A"] < op.beliefs["A"]

    def test_full_discount_preserves(self):
        categories = ["A", "B"]
        op = MultinomialOpinion.from_observation("A", categories, 0.4)

        discounted = op.discount_by_factor(1.0)

        assert math.isclose(discounted.uncertainty, op.uncertainty)
        assert math.isclose(discounted.beliefs["A"], op.beliefs["A"])

    def test_zero_discount_gives_vacuous(self):
        categories = ["A", "B"]
        op = MultinomialOpinion.from_observation("A", categories, 0.6)

        discounted = op.discount_by_factor(0.0)

        assert math.isclose(discounted.uncertainty, 1.0)


class TestMultinomialUpdateFromEvidence:
    """Tests for update_from_evidence convenience method."""

    def test_update_from_evidence(self):
        categories = ["TAG", "LAG", "Fish"]
        op = MultinomialOpinion.vacuous(categories)

        updated = op.update_from_evidence("Fish", strength=0.3)

        # Belief in Fish should increase
        assert updated.beliefs["Fish"] > op.beliefs["Fish"]
        # Uncertainty should decrease
        assert updated.uncertainty < op.uncertainty

    def test_repeated_evidence_accumulates(self):
        categories = ["A", "B", "C"]
        op = MultinomialOpinion.vacuous(categories)

        # Multiple observations of same category
        for _ in range(5):
            op = op.update_from_evidence("A", strength=0.2)

        # Should now strongly believe A
        best_cat, prob = op.most_likely_category()
        assert best_cat == "A"
        assert prob > 0.5


class TestMultinomialBelief:
    """Tests for MultinomialBelief wrapper."""

    def test_multinomial_belief_creation(self):
        categories = ["TAG", "LAG", "NIT", "Fish"]
        op = MultinomialOpinion.vacuous(categories)
        belief = MultinomialBelief("player_type", op, "classification")

        assert belief.label == "player_type"
        assert belief.category == "classification"

    def test_multinomial_belief_to_agent_format(self):
        categories = ["TAG", "LAG", "Fish"]
        op = MultinomialOpinion.from_observation("TAG", categories, 0.5)
        belief = MultinomialBelief("player_type", op)

        formatted = belief.to_agent_format()

        assert "player_type" in formatted
        assert "TAG" in formatted

    def test_multinomial_belief_knowledge(self):
        categories = ["A", "B"]
        op = MultinomialOpinion.from_observation("A", categories, 0.4)
        belief = MultinomialBelief("test", op)

        assert belief.knowledge == op.knowledge
