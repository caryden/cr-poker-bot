"""
Subjective Logic (SL) implementation for belief representation and reasoning.

Subjective Logic represents beliefs as opinions with explicit uncertainty,
using the tuple (b, d, u, a) where:
- b = belief mass (evidence FOR the proposition)
- d = disbelief mass (evidence AGAINST the proposition)
- u = uncertainty mass (lack of evidence)
- a = base rate (prior probability when uncertain)

Constraint: b + d + u = 1

Key insight: High disbelief (d) is valuable for narrowing search space.
Knowledge = b + d = 1 - u represents total evidence accumulated.

Reference: Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning
Under Uncertainty.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class Opinion:
    """
    A subjective logic opinion about a proposition.

    Attributes:
        belief: Evidence supporting the proposition (0 to 1)
        disbelief: Evidence against the proposition (0 to 1)
        uncertainty: Lack of evidence (0 to 1)
        base_rate: Prior probability when uncertain (0 to 1)

    Invariant: belief + disbelief + uncertainty = 1
    """
    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.5

    def __post_init__(self):
        """Validate opinion constraints."""
        total = self.belief + self.disbelief + self.uncertainty
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Opinion must sum to 1.0, got {total:.6f} "
                f"(b={self.belief}, d={self.disbelief}, u={self.uncertainty})"
            )
        if not (0 <= self.belief <= 1):
            raise ValueError(f"Belief must be in [0,1], got {self.belief}")
        if not (0 <= self.disbelief <= 1):
            raise ValueError(f"Disbelief must be in [0,1], got {self.disbelief}")
        if not (0 <= self.uncertainty <= 1):
            raise ValueError(f"Uncertainty must be in [0,1], got {self.uncertainty}")
        if not (0 <= self.base_rate <= 1):
            raise ValueError(f"Base rate must be in [0,1], got {self.base_rate}")

    @property
    def knowledge(self) -> float:
        """
        Total evidence accumulated: knowledge = b + d = 1 - u.

        Higher knowledge means more informative opinion (less uncertainty).
        Used for sorting beliefs by informativeness.
        """
        return self.belief + self.disbelief

    @property
    def projected_probability(self) -> float:
        """
        Project opinion to probability: P = b + a*u.

        This is the expected probability accounting for uncertainty.
        """
        return self.belief + self.base_rate * self.uncertainty

    @property
    def is_certain(self) -> bool:
        """True if uncertainty is effectively zero."""
        return self.uncertainty < 0.01

    @property
    def is_vacuous(self) -> bool:
        """True if this is a vacuous (maximally uncertain) opinion."""
        return self.uncertainty > 0.99

    @property
    def is_dogmatic(self) -> bool:
        """True if there's zero uncertainty (fully committed)."""
        return math.isclose(self.uncertainty, 0.0, abs_tol=1e-6)

    def to_tuple(self) -> tuple[float, float, float]:
        """Return (b, d, u) tuple."""
        return (self.belief, self.disbelief, self.uncertainty)

    def to_tuple_str(self) -> str:
        """
        Raw tuple format for agent consumption.

        Format: (b=0.XX, d=0.XX, u=0.XX)
        """
        return f"(b={self.belief:.2f}, d={self.disbelief:.2f}, u={self.uncertainty:.2f})"

    def __str__(self) -> str:
        return self.to_tuple_str()

    def __repr__(self) -> str:
        return f"Opinion{self.to_tuple_str()}"

    # ==================== Factory Methods ====================

    @classmethod
    def vacuous(cls, base_rate: float = 0.5) -> Opinion:
        """
        Create a vacuous (maximally uncertain) opinion.

        Use when you have no evidence whatsoever.
        """
        return cls(0.0, 0.0, 1.0, base_rate)

    @classmethod
    def dogmatic_belief(cls, base_rate: float = 0.5) -> Opinion:
        """Create a dogmatic (certain) belief."""
        return cls(1.0, 0.0, 0.0, base_rate)

    @classmethod
    def dogmatic_disbelief(cls, base_rate: float = 0.5) -> Opinion:
        """Create a dogmatic (certain) disbelief."""
        return cls(0.0, 1.0, 0.0, base_rate)

    @classmethod
    def from_probability(cls, prob: float, confidence: float = 0.5,
                         base_rate: Optional[float] = None) -> Opinion:
        """
        Create opinion from probability with given confidence level.

        Args:
            prob: Probability estimate (0 to 1)
            confidence: How certain we are (0 = vacuous, 1 = dogmatic)
            base_rate: Prior probability (defaults to prob)

        Returns:
            Opinion with appropriate (b, d, u) based on confidence
        """
        if base_rate is None:
            base_rate = prob

        # Confidence determines how much uncertainty we have
        u = 1.0 - confidence

        # Distribute remaining mass between b and d based on prob
        remaining = confidence
        b = remaining * prob
        d = remaining * (1 - prob)

        return cls(b, d, u, base_rate)

    @classmethod
    def from_frequency(cls, successes: int, failures: int,
                       base_rate: float = 0.5) -> Opinion:
        """
        Create opinion from observed frequencies.

        Uses the standard mapping from beta distribution:
        b = successes / (total + W)
        d = failures / (total + W)
        u = W / (total + W)

        Where W is the weight of prior (default 2 for non-informative).

        Args:
            successes: Number of positive observations
            failures: Number of negative observations
            base_rate: Prior probability

        Returns:
            Opinion with uncertainty inversely proportional to sample size
        """
        total = successes + failures
        if total == 0:
            return cls.vacuous(base_rate)

        W = 2  # Weight of prior (non-informative)
        b = successes / (total + W)
        d = failures / (total + W)
        u = W / (total + W)

        return cls(b, d, u, base_rate)

    # ==================== SL Operators ====================

    def cumulative_fusion(self, other: Opinion) -> Opinion:
        """
        Cumulative fusion: ω_A ⊕ ω_B

        Combines two opinions from independent sources as if they
        are accumulated evidence. More evidence = less uncertainty.

        This is the primary operator for belief revision.

        Args:
            other: Another opinion to fuse with

        Returns:
            Fused opinion with reduced uncertainty
        """
        u_A, u_B = self.uncertainty, other.uncertainty
        b_A, b_B = self.belief, other.belief
        d_A, d_B = self.disbelief, other.disbelief

        # Handle dogmatic opinions
        if u_A == 0 and u_B == 0:
            # Both dogmatic - use averaging
            return Opinion(
                (b_A + b_B) / 2,
                (d_A + d_B) / 2,
                0.0,
                (self.base_rate + other.base_rate) / 2
            )

        # Normalization factor
        k = u_A + u_B - u_A * u_B
        if k == 0:
            return self  # Shouldn't happen given above check

        # Fusion formulas
        b = (b_A * u_B + b_B * u_A) / k
        d = (d_A * u_B + d_B * u_A) / k
        u = (u_A * u_B) / k

        # Average base rates
        a = (self.base_rate + other.base_rate) / 2

        return Opinion(b, d, u, a)

    def __add__(self, other: Opinion) -> Opinion:
        """Operator overload for cumulative fusion: ω1 + ω2 = ω1 ⊕ ω2"""
        return self.cumulative_fusion(other)

    def averaging_fusion(self, other: Opinion) -> Opinion:
        """
        Averaging fusion: ω_A ⊙ ω_B

        Combines opinions by averaging, appropriate when sources
        may have dependent evidence.

        Args:
            other: Another opinion to fuse with

        Returns:
            Averaged opinion
        """
        u_A, u_B = self.uncertainty, other.uncertainty
        b_A, b_B = self.belief, other.belief
        d_A, d_B = self.disbelief, other.disbelief

        # Handle dogmatic opinions
        if u_A == 0 and u_B == 0:
            return Opinion(
                (b_A + b_B) / 2,
                (d_A + d_B) / 2,
                0.0,
                (self.base_rate + other.base_rate) / 2
            )

        # Normalization factor for averaging
        k = u_A + u_B
        if k == 0:
            return self

        # Averaging formulas
        b = (b_A * u_B + b_B * u_A) / k
        d = (d_A * u_B + d_B * u_A) / k
        u = (2 * u_A * u_B) / k

        a = (self.base_rate + other.base_rate) / 2

        return Opinion(b, d, u, a)

    def trust_discount(self, trust: Opinion) -> Opinion:
        """
        Trust discounting: ω_A^t

        Discount this opinion based on trust in the source.
        Lower trust → more uncertainty in the result.

        Args:
            trust: Opinion about trustworthiness of source

        Returns:
            Discounted opinion with increased uncertainty
        """
        # Project trust to scalar discount factor
        t = trust.projected_probability

        # Discounting preserves proportions but increases uncertainty
        b = t * self.belief
        d = t * self.disbelief
        u = 1 - t * (self.belief + self.disbelief)

        return Opinion(b, d, u, self.base_rate)

    def discount_by_factor(self, factor: float) -> Opinion:
        """
        Simple scalar discounting.

        Args:
            factor: Discount factor (0 = vacuous, 1 = unchanged)

        Returns:
            Discounted opinion
        """
        if not 0 <= factor <= 1:
            raise ValueError(f"Discount factor must be in [0,1], got {factor}")

        b = factor * self.belief
        d = factor * self.disbelief
        u = 1 - factor * (self.belief + self.disbelief)

        return Opinion(b, d, u, self.base_rate)

    def negation(self) -> Opinion:
        """
        Logical negation: ¬ω

        Swaps belief and disbelief, inverts base rate.
        """
        return Opinion(
            self.disbelief,
            self.belief,
            self.uncertainty,
            1 - self.base_rate
        )

    def __neg__(self) -> Opinion:
        """Operator overload for negation: -ω = ¬ω"""
        return self.negation()

    def deduction(self, conditional_true: Opinion,
                  conditional_false: Opinion) -> Opinion:
        """
        Deduction operator for conditional reasoning.

        Given:
        - Self: opinion about X
        - conditional_true: opinion about Y given X is true
        - conditional_false: opinion about Y given X is false

        Returns: opinion about Y

        This enables chains of reasoning.
        """
        p_x = self.projected_probability
        p_y_given_x = conditional_true.projected_probability
        p_y_given_not_x = conditional_false.projected_probability

        # Projected probability of Y
        p_y = p_x * p_y_given_x + (1 - p_x) * p_y_given_not_x

        # Uncertainty propagates
        u_y = self.uncertainty + (1 - self.uncertainty) * (
            p_x * conditional_true.uncertainty +
            (1 - p_x) * conditional_false.uncertainty
        )

        # Derive belief and disbelief from probability and uncertainty
        b = (1 - u_y) * p_y
        d = (1 - u_y) * (1 - p_y)

        return Opinion(b, d, u_y, p_y)

    # ==================== Comparison Methods ====================

    def confidence_weighted_value(self) -> float:
        """
        Value weighted by confidence, useful for sorting.

        Returns projected probability weighted by knowledge.
        High knowledge + high belief = high positive value.
        High knowledge + high disbelief = high negative value.
        """
        return self.knowledge * (self.belief - self.disbelief)

    def entropy(self) -> float:
        """
        Shannon entropy of the projected probability distribution.

        Lower entropy = more decisive opinion.
        """
        p = self.projected_probability
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# ============== Multinomial Opinions ==============

@dataclass
class MultinomialOpinion:
    """
    Multinomial subjective logic opinion over k mutually exclusive outcomes.

    Used for categorical classifications where outcomes are mutually exclusive
    (e.g., player type: TAG, LAG, NIT, Fish - a player can only be one).

    Based on Dirichlet distribution, generalizes binomial SL to k categories.

    Attributes:
        beliefs: Dict mapping category -> belief mass (evidence FOR that category)
        uncertainty: Uncommitted belief mass (lack of evidence)
        base_rates: Dict mapping category -> prior probability

    Constraint: sum(beliefs.values()) + uncertainty = 1
    Constraint: sum(base_rates.values()) = 1
    """
    beliefs: dict[str, float]
    uncertainty: float
    base_rates: dict[str, float]

    def __post_init__(self):
        """Validate multinomial opinion constraints."""
        # Check belief sum + uncertainty = 1
        belief_sum = sum(self.beliefs.values())
        total = belief_sum + self.uncertainty
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Beliefs + uncertainty must sum to 1.0, got {total:.6f}"
            )

        # Check base rates sum to 1
        base_sum = sum(self.base_rates.values())
        if not math.isclose(base_sum, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Base rates must sum to 1.0, got {base_sum:.6f}"
            )

        # Check all values in valid range
        for cat, b in self.beliefs.items():
            if not (0 <= b <= 1):
                raise ValueError(f"Belief for {cat} must be in [0,1], got {b}")

        if not (0 <= self.uncertainty <= 1):
            raise ValueError(f"Uncertainty must be in [0,1], got {self.uncertainty}")

        for cat, a in self.base_rates.items():
            if not (0 <= a <= 1):
                raise ValueError(f"Base rate for {cat} must be in [0,1], got {a}")

        # Ensure same categories in beliefs and base_rates
        if set(self.beliefs.keys()) != set(self.base_rates.keys()):
            raise ValueError("Beliefs and base_rates must have same categories")

    @property
    def categories(self) -> list[str]:
        """Get all category names."""
        return list(self.beliefs.keys())

    @property
    def knowledge(self) -> float:
        """
        Total evidence accumulated: knowledge = 1 - uncertainty.

        Higher knowledge = more informative opinion.
        """
        return 1 - self.uncertainty

    @property
    def is_vacuous(self) -> bool:
        """True if maximally uncertain (no evidence)."""
        return self.uncertainty > 0.99

    @property
    def is_dogmatic(self) -> bool:
        """True if fully committed (no uncertainty)."""
        return math.isclose(self.uncertainty, 0.0, abs_tol=1e-6)

    def projected_probability(self, category: str) -> float:
        """
        Project opinion to probability for a category: P(x) = b_x + a_x * u

        Args:
            category: The category to get probability for

        Returns:
            Projected probability for that category
        """
        b = self.beliefs.get(category, 0.0)
        a = self.base_rates.get(category, 0.0)
        return b + a * self.uncertainty

    def all_projected_probabilities(self) -> dict[str, float]:
        """Get projected probabilities for all categories."""
        return {cat: self.projected_probability(cat) for cat in self.categories}

    def most_likely_category(self) -> tuple[str, float]:
        """
        Get the most likely category and its probability.

        Returns:
            (category_name, probability) tuple
        """
        probs = self.all_projected_probabilities()
        best = max(probs.items(), key=lambda x: x[1])
        return best

    def to_dict(self) -> dict[str, float]:
        """Return beliefs as dict for inspection."""
        return dict(self.beliefs)

    def to_tuple_str(self) -> str:
        """
        Format for agent consumption.

        Format: {CAT1: b1, CAT2: b2, ..., u: uncertainty}
        """
        parts = [f"{cat}={b:.2f}" for cat, b in sorted(self.beliefs.items())]
        parts.append(f"u={self.uncertainty:.2f}")
        return "{" + ", ".join(parts) + "}"

    def __str__(self) -> str:
        return self.to_tuple_str()

    def __repr__(self) -> str:
        return f"MultinomialOpinion{self.to_tuple_str()}"

    # ==================== Factory Methods ====================

    @classmethod
    def vacuous(cls, categories: list[str],
                base_rates: Optional[dict[str, float]] = None) -> 'MultinomialOpinion':
        """
        Create a vacuous (maximally uncertain) multinomial opinion.

        Args:
            categories: List of category names
            base_rates: Prior probabilities (uniform if not specified)
        """
        if base_rates is None:
            # Uniform priors
            base_rates = {cat: 1.0 / len(categories) for cat in categories}

        beliefs = {cat: 0.0 for cat in categories}
        return cls(beliefs, 1.0, base_rates)

    @classmethod
    def dogmatic(cls, category: str, all_categories: list[str],
                 base_rates: Optional[dict[str, float]] = None) -> 'MultinomialOpinion':
        """
        Create a dogmatic opinion certain of one category.

        Args:
            category: The certain category
            all_categories: All possible categories
            base_rates: Prior probabilities
        """
        if base_rates is None:
            base_rates = {cat: 1.0 / len(all_categories) for cat in all_categories}

        beliefs = {cat: (1.0 if cat == category else 0.0) for cat in all_categories}
        return cls(beliefs, 0.0, base_rates)

    @classmethod
    def from_observation(cls, observed_category: str, all_categories: list[str],
                        confidence: float = 0.3,
                        base_rates: Optional[dict[str, float]] = None) -> 'MultinomialOpinion':
        """
        Create opinion from a single observation.

        Args:
            observed_category: The category suggested by observation
            all_categories: All possible categories
            confidence: How much belief mass to assign (rest is uncertainty)
            base_rates: Prior probabilities
        """
        if base_rates is None:
            base_rates = {cat: 1.0 / len(all_categories) for cat in all_categories}

        beliefs = {cat: 0.0 for cat in all_categories}
        beliefs[observed_category] = confidence
        uncertainty = 1.0 - confidence

        return cls(beliefs, uncertainty, base_rates)

    # ==================== SL Operators ====================

    def cumulative_fusion(self, other: 'MultinomialOpinion') -> 'MultinomialOpinion':
        """
        Cumulative fusion for multinomial opinions.

        Combines evidence from independent sources, reducing uncertainty.

        Args:
            other: Another multinomial opinion with same categories

        Returns:
            Fused opinion with accumulated evidence
        """
        if set(self.categories) != set(other.categories):
            raise ValueError("Cannot fuse opinions with different categories")

        u_A, u_B = self.uncertainty, other.uncertainty

        # Handle dogmatic cases
        if u_A == 0 and u_B == 0:
            # Both dogmatic - average beliefs
            new_beliefs = {
                cat: (self.beliefs[cat] + other.beliefs[cat]) / 2
                for cat in self.categories
            }
            new_base = {
                cat: (self.base_rates[cat] + other.base_rates[cat]) / 2
                for cat in self.categories
            }
            return MultinomialOpinion(new_beliefs, 0.0, new_base)

        # Normalization factor
        k = u_A + u_B - u_A * u_B
        if k == 0:
            return self

        # Fusion formula for each category
        new_beliefs = {}
        for cat in self.categories:
            b_A = self.beliefs[cat]
            b_B = other.beliefs[cat]
            new_beliefs[cat] = (b_A * u_B + b_B * u_A) / k

        new_u = (u_A * u_B) / k

        # Average base rates
        new_base = {
            cat: (self.base_rates[cat] + other.base_rates[cat]) / 2
            for cat in self.categories
        }

        return MultinomialOpinion(new_beliefs, new_u, new_base)

    def __add__(self, other: 'MultinomialOpinion') -> 'MultinomialOpinion':
        """Operator overload for cumulative fusion."""
        return self.cumulative_fusion(other)

    def discount_by_factor(self, factor: float) -> 'MultinomialOpinion':
        """
        Discount opinion by a scalar factor.

        Args:
            factor: Discount factor (0 = vacuous, 1 = unchanged)

        Returns:
            Discounted opinion with increased uncertainty
        """
        if not 0 <= factor <= 1:
            raise ValueError(f"Discount factor must be in [0,1], got {factor}")

        new_beliefs = {cat: factor * b for cat, b in self.beliefs.items()}
        new_u = 1 - factor * (1 - self.uncertainty)

        return MultinomialOpinion(new_beliefs, new_u, dict(self.base_rates))

    def update_from_evidence(self, evidence_category: str,
                            strength: float = 0.2) -> 'MultinomialOpinion':
        """
        Update opinion based on new evidence favoring a category.

        This is a convenience method that creates an evidence opinion
        and fuses it with the current opinion.

        Args:
            evidence_category: Category supported by evidence
            strength: Strength of evidence (0 to 1)

        Returns:
            Updated opinion
        """
        evidence = MultinomialOpinion.from_observation(
            evidence_category, self.categories, strength, dict(self.base_rates)
        )
        return self.cumulative_fusion(evidence)


@dataclass
class MultinomialBelief:
    """
    A labeled multinomial belief for categorical classifications.

    Used for mutually exclusive categories like player type.
    """
    label: str
    opinion: MultinomialOpinion
    category: str = "classification"

    def __str__(self) -> str:
        best_cat, prob = self.opinion.most_likely_category()
        return f"{self.label}: {best_cat} ({prob:.0%}) {self.opinion}"

    @property
    def knowledge(self) -> float:
        """Delegate to opinion's knowledge."""
        return self.opinion.knowledge

    def to_agent_format(self) -> str:
        """
        Format for agent consumption.

        Shows most likely category plus full distribution.
        """
        best_cat, prob = self.opinion.most_likely_category()
        return f"{self.label}: most likely {best_cat} ({prob:.0%}), dist={self.opinion.to_tuple_str()}"


@dataclass
class Belief:
    """
    A labeled belief with subjective logic opinion.

    Wraps an Opinion with a human-readable label for use in
    belief state management.
    """
    label: str
    opinion: Opinion
    category: str = "general"  # e.g., "player_type", "hand_range", "tendency"

    def __str__(self) -> str:
        return f"{self.label}: {self.opinion}"

    @property
    def knowledge(self) -> float:
        """Delegate to opinion's knowledge."""
        return self.opinion.knowledge

    def to_agent_format(self) -> str:
        """
        Format belief for agent consumption.

        Format: "label (b=X.XX, d=X.XX, u=X.XX)"
        """
        return f"{self.label} {self.opinion.to_tuple_str()}"


def sort_beliefs_by_knowledge(beliefs: list[Belief],
                               descending: bool = True) -> list[Belief]:
    """
    Sort beliefs by knowledge (b + d), most informative first.

    High knowledge beliefs (low uncertainty) should be prioritized
    in agent reasoning as they provide more actionable constraints.
    """
    return sorted(beliefs, key=lambda b: b.knowledge, reverse=descending)


def format_beliefs_for_agent(beliefs: list[Belief],
                              sort_by_knowledge: bool = True) -> str:
    """
    Format beliefs as conditioning context for agent.

    Returns beliefs sorted by knowledge with raw (b,d,u) tuples.
    """
    if sort_by_knowledge:
        beliefs = sort_beliefs_by_knowledge(beliefs)

    lines = []
    for belief in beliefs:
        lines.append(f"  {belief.to_agent_format()}")

    return "\n".join(lines)
