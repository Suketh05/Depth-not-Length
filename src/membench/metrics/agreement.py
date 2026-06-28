r"""Inter-arm agreement: do different memory arms agree on which decision to apply.

When several memory arms (the *raters*) each pick a governing decision for the same
task (the *subject/unit*), we want a chance-corrected measure of how much they agree
beyond what random labelling would produce. Two standard nominal-scale coefficients
are implemented:

**Fleiss' kappa** -- agreement among a *fixed* number of raters per subject.

.. math::

    \bar P = \frac{1}{N}\sum_i P_i,\qquad
    P_i = \frac{1}{n(n-1)}\Bigl(\sum_j n_{ij}^2 - n\Bigr),\qquad
    P_e = \sum_j p_j^2,\qquad
    \kappa = \frac{\bar P - P_e}{1 - P_e}

where :math:`N` subjects are each rated by :math:`n` raters into :math:`k`
categories, :math:`n_{ij}` is the number of raters assigning subject :math:`i` to
category :math:`j`, and :math:`p_j` is the overall proportion of assignments to
category :math:`j`. :math:`\kappa = 1` is perfect agreement and :math:`\kappa = 0`
is exactly chance.

**Krippendorff's alpha** (nominal metric) -- a more general reliability coefficient
that tolerates a variable number of raters per unit and missing values:

.. math::

    \alpha = 1 - \frac{D_o}{D_e},\qquad
    D_o = \sum_{c \neq k} o_{ck},\qquad
    D_e = \frac{1}{n-1}\sum_{c \neq k} n_c\, n_k

where :math:`o_{ck}` is the coincidence matrix (each unit with :math:`m_u` pairable
values contributes its within-unit value pairs weighted by :math:`1/(m_u-1)`),
:math:`n_c = \sum_k o_{ck}`, and :math:`n = \sum_c n_c`. :math:`\alpha = 1` is
perfect reliability.

References
----------
Fleiss, J. L. (1971). "Measuring nominal scale agreement among many raters."
*Psychological Bulletin*, 76(5), 378-382. https://doi.org/10.1037/h0031619

Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology*
(2nd ed.), chap. 11. Sage. See also Hayes & Krippendorff (2007),
"Answering the call for a standard reliability measure for coding data,"
*Communication Methods and Measures*, 1(1), 77-89.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

__all__ = [
    "FleissKappa",
    "fleiss_kappa",
    "krippendorff_alpha",
]

Rating = Hashable
RatingMatrix = Sequence[Sequence[Rating]]


@dataclass(frozen=True, slots=True)
class FleissKappa:
    r"""Fleiss' kappa together with the intermediate quantities it is built from.

    Attributes
    ----------
    kappa : float
        Chance-corrected agreement; 1.0 is perfect, 0.0 is exactly chance, and
        negative values indicate worse-than-chance agreement.
    p_bar : float
        Mean observed agreement across subjects (:math:`\bar P`).
    p_e : float
        Expected (chance) agreement (:math:`P_e`).
    n_subjects : int
        Number of subjects/units rated.
    n_raters : int
        Number of raters per subject (constant for Fleiss' kappa).
    n_categories : int
        Number of distinct categories observed.
    """

    kappa: float
    p_bar: float
    p_e: float
    n_subjects: int
    n_raters: int
    n_categories: int


def fleiss_kappa(ratings: RatingMatrix) -> FleissKappa:
    """Compute Fleiss' kappa for a subjects-by-raters matrix of categorical labels.

    Parameters
    ----------
    ratings : Sequence[Sequence[Hashable]]
        Rows are subjects/units, columns are raters; each entry is the category the
        rater assigned. Every subject must be rated by the same number of raters
        (Fleiss' kappa assumption) and no entry may be missing.

    Returns
    -------
    FleissKappa
        The kappa value and the ``p_bar`` / ``p_e`` components used to derive it.

    Raises
    ------
    ValueError
        If there are no subjects, fewer than two raters, the rater count is not
        constant across subjects, or any rating is missing (``None``).
    """
    units = [list(row) for row in ratings]
    if not units:
        raise ValueError("need at least one subject")
    n_raters = len(units[0])
    if n_raters < 2:
        raise ValueError("Fleiss' kappa needs at least two raters per subject")
    if any(len(row) != n_raters for row in units):
        raise ValueError("every subject must be rated by the same number of raters")
    if any(value is None for row in units for value in row):
        raise ValueError("Fleiss' kappa does not accept missing ratings")

    n_subjects = len(units)
    category_totals: Counter[Rating] = Counter()
    sum_agreement = 0.0
    for row in units:
        counts = Counter(row)
        sum_sq = sum(count * count for count in counts.values())
        # P_i: the proportion of agreeing rater pairs for this subject.
        sum_agreement += (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        category_totals.update(counts)

    p_bar = sum_agreement / n_subjects
    total_assignments = n_subjects * n_raters
    p_e = sum((total / total_assignments) ** 2 for total in category_totals.values())

    denominator = 1.0 - p_e
    # When only one category is ever used, agreement is trivially perfect.
    kappa = 1.0 if denominator == 0.0 else (p_bar - p_e) / denominator
    return FleissKappa(
        kappa=kappa,
        p_bar=p_bar,
        p_e=p_e,
        n_subjects=n_subjects,
        n_raters=n_raters,
        n_categories=len(category_totals),
    )


def krippendorff_alpha(ratings: RatingMatrix) -> float:
    """Compute Krippendorff's alpha (nominal metric) for a units-by-raters matrix.

    Unlike Fleiss' kappa this tolerates a variable number of raters per unit and
    missing values (``None`` entries are skipped); units with fewer than two
    pairable values contribute nothing.

    Parameters
    ----------
    ratings : Sequence[Sequence[Hashable]]
        Rows are units, columns are raters; each entry is the assigned category, or
        ``None`` for a missing rating.

    Returns
    -------
    float
        Alpha; 1.0 is perfect reliability, 0.0 is chance, negative is systematic
        disagreement. Returns 1.0 when every value is identical (no variation).

    Raises
    ------
    ValueError
        If fewer than two pairable values exist across all units.
    """
    coincidence: dict[tuple[Rating, Rating], float] = {}
    for row in ratings:
        values = [value for value in row if value is not None]
        pairable = len(values)
        if pairable < 2:
            continue
        counts = Counter(values)
        for c, n_c in counts.items():
            for k, n_k in counts.items():
                pairs = n_c * (n_c - 1) if c == k else n_c * n_k
                key = (c, k)
                coincidence[key] = coincidence.get(key, 0.0) + pairs / (pairable - 1)

    categories = {c for c, _ in coincidence}
    marginals = {c: sum(coincidence.get((c, k), 0.0) for k in categories) for c in categories}
    grand_total = sum(marginals.values())
    if grand_total < 2:
        raise ValueError("need at least two pairable values to compute alpha")

    observed = sum(value for (c, k), value in coincidence.items() if c != k)
    expected = sum(
        marginals[c] * marginals[k] for c in categories for k in categories if c != k
    ) / (grand_total - 1)

    # No off-diagonal expectation means there is no variation: perfectly reliable.
    return 1.0 if expected == 0.0 else 1.0 - observed / expected
