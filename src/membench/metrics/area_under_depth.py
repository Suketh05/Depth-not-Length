r"""Area under the compliance-vs-depth curve (AUDC) and the depth half-life.

The depth thesis predicts compliance degrades as the governing-decision chain gets
deeper. Two single-number summaries of that whole curve are useful for ranking arms:

* **AUDC** -- the area under the compliance-vs-depth curve, computed with the
  composite **trapezoidal rule** (Press et al., *Numerical Recipes*, 3rd ed.,
  2007, sec. 4.1; Atkinson, *An Introduction to Numerical Analysis*, 2nd ed.,
  1989). For depths :math:`d_0 < d_1 < \dots < d_n` with values :math:`v_i`,

  .. math::

      \text{AUDC}_\text{raw} = \sum_{i=0}^{n-1}
          \tfrac{1}{2}\,(d_{i+1}-d_i)\,(v_i + v_{i+1}).

  Normalising by the depth span :math:`d_n - d_0` turns this into the
  interval-weighted **mean compliance** over the measured depth range,

  .. math::

      \text{AUDC} = \frac{1}{d_n - d_0}\sum_{i=0}^{n-1}
          \tfrac{1}{2}\,(d_{i+1}-d_i)\,(v_i + v_{i+1}),

  which is exactly the mean-value-of-a-function identity
  :math:`\bar f = \frac{1}{b-a}\int_a^b f`. A flat curve at constant ``c`` therefore
  has normalised AUDC ``c`` -- the natural invariant.

* **Depth half-life** -- by analogy with exponential-decay half-life, the depth at
  which compliance first falls to **half its shallowest (``d=1``) value**, found by
  linear interpolation between the two bracketing measured depths. ``None`` is
  returned when the curve never reaches half within the measured range (or the
  baseline is non-positive), since extrapolating past the data would be fabrication.

Inputs follow the metrics-package convention: either a ``depth -> value`` mapping or
a sequence of :class:`~membench.analysis.tables.ScoredRow` (compliance averaged per
depth). No data is invented -- every value comes from the supplied curve.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from membench.analysis.tables import ScoredRow

    CurveInput = Mapping[int, float] | Iterable[ScoredRow]

__all__ = [
    "area_under_depth_curve",
    "depth_half_life",
    "depth_value_curve",
]


def _compliance_rate(row: ScoredRow) -> float:
    """Default value extractor: a row's ``compliance_rate``."""
    return row.compliance_rate


def depth_value_curve(
    rows: Iterable[ScoredRow],
    *,
    value: Callable[[ScoredRow], float] = _compliance_rate,
) -> dict[int, float]:
    """Average a per-row value by depth into a sorted ``depth -> value`` curve.

    Parameters
    ----------
    rows : Iterable[ScoredRow]
        Scored rows to summarise. Rows at the same ``depth`` are averaged.
    value : Callable[[ScoredRow], float], optional
        Selects the value to curve; defaults to ``row.compliance_rate``.

    Returns
    -------
    dict[int, float]
        Depths (ascending) mapped to their mean value.

    Raises
    ------
    ValueError
        If ``rows`` is empty.
    """
    groups: dict[int, list[float]] = {}
    for row in rows:
        groups.setdefault(int(row.depth), []).append(float(value(row)))
    if not groups:
        raise ValueError("need at least one scored row to build a depth curve")
    return {depth: statistics.fmean(groups[depth]) for depth in sorted(groups)}


def _coerce_curve(
    curve: CurveInput,
    value: Callable[[ScoredRow], float],
) -> dict[int, float]:
    """Normalise either input form into a sorted ``depth -> value`` mapping."""
    if isinstance(curve, Mapping):
        if not curve:
            raise ValueError("depth curve must contain at least one point")
        return {int(depth): float(curve[depth]) for depth in sorted(curve)}
    return depth_value_curve(curve, value=value)


def area_under_depth_curve(
    curve: CurveInput,
    *,
    value: Callable[[ScoredRow], float] = _compliance_rate,
    normalize: bool = True,
) -> float:
    """Trapezoidal area under the compliance-vs-depth curve.

    Parameters
    ----------
    curve : Mapping[int, float] | Iterable[ScoredRow]
        Either a ``depth -> value`` mapping or scored rows (averaged per depth via
        :func:`depth_value_curve`).
    value : Callable[[ScoredRow], float], optional
        Value extractor used only when ``curve`` is a row sequence; defaults to
        ``row.compliance_rate``.
    normalize : bool, optional
        When ``True`` (default), divide the raw area by the depth span
        ``d_max - d_min`` so the result is the interval-weighted mean value over the
        measured depth range (a flat curve at ``c`` yields ``c``). When ``False``,
        return the raw trapezoidal area.

    Returns
    -------
    float
        Normalised AUDC (mean value) or raw trapezoidal area. With a single measured
        depth the raw area is ``0.0`` and the normalised value is that point's value.

    Notes
    -----
    Composite trapezoidal rule; see the module docstring for the formula and
    references.
    """
    points = _coerce_curve(curve, value)
    depths = list(points)
    values = [points[depth] for depth in depths]
    if len(depths) == 1:
        return values[0] if normalize else 0.0
    area = sum(
        (d1 - d0) * (v0 + v1) / 2.0
        for d0, d1, v0, v1 in zip(depths[:-1], depths[1:], values[:-1], values[1:], strict=True)
    )
    if not normalize:
        return area
    return area / (depths[-1] - depths[0])


def depth_half_life(
    curve: CurveInput,
    *,
    value: Callable[[ScoredRow], float] = _compliance_rate,
) -> float | None:
    """Depth at which the value first falls to half its shallowest-depth value.

    The baseline is the value at the smallest measured depth (``d=1`` by the Task
    convention). The crossing depth where the curve reaches ``baseline / 2`` is found
    by linear interpolation between the two bracketing measured depths.

    Parameters
    ----------
    curve : Mapping[int, float] | Iterable[ScoredRow]
        A ``depth -> value`` mapping or scored rows (averaged per depth).
    value : Callable[[ScoredRow], float], optional
        Value extractor used only for row input; defaults to ``row.compliance_rate``.

    Returns
    -------
    float | None
        The interpolated half-life depth, or ``None`` if the baseline is
        non-positive or the curve never falls to half within the measured range
        (extrapolating beyond the data is deliberately avoided).

    Notes
    -----
    Assumes a (broadly) decreasing curve and reports the *first* downward crossing of
    the half-baseline level.
    """
    points = _coerce_curve(curve, value)
    depths = list(points)
    values = [points[depth] for depth in depths]
    baseline = values[0]
    if baseline <= 0.0:
        return None
    target = baseline / 2.0
    for d0, d1, v0, v1 in zip(depths[:-1], depths[1:], values[:-1], values[1:], strict=True):
        if v1 <= target <= v0:
            denom = v0 - v1
            if denom == 0.0:
                return float(d0)
            return d0 + (v0 - target) / denom * (d1 - d0)
    return None
