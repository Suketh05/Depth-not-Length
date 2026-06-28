r"""Recovery of the *latest* decision along a chain of superseding decisions.

When a decision is revised, the new ruling **supersedes** the old one; over time a
single topic accretes a chain of :math:`n` decisions
:math:`v_1 \to v_2 \to \dots \to v_n`, where each :math:`v_{k}` overrides
:math:`v_{k-1}` and only the head :math:`v_n` is in force. Acting correctly means
recovering that head and *not* a stale ancestor. The two memory regimes do this
very differently.

**Structured memory (follow the supersession edge).** A typed store keeps an
explicit ``supersedes`` edge from each version to its successor. Landing on the
oldest node and walking forward to the head traverses
:math:`h = n - 1` edges; if each hop is followed correctly with probability
:math:`q` independently, the head is recovered with the **series-system
reliability** of the path,

.. math::

    P_\text{struct}(n) = q^{\,h} = q^{\,n-1}.

This is exactly :func:`membench.theory.recovery.structured_recovery` applied to
the ``n - 1`` supersession hops: a path of independent components fails unless
every component holds, so its reliability is the product of the per-component
reliabilities (Barlow & Proschan, *Statistical Theory of Reliability and Life
Testing*, 1975, ch. 2).

**Similarity memory (pick the most similar).** All :math:`n` versions concern the
same topic and are therefore mutually near-identical in embedding space, so a
nearest-neighbour retriever has no recency signal to separate the head from its
ancestors: it draws what is effectively a uniform pick among the :math:`n`
indistinguishable candidates,

.. math::

    P_\text{sim}(n) = \frac{1}{n}.

This is the standard *closest/uniform baseline* for choosing one correct item out
of :math:`n` equally-similar candidates. The supersession semantics — "the most
recent ruling wins" — is the recency principle of belief revision (Alchourrón,
Gärdenfors & Makinson, "On the Logic of Theory Change", *J. Symbolic Logic*
50(2), 1985) and of reason-maintenance systems (Doyle, "A Truth Maintenance
System", *Artificial Intelligence* 12(3), 1979); a content-similarity retriever is
blind to it.

**Framing.** The membench theory note reports that following the supersession edge
recovers the head far more reliably (around 92.3% in the measured runs) than a
similarity retriever, which lands in a 64-69% band on the same chains. Those are
*measured* figures and are quoted here only for context; this module asserts the
closed forms above, not the empirical numbers.

Structured memory overtakes the uniform baseline once :math:`q` clears the
break-even :math:`q^{n-1} = 1/n`, i.e. :math:`q = n^{-1/(n-1)}` (e.g. ``0.5`` at
:math:`n = 2`); for the high per-hop reliabilities typed stores actually achieve,
the structured edge dominates across the whole chain.
"""

from __future__ import annotations

from dataclasses import dataclass

from membench.theory.recovery import structured_recovery

__all__ = [
    "SupersessionModel",
    "SupersessionRecovery",
    "edge_hops",
    "similarity_supersession_recovery",
    "structured_supersession_recovery",
]


def _check_chain_length(n: int) -> None:
    if n < 1:
        raise ValueError(f"chain length n must be >= 1, got {n}")


def _check_q(q: float) -> None:
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")


def _check_baseline(baseline: float) -> None:
    if not 0.0 <= baseline <= 1.0:
        raise ValueError(f"baseline must be in [0, 1], got {baseline}")


def edge_hops(n: int) -> int:
    r"""Return the number of supersession edges from the oldest version to the head.

    A chain of ``n`` versions :math:`v_1 \to \dots \to v_n` has :math:`n - 1`
    ``supersedes`` edges; walking from the oldest node to the in-force head
    therefore traverses ``n - 1`` hops (``0`` when there is a single decision).

    Parameters
    ----------
    n
        Number of decisions in the supersession chain (>= 1).

    Returns
    -------
    int
        The hop count ``n - 1``.
    """
    _check_chain_length(n)
    return n - 1


def structured_supersession_recovery(n: int, q: float) -> float:
    r"""Probability a structured store recovers the head of a supersession chain.

    Walks the ``n - 1`` ``supersedes`` edges from the oldest version to the head,
    each followed correctly with probability ``q`` independently, giving the
    series-system reliability :math:`P_\text{struct}(n) = q^{\,n-1}`. For
    ``n - 1 >= 1`` this is :func:`membench.theory.recovery.structured_recovery`
    evaluated at the hop count; a single-decision chain (``n = 1``) is recovered
    trivially with probability ``1``.

    Parameters
    ----------
    n
        Number of decisions in the supersession chain (>= 1).
    q
        Per-edge recovery (link-follow) probability, in :math:`(0, 1]`.

    Returns
    -------
    float
        The head-recovery probability :math:`q^{\,n-1}`.
    """
    _check_chain_length(n)
    _check_q(q)
    hops = edge_hops(n)
    if hops == 0:
        return 1.0
    return structured_recovery(hops, q)


def similarity_supersession_recovery(n: int, baseline: float | None = None) -> float:
    r"""Probability a similarity retriever lands on the head of the chain.

    The ``n`` versions are mutually near-identical in embedding space, so the
    retriever has no recency signal and draws a uniform pick among them:
    :math:`P_\text{sim}(n) = 1/n`. A fixed empirical ``baseline`` may be supplied
    instead (e.g. a measured selection rate independent of ``n``).

    Parameters
    ----------
    n
        Number of decisions in the supersession chain (>= 1).
    baseline
        Optional fixed selection probability in :math:`[0, 1]`. When ``None``
        (default) the uniform :math:`1/n` baseline is used.

    Returns
    -------
    float
        The probability the most-similar pick is the in-force head.
    """
    _check_chain_length(n)
    if baseline is None:
        return 1.0 / n
    _check_baseline(baseline)
    return baseline


@dataclass(frozen=True, slots=True)
class SupersessionRecovery:
    r"""Head-recovery comparison for one supersession chain of length ``n``.

    Attributes
    ----------
    n
        Number of decisions in the chain.
    edge_hops
        Supersession edges traversed to reach the head (``n - 1``).
    q
        Per-edge recovery probability used for the structured store.
    p_struct
        Structured head-recovery probability :math:`q^{\,n-1}`.
    p_sim
        Similarity head-recovery probability (uniform ``1/n`` or fixed baseline).
    advantage
        Absolute gain :math:`P_\text{struct} - P_\text{sim}` (may be negative when
        ``q`` is below break-even).
    structured_wins
        Whether the structured store is at least as likely to recover the head.
    """

    n: int
    edge_hops: int
    q: float
    p_struct: float
    p_sim: float
    advantage: float
    structured_wins: bool


@dataclass(frozen=True, slots=True)
class SupersessionModel:
    r"""Bundles the per-edge reliability and the similarity baseline.

    Parameters
    ----------
    q
        Structured per-edge recovery probability, in :math:`(0, 1]`.
    similarity_baseline
        Optional fixed similarity selection rate in :math:`[0, 1]`; ``None``
        (default) uses the uniform :math:`1/n` baseline.
    """

    q: float
    similarity_baseline: float | None = None

    def __post_init__(self) -> None:
        _check_q(self.q)
        if self.similarity_baseline is not None:
            _check_baseline(self.similarity_baseline)

    def p_struct(self, n: int) -> float:
        """Structured head-recovery probability at chain length ``n``."""
        return structured_supersession_recovery(n, self.q)

    def p_sim(self, n: int) -> float:
        """Similarity head-recovery probability at chain length ``n``."""
        return similarity_supersession_recovery(n, self.similarity_baseline)

    def advantage(self, n: int) -> float:
        r"""Absolute gain :math:`P_\text{struct}(n) - P_\text{sim}(n)`."""
        return self.p_struct(n) - self.p_sim(n)

    def break_even_q(self, n: int) -> float:
        r"""Smallest ``q`` at which structure ties the similarity baseline.

        Solves :math:`q^{\,n-1} = P_\text{sim}(n)`, giving
        :math:`q = P_\text{sim}(n)^{1/(n-1)}` (e.g. :math:`n^{-1/(n-1)}` for the
        uniform baseline). A single-decision chain (``n = 1``) is always recovered,
        so any ``q`` suffices and the break-even is ``0``.

        Parameters
        ----------
        n
            Number of decisions in the chain (>= 1).

        Returns
        -------
        float
            The break-even per-edge reliability.
        """
        hops = edge_hops(n)
        if hops == 0:
            return 0.0
        return float(self.p_sim(n) ** (1.0 / hops))

    def recovery(self, n: int) -> SupersessionRecovery:
        """Assemble the full :class:`SupersessionRecovery` comparison at length ``n``."""
        hops = edge_hops(n)
        p_struct = self.p_struct(n)
        p_sim = self.p_sim(n)
        return SupersessionRecovery(
            n=n,
            edge_hops=hops,
            q=self.q,
            p_struct=p_struct,
            p_sim=p_sim,
            advantage=p_struct - p_sim,
            structured_wins=p_struct >= p_sim,
        )
