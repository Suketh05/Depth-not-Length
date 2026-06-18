"""The ``MemorySystem`` contract, the arm registry, and shared budget packing.

Every memory arm implements exactly two methods — ``write`` (ingest a task's
corpus) and ``retrieve`` (return context within a caller-supplied token budget) —
and nothing else. That uniformity is what lets the harness treat the memory system
as the single swappable variable.

Two pieces of shared machinery live here so that fairness is enforced structurally
rather than by convention:

* :func:`pack_to_budget` is the *one* truncation rule used by every arm. Arms
  differ only in how they *rank* candidates; the rule that turns a ranking into a
  budget-bounded context is identical for all of them. If packing differed between
  arms, the token budget would no longer be a fair constraint.
* :class:`ArmRegistry` records each arm's capabilities (family, whether it follows
  explicit links, whether it uses embeddings, whether it needs the network). The
  analysis layer reads this to build the methodology table and to separate
  measured arms from network-bound competitor adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from membench.types import MemoryItem, RetrievedContext

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "REGISTRY",
    "ArmCapabilities",
    "ArmFamily",
    "ArmRegistry",
    "ArmSpec",
    "MemorySystem",
    "available_arms",
    "build_arm",
    "get_spec",
    "iter_specs",
    "pack_to_budget",
    "register_arm",
]

DEFAULT_CHARS_PER_TOKEN = 4
"""Shared chars-per-token heuristic for budget packing (identical across arms)."""


class ArmFamily(str, Enum):
    """Coarse taxonomy of memory arms, used for grouping in tables and figures."""

    CONTROL = "control"
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    GRAPH = "graph"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class ArmCapabilities:
    """Declared properties of a memory arm.

    These are descriptive metadata, not behaviour: the analysis layer uses them to
    explain *why* an arm behaves as it does (e.g. only ``follows_links`` arms can
    traverse a stripped justification edge) and to flag arms that require the
    network so measured and vendor-reported results stay separated.
    """

    family: ArmFamily
    follows_links: bool
    uses_embeddings: bool
    requires_network: bool = False
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """A registered arm: its name, constructor, capabilities, and description."""

    name: str
    factory: Callable[..., MemorySystem]
    capabilities: ArmCapabilities
    description: str


class MemorySystem(ABC):
    """Base class every memory arm subclasses.

    ``budget_tokens`` is a parameter of :meth:`retrieve`, never a property of the
    arm: the harness passes the same budget to every arm for a given task, so that
    memory architecture is the only variable being compared. An arm must never
    read its own budget from configuration or choose one internally.
    """

    @abstractmethod
    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest a task's memory corpus before any retrieval happens."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return assembled context for ``query`` within ``budget_tokens``."""
        raise NotImplementedError


def pack_to_budget(
    ranked: Iterable[tuple[str, str]],
    budget_tokens: int,
    *,
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    separator: str = "\n\n",
) -> RetrievedContext:
    """Pack a ranked list of items into a token budget (truncate-at-first-overflow).

    This is the shared fairness primitive. Items are consumed in the order given —
    the *ranking* is each arm's contribution — and included greedily until the next
    item would exceed ``budget_tokens``. The first overflowing item stops packing;
    later items are not back-filled even if they would fit, so the rule is a simple
    deterministic prefix of the ranking.

    Parameters
    ----------
    ranked
        Ordered ``(item_id, text)`` pairs, best first.
    budget_tokens
        Token budget shared across all arms for this task. Must be non-negative.
    chars_per_token
        Heuristic divisor for the token estimate; identical across arms.
    separator
        Joins the included texts into the assembled context string.

    Returns
    -------
    RetrievedContext
        The assembled text and the ids of exactly the items that were included.
    """
    if budget_tokens < 0:
        raise ValueError(f"budget_tokens must be non-negative, got {budget_tokens}")
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")

    texts: list[str] = []
    ids: list[str] = []
    used = 0
    for item_id, text in ranked:
        cost = len(text) // chars_per_token
        if used + cost > budget_tokens:
            break
        texts.append(text)
        ids.append(item_id)
        used += cost
    return RetrievedContext(text=separator.join(texts), ids=tuple(ids))


class ArmRegistry:
    """A registry mapping arm names to their specs.

    A single module-global :data:`REGISTRY` is populated by the ``@register_arm``
    decorator as arm modules import. Tests can construct an isolated
    ``ArmRegistry`` to avoid mutating global state.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ArmSpec] = {}

    def register(
        self,
        name: str,
        *,
        family: ArmFamily,
        follows_links: bool,
        uses_embeddings: bool,
        requires_network: bool = False,
        deterministic: bool = True,
    ) -> Callable[[Callable[..., MemorySystem]], Callable[..., MemorySystem]]:
        """Decorate a ``MemorySystem`` subclass (or factory) to register it.

        The decorated object is returned unchanged, so registration is transparent
        to importers of the class.
        """

        def decorate(factory: Callable[..., MemorySystem]) -> Callable[..., MemorySystem]:
            if name in self._specs:
                raise ValueError(f"arm {name!r} is already registered")
            description = (factory.__doc__ or "").strip().splitlines()[0] if factory.__doc__ else ""
            self._specs[name] = ArmSpec(
                name=name,
                factory=factory,
                capabilities=ArmCapabilities(
                    family=family,
                    follows_links=follows_links,
                    uses_embeddings=uses_embeddings,
                    requires_network=requires_network,
                    deterministic=deterministic,
                ),
                description=description,
            )
            return factory

        return decorate

    def build(self, name: str, /, **kwargs: object) -> MemorySystem:
        """Construct an arm instance by registered name."""
        return self.get(name).factory(**kwargs)

    def get(self, name: str) -> ArmSpec:
        """Return the spec for ``name`` or raise ``KeyError`` with the known names."""
        try:
            return self._specs[name]
        except KeyError:
            known = ", ".join(sorted(self._specs)) or "(none registered)"
            raise KeyError(f"unknown arm {name!r}; registered arms: {known}") from None

    def names(self) -> list[str]:
        """Sorted list of registered arm names."""
        return sorted(self._specs)

    def specs(self) -> Iterator[ArmSpec]:
        """Iterate registered specs in name order."""
        for name in self.names():
            yield self._specs[name]


REGISTRY = ArmRegistry()
"""The process-wide arm registry populated at import time by ``@register_arm``."""


def register_arm(
    name: str,
    *,
    family: ArmFamily,
    follows_links: bool,
    uses_embeddings: bool,
    requires_network: bool = False,
    deterministic: bool = True,
) -> Callable[[Callable[..., MemorySystem]], Callable[..., MemorySystem]]:
    """Register an arm in the global :data:`REGISTRY` (see :meth:`ArmRegistry.register`)."""
    return REGISTRY.register(
        name,
        family=family,
        follows_links=follows_links,
        uses_embeddings=uses_embeddings,
        requires_network=requires_network,
        deterministic=deterministic,
    )


def build_arm(name: str, /, **kwargs: object) -> MemorySystem:
    """Construct an arm from the global registry by name."""
    return REGISTRY.build(name, **kwargs)


def get_spec(name: str) -> ArmSpec:
    """Return the spec for ``name`` from the global registry."""
    return REGISTRY.get(name)


def available_arms() -> list[str]:
    """Sorted names of all arms registered in the global registry."""
    return REGISTRY.names()


def iter_specs() -> Iterator[ArmSpec]:
    """Iterate specs in the global registry, in name order."""
    return REGISTRY.specs()
