r"""Centralized determinism: seeding, scoped RNG state, and a stable hash.

Reproducibility is a first-class requirement for this benchmark: the same seed must
produce the same tasks, the same bootstrap resamples, and the same control-arm draws
on any machine and in any process. This module concentrates the three primitives that
guarantee that.

``seed_all``
    Seeds the two random number generators the benchmark touches -- Python's
    :mod:`random` (the Mersenne Twister MT19937 of Matsumoto & Nishimura, 1998) and
    NumPy's legacy global generator (:func:`numpy.random.seed`, also MT19937). After a
    call, ``random.random()`` and ``numpy.random.random()`` replay identical streams.

``seeded``
    A context manager that *saves* both generators' full internal state on entry,
    re-seeds them for a reproducible block, and *restores* the saved state on exit
    (even if the block raises). The surrounding ("outer") RNG stream is therefore
    unaffected by anything drawn inside the block -- the standard save/seed/restore
    idiom for local determinism without global contamination.

``stable_hash``
    A deterministic content hash. CPython's builtin :func:`hash` for ``str``/``bytes``
    is *salted* per process (SipHash keyed by ``PYTHONHASHSEED``; Aumasson & Bernstein,
    SipHash, 2012), so it is **not** reproducible across processes and must never be
    used for cache keys or provenance. ``stable_hash`` instead serialises the object
    to canonical bytes and digests them with SHA-256 (NIST FIPS 180-4), which is fixed
    across processes, machines, and Python versions.

References
----------
* M. Matsumoto and T. Nishimura, "Mersenne Twister: a 623-dimensionally
  equidistributed uniform pseudo-random number generator", *ACM TOMACS* 8(1), 1998.
* National Institute of Standards and Technology, FIPS PUB 180-4, *Secure Hash
  Standard (SHS)*, 2015. The "abc" digest used in the tests is the test vector from
  FIPS 180-2, Appendix B.1.
* J.-P. Aumasson and D. J. Bernstein, "SipHash: a fast short-input PRF", 2012 --
  the keyed hash behind CPython's salted ``hash`` for ``str``/``bytes``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = ["seed_all", "seeded", "stable_hash"]


def seed_all(seed: int) -> None:
    """Seed every random number generator the benchmark uses.

    Seeds Python's :mod:`random` module and NumPy's legacy global generator so that
    subsequent ``random.random()`` and ``numpy.random.*`` draws replay identical
    streams for a given ``seed``.

    Parameters
    ----------
    seed
        The seed value applied to both generators.

    Notes
    -----
    This does **not** touch ``PYTHONHASHSEED`` (the salt for the builtin ``hash``),
    which cannot be changed once the interpreter has started. Use :func:`stable_hash`
    wherever a process-stable digest is required.
    """
    random.seed(seed)
    # NPY002: seeding the *legacy global* generator is the explicit purpose here --
    # it is what makes ``np.random.*`` (not just a local Generator) reproducible.
    np.random.seed(seed)  # noqa: NPY002


@contextlib.contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Run a block under a fixed seed, restoring the outer RNG state afterwards.

    On entry the full internal state of Python's :mod:`random` and NumPy's legacy
    global generator is captured; both are then re-seeded via :func:`seed_all`. On
    exit -- including when the block raises -- the captured state is restored, so draws
    made inside the block do not perturb the surrounding RNG streams.

    Parameters
    ----------
    seed
        The seed applied inside the ``with`` block.

    Yields
    ------
    None
        The context manager has no value to expose; it manages global RNG state.

    Examples
    --------
    >>> import random
    >>> with seeded(0):
    ...     inside = random.random()
    >>> with seeded(0):
    ...     repeat = random.random()
    >>> inside == repeat
    True
    """
    py_state = random.getstate()
    np_state = np.random.get_state()  # noqa: NPY002  (snapshot the legacy global state)
    try:
        seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)  # noqa: NPY002  (restore the legacy global state)


def _canonical_bytes(obj: object) -> bytes:
    """Serialise ``obj`` to deterministic bytes for hashing.

    ``bytes`` pass through unchanged, ``str`` is UTF-8 encoded as-is, and any other
    object is encoded as canonical JSON (keys sorted, no insignificant whitespace) so
    that mappings with the same contents always serialise identically.

    Parameters
    ----------
    obj
        The object to serialise. Non-``str``/``bytes`` values must be JSON-serialisable
        (a ``default=str`` fallback stringifies anything else, e.g. ``Path``).

    Returns
    -------
    bytes
        The canonical byte representation fed to the hash.
    """
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, str):
        return obj.encode("utf-8")
    payload: Any = obj
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def stable_hash(obj: object, *, algorithm: str = "sha256") -> str:
    """Return a process-stable hex digest of ``obj``.

    The object is serialised to canonical bytes (see :func:`_canonical_bytes`) and
    digested with ``algorithm`` (SHA-256 by default; NIST FIPS 180-4). Unlike the
    builtin :func:`hash`, the result is identical across processes, machines, and
    Python versions, making it safe for cache keys and run provenance.

    Parameters
    ----------
    obj
        The object to hash. ``str`` and ``bytes`` hash their UTF-8/raw bytes directly;
        other objects hash their canonical-JSON encoding.
    algorithm
        Any name accepted by :func:`hashlib.new` (e.g. ``"sha256"``, ``"blake2b"``).

    Returns
    -------
    str
        The lowercase hexadecimal digest.

    Examples
    --------
    >>> stable_hash("abc")  # FIPS 180-2 Appendix B.1 SHA-256 test vector
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    """
    return hashlib.new(algorithm, _canonical_bytes(obj)).hexdigest()
