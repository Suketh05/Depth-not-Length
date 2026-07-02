"""Optional native cosine-top-k kernel with a NumPy reference and fallback.

The NumPy reference (:func:`cosine_topk_reference`) is authoritative and always
available; it is what produces the reported numbers. When the compiled shared
library (``native/c``) is present it is loaded via ctypes and used for speed on
large sweeps, but it is validated to match the reference byte-for-byte, so the
two are interchangeable. Absence of the library is not an error — callers use
:func:`native_available` to decide, and the benchmark itself always has a working
path.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "cohens_h_native",
    "cosine_topk_native",
    "cosine_topk_reference",
    "crossover_search_native",
    "gap_native",
    "native_available",
    "native_library_path",
    "similarity_recovery_native",
    "structured_recovery_native",
    "wilson_interval_native",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB: ctypes.CDLL | None = None
_LOAD_ATTEMPTED = False


def _candidate_paths() -> list[Path]:
    override = os.environ.get("MEMBENCH_NATIVE_LIB")
    candidates = [Path(override)] if override else []
    build_dir = _REPO_ROOT / "native" / "c" / "build"
    candidates += [build_dir / "libmembench_native.dylib", build_dir / "libmembench_native.so"]
    return candidates


def native_library_path() -> Path | None:
    """Return the path to the compiled native library, or ``None`` if not built."""
    for path in _candidate_paths():
        if path.is_file():
            return path
    return None


def _load() -> ctypes.CDLL | None:
    global _LIB, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIB
    _LOAD_ATTEMPTED = True
    path = native_library_path()
    if path is None:
        return None
    lib = ctypes.CDLL(str(path))
    lib.mb_version.restype = ctypes.c_int
    lib.mb_cosine_topk.restype = ctypes.c_int
    lib.mb_cosine_topk.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # corpus
        ctypes.c_int,  # n
        ctypes.c_int,  # dim
        ctypes.POINTER(ctypes.c_float),  # query
        ctypes.c_int,  # k
        ctypes.POINTER(ctypes.c_int),  # out_idx
        ctypes.POINTER(ctypes.c_float),  # out_score
    ]
    # Depth-theory kernels (membench.theory.*). Scalar double in/out; restype
    # MUST be set or ctypes truncates the return to a C int.
    lib.mb_similarity_recovery.restype = ctypes.c_double
    lib.mb_similarity_recovery.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.mb_structured_recovery.restype = ctypes.c_double
    lib.mb_structured_recovery.argtypes = [ctypes.c_double, ctypes.c_int]
    lib.mb_gap.restype = ctypes.c_double
    lib.mb_gap.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.mb_crossover_search.restype = ctypes.c_int
    lib.mb_crossover_search.argtypes = [
        ctypes.c_double,  # s0
        ctypes.c_double,  # rho
        ctypes.c_double,  # q
        ctypes.c_double,  # tau
        ctypes.c_int,  # max_d
    ]
    # Inference kernels (membench.metrics.factorization, membench.stats).
    lib.mb_wilson_low.restype = ctypes.c_double
    lib.mb_wilson_low.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.mb_wilson_high.restype = ctypes.c_double
    lib.mb_wilson_high.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.mb_cohens_h.restype = ctypes.c_double
    lib.mb_cohens_h.argtypes = [ctypes.c_double, ctypes.c_double]
    _LIB = lib
    return lib


def native_available() -> bool:
    """Return whether the compiled native kernel is loadable."""
    return _load() is not None


def cosine_topk_reference(
    corpus: NDArray[np.float64],
    query: NDArray[np.float64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return the top-``k`` corpus rows by cosine similarity (NumPy reference).

    Assumes rows of ``corpus`` and ``query`` are L2-normalised, so the dot product
    is the cosine similarity. Returns ``(indices, scores)`` in descending order.
    """
    if k <= 0 or corpus.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    scores = corpus @ query
    kk = min(k, scores.shape[0])
    top = np.argpartition(-scores, kk - 1)[:kk]
    order = np.argsort(-scores[top], kind="stable")
    idx = top[order]
    return idx.astype(np.int64), np.asarray(scores[idx], dtype=np.float64)


def cosine_topk_native(
    corpus: NDArray[np.float64],
    query: NDArray[np.float64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return the top-``k`` rows using the compiled kernel.

    Raises ``RuntimeError`` if the native library is unavailable; callers should
    gate on :func:`native_available` or use :func:`cosine_topk_reference`.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("native kernel not available; build native/c or use the reference")
    if k <= 0 or corpus.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    corpus32 = np.ascontiguousarray(corpus, dtype=np.float32)
    query32 = np.ascontiguousarray(query, dtype=np.float32)
    n, dim = int(corpus32.shape[0]), int(corpus32.shape[1])
    kk = min(k, n)
    out_idx = np.empty(kk, dtype=np.int32)
    out_score = np.empty(kk, dtype=np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    lib.mb_cosine_topk(
        corpus32.ctypes.data_as(c_float_p),
        ctypes.c_int(n),
        ctypes.c_int(dim),
        query32.ctypes.data_as(c_float_p),
        ctypes.c_int(kk),
        out_idx.ctypes.data_as(c_int_p),
        out_score.ctypes.data_as(c_float_p),
    )
    return out_idx.astype(np.int64), out_score.astype(np.float64)


def _require_lib() -> ctypes.CDLL:
    lib = _load()
    if lib is None:
        raise RuntimeError("native kernel not available; build native/c or use the reference")
    return lib


def similarity_recovery_native(s0: float, rho: float, d: int) -> float:
    r"""Native ``P_sim(d) = s0^d * rho^{d(d+1)/2}`` (paper eq:simceiling).

    Parity mirror of :func:`membench.theory.recovery.similarity_recovery`. Raises
    ``RuntimeError`` if the native library is not built (gate on
    :func:`native_available`).
    """
    return float(_require_lib().mb_similarity_recovery(s0, rho, d))


def structured_recovery_native(q: float, d: int) -> float:
    r"""Native ``P_struct(d) = q^d`` (paper Thm thm:struct).

    Parity mirror of :func:`membench.theory.recovery.structured_recovery`.
    """
    return float(_require_lib().mb_structured_recovery(q, d))


def gap_native(s0: float, rho: float, q: float, d: float) -> float:
    r"""Native structured-advantage gap ``g(d) = q^d - s0^d rho^{d(d+1)/2}``.

    Paper eq:gd. Continuous ``d`` allowed, matching
    :func:`membench.theory.crossover.g`.
    """
    return float(_require_lib().mb_gap(s0, rho, q, d))


def crossover_search_native(
    s0: float, rho: float, q: float, tau: float, max_d: int
) -> int | None:
    r"""Native crossover depth ``d*`` (paper prop:cross).

    Smallest integer ``d`` in ``[1, max_d]`` with ``g(d) > tau``, or ``None`` when
    structure never clears the overhead on the scanned range. Parity mirror of
    :func:`membench.theory.crossover.find_crossover_depth`'s ``d_star`` (the C
    kernel returns 0 for the no-crossover case, mapped to ``None`` here).
    """
    result = int(_require_lib().mb_crossover_search(s0, rho, q, tau, max_d))
    return result if result != 0 else None


def wilson_interval_native(
    successes: float,
    trials: float,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    r"""Native Wilson score interval ``(low, high)`` for a binomial proportion.

    Parity mirror of :func:`membench.metrics.factorization.wilson_interval`
    (Wilson 1927; the interval reported for the use-factor kappa, tab:realret).
    The standard-normal quantile ``z`` is computed here with the same
    ``scipy.stats.norm.ppf`` call as the reference and passed into the kernel, so
    the two agree to floating-point tolerance.
    """
    from scipy import stats  # local import: keeps the native loader lightweight

    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0:
        raise ValueError("successes must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    lib = _require_lib()
    low = float(lib.mb_wilson_low(successes, trials, z))
    high = float(lib.mb_wilson_high(successes, trials, z))
    return low, high


def cohens_h_native(p1: float, p2: float) -> float:
    r"""Native Cohen's h ``2 asin(sqrt(p1)) - 2 asin(sqrt(p2))`` (paper fig:cohensh).

    Parity mirror of :func:`membench.stats.effect_size.cohens_h`.
    """
    return float(_require_lib().mb_cohens_h(p1, p2))
