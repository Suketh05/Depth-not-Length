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
    "cosine_topk_native",
    "cosine_topk_reference",
    "native_available",
    "native_library_path",
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
