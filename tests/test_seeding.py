"""Tests for the centralized determinism module.

The hashing golden values are independent textbook constants, not values produced by
running ``stable_hash`` itself:

* SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
  -- the canonical test vector from NIST FIPS 180-2, Appendix B.1.
* SHA-256("")    = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  -- the well-known SHA-256 digest of the empty byte string.

``stable_hash`` UTF-8 encodes a ``str`` as-is before digesting, so a plain ASCII input
hashes to exactly the textbook digest of those bytes; both can be confirmed with
``printf 'abc' | shasum -a 256`` independently of this code.
"""

from __future__ import annotations

import random

import numpy as np

from membench.seeding import seed_all, seeded, stable_hash

SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestStableHash:
    def test_matches_nist_abc_vector(self) -> None:
        # Hand-derived golden: FIPS 180-2 Appendix B.1 SHA-256("abc").
        assert stable_hash("abc") == SHA256_ABC

    def test_matches_empty_string_vector(self) -> None:
        # Hand-derived golden: SHA-256 of the empty byte string.
        assert stable_hash("") == SHA256_EMPTY

    def test_bytes_and_str_agree_with_textbook(self) -> None:
        assert stable_hash(b"abc") == SHA256_ABC

    def test_is_deterministic_for_structured_objects(self) -> None:
        a = stable_hash({"x": 1, "y": [2, 3]})
        b = stable_hash({"x": 1, "y": [2, 3]})
        assert a == b
        assert len(a) == 64  # sha256 -> 32 bytes -> 64 hex chars

    def test_key_order_does_not_change_digest(self) -> None:
        # Canonical JSON sorts keys, so insertion order is irrelevant.
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

    def test_distinct_inputs_differ(self) -> None:
        assert stable_hash("abc") != stable_hash("abd")


class TestSeedAll:
    def test_python_random_replays(self) -> None:
        seed_all(123)
        first = [random.random() for _ in range(3)]
        seed_all(123)
        second = [random.random() for _ in range(3)]
        assert first == second

    def test_numpy_random_replays(self) -> None:
        seed_all(123)
        first = np.random.random(5)  # noqa: NPY002
        seed_all(123)
        second = np.random.random(5)  # noqa: NPY002
        assert np.array_equal(first, second)

    def test_different_seeds_diverge(self) -> None:
        seed_all(1)
        a = random.random()
        seed_all(2)
        b = random.random()
        assert a != b


class TestSeeded:
    def test_two_inner_draws_are_reproducible(self) -> None:
        with seeded(0):
            first = [random.random(), random.random()]
        with seeded(0):
            second = [random.random(), random.random()]
        assert first == second

    def test_restores_python_state(self) -> None:
        before = random.getstate()
        with seeded(0):
            random.random()
            random.random()
        assert random.getstate() == before

    def test_restores_numpy_state(self) -> None:
        before = np.random.get_state()  # noqa: NPY002
        with seeded(0):
            np.random.random(4)  # noqa: NPY002
        after = np.random.get_state()  # noqa: NPY002
        # Legacy state is a tuple whose second element is the MT19937 key array.
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]

    def test_outer_stream_is_unaffected(self) -> None:
        # Reference: the outer stream with no nested seeded() block.
        random.seed(7)
        reference = [random.random(), random.random()]
        # Same outer seed, but interpose a seeded() block between the two draws.
        random.seed(7)
        with_block = [random.random()]
        with seeded(999):
            random.random()
            random.random()
        with_block.append(random.random())
        assert with_block == reference

    def test_state_restored_even_if_block_raises(self) -> None:
        before = random.getstate()
        try:
            with seeded(0):
                random.random()
                raise ValueError("boom")
        except ValueError:
            pass
        assert random.getstate() == before
