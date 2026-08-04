"""Core contracts: config identity, invariants, factorial enumeration."""

from __future__ import annotations

import pytest

from parsimony.core.config import (
    ParsimonyConfig,
    baseline,
    factorial_cells,
    full_stack,
    with_cache_lookup,
)
from parsimony.core.errors import ConfigError
from parsimony.core.types import InvariantClass, Invariants


class TestConfigHash:
    def test_is_stable_across_identical_configs(self):
        assert full_stack().config_hash == full_stack().config_hash

    def test_differs_when_a_module_is_toggled(self):
        assert baseline().config_hash != full_stack().config_hash

    def test_differs_when_a_threshold_changes(self):
        from dataclasses import replace

        from parsimony.core.config import CacheConfig

        a = full_stack()
        b = replace(a, cache=CacheConfig(tau_hi=0.99))
        assert a.config_hash != b.config_hash

    def test_ignores_cosmetic_label(self):
        """An ablation cell is identified by its settings, not by its name."""
        a = full_stack()
        b = a.with_modules(a.enabled_modules, label="a totally different name")
        assert a.config_hash == b.config_hash

    def test_is_insensitive_to_frozenset_ordering(self):
        a = ParsimonyConfig(enabled_modules=frozenset({"M1", "M2"}))
        b = ParsimonyConfig(enabled_modules=frozenset({"M2", "M1"}))
        assert a.config_hash == b.config_hash


class TestConfigValidation:
    def test_rejects_unknown_cache_lookup_mode(self):
        with pytest.raises(ConfigError):
            ParsimonyConfig(cache_lookup_on="SOMETIMES")

    def test_rejects_duplicate_stages(self):
        with pytest.raises(ConfigError):
            ParsimonyConfig(stage_order=("m1_tier1", "m1_tier1"))


class TestFactorialCells:
    def test_enumerates_two_to_the_n(self):
        cells = list(factorial_cells(axes=("M1", "M2", "M5"), always_on=frozenset()))
        assert len(cells) == 8

    def test_every_cell_has_a_distinct_hash(self):
        cells = list(factorial_cells(axes=("M1", "M2", "M5"), always_on=frozenset()))
        assert len({c.config_hash for c in cells}) == len(cells)

    def test_contains_a_baseline_with_no_modules(self):
        cells = list(factorial_cells(axes=("M1", "M2"), always_on=frozenset()))
        base = next(c for c in cells if c.label == "baseline")
        assert base.enabled_modules == frozenset()


class TestCacheLookupReordering:
    def test_raw_puts_cache_before_compression(self):
        order = with_cache_lookup(full_stack(), "RAW").stage_order
        assert order.index("m2_cache") < order.index("m1_tier1")

    def test_compressed_puts_cache_after_compression(self):
        order = with_cache_lookup(full_stack(), "COMPRESSED").stage_order
        assert order.index("m2_cache") > order.index("m1_tier3")

    def test_both_arms_contain_the_same_stages(self):
        """The two Gap 3 arms must differ ONLY in ordering, or the comparison
        is confounded by which modules ran."""
        raw = with_cache_lookup(full_stack(), "RAW").stage_order
        comp = with_cache_lookup(full_stack(), "COMPRESSED").stage_order
        assert sorted(raw) == sorted(comp)
        assert raw != comp


class TestInvariants:
    def test_reports_nothing_lost_when_text_is_preserved(self):
        inv = Invariants(numbers=frozenset({"42"}), entities=frozenset({"Paris"}))
        assert inv.missing_from("Paris has 42 things") == {}

    def test_detects_a_dropped_number(self):
        inv = Invariants(numbers=frozenset({"42", "7"}))
        lost = inv.missing_from("only 42 survives")
        assert lost[InvariantClass.NUMBER] == frozenset({"7"})

    def test_does_not_match_a_number_inside_a_longer_number(self):
        """'210' must not be considered present in '1210' — a word-boundary
        failure here would let the gate pass genuine losses."""
        inv = Invariants(numbers=frozenset({"210"}))
        assert InvariantClass.NUMBER in inv.missing_from("the value is 1210")

    def test_detects_a_dropped_negation(self):
        inv = Invariants(negations=frozenset({"not"}))
        lost = inv.missing_from("it is safe")
        assert lost[InvariantClass.NEGATION] == frozenset({"not"})

    def test_union_merges_all_classes(self):
        a = Invariants(numbers=frozenset({"1"}), entities=frozenset({"A"}))
        b = Invariants(numbers=frozenset({"2"}), negations=frozenset({"no"}))
        merged = a.union(b)
        assert merged.numbers == frozenset({"1", "2"})
        assert merged.total() == 4
