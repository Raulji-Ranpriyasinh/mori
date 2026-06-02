"""
Unit tests for AI-4.5 & AI-4.7: Food Data Layer Calibration

Tests the soft-coupling rule that decides when to trust LLM vs USDA data.
"""

import pytest
from calibration import decide, pct_diff, apply_coupling_mode


class TestPctDiff:
    """Tests for percentage difference calculation."""

    def test_identical_values(self):
        """Identical values should have 0% difference."""
        assert pct_diff(100, 100) == 0.0
        assert pct_diff(0, 0) == 0.0

    def test_small_difference(self):
        """Small differences should calculate correctly."""
        # |100 - 90| / 100 = 0.1
        assert pct_diff(100, 90) == 0.1
        # |50 - 45| / 50 = 0.1
        assert pct_diff(50, 45) == 0.1

    def test_25_percent_difference(self):
        """Exactly 25% difference boundary."""
        # |100 - 75| / 100 = 0.25
        assert pct_diff(100, 75) == 0.25
        # |100 - 125| / 100 = 0.25
        assert pct_diff(100, 125) == 0.25

    def test_large_difference(self):
        """Large differences should calculate correctly."""
        # |100 - 50| / 100 = 0.5
        assert pct_diff(100, 50) == 0.5
        # |100 - 200| / 100 = 1.0
        assert pct_diff(100, 200) == 1.0

    def test_zero_llm_value_with_nonzero_usda(self):
        """LLM is 0 but USDA has value -> 100% difference."""
        assert pct_diff(0, 50) == 1.0

    def test_zero_llm_value_with_zero_usda(self):
        """Both are 0 -> 0% difference."""
        assert pct_diff(0, 0) == 0.0

    def test_negative_values(self):
        """Handle negative values (shouldn't happen but be safe)."""
        # abs(-100 - (-90)) / abs(-100) = 10/100 = 0.1
        assert pct_diff(-100, -90) == 0.1


class TestDecide:
    """Tests for the soft-coupling decision function."""

    def test_usda_missing_returns_llm(self):
        """When USDA data is missing, use LLM value."""
        value, source, reason = decide(50, None)
        assert value == 50
        assert source == "llm"
        assert reason == "usda_missing"

    def test_usda_missing_with_zero_llm(self):
        """When USDA is missing and LLM is 0, still use LLM."""
        value, source, reason = decide(0, None)
        assert value == 0
        assert source == "llm"
        assert reason == "usda_missing"

    def test_within_threshold_uses_usda(self):
        """When within 25%, prefer USDA value."""
        # 10% difference -> use USDA
        value, source, reason = decide(100, 90)
        assert value == 90
        assert source == "usda"
        assert reason == "within_25%"

    def test_exactly_at_threshold_uses_usda(self):
        """Exactly 25% difference should use USDA (boundary condition)."""
        value, source, reason = decide(100, 75)
        assert value == 75
        assert source == "usda"
        assert reason == "within_25%"

    def test_above_threshold_uses_llm(self):
        """When above 25%, keep LLM value."""
        # 30% difference -> use LLM
        value, source, reason = decide(100, 70)
        assert value == 100
        assert source == "llm"
        assert reason == "deviation_too_high"

    def test_usda_higher_within_threshold(self):
        """USDA higher than LLM but within threshold."""
        # 20% difference (USDA higher) -> use USDA
        value, source, reason = decide(100, 120)
        assert value == 120
        assert source == "usda"
        assert reason == "within_25%"

    def test_usda_higher_above_threshold(self):
        """USDA much higher than LLM -> keep LLM."""
        # 50% difference (USDA higher) -> use LLM
        value, source, reason = decide(100, 150)
        assert value == 100
        assert source == "llm"
        assert reason == "deviation_too_high"

    def test_custom_threshold(self):
        """Custom threshold should work."""
        # 20% difference with 15% threshold -> use LLM
        value, source, reason = decide(100, 80, threshold=0.15)
        assert value == 100
        assert source == "llm"
        assert reason == "deviation_too_high"

        # 20% difference with 25% threshold -> use USDA
        value, source, reason = decide(100, 80, threshold=0.25)
        assert value == 80
        assert source == "usda"
        assert reason == "within_25%"

    def test_zero_values_both(self):
        """Both zero -> use USDA (0% difference)."""
        value, source, reason = decide(0, 0)
        assert value == 0
        assert source == "usda"
        assert reason == "within_25%"

    def test_llm_zero_usda_nonzero(self):
        """LLM is 0, USDA has value -> 100% diff, use LLM."""
        value, source, reason = decide(0, 50)
        assert value == 0
        assert source == "llm"
        assert reason == "deviation_too_high"

    def test_small_absolute_values(self):
        """Test with small macro values."""
        # |5 - 4| / 5 = 0.2 -> use USDA
        value, source, reason = decide(5, 4)
        assert value == 4
        assert source == "usda"
        assert reason == "within_25%"

    def test_floating_point_precision(self):
        """Test floating point handling."""
        value, source, reason = decide(10.5, 10.0)
        # |10.5 - 10.0| / 10.5 ≈ 0.0476 -> use USDA
        assert value == 10.0
        assert source == "usda"
        assert reason == "within_25%"


class TestProvenanceTracking:
    """Tests verifying provenance metadata is correctly tracked."""

    def test_all_sources_represented(self):
        """Verify all possible sources can be returned."""
        sources_seen = set()
        reasons_seen = set()

        # Force usda_missing
        _, src, rsn = decide(50, None)
        sources_seen.add(src)
        reasons_seen.add(rsn)

        # Force within_25%
        _, src, rsn = decide(100, 95)
        sources_seen.add(src)
        reasons_seen.add(rsn)

        # Force deviation_too_high
        _, src, rsn = decide(100, 50)
        sources_seen.add(src)
        reasons_seen.add(rsn)

        assert sources_seen == {"llm", "usda"}
        assert reasons_seen == {"usda_missing", "within_25%", "deviation_too_high"}


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_very_large_values(self):
        """Test with very large macro values."""
        value, source, reason = decide(10000, 9500)
        assert value == 9500
        assert source == "usda"

    def test_very_small_difference(self):
        """Test with tiny difference."""
        value, source, reason = decide(100, 99.9)
        assert value == 99.9
        assert source == "usda"

    def test_negative_usda_value(self):
        """Negative USDA value (data error) should still process."""
        value, source, reason = decide(100, -10)
        # |100 - (-10)| / 100 = 1.1 -> 110% diff
        assert value == 100
        assert source == "llm"
        assert reason == "deviation_too_high"


class TestCouplingModes:
    """Tests for A/B testing coupling modes."""

    def test_soft_mode_default(self):
        """Soft mode applies 25% threshold."""
        # 20% difference -> use USDA in soft mode
        value, source, reason = apply_coupling_mode(100, 80, coupling_mode="soft")
        assert value == 80
        assert source == "usda"

    def test_strict_mode_always_usda(self):
        """Strict mode always uses USDA when available."""
        # 50% difference -> still use USDA in strict mode
        value, source, reason = apply_coupling_mode(100, 50, coupling_mode="strict")
        assert value == 50
        assert source == "usda"
        assert reason == "strict_mode"

    def test_llm_only_mode_never_usda(self):
        """LLM-only mode never uses USDA."""
        value, source, reason = apply_coupling_mode(100, 95, coupling_mode="llm_only")
        assert value == 100
        assert source == "llm"
        assert reason == "llm_only_mode"

    def test_usda_missing_all_modes_same(self):
        """When USDA is missing, all modes return LLM."""
        for mode in ["soft", "strict", "llm_only"]:
            value, source, reason = apply_coupling_mode(100, None, coupling_mode=mode)
            assert value == 100
            assert source == "llm"
            assert reason == "usda_missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
