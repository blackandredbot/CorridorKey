"""Feature: depth-keying-pipeline, Property 17: CLI parameter range validation

For any depth-specific parameter value outside its documented valid range
(e.g., depth_threshold < 0 or > 1, depth_falloff < 0 or > 0.5,
flow_method not in {"farneback", "raft"}, fusion_mode not in
{"blend", "max", "min"}, weights not summing to 1.0), the system SHALL
reject the input with a descriptive error rather than proceeding.

The validation lives in DepthKeyingConfig.__post_init__, which is the same
validation the CLI triggers when constructing the config from flags.

Validates: Requirements 6.5
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from CorridorKeyModule.depth import DepthKeyingConfig


# ---------------------------------------------------------------------------
# Strategies for generating out-of-range values
# ---------------------------------------------------------------------------

# Floats outside [0.0, 1.0]
_out_of_unit_range = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.001, allow_nan=False, allow_infinity=False),
)

# Floats outside [0.0, 0.5]
_out_of_falloff_range = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.501, allow_nan=False, allow_infinity=False),
)

# Invalid flow methods
_invalid_flow_method = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in {"farneback", "raft"}
)

# Invalid fusion modes
_invalid_fusion_mode = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in {"blend", "max", "min"}
)

# Buffer size < 2
_invalid_buffer_size = st.integers(max_value=1)

# Consistency threshold <= 0
_invalid_consistency = st.floats(
    max_value=0.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestDepthThresholdValidation:
    """depth_threshold must be in [0.0, 1.0]."""

    @settings(max_examples=100)
    @given(value=_out_of_unit_range)
    def test_rejects_out_of_range_depth_threshold(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="depth_threshold"):
            DepthKeyingConfig(depth_threshold=value)


class TestDepthFalloffValidation:
    """depth_falloff must be in [0.0, 0.5]."""

    @settings(max_examples=100)
    @given(value=_out_of_falloff_range)
    def test_rejects_out_of_range_depth_falloff(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="depth_falloff"):
            DepthKeyingConfig(depth_falloff=value)


class TestFlowMethodValidation:
    """flow_method must be one of {"farneback", "raft"}."""

    @settings(max_examples=100)
    @given(value=_invalid_flow_method)
    def test_rejects_invalid_flow_method(self, value: str) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="flow_method"):
            DepthKeyingConfig(flow_method=value)


class TestFusionModeValidation:
    """fusion_mode must be one of {"blend", "max", "min"}."""

    @settings(max_examples=100)
    @given(value=_invalid_fusion_mode)
    def test_rejects_invalid_fusion_mode(self, value: str) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="fusion_mode"):
            DepthKeyingConfig(fusion_mode=value)


class TestWeightSumValidation:
    """parallax_weight + persistence_weight + stability_weight must == 1.0."""

    @settings(max_examples=100)
    @given(
        pw=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        per=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        sw=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_rejects_weights_not_summing_to_one(
        self, pw: float, per: float, sw: float
    ) -> None:
        """**Validates: Requirements 6.5**"""
        assume(abs(pw + per + sw - 1.0) > 1e-9)
        with pytest.raises(ValueError, match="must equal 1.0"):
            DepthKeyingConfig(
                parallax_weight=pw,
                persistence_weight=per,
                stability_weight=sw,
            )


class TestIndividualWeightValidation:
    """Each weight must be in [0.0, 1.0]."""

    @settings(max_examples=100)
    @given(value=_out_of_unit_range)
    def test_rejects_out_of_range_parallax_weight(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="parallax_weight"):
            DepthKeyingConfig(parallax_weight=value)

    @settings(max_examples=100)
    @given(value=_out_of_unit_range)
    def test_rejects_out_of_range_persistence_weight(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="persistence_weight"):
            DepthKeyingConfig(persistence_weight=value)

    @settings(max_examples=100)
    @given(value=_out_of_unit_range)
    def test_rejects_out_of_range_stability_weight(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="stability_weight"):
            DepthKeyingConfig(stability_weight=value)


class TestRefinementStrengthValidation:
    """refinement_strength must be in [0.0, 1.0]."""

    @settings(max_examples=100)
    @given(value=_out_of_unit_range)
    def test_rejects_out_of_range_refinement_strength(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="refinement_strength"):
            DepthKeyingConfig(refinement_strength=value)


class TestBufferSizeValidation:
    """cube_buffer_size must be >= 2."""

    @settings(max_examples=100)
    @given(value=_invalid_buffer_size)
    def test_rejects_buffer_size_below_minimum(self, value: int) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="cube_buffer_size"):
            DepthKeyingConfig(cube_buffer_size=value)


class TestConsistencyThresholdValidation:
    """consistency_threshold must be > 0."""

    @settings(max_examples=100)
    @given(value=_invalid_consistency)
    def test_rejects_non_positive_consistency_threshold(self, value: float) -> None:
        """**Validates: Requirements 6.5**"""
        with pytest.raises(ValueError, match="consistency_threshold"):
            DepthKeyingConfig(consistency_threshold=value)
