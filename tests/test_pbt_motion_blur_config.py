"""Feature: motion-blur-alpha-refinement, Property 15: CLI parameter validation

For any blur_threshold <= 0, temporal_smoothing outside (0.0, 1.0],
division_epsilon <= 0, blur_dilation < 0, kernel_profile not in
{"linear", "cosine", "gaussian"}, plate_search_radius < 1, or
plate_alpha_threshold outside (0.0, 1.0], constructing a MotionBlurConfig
shall raise ValueError.

Validates: Requirements 7.2, 7.5, 7.7
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from CorridorKeyModule.depth import MotionBlurConfig
from CorridorKeyModule.depth.data_models import _VALID_KERNEL_PROFILES


# ---------------------------------------------------------------------------
# Strategies for generating out-of-range values
# ---------------------------------------------------------------------------

# blur_threshold <= 0
_invalid_blur_threshold = st.floats(
    max_value=0.0, allow_nan=False, allow_infinity=False
)

# temporal_smoothing outside (0.0, 1.0]
_invalid_temporal_smoothing = st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.001, allow_nan=False, allow_infinity=False),
)

# division_epsilon <= 0
_invalid_division_epsilon = st.floats(
    max_value=0.0, allow_nan=False, allow_infinity=False
)

# blur_dilation < 0
_invalid_blur_dilation = st.integers(max_value=-1)

# kernel_profile not in valid set
_invalid_kernel_profile = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in _VALID_KERNEL_PROFILES
)

# plate_search_radius < 1
_invalid_plate_search_radius = st.integers(max_value=0)

# plate_alpha_threshold outside (0.0, 1.0]
_invalid_plate_alpha_threshold = st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.001, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestBlurThresholdValidation:
    """blur_threshold must be > 0."""

    @settings(max_examples=100)
    @given(value=_invalid_blur_threshold)
    def test_rejects_non_positive_blur_threshold(self, value: float) -> None:
        """**Validates: Requirements 7.2**"""
        with pytest.raises(ValueError, match="blur_threshold"):
            MotionBlurConfig(blur_threshold=value)


class TestKernelProfileValidation:
    """kernel_profile must be one of {"linear", "cosine", "gaussian"}."""

    @settings(max_examples=100)
    @given(value=_invalid_kernel_profile)
    def test_rejects_invalid_kernel_profile(self, value: str) -> None:
        """**Validates: Requirements 7.2**"""
        with pytest.raises(ValueError, match="kernel_profile"):
            MotionBlurConfig(kernel_profile=value)


class TestTemporalSmoothingValidation:
    """temporal_smoothing must be in (0.0, 1.0]."""

    @settings(max_examples=100)
    @given(value=_invalid_temporal_smoothing)
    def test_rejects_out_of_range_temporal_smoothing(self, value: float) -> None:
        """**Validates: Requirements 7.2**"""
        with pytest.raises(ValueError, match="temporal_smoothing"):
            MotionBlurConfig(temporal_smoothing=value)


class TestDivisionEpsilonValidation:
    """division_epsilon must be > 0."""

    @settings(max_examples=100)
    @given(value=_invalid_division_epsilon)
    def test_rejects_non_positive_division_epsilon(self, value: float) -> None:
        """**Validates: Requirements 7.2**"""
        with pytest.raises(ValueError, match="division_epsilon"):
            MotionBlurConfig(division_epsilon=value)


class TestBlurDilationValidation:
    """blur_dilation must be >= 0."""

    @settings(max_examples=100)
    @given(value=_invalid_blur_dilation)
    def test_rejects_negative_blur_dilation(self, value: int) -> None:
        """**Validates: Requirements 7.2**"""
        with pytest.raises(ValueError, match="blur_dilation"):
            MotionBlurConfig(blur_dilation=value)


class TestPlateSearchRadiusValidation:
    """plate_search_radius must be >= 1."""

    @settings(max_examples=100)
    @given(value=_invalid_plate_search_radius)
    def test_rejects_plate_search_radius_below_minimum(self, value: int) -> None:
        """**Validates: Requirements 7.7**"""
        with pytest.raises(ValueError, match="plate_search_radius"):
            MotionBlurConfig(plate_search_radius=value)


class TestPlateAlphaThresholdValidation:
    """plate_alpha_threshold must be in (0.0, 1.0]."""

    @settings(max_examples=100)
    @given(value=_invalid_plate_alpha_threshold)
    def test_rejects_out_of_range_plate_alpha_threshold(self, value: float) -> None:
        """**Validates: Requirements 7.7**"""
        with pytest.raises(ValueError, match="plate_alpha_threshold"):
            MotionBlurConfig(plate_alpha_threshold=value)
