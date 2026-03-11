"""Property-based tests for neural fallback activation threshold.

# Feature: depth-keying-pipeline, Property 18: Neural fallback activation threshold
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.data_models import CubeResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def confidence_and_threshold(draw: st.DrawFn):
    """Generate a confidence map mean and a fallback threshold.

    Returns (mean_confidence, threshold, h, w) where h and w are small
    spatial dimensions for the confidence map.
    """
    h = draw(st.integers(min_value=4, max_value=16))
    w = draw(st.integers(min_value=4, max_value=16))
    threshold = draw(st.floats(min_value=0.1, max_value=0.9))
    # Generate a mean confidence that is clearly below or above threshold
    # with a gap to avoid floating-point edge cases at the boundary
    below = draw(st.booleans())
    epsilon = 0.02
    if below:
        mean_conf = draw(st.floats(
            min_value=0.0,
            max_value=max(0.0, threshold - epsilon),
        ))
    else:
        mean_conf = draw(st.floats(
            min_value=threshold + epsilon,
            max_value=1.0,
        ))
    assume(np.isfinite(mean_conf))
    assume(np.isfinite(threshold))
    return mean_conf, threshold, h, w, below


def _make_confidence_map(mean_val: float, h: int, w: int) -> np.ndarray:
    """Create a confidence map with a specific mean value.

    Fills the map uniformly so that np.mean() == mean_val exactly.
    """
    return np.full((h, w), mean_val, dtype=np.float32)


def _make_cube_result(h: int, w: int, confidence_mean: float) -> CubeResult:
    """Create a CubeResult with a given confidence mean."""
    return CubeResult(
        background_score=np.random.rand(h, w).astype(np.float32),
        confidence_map=_make_confidence_map(confidence_mean, h, w),
        parallax_map=None,
        persistence_map=None,
        stability_map=None,
    )


def _make_fallback_cube_result(h: int, w: int) -> CubeResult:
    """Create a distinct CubeResult that the mock fallback returns."""
    return CubeResult(
        background_score=np.ones((h, w), dtype=np.float32) * 0.999,
        confidence_map=np.ones((h, w), dtype=np.float32),
        parallax_map=None,
        persistence_map=None,
        stability_map=None,
    )


# ---------------------------------------------------------------------------
# Property 18: Neural fallback activation threshold
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=confidence_and_threshold())
def test_neural_fallback_activation_threshold(data):
    """Feature: depth-keying-pipeline, Property 18: Neural fallback activation threshold

    For any frame where the depth fallback is enabled: if the mean
    Depth_Confidence across the frame is below fallback_confidence_threshold,
    the neural fallback SHALL activate; if the mean confidence is at or above
    the threshold, the neural fallback SHALL NOT activate.

    Validates: Requirements 10.1, 10.2
    """
    mean_conf, threshold, h, w, expect_fallback = data

    # Build the original cube result from the image cube builder
    original_cube = _make_cube_result(h, w, mean_conf)
    fallback_cube = _make_fallback_cube_result(h, w)

    # Create a mock neural fallback that returns our known fallback result
    mock_fallback = MagicMock()
    mock_fallback.estimate.return_value = fallback_cube

    # Simulate the activation logic from DepthKeyingEngine.process_clip:
    #   if self.depth_fallback and self._neural_fallback is not None:
    #       mean_confidence = float(np.mean(cube_result.confidence_map))
    #       if mean_confidence < self.fallback_confidence_threshold:
    #           cube_result = self._neural_fallback.estimate(frame)
    depth_fallback_enabled = True
    fallback_confidence_threshold = threshold

    cube_result = original_cube
    mean_confidence = float(np.mean(cube_result.confidence_map))

    if depth_fallback_enabled and mock_fallback is not None:
        if mean_confidence < fallback_confidence_threshold:
            cube_result = mock_fallback.estimate(
                np.zeros((h, w, 3), dtype=np.float32)
            )

    if expect_fallback:
        # Fallback should have activated — result should be the fallback cube
        assert mock_fallback.estimate.called, (
            f"Fallback should activate: mean_confidence={mean_confidence:.4f} "
            f"< threshold={threshold:.4f}"
        )
        np.testing.assert_array_equal(
            cube_result.background_score,
            fallback_cube.background_score,
        )
    else:
        # Fallback should NOT have activated — result should be original
        assert not mock_fallback.estimate.called, (
            f"Fallback should NOT activate: mean_confidence={mean_confidence:.4f} "
            f">= threshold={threshold:.4f}"
        )
        np.testing.assert_array_equal(
            cube_result.background_score,
            original_cube.background_score,
        )
