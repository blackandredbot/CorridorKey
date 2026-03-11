"""Property-based tests for DepthThresholder.

# Feature: depth-keying-pipeline, Property 11: Background_Score threshold piecewise function
# Feature: depth-keying-pipeline, Property 12: Confidence modulation formula
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from CorridorKeyModule.depth.data_models import CubeResult
from CorridorKeyModule.depth.depth_thresholder import DepthThresholder


# ---------------------------------------------------------------------------
# Property 11: Background_Score threshold piecewise function
# ---------------------------------------------------------------------------

@st.composite
def threshold_piecewise_data(draw):
    """Generate random Background_Score maps with confidence = 1.0,
    random threshold in [0, 1], and random falloff in [0, 0.5]."""
    h = draw(st.integers(min_value=4, max_value=32))
    w = draw(st.integers(min_value=4, max_value=32))

    bg_score = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    assume(np.all(np.isfinite(bg_score)))

    threshold = draw(st.floats(min_value=0.0, max_value=1.0,
                              allow_subnormal=False))
    falloff = draw(st.floats(min_value=0.0, max_value=0.5,
                             allow_subnormal=False))
    assume(np.isfinite(threshold) and np.isfinite(falloff))
    # Ensure threshold and falloff are representable in float32
    threshold = float(np.float32(threshold))
    falloff = float(np.float32(falloff))

    return bg_score, threshold, falloff, h, w


@settings(max_examples=100)
@given(data=threshold_piecewise_data())
def test_background_score_threshold_piecewise(data):
    """Feature: depth-keying-pipeline, Property 11: Background_Score threshold piecewise function

    For any Background_Score map, threshold in [0, 1], and falloff in [0, 0.5],
    the Depth_Thresholder (with confidence = 1.0 everywhere) SHALL produce
    alpha values satisfying:
    - background_score >= threshold → alpha == 0.0 (background)
    - background_score < threshold - falloff → alpha == 1.0 (foreground)
    - threshold - falloff <= background_score < threshold → smooth cosine
      interpolation between 1.0 and 0.0

    All within floating-point tolerance of 1e-7.

    **Validates: Requirements 3.2, 3.3, 3.4, 3.7**
    """
    bg_score, threshold, falloff, h, w = data

    confidence = np.ones((h, w), dtype=np.float32)
    cube = CubeResult(
        background_score=bg_score,
        confidence_map=confidence,
    )

    thresholder = DepthThresholder(
        depth_threshold=threshold,
        depth_falloff=falloff,
        low_confidence_alpha=0.0,
    )
    alpha = thresholder.apply(cube)

    # Verify output shape and dtype
    assert alpha.shape == (h, w)
    assert alpha.dtype == np.float32

    zone_bottom = threshold - falloff

    for y in range(h):
        for x in range(w):
            score = float(bg_score[y, x])
            a = float(alpha[y, x])

            if score >= threshold:
                # Background region
                assert abs(a - 0.0) < 1e-6, (
                    f"At ({y},{x}): score={score:.8f} >= threshold={threshold:.8f}, "
                    f"expected alpha=0.0, got {a:.8f}"
                )
            elif score < zone_bottom:
                # Foreground region
                assert abs(a - 1.0) < 1e-6, (
                    f"At ({y},{x}): score={score:.8f} < zone_bottom={zone_bottom:.8f}, "
                    f"expected alpha=1.0, got {a:.8f}"
                )
            else:
                # Transition zone — cosine interpolation
                if falloff > 0:
                    t = (score - zone_bottom) / falloff
                    expected = 0.5 * (1.0 + np.cos(np.pi * t))
                    assert abs(a - expected) < 1e-6, (
                        f"At ({y},{x}): score={score:.8f} in transition zone, "
                        f"expected alpha={expected:.8f}, got {a:.8f}"
                    )

    # All alpha values must be in [0.0, 1.0]
    assert alpha.min() >= 0.0 - 1e-6
    assert alpha.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Property 12: Confidence modulation formula
# ---------------------------------------------------------------------------

@st.composite
def confidence_modulation_data(draw):
    """Generate random CubeResults with varying confidence values."""
    h = draw(st.integers(min_value=4, max_value=32))
    w = draw(st.integers(min_value=4, max_value=32))

    bg_score = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    confidence = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    assume(np.all(np.isfinite(bg_score)))
    assume(np.all(np.isfinite(confidence)))

    threshold = draw(st.floats(min_value=0.0, max_value=1.0,
                              allow_subnormal=False))
    falloff = draw(st.floats(min_value=0.0, max_value=0.5,
                             allow_subnormal=False))
    low_conf_alpha = draw(st.floats(min_value=0.0, max_value=1.0,
                                    allow_subnormal=False))
    assume(np.isfinite(threshold) and np.isfinite(falloff) and np.isfinite(low_conf_alpha))
    threshold = float(np.float32(threshold))
    falloff = float(np.float32(falloff))
    low_conf_alpha = float(np.float32(low_conf_alpha))

    return bg_score, confidence, threshold, falloff, low_conf_alpha, h, w


@settings(max_examples=100)
@given(data=confidence_modulation_data())
def test_confidence_modulation_formula(data):
    """Feature: depth-keying-pipeline, Property 12: Confidence modulation formula

    For any CubeResult with arbitrary confidence values, the Depth_Thresholder
    output SHALL satisfy:
    final_alpha[y, x] == raw_alpha[y, x] * confidence[y, x]
                       + low_confidence_alpha * (1 - confidence[y, x])
    within floating-point tolerance, where raw_alpha is the result of the
    piecewise threshold function.

    **Validates: Requirements 3.6**
    """
    bg_score, confidence, threshold, falloff, low_conf_alpha, h, w = data

    # First compute raw_alpha with confidence = 1.0 to get the piecewise result
    ones_confidence = np.ones((h, w), dtype=np.float32)
    cube_raw = CubeResult(
        background_score=bg_score,
        confidence_map=ones_confidence,
    )
    thresholder = DepthThresholder(
        depth_threshold=threshold,
        depth_falloff=falloff,
        low_confidence_alpha=low_conf_alpha,
    )
    raw_alpha = thresholder.apply(cube_raw)

    # Now compute with actual confidence
    cube_conf = CubeResult(
        background_score=bg_score,
        confidence_map=confidence,
    )
    final_alpha = thresholder.apply(cube_conf)

    # Verify the confidence modulation formula
    expected = raw_alpha * confidence + low_conf_alpha * (1.0 - confidence)
    expected = np.clip(expected, 0.0, 1.0).astype(np.float32)

    np.testing.assert_allclose(
        final_alpha,
        expected,
        atol=1e-6,
        err_msg="Confidence modulation formula violated",
    )
