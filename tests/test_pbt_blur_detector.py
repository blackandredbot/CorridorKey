"""Feature: motion-blur-alpha-refinement, Properties 1, 2, 3: Blur Region Detector

Property 1: Blur magnitude thresholding classifies correctly
Property 2: Blur mask preserves spatial resolution
Property 3: Dilation expands the blur mask

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.blur_region_detector import BlurRegionDetector


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def dimensions(draw: st.DrawFn) -> tuple[int, int]:
    """Generate small (H, W) dimensions in [4, 64]."""
    h = draw(st.integers(min_value=4, max_value=64))
    w = draw(st.integers(min_value=4, max_value=64))
    return h, w


@st.composite
def flow_fields(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W, 2] flow field with values in [-20, 20]."""
    data = draw(
        st.lists(
            st.floats(
                min_value=-20.0, max_value=20.0,
                allow_nan=False, allow_infinity=False,
            ),
            min_size=h * w * 2,
            max_size=h * w * 2,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w, 2)


@st.composite
def alpha_mattes(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W] alpha matte with values in [0.0, 1.0].

    Includes edge values 0.0 and 1.0 to exercise boundary conditions.
    """
    data = draw(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            ),
            min_size=h * w,
            max_size=h * w,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w)


@st.composite
def blur_thresholds(draw: st.DrawFn) -> float:
    """Generate a positive blur threshold."""
    return draw(
        st.floats(min_value=0.1, max_value=15.0, allow_nan=False, allow_infinity=False)
    )


# ---------------------------------------------------------------------------
# Property 1: Blur magnitude thresholding classifies correctly
# ---------------------------------------------------------------------------


class TestBlurMagnitudeThresholding:
    """Property 1: For any flow field and blur_threshold > 0, the blur mask
    marks a pixel as 1.0 iff sqrt(dx² + dy²) > blur_threshold AND alpha
    in (0.0, 1.0).

    IMPORTANT: Uses dilation_radius=0 to test pure thresholding without
    dilation effects.

    **Validates: Requirements 1.1, 1.2, 1.4**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_thresholding_classifies_correctly(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 1: Blur magnitude
        thresholding classifies correctly.

        **Validates: Requirements 1.1, 1.2, 1.4**
        """
        h, w = data.draw(dimensions())
        flow = data.draw(flow_fields(h, w))
        alpha = data.draw(alpha_mattes(h, w))
        threshold = data.draw(blur_thresholds())

        detector = BlurRegionDetector(
            blur_threshold=threshold, dilation_radius=0
        )
        mask = detector.detect(flow, alpha)

        # Compute expected classification per pixel
        dx = flow[:, :, 0]
        dy = flow[:, :, 1]
        magnitude = np.sqrt(dx * dx + dy * dy)

        expected = (
            (magnitude > threshold) & (alpha > 0.0) & (alpha < 1.0)
        ).astype(np.float32)

        np.testing.assert_array_equal(
            mask, expected,
            err_msg=(
                f"Thresholding mismatch with threshold={threshold}. "
                f"Mask sum={mask.sum()}, expected sum={expected.sum()}"
            ),
        )


# ---------------------------------------------------------------------------
# Property 2: Blur mask preserves spatial resolution
# ---------------------------------------------------------------------------


class TestBlurMaskSpatialResolution:
    """Property 2: For any alpha [H, W] and flow [H, W, 2], output blur mask
    has shape [H, W] with dtype float32.

    **Validates: Requirements 1.3**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_output_shape_and_dtype(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 2: Blur mask
        preserves spatial resolution.

        **Validates: Requirements 1.3**
        """
        h, w = data.draw(dimensions())
        flow = data.draw(flow_fields(h, w))
        alpha = data.draw(alpha_mattes(h, w))

        detector = BlurRegionDetector(blur_threshold=2.0, dilation_radius=3)
        mask = detector.detect(flow, alpha)

        assert mask.shape == (h, w), (
            f"Shape mismatch: expected ({h}, {w}), got {mask.shape}"
        )
        assert mask.dtype == np.float32, (
            f"Dtype mismatch: expected float32, got {mask.dtype}"
        )


# ---------------------------------------------------------------------------
# Property 3: Dilation expands the blur mask
# ---------------------------------------------------------------------------


class TestDilationExpansion:
    """Property 3: For any blur mask, dilation with radius > 0 produces a
    superset — every 1.0 pixel remains 1.0, total count >= original.

    **Validates: Requirements 1.5**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_dilation_is_superset(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 3: Dilation
        expands the blur mask.

        **Validates: Requirements 1.5**
        """
        h, w = data.draw(dimensions())
        flow = data.draw(flow_fields(h, w))
        alpha = data.draw(alpha_mattes(h, w))
        dilation_radius = data.draw(
            st.integers(min_value=1, max_value=5)
        )

        # Get undilated mask
        detector_no_dilation = BlurRegionDetector(
            blur_threshold=2.0, dilation_radius=0
        )
        mask_undilated = detector_no_dilation.detect(flow, alpha)

        # Get dilated mask
        detector_dilated = BlurRegionDetector(
            blur_threshold=2.0, dilation_radius=dilation_radius
        )
        mask_dilated = detector_dilated.detect(flow, alpha)

        # Every pixel that was 1.0 before dilation must remain 1.0
        assert np.all(mask_dilated[mask_undilated == 1.0] == 1.0), (
            "Dilation removed a pixel that was 1.0 in the undilated mask"
        )

        # Total count of 1.0 pixels must be >= original
        assert mask_dilated.sum() >= mask_undilated.sum(), (
            f"Dilated mask has fewer 1.0 pixels ({mask_dilated.sum()}) "
            f"than undilated ({mask_undilated.sum()})"
        )
