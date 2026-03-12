"""Feature: motion-blur-alpha-refinement, Properties 12, 13: Temporal Coherence Filter

Property 12: Temporal EMA formula correctness
Property 13: Temporal smoothing only affects blur-masked pixels

Validates: Requirements 5.2, 5.3, 5.4
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.temporal_coherence import TemporalCoherenceFilter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def dimensions(draw: st.DrawFn) -> tuple[int, int]:
    """Generate small (H, W) dimensions in [4, 16]."""
    h = draw(st.integers(min_value=4, max_value=16))
    w = draw(st.integers(min_value=4, max_value=16))
    return h, w


@st.composite
def alpha_mattes(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W] alpha matte with values in [0.0, 1.0]."""
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
def blur_masks(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W] binary mask with values in {0.0, 1.0}."""
    data = draw(
        st.lists(
            st.sampled_from([0.0, 1.0]),
            min_size=h * w,
            max_size=h * w,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w)


# ---------------------------------------------------------------------------
# Property 12: Temporal EMA formula correctness (Task 6.2)
# ---------------------------------------------------------------------------


class TestTemporalEMAFormulaCorrectness:
    """Property 12: For any sequence of two or more alpha mattes and a
    temporal_smoothing weight w in (0.0, 1.0], the output of
    TemporalCoherenceFilter.smooth() for frame N (N > 0) at each
    blur-masked pixel shall equal w * current_alpha + (1 - w) *
    previous_smoothed_alpha, and for frame 0 shall equal the input
    alpha exactly.

    **Validates: Requirements 5.2, 5.4**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_ema_formula_correctness(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 12: Temporal EMA
        formula correctness.

        **Validates: Requirements 5.2, 5.4**
        """
        h, w = data.draw(dimensions())
        num_frames = data.draw(st.integers(min_value=2, max_value=6))
        weight = data.draw(
            st.floats(
                min_value=0.01, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate a sequence of alpha mattes and a single blur mask.
        alphas = [data.draw(alpha_mattes(h, w)) for _ in range(num_frames)]
        mask = data.draw(blur_masks(h, w))

        filt = TemporalCoherenceFilter(temporal_smoothing=weight)

        prev_smoothed = None
        for i, alpha in enumerate(alphas):
            result = filt.smooth(alpha, mask)

            if i == 0:
                # Frame 0: output equals input exactly.
                np.testing.assert_array_equal(
                    result, alpha,
                    err_msg=f"Frame 0 output should equal input exactly",
                )
            else:
                # Frame N > 0: verify EMA at blur-masked pixels.
                assert prev_smoothed is not None
                blur_pixels = mask == 1.0
                expected = (
                    weight * alpha[blur_pixels]
                    + (1.0 - weight) * prev_smoothed[blur_pixels]
                )
                np.testing.assert_allclose(
                    result[blur_pixels],
                    expected,
                    atol=1e-6,
                    err_msg=(
                        f"EMA formula incorrect at frame {i} for "
                        f"blur-masked pixels (w={weight})"
                    ),
                )

            prev_smoothed = result.copy()


# ---------------------------------------------------------------------------
# Property 13: Temporal smoothing only affects blur-masked pixels (Task 6.3)
# ---------------------------------------------------------------------------


class TestBlurMaskOnlySmoothing:
    """Property 13: For any alpha matte and blur mask, pixels where
    blur_mask == 0.0 shall pass through TemporalCoherenceFilter.smooth()
    unchanged — i.e., the output at those pixels shall be identical to
    the input.

    **Validates: Requirements 5.3**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_non_masked_pixels_unchanged(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 13: Temporal
        smoothing only affects blur-masked pixels.

        **Validates: Requirements 5.3**
        """
        h, w = data.draw(dimensions())
        num_frames = data.draw(st.integers(min_value=1, max_value=5))
        weight = data.draw(
            st.floats(
                min_value=0.01, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            )
        )

        alphas = [data.draw(alpha_mattes(h, w)) for _ in range(num_frames)]
        mask = data.draw(blur_masks(h, w))

        filt = TemporalCoherenceFilter(temporal_smoothing=weight)

        non_blur = mask == 0.0
        for alpha in alphas:
            result = filt.smooth(alpha, mask)

            # Non-masked pixels must pass through unchanged.
            np.testing.assert_array_equal(
                result[non_blur],
                alpha[non_blur],
                err_msg="Non-blur-masked pixels were modified by smooth()",
            )
