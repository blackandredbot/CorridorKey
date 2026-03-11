"""Property-based tests for MaskRefiner.

# Feature: depth-keying-pipeline, Property 13: Despeckle and hole fill size enforcement
# Feature: depth-keying-pipeline, Property 14: Refinement bypass at zero strength
# Feature: depth-keying-pipeline, Property 15: Resolution-scaled tightening is no-op at full resolution
"""

from __future__ import annotations

import numpy as np
import cv2
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from CorridorKeyModule.depth.mask_refiner import MaskRefiner


# ---------------------------------------------------------------------------
# Property 13: Despeckle and hole fill size enforcement
# ---------------------------------------------------------------------------

@st.composite
def binary_matte_with_specks_and_holes(draw):
    """Generate a binary matte (0.0 or 1.0) with small specks and holes
    of known sizes, plus configurable despeckle/hole fill thresholds."""
    h = draw(st.integers(min_value=64, max_value=128))
    w = draw(st.integers(min_value=64, max_value=128))
    despeckle_size = draw(st.integers(min_value=10, max_value=60))
    hole_fill_size = draw(st.integers(min_value=10, max_value=60))

    # Start with a mostly-background matte
    matte = np.zeros((h, w), dtype=np.float32)

    # Add a large foreground block in the center (guaranteed > despeckle_size)
    block_size = max(despeckle_size + 20, 30)
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - block_size // 2)
    y1 = min(h, cy + block_size // 2)
    x0 = max(0, cx - block_size // 2)
    x1 = min(w, cx + block_size // 2)
    matte[y0:y1, x0:x1] = 1.0

    # Add small foreground specks (should be removed by despeckle)
    num_specks = draw(st.integers(min_value=1, max_value=5))
    for _ in range(num_specks):
        speck_size = draw(st.integers(min_value=1, max_value=max(1, despeckle_size - 1)))
        sy = draw(st.integers(min_value=0, max_value=max(0, h - 4)))
        sx = draw(st.integers(min_value=0, max_value=max(0, w - 4)))
        # Place speck away from the main block to keep it isolated
        side = int(np.ceil(np.sqrt(speck_size)))
        ey = min(h, sy + side)
        ex = min(w, sx + side)
        # Only place if it doesn't overlap the main block
        if ey <= y0 or sy >= y1 or ex <= x0 or sx >= x1:
            matte[sy:ey, sx:ex] = 1.0

    # Add small holes inside the foreground block (should be filled)
    num_holes = draw(st.integers(min_value=1, max_value=3))
    for _ in range(num_holes):
        hole_size = draw(st.integers(min_value=1, max_value=max(1, hole_fill_size - 1)))
        side = int(np.ceil(np.sqrt(hole_size)))
        # Place hole inside the foreground block
        if y1 - y0 > side + 2 and x1 - x0 > side + 2:
            hy = draw(st.integers(min_value=y0 + 1, max_value=max(y0 + 1, y1 - side - 1)))
            hx = draw(st.integers(min_value=x0 + 1, max_value=max(x0 + 1, x1 - side - 1)))
            matte[hy:hy + side, hx:hx + side] = 0.0

    return matte, despeckle_size, hole_fill_size, h, w


@settings(max_examples=100)
@given(data=binary_matte_with_specks_and_holes())
def test_despeckle_and_hole_fill_size_enforcement(data):
    """Feature: depth-keying-pipeline, Property 13: Despeckle and hole fill size enforcement

    For any binary-ish alpha matte, after MaskRefiner.refine() with
    refinement_strength=1.0, there SHALL be no connected foreground regions
    (alpha > 0.5) smaller than despeckle_size pixels, and no connected
    background holes (alpha < 0.5) within the foreground region smaller
    than hole_fill_size pixels.

    **Validates: Requirements 4.3, 4.4**
    """
    matte, despeckle_size, hole_fill_size, h, w = data

    # Use a simple RGB guide (zeros) so guided filter doesn't alter the
    # binary structure significantly — we're testing despeckle/hole fill.
    rgb_guide = np.zeros((h, w, 3), dtype=np.float32)

    refiner = MaskRefiner(
        refinement_strength=1.0,
        despeckle_size=despeckle_size,
        hole_fill_size=hole_fill_size,
        processing_resolution=None,
        original_resolution=None,
    )
    refined = refiner.refine(matte, rgb_guide)

    # Check output basics
    assert refined.shape == (h, w)
    assert refined.dtype == np.float32
    assert refined.min() >= -1e-6
    assert refined.max() <= 1.0 + 1e-6

    # Binarize the refined output at 0.5 threshold
    fg_mask = (refined > 0.5).astype(np.uint8) * 255

    # Verify no small foreground regions remain
    num_fg, labels_fg, stats_fg, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )
    for i in range(1, num_fg):
        area = stats_fg[i, cv2.CC_STAT_AREA]
        assert area >= despeckle_size, (
            f"Foreground component {i} has area {area} < despeckle_size {despeckle_size}"
        )

    # Verify no small background holes remain
    bg_mask = (refined <= 0.5).astype(np.uint8) * 255
    num_bg, labels_bg, stats_bg, _ = cv2.connectedComponentsWithStats(
        bg_mask, connectivity=8
    )
    # The largest background component is the outer background — skip it.
    # Only check interior holes (components that are NOT the largest bg component).
    if num_bg > 1:
        # Find the largest background component (likely the outer background)
        bg_areas = [stats_bg[i, cv2.CC_STAT_AREA] for i in range(1, num_bg)]
        if bg_areas:
            largest_bg_idx = np.argmax(bg_areas) + 1  # +1 because we skip label 0
            for i in range(1, num_bg):
                if i == largest_bg_idx:
                    continue  # skip the main outer background
                area = stats_bg[i, cv2.CC_STAT_AREA]
                assert area >= hole_fill_size, (
                    f"Background hole {i} has area {area} < hole_fill_size {hole_fill_size}"
                )


# ---------------------------------------------------------------------------
# Property 14: Refinement bypass at zero strength
# ---------------------------------------------------------------------------

@st.composite
def random_alpha_and_guide(draw):
    """Generate random alpha mattes and RGB guides."""
    h = draw(st.integers(min_value=4, max_value=64))
    w = draw(st.integers(min_value=4, max_value=64))

    alpha = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    assume(np.all(np.isfinite(alpha)))

    rgb = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w, 3),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    assume(np.all(np.isfinite(rgb)))

    return alpha, rgb, h, w


@settings(max_examples=100)
@given(data=random_alpha_and_guide())
def test_refinement_bypass_at_zero_strength(data):
    """Feature: depth-keying-pipeline, Property 14: Refinement bypass at zero strength

    For any alpha matte and RGB guide frame, MaskRefiner.refine() with
    refinement_strength=0.0 SHALL return an output numerically identical
    to the input alpha matte.

    **Validates: Requirements 4.5**
    """
    alpha, rgb, h, w = data

    refiner = MaskRefiner(
        refinement_strength=0.0,
        despeckle_size=50,
        hole_fill_size=50,
    )
    result = refiner.refine(alpha, rgb)

    assert result.shape == (h, w)
    assert result.dtype == np.float32
    assert np.array_equal(result, alpha.astype(np.float32)), (
        "Output must be numerically identical to input when refinement_strength=0.0"
    )


# ---------------------------------------------------------------------------
# Property 15: Resolution-scaled tightening is no-op at full resolution
# ---------------------------------------------------------------------------

@st.composite
def alpha_with_matching_resolution(draw):
    """Generate random mattes with processing_resolution == original_resolution."""
    h = draw(st.integers(min_value=4, max_value=64))
    w = draw(st.integers(min_value=4, max_value=64))

    alpha = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    assume(np.all(np.isfinite(alpha)))

    rgb = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w, 3),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    assume(np.all(np.isfinite(rgb)))

    resolution = draw(st.integers(min_value=100, max_value=4096))

    return alpha, rgb, h, w, resolution


@settings(max_examples=100)
@given(data=alpha_with_matching_resolution())
def test_resolution_tightening_noop_at_full_resolution(data):
    """Feature: depth-keying-pipeline, Property 15: Resolution-scaled tightening is no-op at full resolution

    For any alpha matte, when processing_resolution == original_resolution,
    the matte tightening correction magnitude SHALL be zero, and the output
    SHALL be identical to the output without tightening applied.

    **Validates: Requirements 4.6**
    """
    alpha, rgb, h, w, resolution = data

    # Refiner with processing_resolution == original_resolution
    refiner_with_res = MaskRefiner(
        refinement_strength=1.0,
        despeckle_size=50,
        hole_fill_size=50,
        processing_resolution=resolution,
        original_resolution=resolution,
    )

    # Refiner with no tightening (processing_resolution=None)
    refiner_no_tighten = MaskRefiner(
        refinement_strength=1.0,
        despeckle_size=50,
        hole_fill_size=50,
        processing_resolution=None,
        original_resolution=None,
    )

    result_with_res = refiner_with_res.refine(alpha.copy(), rgb.copy())
    result_no_tighten = refiner_no_tighten.refine(alpha.copy(), rgb.copy())

    assert result_with_res.shape == (h, w)
    assert result_with_res.dtype == np.float32

    np.testing.assert_array_equal(
        result_with_res,
        result_no_tighten,
        err_msg=(
            "When processing_resolution == original_resolution, output must "
            "match the output without tightening applied."
        ),
    )
