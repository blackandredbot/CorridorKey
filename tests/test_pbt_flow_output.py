"""Property-based tests for OpticalFlowEngine output invariants.

# Feature: depth-keying-pipeline, Property 1: Flow field output invariants
# Feature: depth-keying-pipeline, Property 2: Occlusion mask consistency with forward-backward error
"""

from __future__ import annotations

import numpy as np
import cv2
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from CorridorKeyModule.depth.optical_flow import OpticalFlowEngine


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Small frame dimensions for speed — Farneback needs at least a few pixels
frame_hw = st.tuples(
    st.integers(min_value=16, max_value=32),  # H
    st.integers(min_value=16, max_value=32),  # W
)


@st.composite
def frame_pairs(draw):
    """Generate a pair of random RGB frames with matching resolution."""
    h, w = draw(frame_hw)
    frame_a = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w, 3),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    frame_b = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w, 3),
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
        )
    )
    # Exclude NaN/Inf — floats strategy with bounded range shouldn't produce
    # them, but be safe.
    assume(np.all(np.isfinite(frame_a)))
    assume(np.all(np.isfinite(frame_b)))
    return frame_a, frame_b


# ---------------------------------------------------------------------------
# Property 1: Flow field output invariants
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=frame_pairs())
def test_flow_field_output_invariants(data):
    """Feature: depth-keying-pipeline, Property 1: Flow field output invariants

    For any two RGB frames of identical resolution (H, W, 3) in float32 [0, 1],
    the OpticalFlowEngine.compute() output SHALL have:
    - forward_flow and backward_flow both of shape [H, W, 2] and dtype float32
    - occlusion_mask of shape [H, W] and dtype float32 with values in {0.0, 1.0}

    **Validates: Requirements 1.1, 1.2, 1.4, 1.5**
    """
    frame_a, frame_b = data
    h, w = frame_a.shape[:2]

    engine = OpticalFlowEngine(method="farneback", consistency_threshold=1.5)
    result = engine.compute(frame_a, frame_b)

    # Shape invariants
    assert result.forward_flow.shape == (h, w, 2), (
        f"forward_flow shape {result.forward_flow.shape} != expected ({h}, {w}, 2)"
    )
    assert result.backward_flow.shape == (h, w, 2), (
        f"backward_flow shape {result.backward_flow.shape} != expected ({h}, {w}, 2)"
    )
    assert result.occlusion_mask.shape == (h, w), (
        f"occlusion_mask shape {result.occlusion_mask.shape} != expected ({h}, {w})"
    )

    # Dtype invariants
    assert result.forward_flow.dtype == np.float32
    assert result.backward_flow.dtype == np.float32
    assert result.occlusion_mask.dtype == np.float32

    # Occlusion mask values must be exactly 0.0 or 1.0
    unique_vals = set(np.unique(result.occlusion_mask))
    assert unique_vals <= {0.0, 1.0}, (
        f"occlusion_mask contains values other than {{0.0, 1.0}}: {unique_vals}"
    )


# ---------------------------------------------------------------------------
# Property 2: Occlusion mask consistency with forward-backward error
# ---------------------------------------------------------------------------


def _reference_occlusion_mask(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Independent reference implementation of the occlusion check.

    For each pixel p, warp the backward flow by the forward flow and check
    ``||forward(p) + backward(warp(p))|| > threshold``.
    """
    h, w = forward_flow.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)

    # Warped coordinates
    wx = xs + forward_flow[..., 0]
    wy = ys + forward_flow[..., 1]

    # Sample backward flow at warped locations
    bw_at_warp_x = cv2.remap(
        backward_flow[..., 0], wx, wy, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    bw_at_warp_y = cv2.remap(
        backward_flow[..., 1], wx, wy, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    error = np.sqrt(
        (forward_flow[..., 0] + bw_at_warp_x) ** 2
        + (forward_flow[..., 1] + bw_at_warp_y) ** 2
    )
    return (error > threshold).astype(np.float32)


@st.composite
def frame_pairs_with_threshold(draw):
    """Generate frame pairs and a consistency threshold."""
    frames = draw(frame_pairs())
    threshold = draw(st.floats(min_value=0.1, max_value=5.0))
    assume(np.isfinite(threshold))
    return frames, threshold


@settings(max_examples=100)
@given(data=frame_pairs_with_threshold())
def test_occlusion_mask_consistency(data):
    """Feature: depth-keying-pipeline, Property 2: Occlusion mask consistency with forward-backward error

    For any two RGB frames and a given consistency threshold, a pixel is marked
    occluded in the output occlusion mask if and only if the forward-backward
    consistency error at that pixel exceeds the threshold.

    **Validates: Requirements 1.6**
    """
    (frame_a, frame_b), threshold = data

    engine = OpticalFlowEngine(method="farneback", consistency_threshold=threshold)
    result = engine.compute(frame_a, frame_b)

    # Independently compute the expected occlusion mask
    expected_mask = _reference_occlusion_mask(
        result.forward_flow, result.backward_flow, threshold
    )

    np.testing.assert_array_equal(
        result.occlusion_mask,
        expected_mask,
        err_msg="Occlusion mask does not match independent forward-backward consistency check",
    )
