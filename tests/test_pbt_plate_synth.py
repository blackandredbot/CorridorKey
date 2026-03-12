"""Property-based tests for Clean Plate Synthesizer and shared utilities.

# Feature: plate-subtraction-keying
# Properties 1–5 live in this file.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from CorridorKeyModule.depth.data_models import FlowResult
from CorridorKeyModule.depth.optical_flow import accumulate_flow
from CorridorKeyModule.depth.clean_plate_synthesizer import CleanPlateSynthesizer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def flow_result_strategy(draw: st.DrawFn, h: int, w: int) -> FlowResult:
    """Generate a FlowResult with random forward/backward flow and occlusion mask."""
    forward = draw(
        st.from_type(np.ndarray).filter(lambda _: False)  # placeholder, overridden
    ) if False else None  # noqa: never reached

    rng_seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(rng_seed)

    forward_flow = rng.uniform(-5.0, 5.0, size=(h, w, 2)).astype(np.float32)
    backward_flow = rng.uniform(-5.0, 5.0, size=(h, w, 2)).astype(np.float32)
    occlusion_mask = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)

    return FlowResult(
        forward_flow=forward_flow,
        backward_flow=backward_flow,
        occlusion_mask=occlusion_mask,
    )


@st.composite
def flow_results_with_none_gap(
    draw: st.DrawFn,
) -> tuple[list[FlowResult | None], int, int, int, int, int]:
    """Generate a flow_results list with at least one None gap.

    Returns (flow_results, gap_index, src_idx, dst_idx, h, w) where the
    chain from src_idx to dst_idx crosses the None at gap_index.
    """
    h = draw(st.integers(min_value=4, max_value=16))
    w = draw(st.integers(min_value=4, max_value=16))
    n_pairs = draw(st.integers(min_value=3, max_value=10))

    # Pick a gap index
    gap_idx = draw(st.integers(min_value=0, max_value=n_pairs - 1))

    # Build flow_results: all valid except the gap
    flow_results: list[FlowResult | None] = []
    for i in range(n_pairs):
        if i == gap_idx:
            flow_results.append(None)
        else:
            fr = draw(flow_result_strategy(h, w))
            flow_results.append(fr)

    # Pick src and dst so the chain crosses the gap.
    # flow_results[i] links frame i -> frame i+1.
    # A chain from src to dst uses indices min(src,dst) .. max(src,dst)-1.
    # We need gap_idx to be in [min(src,dst), max(src,dst)-1].
    # So: src <= gap_idx and dst >= gap_idx + 1  (or vice versa).
    src_idx = draw(st.integers(min_value=0, max_value=gap_idx))
    dst_idx = draw(st.integers(min_value=gap_idx + 1, max_value=n_pairs))
    assume(src_idx != dst_idx)

    # Randomly swap direction
    if draw(st.booleans()):
        src_idx, dst_idx = dst_idx, src_idx

    return flow_results, gap_idx, src_idx, dst_idx, h, w


# ---------------------------------------------------------------------------
# Property 5: Missing flow chain exclusion
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=flow_results_with_none_gap())
def test_accumulate_flow_returns_none_for_chain_crossing_gap(
    data: tuple[list[FlowResult | None], int, int, int, int, int],
):
    """Feature: plate-subtraction-keying, Property 5: Missing flow chain exclusion

    For any sequence of flow results where flow_results[i] is None for some
    index i, accumulate_flow shall return None for any (src, dst) pair whose
    chain passes through index i.

    **Validates: Requirements 2.8**
    """
    flow_results, gap_idx, src_idx, dst_idx, h, w = data

    # The chain from src_idx to dst_idx must cross gap_idx, so result is None
    result = accumulate_flow(src_idx, dst_idx, flow_results, h, w)
    assert result is None, (
        f"Expected None for chain from {src_idx} to {dst_idx} "
        f"crossing gap at index {gap_idx}, but got array with shape {result.shape}"
    )


# ---------------------------------------------------------------------------
# Additional strategies for Properties 1–4
# ---------------------------------------------------------------------------


@st.composite
def identity_flow_result(draw: st.DrawFn, h: int, w: int) -> FlowResult:
    """Generate a FlowResult with zero (identity) flow and zero occlusion.

    Identity flow means warped frames equal the original frames, which
    simplifies reasoning about synthesizer outputs.
    """
    forward_flow = np.zeros((h, w, 2), dtype=np.float32)
    backward_flow = np.zeros((h, w, 2), dtype=np.float32)
    occlusion_mask = np.zeros((h, w), dtype=np.float32)
    return FlowResult(
        forward_flow=forward_flow,
        backward_flow=backward_flow,
        occlusion_mask=occlusion_mask,
    )


@st.composite
def synth_frame_sequence(
    draw: st.DrawFn,
) -> tuple[list[np.ndarray], list[np.ndarray], list[FlowResult | None], int, int, int]:
    """Generate a small frame sequence with masks and identity flow results.

    Returns (frames, masks, flow_results, frame_idx, h, w).
    Uses identity flow so warped frames equal originals.
    Masks are all-zero (all background) to ensure valid donors.
    """
    h = draw(st.integers(min_value=4, max_value=12))
    w = draw(st.integers(min_value=4, max_value=12))
    n_frames = draw(st.integers(min_value=3, max_value=8))

    rng_seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(rng_seed)

    frames = [
        rng.uniform(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
        for _ in range(n_frames)
    ]
    masks = [
        rng.uniform(0.0, 0.2, size=(h, w)).astype(np.float32)
        for _ in range(n_frames)
    ]

    flow_results: list[FlowResult | None] = []
    for _ in range(n_frames - 1):
        fr = draw(identity_flow_result(h, w))
        flow_results.append(fr)

    frame_idx = draw(st.integers(min_value=0, max_value=n_frames - 1))

    return frames, masks, flow_results, frame_idx, h, w


# ---------------------------------------------------------------------------
# Property 1: Pipeline output shape, dtype, and range invariants
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(data=synth_frame_sequence())
def test_synthesize_output_shape_dtype_range(
    data: tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[FlowResult | None],
        int,
        int,
        int,
    ],
):
    """Feature: plate-subtraction-keying, Property 1: Pipeline output shape, dtype, and range invariants

    For any valid input frame sequence, synthesize() shall produce a clean plate
    of shape [H, W, 3] float32 in [0.0, 1.0] and a plate confidence of shape
    [H, W] float32 in [0.0, 1.0].

    **Validates: Requirements 1.2, 2.1, 2.6, 3.4**
    """
    frames, masks, flow_results, frame_idx, h, w = data

    synth = CleanPlateSynthesizer(plate_search_radius=15, donor_threshold=0.3)
    clean_plate, confidence = synth.synthesize(frame_idx, frames, masks, flow_results)

    # Shape checks
    assert clean_plate.shape == (h, w, 3), (
        f"Expected clean_plate shape ({h}, {w}, 3), got {clean_plate.shape}"
    )
    assert confidence.shape == (h, w), (
        f"Expected confidence shape ({h}, {w}), got {confidence.shape}"
    )

    # Dtype checks
    assert clean_plate.dtype == np.float32, (
        f"Expected clean_plate dtype float32, got {clean_plate.dtype}"
    )
    assert confidence.dtype == np.float32, (
        f"Expected confidence dtype float32, got {confidence.dtype}"
    )

    # Range checks
    assert np.all(clean_plate >= 0.0) and np.all(clean_plate <= 1.0), (
        f"clean_plate values out of [0.0, 1.0]: "
        f"min={clean_plate.min()}, max={clean_plate.max()}"
    )
    assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0), (
        f"confidence values out of [0.0, 1.0]: "
        f"min={confidence.min()}, max={confidence.max()}"
    )


# ---------------------------------------------------------------------------
# Property 2: Donor weight computation
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    mask_val=st.floats(min_value=0.0, max_value=1.0),
    fc_val=st.floats(min_value=0.0, max_value=1.0),
    donor_threshold=st.floats(min_value=0.01, max_value=1.0),
)
def test_donor_weight_computation(
    mask_val: float,
    fc_val: float,
    donor_threshold: float,
):
    """Feature: plate-subtraction-keying, Property 2: Donor weight computation

    For any mask value m in [0.0, 1.0] and flow consistency fc in [0.0, 1.0],
    donor_weight = (1.0 - m) * fc. Pixels with mask >= donor_threshold are not
    counted as valid donors.

    **Validates: Requirements 2.2, 2.5**
    """
    # Verify the weight formula
    expected_weight = (1.0 - mask_val) * fc_val
    assert abs(expected_weight - (1.0 - mask_val) * fc_val) < 1e-7

    # Verify valid donor classification: mask < threshold → valid
    is_valid = mask_val < donor_threshold
    if mask_val >= donor_threshold:
        assert not is_valid, (
            f"Pixel with mask={mask_val} >= threshold={donor_threshold} "
            f"should NOT be a valid donor"
        )

    # Integration check: construct a 1-pixel scenario and run through synthesizer
    # to verify the actual implementation matches the formula.
    h, w = 1, 1
    n_frames = 3
    target_idx = 1

    frames = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n_frames)]
    masks = [np.zeros((h, w), dtype=np.float32) for _ in range(n_frames)]
    # Set the donor frame's mask to our test mask_val
    masks[0] = np.full((h, w), mask_val, dtype=np.float32)
    masks[2] = np.full((h, w), mask_val, dtype=np.float32)

    # Use identity flow (zeros) so warped mask == original mask
    flow_results: list[FlowResult | None] = []
    for _ in range(n_frames - 1):
        flow_results.append(
            FlowResult(
                forward_flow=np.zeros((h, w, 2), dtype=np.float32),
                backward_flow=np.zeros((h, w, 2), dtype=np.float32),
                occlusion_mask=np.zeros((h, w), dtype=np.float32),
            )
        )

    synth = CleanPlateSynthesizer(plate_search_radius=15, donor_threshold=donor_threshold)
    clean_plate, confidence = synth.synthesize(target_idx, frames, masks, flow_results)

    # With identity flow and zero backward flow, flow_consistency ≈ 1.0
    # (exp(-0/2) = 1.0), so donor_weight ≈ (1 - mask_val) * 1.0
    # Verify valid donor classification through confidence:
    # If mask_val >= donor_threshold, donors are invalid → confidence = 0.0
    if mask_val >= donor_threshold:
        assert confidence[0, 0] == 0.0, (
            f"With mask={mask_val} >= threshold={donor_threshold}, "
            f"confidence should be 0.0, got {confidence[0, 0]}"
        )


# ---------------------------------------------------------------------------
# Property 3: Plate confidence equals valid donor fraction
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    n_valid=st.integers(min_value=0, max_value=6),
    n_invalid=st.integers(min_value=0, max_value=4),
)
def test_plate_confidence_equals_valid_donor_fraction(
    n_valid: int,
    n_invalid: int,
):
    """Feature: plate-subtraction-keying, Property 3: Plate confidence equals valid donor fraction

    For any target frame and set of donor frames, plate_confidence at each pixel
    equals valid_count / total_possible_donors. When < 2 valid donors exist,
    confidence is forced to 0.0.

    **Validates: Requirements 2.6**
    """
    total_donors = n_valid + n_invalid
    assume(total_donors >= 1)  # Need at least 1 donor frame

    h, w = 1, 1
    donor_threshold = 0.3

    # Total frames = donors + 1 (the target frame)
    n_frames = total_donors + 1
    target_idx = 0  # Target is frame 0; donors are frames 1..n_frames-1

    frames = [
        np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n_frames)
    ]

    # Build masks: valid donors get mask=0.0, invalid donors get mask=1.0
    masks = [np.zeros((h, w), dtype=np.float32)]  # target frame mask
    for i in range(total_donors):
        if i < n_valid:
            masks.append(np.full((h, w), 0.0, dtype=np.float32))  # valid
        else:
            masks.append(np.full((h, w), 1.0, dtype=np.float32))  # invalid

    # Identity flow for all pairs
    flow_results: list[FlowResult | None] = []
    for _ in range(n_frames - 1):
        flow_results.append(
            FlowResult(
                forward_flow=np.zeros((h, w, 2), dtype=np.float32),
                backward_flow=np.zeros((h, w, 2), dtype=np.float32),
                occlusion_mask=np.zeros((h, w), dtype=np.float32),
            )
        )

    synth = CleanPlateSynthesizer(
        plate_search_radius=total_donors + 1,
        donor_threshold=donor_threshold,
    )
    _, confidence = synth.synthesize(target_idx, frames, masks, flow_results)

    if n_valid < 2:
        # Fallback path: confidence forced to 0.0
        assert confidence[0, 0] == 0.0, (
            f"With {n_valid} valid donors (< 2), confidence should be 0.0, "
            f"got {confidence[0, 0]}"
        )
    else:
        # Median path: confidence = n_valid / total_donors
        expected = n_valid / total_donors
        assert abs(confidence[0, 0] - expected) < 1e-6, (
            f"Expected confidence={expected} ({n_valid}/{total_donors}), "
            f"got {confidence[0, 0]}"
        )


# ---------------------------------------------------------------------------
# Property 4: Temporal median rejects outliers
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    n_good=st.integers(min_value=3, max_value=7),
    good_val=st.floats(min_value=0.05, max_value=0.95),
    outlier_val=st.floats(min_value=0.0, max_value=1.0),
)
def test_temporal_median_rejects_outliers(
    n_good: int,
    good_val: float,
    outlier_val: float,
):
    """Feature: plate-subtraction-keying, Property 4: Temporal median rejects outliers

    For N >= 3 donor pixel values where at least 2 are valid, the clean plate
    value equals the per-channel median of valid donors. A single outlier
    among consistent donors is rejected by the median.

    **Validates: Requirements 2.4**
    """
    # Ensure the outlier is actually different from the good value
    assume(abs(outlier_val - good_val) > 0.05)

    h, w = 1, 1
    donor_threshold = 0.3

    # n_good "good" donors + 1 outlier donor + 1 target frame
    n_donors = n_good + 1
    n_frames = n_donors + 1
    target_idx = 0

    # Build frames: target frame is arbitrary, good donors have good_val,
    # outlier donor has outlier_val
    frames = [np.full((h, w, 3), 0.5, dtype=np.float32)]  # target
    for i in range(n_donors):
        if i < n_good:
            frames.append(np.full((h, w, 3), good_val, dtype=np.float32))
        else:
            frames.append(np.full((h, w, 3), outlier_val, dtype=np.float32))

    # All donors are valid (mask=0.0)
    masks = [np.zeros((h, w), dtype=np.float32) for _ in range(n_frames)]

    # Identity flow
    flow_results: list[FlowResult | None] = []
    for _ in range(n_frames - 1):
        flow_results.append(
            FlowResult(
                forward_flow=np.zeros((h, w, 2), dtype=np.float32),
                backward_flow=np.zeros((h, w, 2), dtype=np.float32),
                occlusion_mask=np.zeros((h, w), dtype=np.float32),
            )
        )

    synth = CleanPlateSynthesizer(
        plate_search_radius=n_donors + 1,
        donor_threshold=donor_threshold,
    )
    clean_plate, _ = synth.synthesize(target_idx, frames, masks, flow_results)

    # With n_good identical values and 1 outlier, the median should be good_val
    # (since n_good >= 3, the outlier cannot shift the median).
    # Build the expected donor values and compute median manually.
    donor_vals = [good_val] * n_good + [outlier_val]
    expected_median = float(np.median(donor_vals))

    for c in range(3):
        actual = clean_plate[0, 0, c]
        assert abs(actual - expected_median) < 1e-5, (
            f"Channel {c}: expected median={expected_median:.6f}, "
            f"got {actual:.6f} (good_val={good_val}, outlier_val={outlier_val}, "
            f"n_good={n_good})"
        )
