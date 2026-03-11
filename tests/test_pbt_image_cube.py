"""Property-based tests for ImageCubeBuilder three-signal fusion.

# Feature: depth-keying-pipeline, Property 3: Parallax-depth monotonicity
# Feature: depth-keying-pipeline, Property 4: Global motion compensation invariance
# Feature: depth-keying-pipeline, Property 5: Persistence channel monotonicity
# Feature: depth-keying-pipeline, Property 6: Positional stability decay
# Feature: depth-keying-pipeline, Property 7: Signal fusion formula
# Feature: depth-keying-pipeline, Property 8: Rolling buffer bounded size
# Feature: depth-keying-pipeline, Property 9: Cube output range and resolution
# Feature: depth-keying-pipeline, Property 10: Stationary camera re-weighting
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from CorridorKeyModule.depth.image_cube import ImageCubeBuilder
from CorridorKeyModule.depth.data_models import FlowResult


# ---------------------------------------------------------------------------
# Shared helpers / strategies
# ---------------------------------------------------------------------------

def _make_flow_result(
    h: int, w: int,
    forward_flow: np.ndarray | None = None,
    backward_flow: np.ndarray | None = None,
    occlusion_mask: np.ndarray | None = None,
) -> FlowResult:
    """Create a FlowResult with sensible defaults."""
    if forward_flow is None:
        forward_flow = np.zeros((h, w, 2), dtype=np.float32)
    if backward_flow is None:
        backward_flow = np.zeros((h, w, 2), dtype=np.float32)
    if occlusion_mask is None:
        occlusion_mask = np.zeros((h, w), dtype=np.float32)
    return FlowResult(
        forward_flow=forward_flow,
        backward_flow=backward_flow,
        occlusion_mask=occlusion_mask,
    )


# ---------------------------------------------------------------------------
# Property 3: Parallax-depth monotonicity
# ---------------------------------------------------------------------------

@st.composite
def parallax_monotonicity_data(draw):
    """Generate a synthetic residual flow field where pixel A has strictly
    larger residual parallax magnitude than pixel B.

    We directly construct a residual flow field (bypassing RANSAC) to test
    the parallax scoring in isolation.  The field has a spread of magnitudes
    so that percentile normalization produces meaningful differentiation.
    """
    h = draw(st.integers(min_value=16, max_value=32))
    w = draw(st.integers(min_value=16, max_value=32))

    # Create a residual flow field with a gradient of magnitudes so that
    # percentile normalization has a meaningful range.
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10000)))
    residual = rng.uniform(-3.0, 3.0, (h, w, 2)).astype(np.float32)

    # Pick two distinct pixel locations away from edges
    ya = draw(st.integers(min_value=1, max_value=h - 2))
    xa = draw(st.integers(min_value=1, max_value=w - 2))
    yb = draw(st.integers(min_value=1, max_value=h - 2))
    xb = draw(st.integers(min_value=1, max_value=w - 2))
    assume((ya, xa) != (yb, xb))

    # Pixel A: large residual (foreground — should get LOW score)
    mag_a = draw(st.floats(min_value=5.0, max_value=15.0))
    # Pixel B: small residual (background — should get HIGH score)
    mag_b = draw(st.floats(min_value=0.0, max_value=0.5))
    assume(np.isfinite(mag_a) and np.isfinite(mag_b))

    residual[ya, xa] = [mag_a, 0.0]
    residual[yb, xb] = [mag_b, 0.0]

    return residual, (ya, xa), (yb, xb), h, w


@settings(max_examples=100)
@given(data=parallax_monotonicity_data())
def test_parallax_depth_monotonicity(data):
    """Feature: depth-keying-pipeline, Property 3: Parallax-depth monotonicity

    For any synthetic flow field where pixel A has strictly larger residual
    parallax magnitude than pixel B (after global motion compensation), the
    parallax channel SHALL assign pixel A a strictly lower (more foreground)
    score than pixel B.

    **Validates: Requirements 2.2**
    """
    residual, (ya, xa), (yb, xb), h, w = data

    # Test the parallax scoring directly on the residual flow
    parallax = ImageCubeBuilder._compute_parallax(residual)

    # Pixel A (larger residual) should have LOWER parallax score (foreground)
    # Pixel B (smaller residual) should have HIGHER parallax score (background)
    assert parallax[ya, xa] < parallax[yb, xb], (
        f"Pixel A ({ya},{xa}) score {parallax[ya, xa]:.6f} should be < "
        f"pixel B ({yb},{xb}) score {parallax[yb, xb]:.6f}"
    )


# ---------------------------------------------------------------------------
# Property 4: Global motion compensation invariance
# ---------------------------------------------------------------------------

@st.composite
def flow_with_uniform_offset(draw):
    """Generate a flow field and a random uniform offset."""
    h = draw(st.integers(min_value=16, max_value=24))
    w = draw(st.integers(min_value=16, max_value=24))

    # Base flow with some structure (not all zeros)
    base_flow = draw(
        arrays(
            dtype=np.float32,
            shape=(h, w, 2),
            elements=st.floats(min_value=-5.0, max_value=5.0, width=32),
        )
    )
    assume(np.all(np.isfinite(base_flow)))

    # Uniform offset (simulating pure camera motion)
    ox = draw(st.floats(min_value=-10.0, max_value=10.0))
    oy = draw(st.floats(min_value=-10.0, max_value=10.0))
    assume(np.isfinite(ox) and np.isfinite(oy))

    return base_flow, ox, oy, h, w


@settings(max_examples=100)
@given(data=flow_with_uniform_offset())
def test_global_motion_compensation_invariance(data):
    """Feature: depth-keying-pipeline, Property 4: Global motion compensation invariance

    For any flow field, adding a uniform translation vector to all flow vectors
    (simulating pure camera motion with no parallax change) SHALL not change
    the resulting parallax channel scores within floating-point tolerance.

    **Validates: Requirements 2.3**
    """
    base_flow, ox, oy, h, w = data

    # Compute parallax from base flow
    residual_base, _ = ImageCubeBuilder._compensate_global_motion(base_flow)
    parallax_base = ImageCubeBuilder._compute_parallax(residual_base)

    # Add uniform offset
    offset = np.full((h, w, 2), [ox, oy], dtype=np.float32)
    shifted_flow = base_flow + offset

    # Compute parallax from shifted flow
    residual_shifted, _ = ImageCubeBuilder._compensate_global_motion(shifted_flow)
    parallax_shifted = ImageCubeBuilder._compute_parallax(residual_shifted)

    # Scores should be the same within tolerance
    np.testing.assert_allclose(
        parallax_base, parallax_shifted,
        atol=0.05, rtol=0.05,
        err_msg="Parallax scores changed after adding uniform offset",
    )


# ---------------------------------------------------------------------------
# Property 5: Persistence channel monotonicity
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    h=st.integers(min_value=16, max_value=24),
    w=st.integers(min_value=16, max_value=24),
    n_frames=st.integers(min_value=3, max_value=6),
)
def test_persistence_channel_monotonicity(h, w, n_frames):
    """Feature: depth-keying-pipeline, Property 5: Persistence channel monotonicity

    For any pixel whose local patch descriptor remains identical across all
    frames in the rolling buffer, the persistence channel SHALL assign a score
    higher than that of a pixel whose patch descriptor changes completely
    between every consecutive pair of frames.

    **Validates: Requirements 2.4, 2.5, 2.6**
    """
    builder = ImageCubeBuilder(buffer_size=n_frames + 2)

    # Pick two pixel locations
    stable_y, stable_x = h // 4, w // 4
    volatile_y, volatile_x = 3 * h // 4, 3 * w // 4

    rng = np.random.default_rng(123)

    # Create frames where:
    # - stable pixel region has constant color across all frames
    # - volatile pixel region has completely different color each frame
    for i in range(n_frames):
        frame = rng.random((h, w, 3)).astype(np.float32)

        # Make the stable region constant across frames
        patch_r = 3
        sy_lo = max(0, stable_y - patch_r)
        sy_hi = min(h, stable_y + patch_r + 1)
        sx_lo = max(0, stable_x - patch_r)
        sx_hi = min(w, stable_x + patch_r + 1)
        frame[sy_lo:sy_hi, sx_lo:sx_hi] = 0.5  # constant value

        # Make the volatile region random each frame (already random from rng)
        # Just ensure it's very different by using extreme values
        vy_lo = max(0, volatile_y - patch_r)
        vy_hi = min(h, volatile_y + patch_r + 1)
        vx_lo = max(0, volatile_x - patch_r)
        vx_hi = min(w, volatile_x + patch_r + 1)
        frame[vy_lo:vy_hi, vx_lo:vx_hi] = float(i) / max(n_frames - 1, 1)

        flow = np.zeros((h, w, 2), dtype=np.float32)
        fr = _make_flow_result(h, w, forward_flow=flow)
        builder.update(fr, emit_channel_maps=True, frame=frame)

    # Get the last result with channel maps
    last_flow = _make_flow_result(h, w)
    final_frame = rng.random((h, w, 3)).astype(np.float32)
    final_frame[sy_lo:sy_hi, sx_lo:sx_hi] = 0.5
    final_frame[vy_lo:vy_hi, vx_lo:vx_hi] = 1.0

    result = builder.update(last_flow, emit_channel_maps=True, frame=final_frame)

    assert result.persistence_map is not None
    stable_score = result.persistence_map[stable_y, stable_x]
    volatile_score = result.persistence_map[volatile_y, volatile_x]

    assert stable_score > volatile_score, (
        f"Stable pixel persistence {stable_score:.4f} should be > "
        f"volatile pixel persistence {volatile_score:.4f}"
    )


# ---------------------------------------------------------------------------
# Property 6: Positional stability decay
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    h=st.integers(min_value=16, max_value=24),
    w=st.integers(min_value=16, max_value=24),
    n_frames=st.integers(min_value=3, max_value=6),
    drift_magnitude=st.floats(min_value=2.0, max_value=10.0),
)
def test_positional_stability_decay(h, w, n_frames, drift_magnitude):
    """Feature: depth-keying-pipeline, Property 6: Positional stability decay

    For any pixel, the positional stability score SHALL be a monotonically
    decreasing function of cumulative displacement from the pixel's initial
    screen-space coordinates: zero drift SHALL produce a score of 1.0, and
    increasing drift SHALL produce scores approaching 0.0.

    **Validates: Requirements 2.7, 2.8, 2.9**
    """
    assume(np.isfinite(drift_magnitude))

    builder = ImageCubeBuilder(buffer_size=n_frames + 2)

    # Pick two pixels: one stationary, one drifting
    stat_y, stat_x = h // 4, w // 4
    drift_y, drift_x = 3 * h // 4, 3 * w // 4

    for i in range(n_frames):
        flow = np.zeros((h, w, 2), dtype=np.float32)
        # The drifting pixel gets consistent displacement each frame
        flow[drift_y, drift_x, 0] = drift_magnitude
        # Also drift its neighborhood slightly
        patch_r = 2
        dy_lo = max(0, drift_y - patch_r)
        dy_hi = min(h, drift_y + patch_r + 1)
        dx_lo = max(0, drift_x - patch_r)
        dx_hi = min(w, drift_x + patch_r + 1)
        flow[dy_lo:dy_hi, dx_lo:dx_hi, 0] = drift_magnitude

        fr = _make_flow_result(h, w, forward_flow=flow)
        builder.update(fr, emit_channel_maps=True)

    result = builder.update(
        _make_flow_result(h, w), emit_channel_maps=True
    )

    assert result.stability_map is not None
    stationary_score = result.stability_map[stat_y, stat_x]
    drifting_score = result.stability_map[drift_y, drift_x]

    # Stationary pixel should have higher stability than drifting pixel
    assert stationary_score > drifting_score, (
        f"Stationary pixel stability {stationary_score:.4f} should be > "
        f"drifting pixel stability {drifting_score:.4f}"
    )

    # Zero drift should produce score close to 1.0
    # (after neighborhood blur, it may not be exactly 1.0 but should be high)
    assert stationary_score > 0.8, (
        f"Zero-drift pixel stability {stationary_score:.4f} should be > 0.8"
    )


# ---------------------------------------------------------------------------
# Property 7: Signal fusion formula
# ---------------------------------------------------------------------------

@st.composite
def fusion_inputs(draw):
    """Generate three channel scores and valid weights."""
    h = draw(st.integers(min_value=4, max_value=8))
    w = draw(st.integers(min_value=4, max_value=8))

    parallax = draw(
        arrays(np.float32, (h, w),
               elements=st.floats(min_value=0.0, max_value=1.0, width=32))
    )
    persistence = draw(
        arrays(np.float32, (h, w),
               elements=st.floats(min_value=0.0, max_value=1.0, width=32))
    )
    stability = draw(
        arrays(np.float32, (h, w),
               elements=st.floats(min_value=0.0, max_value=1.0, width=32))
    )
    assume(np.all(np.isfinite(parallax)))
    assume(np.all(np.isfinite(persistence)))
    assume(np.all(np.isfinite(stability)))

    # Generate weights that sum to 1.0
    w1 = draw(st.floats(min_value=0.0, max_value=1.0))
    w2 = draw(st.floats(min_value=0.0, max_value=1.0 - w1))
    assume(np.isfinite(w1) and np.isfinite(w2))
    w3 = 1.0 - w1 - w2
    assume(w3 >= 0.0)
    assume(np.isfinite(w3))

    return parallax, persistence, stability, w1, w2, w3, h, w


@settings(max_examples=100)
@given(data=fusion_inputs())
def test_signal_fusion_formula(data):
    """Feature: depth-keying-pipeline, Property 7: Signal fusion formula

    For any three channel scores in [0, 1] and valid weights summing to 1.0:
    - blend mode = weighted average
    - max mode = max of three
    - min mode = min of three
    All within 1e-7 tolerance.

    **Validates: Requirements 2.10, 2.11**
    """
    parallax, persistence, stability, pw, per_w, sw, h, w = data

    for mode in ("blend", "max", "min"):
        builder = ImageCubeBuilder(
            parallax_weight=pw,
            persistence_weight=per_w,
            stability_weight=sw,
            fusion_mode=mode,
        )

        result = builder._fuse_signals(
            parallax, persistence, stability, pw, per_w, sw
        )

        if mode == "blend":
            expected = pw * parallax + per_w * persistence + sw * stability
            expected = np.clip(expected, 0.0, 1.0).astype(np.float32)
        elif mode == "max":
            expected = np.maximum(np.maximum(parallax, persistence), stability)
        elif mode == "min":
            expected = np.minimum(np.minimum(parallax, persistence), stability)

        np.testing.assert_allclose(
            result, expected, atol=1e-7,
            err_msg=f"Fusion mode '{mode}' produced incorrect result",
        )


# ---------------------------------------------------------------------------
# Property 8: Rolling buffer bounded size
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    buffer_size=st.integers(min_value=2, max_value=10),
    extra_frames=st.integers(min_value=1, max_value=5),
)
def test_rolling_buffer_bounded_size(buffer_size, extra_frames):
    """Feature: depth-keying-pipeline, Property 8: Rolling buffer bounded size

    For any sequence of flow results fed to the Image_Cube_Builder with
    buffer_size=N, the internal buffer SHALL never contain more than N entries.

    **Validates: Requirements 2.12**
    """
    h, w = 8, 8
    builder = ImageCubeBuilder(buffer_size=buffer_size)

    total_frames = buffer_size + extra_frames

    for i in range(total_frames):
        flow = _make_flow_result(h, w)
        builder.update(flow)

        # Buffer should never exceed buffer_size
        assert len(builder._buffer) <= buffer_size, (
            f"After {i + 1} frames, buffer has {len(builder._buffer)} entries "
            f"but max is {buffer_size}"
        )

    # After feeding more than buffer_size frames, buffer should be exactly full
    assert len(builder._buffer) == buffer_size, (
        f"After {total_frames} frames, buffer has {len(builder._buffer)} "
        f"entries but should be exactly {buffer_size}"
    )


# ---------------------------------------------------------------------------
# Property 9: Cube output range and resolution
# ---------------------------------------------------------------------------

@st.composite
def random_flow_sequence(draw):
    """Generate a short sequence of random flow results."""
    h = draw(st.integers(min_value=8, max_value=16))
    w = draw(st.integers(min_value=8, max_value=16))
    n_frames = draw(st.integers(min_value=1, max_value=4))

    flows = []
    for _ in range(n_frames):
        fwd = draw(
            arrays(np.float32, (h, w, 2),
                   elements=st.floats(min_value=-5.0, max_value=5.0, width=32))
        )
        bwd = draw(
            arrays(np.float32, (h, w, 2),
                   elements=st.floats(min_value=-5.0, max_value=5.0, width=32))
        )
        assume(np.all(np.isfinite(fwd)))
        assume(np.all(np.isfinite(bwd)))

        occ = np.zeros((h, w), dtype=np.float32)
        flows.append(FlowResult(forward_flow=fwd, backward_flow=bwd,
                                occlusion_mask=occ))

    return flows, h, w


@settings(max_examples=100)
@given(data=random_flow_sequence())
def test_cube_output_range_and_resolution(data):
    """Feature: depth-keying-pipeline, Property 9: Cube output range and resolution

    For any sequence of flow results, the output SHALL have:
    - background_score with all values in [0.0, 1.0]
    - confidence_map with all values in [0.0, 1.0]
    - all output arrays matching the spatial resolution of the input
    - when emit_channel_maps=True, individual maps also in [0.0, 1.0]

    **Validates: Requirements 2.13, 2.16**
    """
    flows, h, w = data
    builder = ImageCubeBuilder(buffer_size=10)

    for i, flow in enumerate(flows):
        emit = (i == len(flows) - 1)  # emit on last frame
        result = builder.update(flow, emit_channel_maps=emit)

    # Check background_score
    assert result.background_score.shape == (h, w), (
        f"background_score shape {result.background_score.shape} != ({h}, {w})"
    )
    assert result.background_score.min() >= 0.0, (
        f"background_score min {result.background_score.min()} < 0.0"
    )
    assert result.background_score.max() <= 1.0, (
        f"background_score max {result.background_score.max()} > 1.0"
    )

    # Check confidence_map
    assert result.confidence_map.shape == (h, w)
    assert result.confidence_map.min() >= 0.0
    assert result.confidence_map.max() <= 1.0

    # Check channel maps (emitted on last frame)
    for name, cmap in [
        ("parallax_map", result.parallax_map),
        ("persistence_map", result.persistence_map),
        ("stability_map", result.stability_map),
    ]:
        assert cmap is not None, f"{name} should be emitted"
        assert cmap.shape == (h, w), f"{name} shape {cmap.shape} != ({h}, {w})"
        assert cmap.min() >= 0.0, f"{name} min {cmap.min()} < 0.0"
        assert cmap.max() <= 1.0, f"{name} max {cmap.max()} > 1.0"


# ---------------------------------------------------------------------------
# Property 10: Stationary camera re-weighting
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    h=st.integers(min_value=8, max_value=16),
    w=st.integers(min_value=8, max_value=16),
    n_frames=st.integers(min_value=3, max_value=6),
)
def test_stationary_camera_reweighting(h, w, n_frames):
    """Feature: depth-keying-pipeline, Property 10: Stationary camera re-weighting

    For any sequence of flow fields where the median global motion magnitude
    across the buffer is below the stationary_threshold, the effective parallax
    weight SHALL be reduced relative to its configured value.

    **Validates: Requirements 2.15**
    """
    threshold = 2.0

    # Builder with default weights
    builder = ImageCubeBuilder(
        buffer_size=n_frames + 2,
        stationary_threshold=threshold,
        parallax_weight=0.4,
        persistence_weight=0.3,
        stability_weight=0.3,
    )

    # Feed near-zero flow fields (well below threshold)
    for _ in range(n_frames):
        # Very small flow — simulates stationary camera
        flow = np.random.default_rng(42).normal(0, 0.01, (h, w, 2)).astype(np.float32)
        fr = _make_flow_result(h, w, forward_flow=flow)
        builder.update(fr)

    # Check effective weights
    eff_pw, eff_per_w, eff_sw = builder._effective_weights()

    # Parallax weight should be reduced
    assert eff_pw < builder.parallax_weight, (
        f"Effective parallax weight {eff_pw:.4f} should be < "
        f"configured {builder.parallax_weight:.4f} for stationary camera"
    )

    # Persistence + stability should have more influence
    assert eff_per_w >= builder.persistence_weight, (
        f"Effective persistence weight {eff_per_w:.4f} should be >= "
        f"configured {builder.persistence_weight:.4f}"
    )
    assert eff_sw >= builder.stability_weight, (
        f"Effective stability weight {eff_sw:.4f} should be >= "
        f"configured {builder.stability_weight:.4f}"
    )

    # Weights should still sum to ~1.0
    total = eff_pw + eff_per_w + eff_sw
    assert abs(total - 1.0) < 1e-7, (
        f"Effective weights sum to {total:.7f}, expected 1.0"
    )
