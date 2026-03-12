"""Feature: motion-blur-alpha-refinement, Properties 7, 8, 9, 10, 11: Alpha Solver

Property 7:  Alpha solver compositing equation consistency
Property 8:  Refined alpha is clamped to [0.0, 1.0] and float32
Property 9:  Epsilon guard preserves original alpha
Property 10: Blur kernel profile correctness
Property 11: Kernel shaping reduces alpha

Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from CorridorKeyModule.depth.alpha_solver import AlphaSolver


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
def rgb_frames(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W, 3] RGB frame with values in [0.0, 1.0]."""
    data = draw(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            ),
            min_size=h * w * 3,
            max_size=h * w * 3,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w, 3)


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
def flow_fields(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W, 2] flow field with values in [-10, 10]."""
    data = draw(
        st.lists(
            st.floats(
                min_value=-10.0, max_value=10.0,
                allow_nan=False, allow_infinity=False,
            ),
            min_size=h * w * 2,
            max_size=h * w * 2,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w, 2)


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


kernel_profiles = st.sampled_from(["linear", "cosine", "gaussian"])
kernel_lengths = st.integers(min_value=2, max_value=200)



# ---------------------------------------------------------------------------
# Property 10: Blur kernel profile correctness (Task 5.2)
# ---------------------------------------------------------------------------


class TestBlurKernelProfileCorrectness:
    """Property 10: For any kernel_profile in {"linear", "cosine", "gaussian"}
    and any kernel length >= 2, the generated kernel shall:
    (a) start at 1.0,
    (b) be monotonically non-increasing,
    (c) have all values in [0.0, 1.0],
    (d) match the specified formula within 1e-6.

    **Validates: Requirements 4.2, 4.4, 4.5, 4.6**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(profile=kernel_profiles, length=kernel_lengths)
    def test_kernel_profile_correctness(self, profile: str, length: int) -> None:
        """Feature: motion-blur-alpha-refinement, Property 10: Blur kernel
        profile correctness.

        **Validates: Requirements 4.2, 4.4, 4.5, 4.6**
        """
        solver = AlphaSolver(kernel_profile=profile)
        kernel = solver._build_kernel(length)

        # (a) Starts at 1.0
        assert abs(kernel[0] - 1.0) < 1e-6, (
            f"Kernel[0] = {kernel[0]}, expected 1.0 for profile={profile}"
        )

        # (b) Monotonically non-increasing
        for i in range(1, len(kernel)):
            assert kernel[i] <= kernel[i - 1] + 1e-6, (
                f"Kernel not monotonically non-increasing at index {i}: "
                f"kernel[{i}]={kernel[i]} > kernel[{i-1}]={kernel[i-1]} "
                f"for profile={profile}, length={length}"
            )

        # (c) All values in [0.0, 1.0]
        assert np.all(kernel >= -1e-6), (
            f"Kernel has negative values: min={kernel.min()} for profile={profile}"
        )
        assert np.all(kernel <= 1.0 + 1e-6), (
            f"Kernel has values > 1.0: max={kernel.max()} for profile={profile}"
        )

        # (d) Matches formula within 1e-6
        t = np.linspace(0.0, 1.0, length, dtype=np.float32)
        if profile == "linear":
            expected = 1.0 - t
        elif profile == "cosine":
            expected = 0.5 * (1.0 + np.cos(np.pi * t))
        elif profile == "gaussian":
            expected = np.exp(-3.0 * t * t)
        else:
            raise AssertionError(f"Unknown profile: {profile}")

        np.testing.assert_allclose(
            kernel, expected.astype(np.float32), atol=1e-6,
            err_msg=f"Kernel values don't match formula for profile={profile}, length={length}",
        )


# ---------------------------------------------------------------------------
# Property 8: Refined alpha is clamped to [0.0, 1.0] and float32 (Task 5.3)
# ---------------------------------------------------------------------------


class TestAlphaClampingAndDtype:
    """Property 8: For any input to solve(), every output pixel is in
    [0.0, 1.0] with dtype float32, shape matches input.

    **Validates: Requirements 3.3, 3.4, 3.7**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_alpha_clamped_and_float32(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 8: Refined alpha
        is clamped to [0.0, 1.0] and float32.

        **Validates: Requirements 3.3, 3.4, 3.7**
        """
        h, w = data.draw(dimensions())
        observed = data.draw(rgb_frames(h, w))
        clean_plate = data.draw(rgb_frames(h, w))
        alpha = data.draw(alpha_mattes(h, w))
        flow = data.draw(flow_fields(h, w))
        mask = data.draw(blur_masks(h, w))

        solver = AlphaSolver(division_epsilon=1e-4, kernel_profile="linear")
        refined_alpha, fg_color = solver.solve(observed, clean_plate, alpha, flow, mask)

        # Shape matches input
        assert refined_alpha.shape == (h, w), (
            f"Alpha shape mismatch: expected ({h}, {w}), got {refined_alpha.shape}"
        )
        assert fg_color.shape == (h, w, 3), (
            f"FG shape mismatch: expected ({h}, {w}, 3), got {fg_color.shape}"
        )

        # dtype is float32
        assert refined_alpha.dtype == np.float32, (
            f"Alpha dtype: expected float32, got {refined_alpha.dtype}"
        )
        assert fg_color.dtype == np.float32, (
            f"FG dtype: expected float32, got {fg_color.dtype}"
        )

        # All values in [0.0, 1.0]
        assert np.all(refined_alpha >= 0.0), (
            f"Alpha has negative values: min={refined_alpha.min()}"
        )
        assert np.all(refined_alpha <= 1.0), (
            f"Alpha has values > 1.0: max={refined_alpha.max()}"
        )


# ---------------------------------------------------------------------------
# Property 9: Epsilon guard preserves original alpha (Task 5.4)
# ---------------------------------------------------------------------------


class TestEpsilonGuard:
    """Property 9: For any pixel where |C_fg_estimate - C_bg| < division_epsilon
    for all 3 channels, refined alpha equals original alpha exactly.

    To test this, we construct inputs where all pixels have very similar
    colours so that the foreground estimate will be close to the background.

    **Validates: Requirements 3.5**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_epsilon_guard_preserves_alpha(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 9: Epsilon guard
        preserves original alpha.

        **Validates: Requirements 3.5**
        """
        h, w = data.draw(dimensions())
        epsilon = 1e-4

        # Create a uniform colour scene — all pixels have the same RGB.
        # This ensures that C_fg_estimate ≈ C_bg for every pixel,
        # triggering the epsilon guard.
        base_color = data.draw(
            st.tuples(
                st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
            )
        )
        color_arr = np.array(base_color, dtype=np.float32)

        observed = np.broadcast_to(color_arr, (h, w, 3)).copy().astype(np.float32)
        clean_plate = np.broadcast_to(color_arr, (h, w, 3)).copy().astype(np.float32)

        alpha = data.draw(alpha_mattes(h, w))

        # Small flow so foreground estimate walks to nearby pixels (same colour)
        flow = np.ones((h, w, 2), dtype=np.float32) * 0.5

        # All pixels are blur-masked
        mask = np.ones((h, w), dtype=np.float32)

        solver = AlphaSolver(division_epsilon=epsilon, kernel_profile="linear")
        refined_alpha, _ = solver.solve(observed, clean_plate, alpha, flow, mask)

        # Since all colours are identical, |C_fg - C_bg| < epsilon for all
        # channels, so the epsilon guard should preserve original alpha.
        np.testing.assert_array_equal(
            refined_alpha, alpha,
            err_msg="Epsilon guard did not preserve original alpha when "
                    "|C_fg - C_bg| < epsilon for all channels",
        )


# ---------------------------------------------------------------------------
# Property 7: Alpha solver compositing equation consistency (Task 5.5)
# ---------------------------------------------------------------------------


class TestCompositingEquationConsistency:
    """Property 7: For any C_obs, C_bg, C_fg where |C_fg - C_bg| >= epsilon
    in at least one channel, the refined alpha and recovered foreground
    satisfy |C_obs - (alpha * C_fg_recovered + (1-alpha) * C_bg)| < 0.01.

    We construct a controlled scenario: a single blur-masked pixel with a
    known opaque neighbour (the foreground estimate source), and verify the
    compositing equation holds after solving.

    **Validates: Requirements 3.1, 3.6**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_compositing_equation_holds(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 7: Alpha solver
        compositing equation consistency.

        **Validates: Requirements 3.1, 3.6**
        """
        epsilon = 1e-4

        # Generate distinct foreground and background colours
        c_fg_val = data.draw(
            st.tuples(
                st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        c_bg_val = data.draw(
            st.tuples(
                st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False),
            )
        )
        c_fg = np.array(c_fg_val, dtype=np.float32)
        c_bg = np.array(c_bg_val, dtype=np.float32)

        # Ensure |C_fg - C_bg| >= epsilon in at least one channel
        assume(np.any(np.abs(c_fg - c_bg) >= epsilon))

        # Generate a true alpha for the blurred pixel
        true_alpha = data.draw(
            st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)
        )

        # Composite: C_obs = alpha * C_fg + (1 - alpha) * C_bg
        c_obs = np.float32(true_alpha) * c_fg + np.float32(1.0 - true_alpha) * c_bg
        c_obs = np.clip(c_obs, 0.0, 1.0).astype(np.float32)

        # Build a small 4x4 image:
        # - pixel (1,1) is the blur-masked pixel with c_obs
        # - pixel (1,2) is fully opaque with c_fg (foreground estimate source)
        # - all other pixels have c_bg
        h, w = 4, 4
        observed = np.broadcast_to(c_bg, (h, w, 3)).copy().astype(np.float32)
        observed[1, 1] = c_obs
        observed[1, 2] = c_fg

        clean_plate = np.broadcast_to(c_bg, (h, w, 3)).copy().astype(np.float32)

        alpha_in = np.zeros((h, w), dtype=np.float32)
        alpha_in[1, 1] = np.float32(true_alpha)  # partially transparent
        alpha_in[1, 2] = 1.0  # fully opaque (foreground source)

        # Flow points from (1,1) toward (1,2): dx=1, dy=0
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[1, 1, 0] = 1.0  # dx
        flow[1, 1, 1] = 0.0  # dy

        # Only pixel (1,1) is blur-masked
        mask = np.zeros((h, w), dtype=np.float32)
        mask[1, 1] = 1.0

        solver = AlphaSolver(division_epsilon=epsilon, kernel_profile="linear")
        refined_alpha, fg_recovered = solver.solve(
            observed, clean_plate, alpha_in, flow, mask
        )

        # Verify compositing equation at pixel (1,1)
        a = refined_alpha[1, 1]
        fg = fg_recovered[1, 1]
        reconstructed = a * fg + (1.0 - a) * c_bg

        error = np.abs(c_obs - reconstructed)
        assert np.all(error < 0.01), (
            f"Compositing equation violated: max error={error.max():.6f}, "
            f"alpha={a:.6f}, c_obs={c_obs}, reconstructed={reconstructed}"
        )


# ---------------------------------------------------------------------------
# Property 11: Kernel shaping reduces alpha (Task 5.6)
# ---------------------------------------------------------------------------


class TestKernelShapingReducesAlpha:
    """Property 11: For any blur-masked pixel, shaped alpha <= unshaped
    solved alpha, since kernel weights are in [0.0, 1.0].

    We compare the solver output with kernel shaping against the unshaped
    solved alpha. The unshaped alpha is obtained by solving with a trivial
    kernel (all 1.0 weights), which we approximate by using a very short
    flow magnitude (< 2.0) that bypasses kernel shaping.

    Alternatively, we verify the property directly: for any blur-masked
    pixel with blur magnitude >= 2.0, the shaped alpha should be <= the
    alpha that would result from solving without kernel shaping.

    **Validates: Requirements 4.1, 4.3**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_kernel_shaping_reduces_alpha(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 11: Kernel shaping
        reduces alpha.

        **Validates: Requirements 4.1, 4.3**
        """
        epsilon = 1e-4
        profile = data.draw(kernel_profiles)

        # Generate distinct foreground and background colours
        c_fg_val = data.draw(
            st.tuples(
                st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        c_bg_val = data.draw(
            st.tuples(
                st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
            )
        )
        c_fg = np.array(c_fg_val, dtype=np.float32)
        c_bg = np.array(c_bg_val, dtype=np.float32)

        assume(np.any(np.abs(c_fg - c_bg) >= epsilon))

        true_alpha = data.draw(
            st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False)
        )

        c_obs = np.float32(true_alpha) * c_fg + np.float32(1.0 - true_alpha) * c_bg
        c_obs = np.clip(c_obs, 0.0, 1.0).astype(np.float32)

        # Build a small 6x6 image with a blur-masked pixel at (2,2)
        # and an opaque pixel at (2,5) along the flow direction
        h, w = 6, 6
        observed = np.broadcast_to(c_bg, (h, w, 3)).copy().astype(np.float32)
        observed[2, 2] = c_obs
        observed[2, 5] = c_fg

        clean_plate = np.broadcast_to(c_bg, (h, w, 3)).copy().astype(np.float32)

        alpha_in = np.zeros((h, w), dtype=np.float32)
        alpha_in[2, 2] = np.float32(true_alpha)
        alpha_in[2, 5] = 1.0  # opaque source

        # Flow with magnitude >= 2.0 to trigger kernel shaping
        flow_mag = data.draw(
            st.floats(min_value=2.5, max_value=8.0, allow_nan=False, allow_infinity=False)
        )
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[2, 2, 0] = np.float32(flow_mag)  # dx pointing toward opaque pixel
        flow[2, 2, 1] = 0.0

        mask = np.zeros((h, w), dtype=np.float32)
        mask[2, 2] = 1.0

        # Solve WITH kernel shaping
        solver_shaped = AlphaSolver(division_epsilon=epsilon, kernel_profile=profile)
        alpha_shaped, _ = solver_shaped.solve(
            observed, clean_plate, alpha_in, flow, mask
        )

        # Solve WITHOUT kernel shaping (flow magnitude < 2.0 bypasses it)
        flow_no_shape = np.zeros((h, w, 2), dtype=np.float32)
        flow_no_shape[2, 2, 0] = 1.0  # magnitude < 2.0, no kernel shaping
        flow_no_shape[2, 2, 1] = 0.0

        solver_unshaped = AlphaSolver(division_epsilon=epsilon, kernel_profile=profile)
        alpha_unshaped, _ = solver_unshaped.solve(
            observed, clean_plate, alpha_in, flow_no_shape, mask
        )

        # Shaped alpha should be <= unshaped alpha at the blur-masked pixel
        shaped_val = alpha_shaped[2, 2]
        unshaped_val = alpha_unshaped[2, 2]

        assert shaped_val <= unshaped_val + 1e-6, (
            f"Kernel shaping did not reduce alpha: "
            f"shaped={shaped_val:.6f} > unshaped={unshaped_val:.6f} "
            f"for profile={profile}, flow_mag={flow_mag}"
        )
