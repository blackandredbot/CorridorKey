"""Feature: motion-blur-alpha-refinement, Properties 4, 5, 6: Clean Plate Provider

Property 4: Clean plate PNG loading applies sRGB-to-linear conversion
Property 5: Clean plate resolution mismatch raises error
Property 6: Static clean plate synthesis selects lowest-alpha frame per pixel

Validates: Requirements 2.2, 2.3, 2.9, 10.4
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.core.color_utils import srgb_to_linear
from CorridorKeyModule.depth.clean_plate_provider import CleanPlateProvider


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def dimensions(draw: st.DrawFn) -> tuple[int, int]:
    """Generate small (H, W) dimensions in [4, 32]."""
    h = draw(st.integers(min_value=4, max_value=32))
    w = draw(st.integers(min_value=4, max_value=32))
    return h, w


@st.composite
def srgb_pixel_arrays(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a uint8 [H, W, 3] sRGB image with values in [0, 255]."""
    data = draw(
        st.lists(
            st.integers(min_value=0, max_value=255),
            min_size=h * w * 3,
            max_size=h * w * 3,
        )
    )
    return np.array(data, dtype=np.uint8).reshape(h, w, 3)


@st.composite
def rgb_frames(draw: st.DrawFn, h: int, w: int) -> np.ndarray:
    """Generate a float32 [H, W, 3] frame with values in [0.0, 1.0]."""
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


# ---------------------------------------------------------------------------
# Property 4: Clean plate PNG loading applies sRGB-to-linear conversion
# ---------------------------------------------------------------------------


class TestPngSrgbToLinearConversion:
    """Property 4: For any PNG with sRGB values, loaded plate equals
    srgb_to_linear() applied to normalized float32 input.

    **Validates: Requirements 2.2, 10.4**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_png_srgb_to_linear(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 4: Clean plate PNG
        loading applies sRGB-to-linear conversion.

        **Validates: Requirements 2.2, 10.4**
        """
        h, w = data.draw(dimensions())
        srgb_img = data.draw(srgb_pixel_arrays(h, w))

        # Write PNG with known sRGB values (cv2 expects BGR)
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "plate.png")
            bgr_img = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(png_path, bgr_img)

            # Load via CleanPlateProvider
            provider = CleanPlateProvider()
            loaded = provider.load(png_path, target_h=h, target_w=w)

            # Compute expected: normalize to [0,1] then srgb_to_linear
            expected = srgb_to_linear(srgb_img.astype(np.float32) / 255.0)

            assert loaded.dtype == np.float32, (
                f"Expected float32, got {loaded.dtype}"
            )
            assert loaded.shape == (h, w, 3), (
                f"Expected shape ({h}, {w}, 3), got {loaded.shape}"
            )
            np.testing.assert_allclose(
                loaded, expected, atol=1e-5,
                err_msg="PNG sRGB-to-linear conversion mismatch",
            )


# ---------------------------------------------------------------------------
# Property 5: Clean plate resolution mismatch raises error
# ---------------------------------------------------------------------------


class TestResolutionMismatchError:
    """Property 5: For any plate with (H_plate, W_plate) != (H_target, W_target),
    load() raises ValueError containing both resolutions.

    **Validates: Requirements 2.3**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_resolution_mismatch_raises(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 5: Clean plate
        resolution mismatch raises error.

        **Validates: Requirements 2.3**
        """
        # Generate plate dimensions
        h_plate, w_plate = data.draw(dimensions())

        # Generate target dimensions that differ from plate
        h_target = data.draw(
            st.integers(min_value=4, max_value=32).filter(lambda x: x != h_plate)
        )
        w_target = data.draw(st.integers(min_value=4, max_value=32))

        # Create a PNG with plate dimensions
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.zeros((h_plate, w_plate, 3), dtype=np.uint8)
            png_path = os.path.join(tmpdir, "plate.png")
            cv2.imwrite(png_path, img)

            provider = CleanPlateProvider()

            with pytest.raises(ValueError, match=str(h_plate)) as exc_info:
                provider.load(png_path, target_h=h_target, target_w=w_target)

            error_msg = str(exc_info.value)
            # Verify both resolutions are mentioned in the error
            assert str(h_plate) in error_msg and str(w_plate) in error_msg, (
                f"Error should contain plate resolution ({h_plate}, {w_plate}), "
                f"got: {error_msg}"
            )
            assert str(h_target) in error_msg and str(w_target) in error_msg, (
                f"Error should contain target resolution ({h_target}, {w_target}), "
                f"got: {error_msg}"
            )


# ---------------------------------------------------------------------------
# Property 6: Static clean plate synthesis selects lowest-alpha frame per pixel
# ---------------------------------------------------------------------------


class TestStaticCleanPlateSynthesis:
    """Property 6: For any sequence of N frames and alphas (N >= 1), static
    plate at pixel (y, x) equals RGB from frame with minimal alpha[y, x].

    **Validates: Requirements 2.9**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(data=st.data())
    def test_static_selects_lowest_alpha(self, data: st.DataObject) -> None:
        """Feature: motion-blur-alpha-refinement, Property 6: Static clean
        plate synthesis selects lowest-alpha frame per pixel.

        **Validates: Requirements 2.9**
        """
        h, w = data.draw(dimensions())
        n_frames = data.draw(st.integers(min_value=1, max_value=5))

        frames = [data.draw(rgb_frames(h, w)) for _ in range(n_frames)]
        alphas_list = [data.draw(alpha_mattes(h, w)) for _ in range(n_frames)]

        provider = CleanPlateProvider()
        plate = provider.synthesize_static(frames, alphas_list)

        assert plate.shape == (h, w, 3), (
            f"Expected shape ({h}, {w}, 3), got {plate.shape}"
        )
        assert plate.dtype == np.float32, (
            f"Expected float32, got {plate.dtype}"
        )

        # Verify per-pixel: plate[y, x] == frames[argmin_alpha][y, x]
        alpha_stack = np.stack(alphas_list, axis=0)  # [N, H, W]
        min_indices = np.argmin(alpha_stack, axis=0)  # [H, W]

        for y in range(h):
            for x in range(w):
                best_frame_idx = min_indices[y, x]
                expected_rgb = frames[best_frame_idx][y, x]
                np.testing.assert_array_equal(
                    plate[y, x], expected_rgb,
                    err_msg=(
                        f"Pixel ({y}, {x}): expected frame {best_frame_idx} "
                        f"RGB={expected_rgb}, got {plate[y, x]}"
                    ),
                )
