"""Property-based tests for plate-subtraction EXR round-trip serialization.

# Feature: plate-subtraction-keying
# Property 12: Clean plate EXR round-trip
# Property 11: Alpha matte EXR round-trip
"""

from __future__ import annotations

import os

# Enable OpenEXR codec before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import tempfile

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.exr_io import (
    read_depth_map,
    read_rgb_exr,
    write_depth_map,
    write_rgb_exr,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def rgb_images(draw: st.DrawFn) -> np.ndarray:
    """Generate a random [H, W, 3] float32 array with values in [0.0, 1.0]."""
    h = draw(st.integers(min_value=4, max_value=32))
    w = draw(st.integers(min_value=4, max_value=32))
    data = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=h * w * 3,
            max_size=h * w * 3,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w, 3)


@st.composite
def alpha_mattes(draw: st.DrawFn) -> np.ndarray:
    """Generate a random [H, W] float32 array with values in [0.0, 1.0]."""
    h = draw(st.integers(min_value=4, max_value=32))
    w = draw(st.integers(min_value=4, max_value=32))
    data = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=h * w,
            max_size=h * w,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w)


# ---------------------------------------------------------------------------
# Property 12: Clean plate EXR round-trip
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=rgb_images())
def test_clean_plate_exr_round_trip(data: np.ndarray):
    """Feature: plate-subtraction-keying, Property 12: Clean plate EXR round-trip

    For any valid clean plate (float32 [H, W, 3] with values in [0.0, 1.0]),
    writing via write_rgb_exr and reading back via read_rgb_exr shall produce
    a numerically equivalent array within tolerance of 1e-7.

    **Validates: Requirements 11.1, 11.2, 11.3**
    """
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        exr_path = tmp.name

    try:
        write_rgb_exr(exr_path, data)
        loaded = read_rgb_exr(exr_path)

        assert loaded.shape == data.shape, (
            f"Shape mismatch: wrote {data.shape}, read {loaded.shape}"
        )
        assert loaded.dtype == np.float32, (
            f"Dtype mismatch: expected float32, got {loaded.dtype}"
        )
        assert np.allclose(data, loaded, atol=1e-7), (
            f"Round-trip fidelity exceeded 1e-7 tolerance. "
            f"Max diff: {np.max(np.abs(data - loaded))}"
        )
    finally:
        if os.path.exists(exr_path):
            os.unlink(exr_path)


# ---------------------------------------------------------------------------
# Property 11: Alpha matte EXR round-trip
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=alpha_mattes())
def test_alpha_matte_exr_round_trip(data: np.ndarray):
    """Feature: plate-subtraction-keying, Property 11: Alpha matte EXR round-trip

    For any valid alpha matte (float32 [H, W] with values in [0.0, 1.0]),
    writing via write_depth_map and reading back via read_depth_map shall
    produce a numerically equivalent array within tolerance of 1e-7.

    **Validates: Requirements 10.1, 10.4**
    """
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        exr_path = tmp.name

    try:
        write_depth_map(exr_path, data)
        loaded = read_depth_map(exr_path)

        assert loaded.shape == data.shape, (
            f"Shape mismatch: wrote {data.shape}, read {loaded.shape}"
        )
        assert loaded.dtype == np.float32, (
            f"Dtype mismatch: expected float32, got {loaded.dtype}"
        )
        assert np.allclose(data, loaded, atol=1e-7), (
            f"Round-trip fidelity exceeded 1e-7 tolerance. "
            f"Max diff: {np.max(np.abs(data - loaded))}"
        )
    finally:
        if os.path.exists(exr_path):
            os.unlink(exr_path)
