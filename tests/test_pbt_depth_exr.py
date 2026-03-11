"""Property-based tests for EXR round-trip serialization.

# Feature: depth-keying-pipeline, Property 16: Depth map EXR round-trip
"""

from __future__ import annotations

import os
import tempfile

# Enable OpenEXR codec before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.exr_io import (
    read_depth_map,
    read_flow_field,
    write_depth_map,
    write_flow_field,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def depth_maps(draw: st.DrawFn) -> np.ndarray:
    """Generate a random float32 depth map with values in [0.0, 1.0].

    Dimensions are kept small (4–64 per side) to keep tests fast while
    still exercising a meaningful range of shapes.
    """
    h = draw(st.integers(min_value=4, max_value=64))
    w = draw(st.integers(min_value=4, max_value=64))
    data = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=h * w,
            max_size=h * w,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w)


@st.composite
def flow_fields(draw: st.DrawFn) -> np.ndarray:
    """Generate a random float32 flow field [H, W, 2] with realistic values.

    Flow values are in [-100, 100] pixel displacement range.
    """
    h = draw(st.integers(min_value=4, max_value=64))
    w = draw(st.integers(min_value=4, max_value=64))
    data = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=h * w * 2,
            max_size=h * w * 2,
        )
    )
    return np.array(data, dtype=np.float32).reshape(h, w, 2)


# ---------------------------------------------------------------------------
# Property 16: Depth map EXR round-trip
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=depth_maps())
def test_depth_map_exr_round_trip(data: np.ndarray):
    """Feature: depth-keying-pipeline, Property 16: Depth map EXR round-trip

    For any valid depth map (float32 [H, W] with values in [0.0, 1.0]),
    writing to a 32-bit float EXR file and reading it back shall produce
    a numerically equivalent array within tolerance of 1e-7.

    Validates: Requirements 9.1, 9.2, 9.3
    """
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_depth_map(tmp_path, data)
        loaded = read_depth_map(tmp_path)

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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=flow_fields())
def test_flow_field_exr_round_trip(data: np.ndarray):
    """Feature: depth-keying-pipeline, Property 16: Depth map EXR round-trip (flow fields)

    Two-channel float32 flow fields shall also survive a write-then-read
    cycle within 1e-7 tolerance.

    Validates: Requirements 9.1, 9.2, 9.3
    """
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_flow_field(tmp_path, data)
        loaded = read_flow_field(tmp_path)

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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
