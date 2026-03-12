"""Feature: motion-blur-alpha-refinement, Property 14: Refined alpha EXR round-trip

For any valid refined alpha (float32 [H, W] in [0.0, 1.0]), writing via
``write_depth_map`` then reading via ``read_depth_map`` produces a
numerically equivalent array within tolerance of 1e-7.

Validates: Requirements 9.1, 9.2, 9.3
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.exr_io import read_depth_map, write_depth_map


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def alpha_mattes(draw: st.DrawFn) -> np.ndarray:
    """Generate a float32 [H, W] alpha matte with values in [0.0, 1.0]."""
    h = draw(st.integers(min_value=4, max_value=32))
    w = draw(st.integers(min_value=4, max_value=32))
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
# Property 14: Refined alpha EXR round-trip (Task 8.2)
# ---------------------------------------------------------------------------


class TestRefinedAlphaEXRRoundTrip:
    """Property 14: For any valid refined alpha matte (float32 [H, W] with
    values in [0.0, 1.0]), writing to a 32-bit float EXR file via
    ``write_depth_map`` and reading back via ``read_depth_map`` shall
    produce a numerically equivalent array within tolerance of 1e-7.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(alpha=alpha_mattes())
    def test_exr_round_trip(self, alpha: np.ndarray, tmp_path_factory) -> None:
        """Feature: motion-blur-alpha-refinement, Property 14: Refined alpha
        EXR round-trip.

        **Validates: Requirements 9.1, 9.2, 9.3**
        """
        tmp_dir = tmp_path_factory.mktemp("exr_roundtrip")
        exr_path = str(tmp_dir / "alpha.exr")

        write_depth_map(exr_path, alpha)
        loaded = read_depth_map(exr_path)

        assert loaded.shape == alpha.shape, (
            f"Shape mismatch: wrote {alpha.shape}, read {loaded.shape}"
        )
        assert loaded.dtype == np.float32, (
            f"Expected float32, got {loaded.dtype}"
        )
        np.testing.assert_allclose(
            loaded, alpha, atol=1e-7,
            err_msg="EXR round-trip produced different values",
        )
