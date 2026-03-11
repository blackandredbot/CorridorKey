"""Property-based tests for DepthKeyingEngine.

# Feature: depth-keying-pipeline, Property 19: Output frame naming preservation
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.depth_keying_engine import DepthKeyingEngine


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Frame stem strategy: alphanumeric names with optional underscores/dashes,
# mimicking real VFX frame naming patterns (e.g., "frame_001", "shot-A-0042").
# Use lowercase only to avoid case-insensitive filesystem collisions (macOS).
_frame_stem = st.from_regex(r"[a-z][a-z0-9_\-]{0,15}", fullmatch=True)


@st.composite
def frame_name_sets(draw: st.DrawFn):
    """Generate a sorted set of unique frame stems (minimum 2).

    Returns a list of unique stem strings, sorted alphabetically.
    """
    n = draw(st.integers(min_value=2, max_value=6))
    stems = draw(
        st.lists(
            _frame_stem,
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return sorted(stems)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_clip(clip_dir: Path, stems: list[str]) -> None:
    """Create a minimal Input/ folder with small solid-color PNG frames."""
    input_dir = clip_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems):
        # Create a tiny 16×16 solid-color image (different per frame)
        color = ((i * 37 + 50) % 256)
        img = np.full((16, 16, 3), color, dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"{stem}.png"), img)


def _mock_resolve_device(requested=None):
    """Always return 'cpu' to avoid GPU requirements in tests."""
    return "cpu"


# ---------------------------------------------------------------------------
# Property 19: Output frame naming preservation
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=frame_name_sets())
def test_output_frame_naming_preservation(data: list[str]):
    """Feature: depth-keying-pipeline, Property 19: Output frame naming preservation

    For any set of input frames in Input/, the output files in DepthMatte/
    and Comp/ shall use the same file stem as the corresponding input frame.

    Validates: Requirements 8.4
    """
    stems = data

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_dir = Path(tmp_dir) / "test_clip"
        _create_synthetic_clip(clip_dir, stems)

        # Instantiate engine with minimal settings (CPU, fast)
        with patch("CorridorKeyModule.depth.depth_keying_engine.resolve_device", _mock_resolve_device):
            engine = DepthKeyingEngine(
                device="cpu",
                flow_method="farneback",
                save_depth_maps=False,
                save_flow=False,
            )

        # Process the clip
        engine.process_clip(str(clip_dir))

        # --- Verify DepthMatte/ output stems ---
        matte_dir = clip_dir / "DepthMatte"
        assert matte_dir.is_dir(), "DepthMatte/ directory should exist"

        matte_stems = sorted(
            os.path.splitext(f)[0] for f in os.listdir(matte_dir)
        )

        # The first frame has no previous frame to pair with, so it produces
        # no output.  Output stems should match input stems[1:] (all frames
        # that had a predecessor to compute flow against).
        expected_output_stems = sorted(stems[1:])
        assert matte_stems == expected_output_stems, (
            f"DepthMatte stems {matte_stems} != expected {expected_output_stems}"
        )

        # --- Verify Comp/ output stems ---
        comp_dir = clip_dir / "Comp"
        assert comp_dir.is_dir(), "Comp/ directory should exist"

        comp_stems = sorted(
            os.path.splitext(f)[0] for f in os.listdir(comp_dir)
        )
        assert comp_stems == expected_output_stems, (
            f"Comp stems {comp_stems} != expected {expected_output_stems}"
        )
