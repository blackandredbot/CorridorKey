"""Property-based tests for PlateSubtractionEngine.

# Feature: plate-subtraction-keying
# Property 13: Minimum frame count validation
# Property 9: Iterative convergence early termination
# Property 14: Progress callback invocation count
# Property 15: Output filename preservation
"""

from __future__ import annotations

import os

# Enable OpenEXR codec before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from CorridorKeyModule.depth.data_models import PlateSubtractionConfig
from CorridorKeyModule.depth.plate_subtraction_engine import PlateSubtractionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PlateSubtractionConfig:
    """Create a PlateSubtractionConfig with sensible fast-test defaults."""
    defaults = dict(
        max_iterations=1,
        plate_search_radius=2,
        cube_buffer_size=2,
    )
    defaults.update(overrides)
    return PlateSubtractionConfig(**defaults)


def _create_frames(input_dir: Path, count: int, size: int = 8) -> list[str]:
    """Write *count* small random PNG frames into *input_dir*.

    Returns the list of filenames written (sorted).
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    rng = np.random.default_rng(42)
    for i in range(count):
        name = f"frame_{i:04d}.png"
        frame = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / name), frame)
        names.append(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Frame stem strategy: lowercase alphanumeric with optional underscores/dashes.
_frame_stem = st.from_regex(r"[a-z][a-z0-9_\-]{0,15}", fullmatch=True)


@st.composite
def frame_name_sets(draw: st.DrawFn) -> list[str]:
    """Generate a sorted list of 3–6 unique frame stems."""
    n = draw(st.integers(min_value=3, max_value=6))
    stems = draw(
        st.lists(_frame_stem, min_size=n, max_size=n, unique=True)
    )
    return sorted(stems)


# ---------------------------------------------------------------------------
# Property 13: Minimum frame count validation
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(n_frames=st.integers(min_value=0, max_value=2))
def test_minimum_frame_count_validation(n_frames: int):
    """Feature: plate-subtraction-keying, Property 13: Minimum frame count validation

    For any input with fewer than 3 frames, PlateSubtractionEngine.process_clip()
    shall raise ValueError with a descriptive message including the frame count.

    **Validates: Requirements 6.6**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_dir = Path(tmp_dir) / "clip"
        input_dir = clip_dir / "Input"
        _create_frames(input_dir, n_frames)

        config = _make_config()
        engine = PlateSubtractionEngine(config, device="cpu")

        with pytest.raises(ValueError, match=str(n_frames)):
            engine.process_clip(str(clip_dir))


# ---------------------------------------------------------------------------
# Property 9: Iterative convergence early termination
# ---------------------------------------------------------------------------


def test_iterative_convergence_early_termination(tmp_path: Path):
    """Feature: plate-subtraction-keying, Property 9: Iterative convergence early termination

    When the alpha matte does not change between iterations (identical frames
    produce identical results), the engine shall terminate after 2 iterations
    rather than running all max_iterations.

    **Validates: Requirements 4.5**
    """
    # Create 4 identical solid-colour frames so the pipeline produces
    # the same alpha on every iteration → convergence after iteration 2.
    clip_dir = tmp_path / "clip"
    input_dir = clip_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    solid = np.full((16, 16, 3), 128, dtype=np.uint8)
    n_frames = 4
    for i in range(n_frames):
        cv2.imwrite(str(input_dir / f"frame_{i:04d}.png"), solid)

    # Use a generous convergence threshold so identical-frame convergence
    # is detected, and set max_iterations high to prove early termination.
    config = _make_config(max_iterations=5, convergence_threshold=0.05)
    engine = PlateSubtractionEngine(config, device="cpu")

    # Track how many times synthesize is called to count iterations.
    original_synthesize = engine.synthesizer.synthesize
    call_count = 0

    def counting_synthesize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_synthesize(*args, **kwargs)

    engine.synthesizer.synthesize = counting_synthesize

    engine.process_clip(str(clip_dir))

    # With N frames, synthesize is called N times per iteration.
    # Convergence should happen well before max_iterations=5 (i.e. < 5*N=20).
    iterations_run = call_count // n_frames
    assert iterations_run < 5, (
        f"Expected early convergence but synthesize was called "
        f"{call_count} times ({iterations_run} iterations for {n_frames} frames), "
        f"which equals max_iterations=5 — no early termination occurred"
    )


# ---------------------------------------------------------------------------
# Property 14: Progress callback invocation count
# ---------------------------------------------------------------------------


def test_progress_callback_invocation_count(tmp_path: Path):
    """Feature: plate-subtraction-keying, Property 14: Progress callback invocation count

    For a clip with N frames processed with max_iterations=1 and no per-frame
    errors, the on_frame_complete callback shall be invoked exactly N times
    with frame indices covering all frames.

    **Validates: Requirements 6.4**
    """
    clip_dir = tmp_path / "clip"
    input_dir = clip_dir / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 4
    rng = np.random.default_rng(99)
    for i in range(n_frames):
        frame = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"frame_{i:04d}.png"), frame)

    config = _make_config(max_iterations=1)
    engine = PlateSubtractionEngine(config, device="cpu")

    calls: list[tuple[int, int]] = []

    def callback(frame_idx: int, total: int) -> None:
        calls.append((frame_idx, total))

    engine.process_clip(str(clip_dir), on_frame_complete=callback)

    # Exactly N callbacks
    assert len(calls) == n_frames, (
        f"Expected {n_frames} callbacks, got {len(calls)}"
    )

    # Each call should report the correct total
    for idx, total in calls:
        assert total == n_frames, (
            f"Callback reported total={total}, expected {n_frames}"
        )

    # Frame indices should cover 0..N-1
    frame_indices = sorted(idx for idx, _ in calls)
    assert frame_indices == list(range(n_frames)), (
        f"Frame indices {frame_indices} != expected {list(range(n_frames))}"
    )


# ---------------------------------------------------------------------------
# Property 15: Output filename preservation
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(data=frame_name_sets())
def test_output_filename_preservation(data: list[str]):
    """Feature: plate-subtraction-keying, Property 15: Output filename preservation

    For any set of input frame filenames, the output matte filenames in
    PlateMatte/ shall use the same stem as the corresponding input frames
    with a .exr extension.

    **Validates: Requirements 10.2**
    """
    stems = data

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_dir = Path(tmp_dir) / "clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(7)
        for stem in stems:
            frame = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"{stem}.png"), frame)

        config = _make_config(max_iterations=1)
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        # Verify PlateMatte/ output stems match input stems
        matte_dir = clip_dir / "PlateMatte"
        assert matte_dir.is_dir(), "PlateMatte/ directory should exist"

        matte_files = sorted(os.listdir(matte_dir))
        matte_stems = sorted(os.path.splitext(f)[0] for f in matte_files)
        matte_exts = [os.path.splitext(f)[1] for f in matte_files]

        assert matte_stems == sorted(stems), (
            f"PlateMatte stems {matte_stems} != expected {sorted(stems)}"
        )

        # All output files should have .exr extension
        for ext in matte_exts:
            assert ext == ".exr", f"Expected .exr extension, got {ext}"
