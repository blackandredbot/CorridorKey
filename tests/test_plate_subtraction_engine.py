"""Unit tests for PlateSubtractionEngine.

Tests end-to-end processing, edge cases, error conditions, iterative
refinement, and optional output flags.

Requirements: 6.1, 6.2, 6.3, 6.5, 6.6, 6.7, 6.8, 7.1, 7.4
"""

from __future__ import annotations

import os

# Enable OpenEXR codec before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from CorridorKeyModule.depth.data_models import PlateSubtractionConfig
from CorridorKeyModule.depth.plate_subtraction_engine import PlateSubtractionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_config(**overrides) -> PlateSubtractionConfig:
    """Create a PlateSubtractionConfig with fast-test defaults."""
    defaults = dict(
        max_iterations=1,
        plate_search_radius=2,
        cube_buffer_size=2,
        difference_threshold=0.05,
        difference_falloff=0.03,
    )
    defaults.update(overrides)
    return PlateSubtractionConfig(**defaults)


def _write_frames(
    input_dir: Path,
    frames: list[np.ndarray],
    prefix: str = "frame",
) -> list[str]:
    """Write uint8 BGR frames as PNGs into *input_dir*. Returns sorted filenames."""
    input_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, frame in enumerate(frames):
        name = f"{prefix}_{i:04d}.png"
        cv2.imwrite(str(input_dir / name), frame)
        names.append(name)
    return sorted(names)


def _random_frames(count: int, size: int = 16, seed: int = 42) -> list[np.ndarray]:
    """Generate *count* random uint8 BGR frames of shape (size, size, 3)."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# 1. End-to-end basic: 4 random frames → verify output dirs and file counts
# ---------------------------------------------------------------------------


class TestEndToEndBasic:
    """End-to-end processing with a small synthetic clip."""

    def test_process_clip_creates_output_dirs_and_files(self, tmp_path: Path):
        """4 random frames → PlateMatte/ and Comp/ with correct file counts."""
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        n = 4
        _write_frames(input_dir, _random_frames(n))

        engine = PlateSubtractionEngine(_fast_config(), device="cpu")
        engine.process_clip(str(clip_dir))

        matte_dir = clip_dir / "PlateMatte"
        comp_dir = clip_dir / "Comp"

        assert matte_dir.is_dir()
        assert comp_dir.is_dir()

        matte_files = sorted(os.listdir(matte_dir))
        comp_files = sorted(os.listdir(comp_dir))

        assert len(matte_files) == n
        assert len(comp_files) == n

        # Mattes are EXR, comps are PNG
        assert all(f.endswith(".exr") for f in matte_files)
        assert all(f.endswith(".png") for f in comp_files)

    def test_output_stems_match_input_stems(self, tmp_path: Path):
        """Output file stems must match input file stems."""
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        frames = _random_frames(3)
        _write_frames(input_dir, frames, prefix="shot")

        engine = PlateSubtractionEngine(_fast_config(), device="cpu")
        engine.process_clip(str(clip_dir))

        input_stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(input_dir)
        )
        matte_stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(clip_dir / "PlateMatte")
        )
        assert matte_stems == input_stems


# ---------------------------------------------------------------------------
# 2. All-background sequence: identical solid frames → near-zero alpha
# ---------------------------------------------------------------------------


class TestAllBackgroundSequence:
    """Identical frames should produce near-zero alpha (everything is background)."""

    def test_identical_frames_produce_low_alpha(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"

        # All frames are the same solid grey
        solid = np.full((16, 16, 3), 128, dtype=np.uint8)
        _write_frames(input_dir, [solid] * 4)

        config = _fast_config(
            difference_threshold=0.05,
            difference_falloff=0.03,
            low_confidence_alpha=0.0,
        )
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        # Read back a matte and check it's mostly low alpha
        matte_dir = clip_dir / "PlateMatte"
        matte_files = sorted(os.listdir(matte_dir))
        assert len(matte_files) > 0

        # Check the last frame (not frame 0 which is bootstrap-special)
        matte = cv2.imread(
            str(matte_dir / matte_files[-1]),
            cv2.IMREAD_UNCHANGED,
        )
        assert matte is not None
        # Mean alpha should be low for identical frames
        assert np.mean(matte) < 0.5, (
            f"Expected low alpha for identical frames, got mean={np.mean(matte):.3f}"
        )


# ---------------------------------------------------------------------------
# 3. All-foreground sequence: very different frames → high alpha
# ---------------------------------------------------------------------------


class TestAllForegroundSequence:
    """Frames that are very different from each other should produce high alpha."""

    def test_diverse_frames_produce_high_alpha(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"

        # Create frames with very different solid colours
        colours = [
            np.full((16, 16, 3), c, dtype=np.uint8)
            for c in [0, 128, 255, 64]
        ]
        _write_frames(input_dir, colours)

        config = _fast_config(
            difference_threshold=0.02,
            difference_falloff=0.01,
            low_confidence_alpha=1.0,
        )
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        # Read a matte from a middle frame
        matte_dir = clip_dir / "PlateMatte"
        matte_files = sorted(os.listdir(matte_dir))
        # Frame 2 (index 2) should have high alpha since neighbours differ
        matte = cv2.imread(
            str(matte_dir / matte_files[2]),
            cv2.IMREAD_UNCHANGED,
        )
        assert matte is not None
        # With low_confidence_alpha=1.0 and very different frames,
        # alpha should be high on average
        assert np.mean(matte) > 0.3, (
            f"Expected high alpha for diverse frames, got mean={np.mean(matte):.3f}"
        )


# ---------------------------------------------------------------------------
# 4. Missing Input/ directory → ValueError
# ---------------------------------------------------------------------------


class TestMissingInputDirectory:
    """process_clip must raise ValueError when Input/ is missing."""

    def test_raises_on_missing_input_dir(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        clip_dir.mkdir()
        # No Input/ subdirectory

        engine = PlateSubtractionEngine(_fast_config(), device="cpu")
        with pytest.raises(ValueError, match="Input directory not found"):
            engine.process_clip(str(clip_dir))


# ---------------------------------------------------------------------------
# 5. Fewer than 3 frames → ValueError
# ---------------------------------------------------------------------------


class TestFewerThanThreeFrames:
    """process_clip must raise ValueError with fewer than 3 frames."""

    @pytest.mark.parametrize("n_frames", [0, 1, 2])
    def test_raises_on_insufficient_frames(self, tmp_path: Path, n_frames: int):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        _write_frames(input_dir, _random_frames(n_frames))

        engine = PlateSubtractionEngine(_fast_config(), device="cpu")
        with pytest.raises(ValueError, match=str(n_frames)):
            engine.process_clip(str(clip_dir))


# ---------------------------------------------------------------------------
# 6. Optional output flags
# ---------------------------------------------------------------------------


class TestOptionalOutputFlags:
    """Verify extra directories are created when save_* flags are True."""

    def test_save_clean_plates(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        n = 3
        _write_frames(input_dir, _random_frames(n))

        config = _fast_config(save_clean_plates=True)
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        plate_dir = clip_dir / "CleanPlate"
        assert plate_dir.is_dir()
        plate_files = os.listdir(plate_dir)
        assert len(plate_files) == n
        assert all(f.endswith(".exr") for f in plate_files)

    def test_save_bootstrap(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        n = 3
        _write_frames(input_dir, _random_frames(n))

        config = _fast_config(save_bootstrap=True)
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        bootstrap_dir = clip_dir / "Bootstrap"
        assert bootstrap_dir.is_dir()
        bootstrap_files = os.listdir(bootstrap_dir)
        assert len(bootstrap_files) == n
        assert all(f.endswith(".exr") for f in bootstrap_files)

    def test_save_flow(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        n = 4
        _write_frames(input_dir, _random_frames(n))

        config = _fast_config(save_flow=True)
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        flow_dir = clip_dir / "FlowField"
        assert flow_dir.is_dir()
        flow_files = os.listdir(flow_dir)
        # Flow is computed between consecutive pairs → n-1 files
        assert len(flow_files) == n - 1
        assert all(f.endswith(".exr") for f in flow_files)

    def test_no_optional_dirs_by_default(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        _write_frames(input_dir, _random_frames(3))

        config = _fast_config()
        engine = PlateSubtractionEngine(config, device="cpu")
        engine.process_clip(str(clip_dir))

        assert not (clip_dir / "CleanPlate").exists()
        assert not (clip_dir / "Bootstrap").exists()
        assert not (clip_dir / "FlowField").exists()


# ---------------------------------------------------------------------------
# 7. Convergence detection
# ---------------------------------------------------------------------------


class TestConvergenceDetection:
    """Identical frames with high max_iterations should converge early."""

    def test_converges_before_max_iterations(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"

        solid = np.full((16, 16, 3), 128, dtype=np.uint8)
        n = 4
        _write_frames(input_dir, [solid] * n)

        # Use max_iterations=5 and a generous threshold so convergence
        # is detected well before the cap.
        config = _fast_config(
            max_iterations=5,
            convergence_threshold=0.05,
        )
        engine = PlateSubtractionEngine(config, device="cpu")

        # Count synthesize calls to determine iterations run
        original_synthesize = engine.synthesizer.synthesize
        call_count = 0

        def counting_synthesize(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_synthesize(*args, **kwargs)

        engine.synthesizer.synthesize = counting_synthesize
        engine.process_clip(str(clip_dir))

        iterations_run = call_count // n
        assert iterations_run < 5, (
            f"Expected convergence before max_iterations=5, "
            f"but ran {iterations_run} iterations"
        )

    def test_max_iterations_cap_respected(self, tmp_path: Path):
        """Even without convergence, iterations must not exceed max_iterations."""
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        n = 3
        _write_frames(input_dir, _random_frames(n))

        max_iter = 2
        config = _fast_config(
            max_iterations=max_iter,
            convergence_threshold=1e-10,  # very tight → won't converge
        )
        engine = PlateSubtractionEngine(config, device="cpu")

        original_synthesize = engine.synthesizer.synthesize
        call_count = 0

        def counting_synthesize(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_synthesize(*args, **kwargs)

        engine.synthesizer.synthesize = counting_synthesize
        engine.process_clip(str(clip_dir))

        iterations_run = call_count // n
        assert iterations_run <= max_iter, (
            f"Ran {iterations_run} iterations, exceeding max_iterations={max_iter}"
        )


# ---------------------------------------------------------------------------
# 8. Per-frame error recovery
# ---------------------------------------------------------------------------


class TestPerFrameErrorRecovery:
    """Engine should continue processing when a single frame errors."""

    def test_continues_after_frame_error(self, tmp_path: Path):
        clip_dir = tmp_path / "clip"
        input_dir = clip_dir / "Input"
        n = 4
        _write_frames(input_dir, _random_frames(n))

        config = _fast_config()
        engine = PlateSubtractionEngine(config, device="cpu")

        # Make synthesize raise on frame index 1
        original_synthesize = engine.synthesizer.synthesize

        def failing_synthesize(frame_idx, *args, **kwargs):
            if frame_idx == 1:
                raise RuntimeError("Synthetic error on frame 1")
            return original_synthesize(frame_idx, *args, **kwargs)

        engine.synthesizer.synthesize = failing_synthesize
        engine.process_clip(str(clip_dir))

        # PlateMatte should still have files for the non-errored frames
        matte_dir = clip_dir / "PlateMatte"
        matte_files = os.listdir(matte_dir)
        # Frame 1 was skipped, so we expect n-1 output files
        assert len(matte_files) == n - 1, (
            f"Expected {n - 1} matte files (1 skipped), got {len(matte_files)}"
        )
