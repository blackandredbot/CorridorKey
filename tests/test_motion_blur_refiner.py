"""Feature: motion-blur-alpha-refinement, Properties 16, 17: MotionBlurRefiner

Property 16: Output frame naming preservation
Property 17: Progress callback invocation count

Validates: Requirements 6.6, 8.4
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from CorridorKeyModule.depth.data_models import MotionBlurConfig
from CorridorKeyModule.depth.exr_io import write_depth_map
from CorridorKeyModule.depth.motion_blur_refiner import MotionBlurRefiner


# ---------------------------------------------------------------------------
# Helpers — synthetic clip directory
# ---------------------------------------------------------------------------


def _create_synthetic_clip(
    clip_dir: str,
    frame_names: list[str],
    height: int = 8,
    width: int = 8,
) -> None:
    """Create a minimal synthetic clip directory for testing.

    Creates ``Input/`` with small PNG frames and ``Alpha/`` with
    corresponding EXR alpha mattes.
    """
    input_dir = os.path.join(clip_dir, "Input")
    alpha_dir = os.path.join(clip_dir, "Alpha")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)

    for name in frame_names:
        stem = os.path.splitext(name)[0]

        # Create a small solid-colour PNG frame (BGR for cv2).
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.imwrite(os.path.join(input_dir, name), frame)

        # Create a corresponding alpha matte as EXR.
        # Use a partially transparent alpha so blur detection has something
        # to work with.
        alpha = np.full((height, width), 0.5, dtype=np.float32)
        write_depth_map(os.path.join(alpha_dir, f"{stem}.exr"), alpha)


# ---------------------------------------------------------------------------
# Property 16: Output frame naming preservation (Task 8.3)
# ---------------------------------------------------------------------------


class TestOutputFrameNaming:
    """Property 16: For any set of input frame filenames, the refined matte
    output filenames shall use the same stem (basename without extension)
    as the corresponding input frames, with a ``.exr`` extension.

    **Validates: Requirements 8.4**
    """

    def test_output_filenames_match_input_stems(self, tmp_path) -> None:
        """Feature: motion-blur-alpha-refinement, Property 16: Output frame
        naming preservation.

        **Validates: Requirements 8.4**
        """
        frame_names = ["frame_001.png", "frame_002.png", "frame_003.png"]
        clip_dir = str(tmp_path / "clip")
        _create_synthetic_clip(clip_dir, frame_names)

        config = MotionBlurConfig()
        refiner = MotionBlurRefiner(config, device="cpu")
        refiner.process_clip(clip_dir)

        matte_dir = os.path.join(clip_dir, "RefinedMatte")
        assert os.path.isdir(matte_dir), "RefinedMatte/ directory not created"

        output_files = sorted(os.listdir(matte_dir))
        expected_files = [
            os.path.splitext(name)[0] + ".exr" for name in frame_names
        ]

        assert output_files == expected_files, (
            f"Output filenames {output_files} do not match expected "
            f"{expected_files}"
        )


# ---------------------------------------------------------------------------
# Property 17: Progress callback invocation count (Task 8.4)
# ---------------------------------------------------------------------------


class TestProgressCallbackCount:
    """Property 17: For any clip with N processable frames,
    ``on_frame_complete`` is invoked exactly N times.

    **Validates: Requirements 6.6**
    """

    def test_callback_invoked_n_times(self, tmp_path) -> None:
        """Feature: motion-blur-alpha-refinement, Property 17: Progress
        callback invocation count.

        **Validates: Requirements 6.6**
        """
        frame_names = ["shot_A.png", "shot_B.png"]
        clip_dir = str(tmp_path / "clip")
        _create_synthetic_clip(clip_dir, frame_names)

        config = MotionBlurConfig()
        refiner = MotionBlurRefiner(config, device="cpu")

        callback_log: list[tuple[int, int]] = []

        def on_complete(idx: int, total: int) -> None:
            callback_log.append((idx, total))

        refiner.process_clip(clip_dir, on_frame_complete=on_complete)

        n_frames = len(frame_names)
        assert len(callback_log) == n_frames, (
            f"Expected {n_frames} callback invocations, got {len(callback_log)}"
        )

        # Verify total is consistent.
        for idx, total in callback_log:
            assert total == n_frames, (
                f"Callback total should be {n_frames}, got {total}"
            )

# ---------------------------------------------------------------------------
# CLI integration tests (Task 9.2)
# ---------------------------------------------------------------------------

import re
from unittest.mock import patch

from typer.testing import CliRunner

from corridorkey_cli import app

_runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class TestCLIMotionBlurFlag:
    """Test that ``--refine-motion-blur`` flag is accepted by the CLI.

    **Validates: Requirements 7.2, 7.5**
    """

    def test_help_shows_refine_motion_blur_flag(self) -> None:
        """The run-inference --help output lists the motion blur flags."""
        result = _runner.invoke(app, ["run-inference", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_ESCAPE.sub("", result.output)
        # Rich may truncate long option names, so check prefixes.
        assert "--refine-motion-blur" in plain
        assert "--blur-threshold" in plain
        assert "--kernel-profile" in plain
        assert "--temporal-smoothing" in plain
        assert "--division-epsilon" in plain
        assert "--blur-dilation" in plain
        assert "--clean-plate" in plain
        assert "--save-refined-fg" in plain
        assert "--plate-search-rad" in plain  # may be truncated
        assert "--plate-alpha-thres" in plain  # may be truncated
        assert "--static-clean-plate" in plain

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_refine_motion_blur_flag_accepted(
        self, mock_run, mock_scan
    ) -> None:
        """--refine-motion-blur is accepted without error when no clips exist."""
        mock_scan.return_value = []

        result = _runner.invoke(
            app,
            [
                "run-inference",
                "--linear",
                "--despill", "5",
                "--despeckle",
                "--refiner", "1.0",
                "--refine-motion-blur",
            ],
        )
        assert result.exit_code == 0


class TestCLIMotionBlurValidation:
    """Test that invalid motion blur parameter values produce descriptive errors.

    **Validates: Requirements 7.2, 7.5**
    """

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_invalid_blur_threshold(self, mock_run, mock_scan) -> None:
        """--blur-threshold with a negative value reports a descriptive error."""
        mock_scan.return_value = []

        result = _runner.invoke(
            app,
            [
                "run-inference",
                "--linear",
                "--despill", "5",
                "--despeckle",
                "--refiner", "1.0",
                "--refine-motion-blur",
                "--blur-threshold", "-1",
            ],
        )
        assert result.exit_code != 0
        plain = _ANSI_ESCAPE.sub("", result.output)
        assert "blur_threshold" in plain

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_invalid_kernel_profile(self, mock_run, mock_scan) -> None:
        """--kernel-profile with an unknown value reports a descriptive error."""
        mock_scan.return_value = []

        result = _runner.invoke(
            app,
            [
                "run-inference",
                "--linear",
                "--despill", "5",
                "--despeckle",
                "--refiner", "1.0",
                "--refine-motion-blur",
                "--kernel-profile", "invalid_profile",
            ],
        )
        assert result.exit_code != 0
        plain = _ANSI_ESCAPE.sub("", result.output)
        assert "kernel_profile" in plain

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_invalid_temporal_smoothing(self, mock_run, mock_scan) -> None:
        """--temporal-smoothing outside (0.0, 1.0] reports a descriptive error."""
        mock_scan.return_value = []

        result = _runner.invoke(
            app,
            [
                "run-inference",
                "--linear",
                "--despill", "5",
                "--despeckle",
                "--refiner", "1.0",
                "--refine-motion-blur",
                "--temporal-smoothing", "0.0",
            ],
        )
        assert result.exit_code != 0
        plain = _ANSI_ESCAPE.sub("", result.output)
        assert "temporal_smoothing" in plain

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_invalid_division_epsilon(self, mock_run, mock_scan) -> None:
        """--division-epsilon <= 0 reports a descriptive error."""
        mock_scan.return_value = []

        result = _runner.invoke(
            app,
            [
                "run-inference",
                "--linear",
                "--despill", "5",
                "--despeckle",
                "--refiner", "1.0",
                "--refine-motion-blur",
                "--division-epsilon", "-0.001",
            ],
        )
        assert result.exit_code != 0
        plain = _ANSI_ESCAPE.sub("", result.output)
        assert "division_epsilon" in plain
