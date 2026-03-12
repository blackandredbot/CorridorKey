"""CLI integration tests for --mode plate-subtraction.

Validates: Requirements 8.1, 8.2, 8.5
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from corridorkey_cli import app

runner = CliRunner()

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _make_clip(name: str = "test_clip", root: str = "/tmp/fake_clip") -> MagicMock:
    clip = MagicMock()
    clip.name = name
    clip.root_path = root
    return clip


# ---------------------------------------------------------------------------
# Flag acceptance (Requirement 8.1)
# ---------------------------------------------------------------------------


class TestModeAcceptance:
    """--mode plate-subtraction is recognized and routes correctly."""

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_mode_plate_subtraction_accepted(self, mock_scan, mock_engine_cls):
        """CLI accepts --mode plate-subtraction without error."""
        mock_scan.return_value = [_make_clip()]
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["run-inference", "--mode", "plate-subtraction"])
        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        mock_engine_cls.assert_called_once()
        mock_engine.process_clip.assert_called_once()

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_mode_plate_subtraction_constructs_engine(self, mock_scan, mock_engine_cls):
        """Engine is constructed with a PlateSubtractionConfig and device."""
        mock_scan.return_value = [_make_clip()]
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["run-inference", "--mode", "plate-subtraction"])
        assert result.exit_code == 0

        _, kwargs = mock_engine_cls.call_args
        assert "config" in kwargs
        assert "device" in kwargs


# ---------------------------------------------------------------------------
# Parameter forwarding (Requirement 8.2)
# ---------------------------------------------------------------------------


class TestParameterForwarding:
    """CLI flags are forwarded to PlateSubtractionConfig correctly."""

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_default_parameters(self, mock_scan, mock_engine_cls):
        """Default CLI values produce a valid config with expected defaults."""
        mock_scan.return_value = [_make_clip()]
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["run-inference", "--mode", "plate-subtraction"])
        assert result.exit_code == 0

        _, kwargs = mock_engine_cls.call_args
        config = kwargs["config"]
        assert config.difference_threshold == 0.05
        assert config.difference_falloff == 0.03
        assert config.color_space_mode == "max_channel"
        assert config.low_confidence_alpha == 1.0
        assert config.donor_threshold == 0.3
        assert config.max_iterations == 2
        assert config.convergence_threshold == 0.001

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_difference_threshold(self, mock_scan, mock_engine_cls):
        """--difference-threshold is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--difference-threshold", "0.1"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.difference_threshold == 0.1

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_difference_falloff(self, mock_scan, mock_engine_cls):
        """--difference-falloff is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--difference-falloff", "0.1"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.difference_falloff == 0.1

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_color_space_mode(self, mock_scan, mock_engine_cls):
        """--color-space-mode luminance is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--color-space-mode", "luminance"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.color_space_mode == "luminance"

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_donor_threshold(self, mock_scan, mock_engine_cls):
        """--donor-threshold is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--donor-threshold", "0.5"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.donor_threshold == 0.5

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_max_iterations(self, mock_scan, mock_engine_cls):
        """--max-iterations is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--max-iterations", "3"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.max_iterations == 3

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_convergence_threshold(self, mock_scan, mock_engine_cls):
        """--convergence-threshold is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--convergence-threshold", "0.01"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.convergence_threshold == 0.01

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_plate_search_radius(self, mock_scan, mock_engine_cls):
        """--plate-search-radius is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--plate-search-radius", "20"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.plate_search_radius == 20

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_custom_low_confidence_alpha(self, mock_scan, mock_engine_cls):
        """--low-confidence-alpha is forwarded to config."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--low-confidence-alpha", "0.5"],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.low_confidence_alpha == 0.5

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_save_flags_forwarded(self, mock_scan, mock_engine_cls):
        """--save-clean-plates and --save-bootstrap are forwarded."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            [
                "run-inference",
                "--mode", "plate-subtraction",
                "--save-clean-plates",
                "--save-bootstrap",
                "--save-flow",
            ],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.save_clean_plates is True
        assert config.save_bootstrap is True
        assert config.save_flow is True

    @patch("corridorkey_cli.PlateSubtractionEngine")
    @patch("corridorkey_cli.scan_clips")
    def test_shared_flags_forwarded(self, mock_scan, mock_engine_cls):
        """Shared flags like --flow-method and --refinement-strength are forwarded."""
        mock_scan.return_value = [_make_clip()]
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(
            app,
            [
                "run-inference",
                "--mode", "plate-subtraction",
                "--flow-method", "farneback",
                "--refinement-strength", "0.8",
            ],
        )
        assert result.exit_code == 0

        config = mock_engine_cls.call_args[1]["config"]
        assert config.flow_method == "farneback"
        assert config.refinement_strength == 0.8


# ---------------------------------------------------------------------------
# Out-of-range parameter rejection (Requirement 8.5)
# ---------------------------------------------------------------------------


class TestOutOfRangeRejection:
    """Invalid parameter values produce descriptive errors."""

    @patch("corridorkey_cli.scan_clips")
    def test_difference_threshold_too_high(self, mock_scan):
        """--difference-threshold 2.0 is rejected."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--difference-threshold", "2.0"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "difference_threshold" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_difference_threshold_zero(self, mock_scan):
        """--difference-threshold 0.0 is rejected (must be > 0)."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--difference-threshold", "0.0"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "difference_threshold" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_difference_falloff_too_high(self, mock_scan):
        """--difference-falloff 0.6 is rejected (max 0.5)."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--difference-falloff", "0.6"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "difference_falloff" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_max_iterations_too_high(self, mock_scan):
        """--max-iterations 10 is rejected (max 5)."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--max-iterations", "10"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "max_iterations" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_max_iterations_zero(self, mock_scan):
        """--max-iterations 0 is rejected (min 1)."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--max-iterations", "0"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "max_iterations" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_donor_threshold_too_high(self, mock_scan):
        """--donor-threshold 1.5 is rejected."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--donor-threshold", "1.5"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "donor_threshold" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_donor_threshold_zero(self, mock_scan):
        """--donor-threshold 0.0 is rejected (must be > 0)."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--donor-threshold", "0.0"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "donor_threshold" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_convergence_threshold_zero(self, mock_scan):
        """--convergence-threshold 0 is rejected (must be > 0)."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--convergence-threshold", "0"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "convergence_threshold" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_invalid_color_space_mode(self, mock_scan):
        """--color-space-mode invalid is rejected."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--color-space-mode", "invalid"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "color_space_mode" in plain

    @patch("corridorkey_cli.scan_clips")
    def test_low_confidence_alpha_too_high(self, mock_scan):
        """--low-confidence-alpha 1.5 is rejected."""
        mock_scan.return_value = [_make_clip()]

        result = runner.invoke(
            app,
            ["run-inference", "--mode", "plate-subtraction", "--low-confidence-alpha", "1.5"],
        )
        assert result.exit_code == 1
        plain = _strip_ansi(result.output)
        assert "low_confidence_alpha" in plain
