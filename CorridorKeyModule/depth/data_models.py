"""Data models for the depth keying pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowResult:
    """Output of optical flow computation between two frames.

    Attributes:
        forward_flow: [H, W, 2] float32 — (dx, dy) in pixel units.
        backward_flow: [H, W, 2] float32 — (dx, dy) in pixel units.
        occlusion_mask: [H, W] float32 — 1.0 = occluded, 0.0 = visible.
    """

    forward_flow: np.ndarray
    backward_flow: np.ndarray
    occlusion_mask: np.ndarray


@dataclass
class CubeResult:
    """Output of the Image_Cube_Builder three-signal fusion.

    Attributes:
        background_score: [H, W] float32 — [0.0, 1.0], high = background.
        confidence_map: [H, W] float32 — [0.0, 1.0].
        parallax_map: [H, W] float32 — [0.0, 1.0], optional individual channel.
        persistence_map: [H, W] float32 — [0.0, 1.0], optional individual channel.
        stability_map: [H, W] float32 — [0.0, 1.0], optional individual channel.
    """

    background_score: np.ndarray
    confidence_map: np.ndarray
    parallax_map: np.ndarray | None = None
    persistence_map: np.ndarray | None = None
    stability_map: np.ndarray | None = None


_VALID_FLOW_METHODS = {"farneback", "raft"}
_VALID_FUSION_MODES = {"blend", "max", "min"}


@dataclass
class DepthKeyingConfig:
    """All user-configurable parameters for the depth keying pipeline.

    Validation is performed in ``__post_init__`` — construction with
    out-of-range values raises ``ValueError`` immediately.
    """

    flow_method: str = "farneback"
    depth_threshold: float = 0.5
    depth_falloff: float = 0.05
    cube_buffer_size: int = 10
    refinement_strength: float = 1.0
    low_confidence_alpha: float = 0.0
    consistency_threshold: float = 1.5
    despeckle_size: int = 50
    hole_fill_size: int = 50
    parallax_weight: float = 0.4
    persistence_weight: float = 0.3
    stability_weight: float = 0.3
    fusion_mode: str = "blend"
    depth_fallback: bool = False
    fallback_confidence_threshold: float = 0.3
    save_depth_maps: bool = False
    save_flow: bool = False

    def __post_init__(self) -> None:  # noqa: C901
        if self.flow_method not in _VALID_FLOW_METHODS:
            raise ValueError(
                f"flow_method must be one of {_VALID_FLOW_METHODS}, "
                f"got {self.flow_method!r}"
            )

        if not 0.0 <= self.depth_threshold <= 1.0:
            raise ValueError(
                f"depth_threshold must be in [0.0, 1.0], got {self.depth_threshold}"
            )

        if not 0.0 <= self.depth_falloff <= 0.5:
            raise ValueError(
                f"depth_falloff must be in [0.0, 0.5], got {self.depth_falloff}"
            )

        if self.cube_buffer_size < 2:
            raise ValueError(
                f"cube_buffer_size must be >= 2, got {self.cube_buffer_size}"
            )

        if not 0.0 <= self.refinement_strength <= 1.0:
            raise ValueError(
                f"refinement_strength must be in [0.0, 1.0], "
                f"got {self.refinement_strength}"
            )

        if not 0.0 <= self.low_confidence_alpha <= 1.0:
            raise ValueError(
                f"low_confidence_alpha must be in [0.0, 1.0], "
                f"got {self.low_confidence_alpha}"
            )

        if self.consistency_threshold <= 0:
            raise ValueError(
                f"consistency_threshold must be > 0, "
                f"got {self.consistency_threshold}"
            )

        if not 0.0 <= self.parallax_weight <= 1.0:
            raise ValueError(
                f"parallax_weight must be in [0.0, 1.0], "
                f"got {self.parallax_weight}"
            )

        if not 0.0 <= self.persistence_weight <= 1.0:
            raise ValueError(
                f"persistence_weight must be in [0.0, 1.0], "
                f"got {self.persistence_weight}"
            )

        if not 0.0 <= self.stability_weight <= 1.0:
            raise ValueError(
                f"stability_weight must be in [0.0, 1.0], "
                f"got {self.stability_weight}"
            )

        weight_sum = self.parallax_weight + self.persistence_weight + self.stability_weight
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"parallax_weight + persistence_weight + stability_weight must "
                f"equal 1.0, got {weight_sum}"
            )

        if self.fusion_mode not in _VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {_VALID_FUSION_MODES}, "
                f"got {self.fusion_mode!r}"
            )

        if not 0.0 <= self.fallback_confidence_threshold <= 1.0:
            raise ValueError(
                f"fallback_confidence_threshold must be in [0.0, 1.0], "
                f"got {self.fallback_confidence_threshold}"
            )

_VALID_KERNEL_PROFILES = {"linear", "cosine", "gaussian"}


@dataclass
class MotionBlurConfig:
    """All user-configurable parameters for the motion blur refinement pipeline.

    Validation is performed in ``__post_init__`` — construction with
    out-of-range values raises ``ValueError`` immediately.
    """

    blur_threshold: float = 2.0
    kernel_profile: str = "linear"
    temporal_smoothing: float = 0.3
    division_epsilon: float = 1e-4
    blur_dilation: int = 3
    save_refined_fg: bool = False
    plate_search_radius: int = 10
    plate_alpha_threshold: float = 0.1
    static_clean_plate: bool = False

    def __post_init__(self) -> None:
        if self.blur_threshold <= 0:
            raise ValueError(
                f"blur_threshold must be > 0, got {self.blur_threshold}"
            )
        if self.kernel_profile not in _VALID_KERNEL_PROFILES:
            raise ValueError(
                f"kernel_profile must be one of {_VALID_KERNEL_PROFILES}, "
                f"got {self.kernel_profile!r}"
            )
        if not 0.0 < self.temporal_smoothing <= 1.0:
            raise ValueError(
                f"temporal_smoothing must be in (0.0, 1.0], got {self.temporal_smoothing}"
            )
        if self.division_epsilon <= 0:
            raise ValueError(
                f"division_epsilon must be > 0, got {self.division_epsilon}"
            )
        if self.blur_dilation < 0:
            raise ValueError(
                f"blur_dilation must be >= 0, got {self.blur_dilation}"
            )
        if self.plate_search_radius < 1:
            raise ValueError(
                f"plate_search_radius must be >= 1, got {self.plate_search_radius}"
            )
        if not 0.0 < self.plate_alpha_threshold <= 1.0:
            raise ValueError(
                f"plate_alpha_threshold must be in (0.0, 1.0], got {self.plate_alpha_threshold}"
            )


_VALID_COLOR_SPACE_MODES = {"max_channel", "luminance"}


@dataclass
class PlateSubtractionConfig:
    """All user-configurable parameters for the plate-subtraction keying pipeline.

    Validation is performed in ``__post_init__`` — construction with
    out-of-range values raises ``ValueError`` immediately.
    """

    # --- Subtraction keyer parameters ---
    difference_threshold: float = 0.05
    difference_falloff: float = 0.03
    color_space_mode: str = "max_channel"
    low_confidence_alpha: float = 1.0

    # --- Clean plate synthesis parameters ---
    plate_search_radius: int = 15
    donor_threshold: float = 0.3

    # --- Iterative refinement ---
    max_iterations: int = 2
    convergence_threshold: float = 0.001

    # --- Bootstrap parameters (passed through to ImageCubeBuilder / DepthThresholder) ---
    flow_method: str = "farneback"
    cube_buffer_size: int = 10
    fusion_mode: str = "blend"
    parallax_weight: float = 0.4
    persistence_weight: float = 0.3
    stability_weight: float = 0.3
    depth_threshold: float = 0.5
    depth_falloff: float = 0.05
    consistency_threshold: float = 1.5

    # --- Refinement ---
    refinement_strength: float = 1.0

    # --- Output options ---
    save_clean_plates: bool = False
    save_bootstrap: bool = False
    save_flow: bool = False

    def __post_init__(self) -> None:  # noqa: C901
        if not 0.0 < self.difference_threshold <= 1.0:
            raise ValueError(
                f"difference_threshold must be in (0.0, 1.0], "
                f"got {self.difference_threshold}"
            )
        if not 0.0 <= self.difference_falloff <= 0.5:
            raise ValueError(
                f"difference_falloff must be in [0.0, 0.5], "
                f"got {self.difference_falloff}"
            )
        if self.color_space_mode not in _VALID_COLOR_SPACE_MODES:
            raise ValueError(
                f"color_space_mode must be one of {_VALID_COLOR_SPACE_MODES}, "
                f"got {self.color_space_mode!r}"
            )
        if not 0.0 <= self.low_confidence_alpha <= 1.0:
            raise ValueError(
                f"low_confidence_alpha must be in [0.0, 1.0], "
                f"got {self.low_confidence_alpha}"
            )
        if self.plate_search_radius < 1:
            raise ValueError(
                f"plate_search_radius must be >= 1, "
                f"got {self.plate_search_radius}"
            )
        if not 0.0 < self.donor_threshold <= 1.0:
            raise ValueError(
                f"donor_threshold must be in (0.0, 1.0], "
                f"got {self.donor_threshold}"
            )
        if not 1 <= self.max_iterations <= 5:
            raise ValueError(
                f"max_iterations must be in [1, 5], "
                f"got {self.max_iterations}"
            )
        if self.convergence_threshold <= 0:
            raise ValueError(
                f"convergence_threshold must be > 0, "
                f"got {self.convergence_threshold}"
            )
        # Bootstrap parameter validation (same rules as DepthKeyingConfig)
        if self.flow_method not in _VALID_FLOW_METHODS:
            raise ValueError(
                f"flow_method must be one of {_VALID_FLOW_METHODS}, "
                f"got {self.flow_method!r}"
            )
        if self.cube_buffer_size < 2:
            raise ValueError(
                f"cube_buffer_size must be >= 2, got {self.cube_buffer_size}"
            )
        if self.fusion_mode not in _VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {_VALID_FUSION_MODES}, "
                f"got {self.fusion_mode!r}"
            )
        weight_sum = self.parallax_weight + self.persistence_weight + self.stability_weight
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"parallax_weight + persistence_weight + stability_weight "
                f"must equal 1.0, got {weight_sum}"
            )
        if not 0.0 <= self.refinement_strength <= 1.0:
            raise ValueError(
                f"refinement_strength must be in [0.0, 1.0], "
                f"got {self.refinement_strength}"
            )

