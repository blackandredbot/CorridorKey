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
