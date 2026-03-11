"""Depth Keying Engine — top-level orchestrator for the depth keying pipeline.

Chains Optical_Flow_Engine → Image_Cube_Builder → Depth_Thresholder →
Mask_Refiner into a complete keying pipeline.  Reads frames from
``clip_dir/Input/``, writes refined alpha mattes to ``clip_dir/DepthMatte/``,
and optionally writes intermediate depth maps, flow fields, and comp previews.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

# Enable OpenEXR codec in OpenCV before any cv2 usage
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from device_utils import resolve_device
from CorridorKeyModule.core.color_utils import (
    create_checkerboard,
    composite_straight,
    linear_to_srgb,
)

from .depth_thresholder import DepthThresholder
from .exr_io import read_depth_map, read_flow_field, write_depth_map, write_flow_field
from .image_cube import ImageCubeBuilder
from .mask_refiner import MaskRefiner
from .neural_fallback import NeuralDepthFallback
from .optical_flow import OpticalFlowEngine

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset(
    (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp")
)


def _is_image_file(filename: str) -> bool:
    """Return True if *filename* has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTENSIONS


class DepthKeyingEngine:
    """Top-level orchestrator for the depth keying pipeline.

    Parameters
    ----------
    device : str
        Device string (``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``).
        Resolved via ``device_utils.resolve_device``, respecting the
        ``CORRIDORKEY_DEVICE`` env var and ``--device`` CLI flag.
    flow_method : str
        Optical flow algorithm: ``"farneback"`` or ``"raft"``.
    depth_threshold : float
        Background_Score cutoff in [0.0, 1.0].
    depth_falloff : float
        Soft transition zone width in [0.0, 0.5].
    cube_buffer_size : int
        Rolling buffer depth for the Image Cube Builder (>= 2).
    refinement_strength : float
        Mask refinement strength in [0.0, 1.0].
    low_confidence_alpha : float
        Alpha bias for low-confidence pixels in [0.0, 1.0].
    consistency_threshold : float
        Forward-backward consistency threshold for occlusion detection (> 0).
    parallax_weight : float
        Weight for the parallax channel in [0.0, 1.0].
    persistence_weight : float
        Weight for the persistence channel in [0.0, 1.0].
    stability_weight : float
        Weight for the positional stability channel in [0.0, 1.0].
    fusion_mode : str
        Signal fusion mode: ``"blend"``, ``"max"``, or ``"min"``.
    depth_fallback : bool
        Enable optional neural depth fallback.
    fallback_confidence_threshold : float
        Mean confidence threshold for neural fallback activation.
    save_depth_maps : bool
        Write Background_Score, confidence, and channel maps to ``DepthMap/``.
    save_flow : bool
        Write flow fields to ``FlowField/``.
    """

    def __init__(
        self,
        device: str = "cpu",
        flow_method: str = "farneback",
        depth_threshold: float = 0.5,
        depth_falloff: float = 0.05,
        cube_buffer_size: int = 10,
        refinement_strength: float = 1.0,
        low_confidence_alpha: float = 0.0,
        consistency_threshold: float = 1.5,
        parallax_weight: float = 0.4,
        persistence_weight: float = 0.3,
        stability_weight: float = 0.3,
        fusion_mode: str = "blend",
        depth_fallback: bool = False,
        fallback_confidence_threshold: float = 0.3,
        save_depth_maps: bool = False,
        save_flow: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.save_depth_maps = save_depth_maps
        self.save_flow = save_flow
        self.depth_fallback = depth_fallback
        self.fallback_confidence_threshold = fallback_confidence_threshold

        # Instantiate pipeline components
        self.flow_engine = OpticalFlowEngine(
            method=flow_method,
            device=self.device,
            consistency_threshold=consistency_threshold,
        )

        self.cube_builder = ImageCubeBuilder(
            buffer_size=cube_buffer_size,
            parallax_weight=parallax_weight,
            persistence_weight=persistence_weight,
            stability_weight=stability_weight,
            fusion_mode=fusion_mode,
        )

        self.thresholder = DepthThresholder(
            depth_threshold=depth_threshold,
            depth_falloff=depth_falloff,
            low_confidence_alpha=low_confidence_alpha,
        )

        self.refiner = MaskRefiner(
            refinement_strength=refinement_strength,
        )

        # Optional neural depth fallback (only instantiated when enabled)
        self._neural_fallback: NeuralDepthFallback | None = None
        if self.depth_fallback:
            self._neural_fallback = NeuralDepthFallback(device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_clip(
        self,
        clip_dir: str,
        *,
        on_frame_complete: Callable[[int, int], None] | None = None,
    ) -> None:
        """Process all frames in ``clip_dir/Input/`` and write results.

        Parameters
        ----------
        clip_dir : str
            Path to the clip directory containing an ``Input/`` subfolder.
        on_frame_complete : callable, optional
            Progress callback ``(frame_index, total_frames) -> None``.

        Raises
        ------
        ValueError
            If fewer than 2 input frames are found.
        """
        clip_path = Path(clip_dir)
        input_dir = clip_path / "Input"

        # --- Scan and sort input frames ---
        if not input_dir.is_dir():
            raise ValueError(
                f"Input directory not found: {input_dir}"
            )

        frame_files = sorted(
            f for f in os.listdir(input_dir) if _is_image_file(f)
        )

        if len(frame_files) < 2:
            raise ValueError(
                f"Depth keying requires at least 2 input frames, "
                f"found {len(frame_files)} in {input_dir}"
            )

        total_frames = len(frame_files)

        # --- Create output directories ---
        matte_dir = clip_path / "DepthMatte"
        comp_dir = clip_path / "Comp"
        matte_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)

        depth_map_dir: Path | None = None
        if self.save_depth_maps:
            depth_map_dir = clip_path / "DepthMap"
            depth_map_dir.mkdir(parents=True, exist_ok=True)

        flow_dir: Path | None = None
        if self.save_flow:
            flow_dir = clip_path / "FlowField"
            flow_dir.mkdir(parents=True, exist_ok=True)

        # --- Reset cube builder for fresh clip ---
        self.cube_builder.reset()

        # --- Process consecutive frame pairs ---
        prev_frame: np.ndarray | None = None
        prev_file: str | None = None

        for idx, frame_file in enumerate(frame_files):
            try:
                frame_path = str(input_dir / frame_file)
                stem = os.path.splitext(frame_file)[0]

                # Read frame as float32 RGB [0, 1]
                raw = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    logger.error(
                        "Failed to read frame %d: %s", idx, frame_path
                    )
                    continue

                # Convert to float32 [0, 1] RGB
                if raw.dtype == np.uint8:
                    frame = raw.astype(np.float32) / 255.0
                else:
                    frame = raw.astype(np.float32)

                # BGR → RGB (OpenCV reads as BGR)
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if prev_frame is None:
                    # First frame — need a pair to compute flow
                    prev_frame = frame
                    prev_file = frame_file
                    # Still call callback for first frame
                    if on_frame_complete is not None:
                        on_frame_complete(idx, total_frames)
                    continue

                # --- Compute optical flow ---
                flow_result = self.flow_engine.compute(prev_frame, frame)

                # --- Update image cube ---
                emit_channels = self.save_depth_maps
                cube_result = self.cube_builder.update(
                    flow_result,
                    emit_channel_maps=emit_channels,
                    frame=frame,
                )

                # --- Neural fallback activation ---
                if self.depth_fallback and self._neural_fallback is not None:
                    mean_confidence = float(np.mean(cube_result.confidence_map))
                    if mean_confidence < self.fallback_confidence_threshold:
                        logger.info(
                            "Neural fallback activated: mean confidence %.3f < threshold %.3f",
                            mean_confidence,
                            self.fallback_confidence_threshold,
                        )
                        cube_result = self._neural_fallback.estimate(frame)

                # --- Apply threshold → raw alpha ---
                raw_alpha = self.thresholder.apply(cube_result)

                # --- Refine matte ---
                refined_alpha = self.refiner.refine(raw_alpha, frame)

                # --- Write refined alpha matte as 32-bit float EXR ---
                matte_path = str(matte_dir / f"{stem}.exr")
                write_depth_map(matte_path, refined_alpha)

                # --- Optionally write depth maps ---
                if depth_map_dir is not None:
                    # Background_Score map
                    bg_path = str(depth_map_dir / f"{stem}_background_score.exr")
                    write_depth_map(bg_path, cube_result.background_score)

                    # Confidence map
                    conf_path = str(depth_map_dir / f"{stem}_confidence.exr")
                    write_depth_map(conf_path, cube_result.confidence_map)

                    # Individual channel maps (if available)
                    if cube_result.parallax_map is not None:
                        par_path = str(depth_map_dir / f"{stem}_parallax.exr")
                        write_depth_map(par_path, cube_result.parallax_map)
                    if cube_result.persistence_map is not None:
                        per_path = str(depth_map_dir / f"{stem}_persistence.exr")
                        write_depth_map(per_path, cube_result.persistence_map)
                    if cube_result.stability_map is not None:
                        stb_path = str(depth_map_dir / f"{stem}_stability.exr")
                        write_depth_map(stb_path, cube_result.stability_map)

                # --- Optionally write flow fields ---
                if flow_dir is not None:
                    flow_path = str(flow_dir / f"{stem}.exr")
                    write_flow_field(flow_path, flow_result.forward_flow)

                # --- Generate comp preview PNG ---
                h, w = frame.shape[:2]
                checkerboard = create_checkerboard(w, h)
                # Expand alpha to [H, W, 1] for compositing
                alpha_3ch = refined_alpha[..., np.newaxis]
                comp = composite_straight(frame, checkerboard, alpha_3ch)
                # Linear → sRGB → uint8 PNG
                comp_srgb = linear_to_srgb(comp)
                comp_uint8 = np.clip(comp_srgb * 255.0, 0, 255).astype(np.uint8)
                comp_bgr = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2BGR)
                comp_path = str(comp_dir / f"{stem}.png")
                cv2.imwrite(comp_path, comp_bgr)

                # --- Update previous frame ---
                prev_frame = frame
                prev_file = frame_file

            except Exception:
                logger.exception(
                    "Error processing frame %d (%s), skipping.",
                    idx,
                    frame_file,
                )
                continue

            # --- Progress callback ---
            if on_frame_complete is not None:
                on_frame_complete(idx, total_frames)
