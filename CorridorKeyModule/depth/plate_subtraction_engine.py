"""Plate Subtraction Engine — two-pass orchestrator for plate-subtraction keying.

Pass 1: Compute optical flow + Bootstrap_Masks for all frames using the
existing ``ImageCubeBuilder`` + ``DepthThresholder``.

Pass 2: For each frame, synthesize a clean plate via ``CleanPlateSynthesizer``,
compute a subtraction-based alpha matte via ``SubtractionKeyer``, and refine
edges via ``MaskRefiner``.  An optional iterative refinement loop replaces the
Bootstrap_Mask with the previous pass's alpha matte and re-runs synthesis +
subtraction, converging toward a more accurate plate and matte.
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

from CorridorKeyModule.core.color_utils import (
    composite_straight,
    create_checkerboard,
    linear_to_srgb,
    srgb_to_linear,
)

from .clean_plate_synthesizer import CleanPlateSynthesizer
from .data_models import PlateSubtractionConfig
from .depth_thresholder import DepthThresholder
from .exr_io import write_depth_map, write_flow_field, write_rgb_exr
from .image_cube import ImageCubeBuilder
from .mask_refiner import MaskRefiner
from .optical_flow import OpticalFlowEngine
from .subtraction_keyer import SubtractionKeyer

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset(
    (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")
)


def _is_image_file(filename: str) -> bool:
    """Return True if *filename* has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTENSIONS


class PlateSubtractionEngine:
    """Two-pass orchestrator for plate-subtraction keying.

    Pass 1: Compute optical flow + Bootstrap_Masks for all frames.
    Pass 2: Synthesize clean plates + compute subtraction mattes.

    Parameters
    ----------
    config : PlateSubtractionConfig
        All pipeline parameters.
    device : str
        Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        config: PlateSubtractionConfig,
        device: str = "cpu",
    ) -> None:
        self.config = config

        self.flow_engine = OpticalFlowEngine(
            method=config.flow_method,
            device=device,
            consistency_threshold=config.consistency_threshold,
        )

        self.cube_builder = ImageCubeBuilder(
            buffer_size=config.cube_buffer_size,
            fusion_mode=config.fusion_mode,
            parallax_weight=config.parallax_weight,
            persistence_weight=config.persistence_weight,
            stability_weight=config.stability_weight,
        )

        self.thresholder = DepthThresholder(
            depth_threshold=config.depth_threshold,
            depth_falloff=config.depth_falloff,
        )

        self.synthesizer = CleanPlateSynthesizer(
            plate_search_radius=config.plate_search_radius,
            donor_threshold=config.donor_threshold,
        )

        self.keyer = SubtractionKeyer(
            difference_threshold=config.difference_threshold,
            difference_falloff=config.difference_falloff,
            low_confidence_alpha=config.low_confidence_alpha,
            color_space_mode=config.color_space_mode,
        )

        self.refiner = MaskRefiner(
            refinement_strength=config.refinement_strength,
        )

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
            If fewer than 3 input frames are found.
        """
        clip_path = Path(clip_dir)
        input_dir = clip_path / "Input"

        # --- Scan and sort input frames ---
        if not input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")

        frame_files = sorted(
            f for f in os.listdir(input_dir) if _is_image_file(f)
        )

        if len(frame_files) < 3:
            raise ValueError(
                f"Plate subtraction keying requires at least 3 input frames, "
                f"found {len(frame_files)} in {input_dir}"
            )

        total_frames = len(frame_files)

        # --- Load all frames ---
        frames = self._load_frames(input_dir, frame_files)
        h, w = frames[0].shape[:2]

        # --- Create output directories ---
        matte_dir = clip_path / "PlateMatte"
        comp_dir = clip_path / "Comp"
        matte_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)

        plate_dir: Path | None = None
        if self.config.save_clean_plates:
            plate_dir = clip_path / "CleanPlate"
            plate_dir.mkdir(parents=True, exist_ok=True)

        bootstrap_dir: Path | None = None
        if self.config.save_bootstrap:
            bootstrap_dir = clip_path / "Bootstrap"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)

        flow_dir: Path | None = None
        if self.config.save_flow:
            flow_dir = clip_path / "FlowField"
            flow_dir.mkdir(parents=True, exist_ok=True)

        # --- Pass 1: Bootstrap ---
        self.cube_builder.reset()
        flow_results = [None] * (total_frames - 1)
        bootstrap_masks: list[np.ndarray] = [
            np.zeros((h, w), dtype=np.float32)
        ]  # frame 0: all background

        for i in range(1, total_frames):
            flow_results[i - 1] = self.flow_engine.compute(
                frames[i - 1], frames[i]
            )
            cube_result = self.cube_builder.update(
                flow_results[i - 1], frame=frames[i]
            )
            bootstrap_masks.append(self.thresholder.apply(cube_result))

        # Optionally save flow fields
        if flow_dir is not None:
            for i, fr in enumerate(flow_results):
                if fr is not None:
                    stem = os.path.splitext(frame_files[i + 1])[0]
                    write_flow_field(
                        str(flow_dir / f"{stem}.exr"),
                        fr.forward_flow,
                    )

        # Optionally save bootstrap masks
        if bootstrap_dir is not None:
            for i, mask in enumerate(bootstrap_masks):
                stem = os.path.splitext(frame_files[i])[0]
                write_depth_map(
                    str(bootstrap_dir / f"{stem}.exr"), mask
                )

        # --- Pass 2: Plate Subtraction (with iterative refinement) ---
        current_masks = bootstrap_masks

        for iteration in range(self.config.max_iterations):
            prev_alphas: list[np.ndarray] = []
            is_last_iteration = (
                iteration == self.config.max_iterations - 1
            )

            for t in range(total_frames):
                try:
                    plate, confidence = self.synthesizer.synthesize(
                        t, frames, current_masks, flow_results
                    )
                    raw_alpha = self.keyer.compute(
                        frames[t], plate, confidence
                    )
                    refined_alpha = self.refiner.refine(
                        raw_alpha, frames[t]
                    )
                    prev_alphas.append(refined_alpha)

                    # Write outputs on last iteration
                    if is_last_iteration:
                        stem = os.path.splitext(frame_files[t])[0]
                        self._write_outputs(
                            stem,
                            refined_alpha,
                            frames[t],
                            plate,
                            matte_dir,
                            comp_dir,
                            plate_dir,
                        )

                    if on_frame_complete is not None:
                        on_frame_complete(t, total_frames)

                except Exception:
                    logger.exception(
                        "Error processing frame %d, skipping", t
                    )
                    prev_alphas.append(current_masks[t])
                    continue

            # Check convergence (need at least 2 iterations to compare)
            if iteration > 0:
                diffs = [
                    np.mean(np.abs(prev_alphas[i] - current_masks[i]))
                    for i in range(total_frames)
                ]
                mean_diff = float(np.mean(diffs))
                if mean_diff < self.config.convergence_threshold:
                    logger.info(
                        "Converged after %d iterations (mean diff %.6f)",
                        iteration + 1,
                        mean_diff,
                    )
                    # Write outputs if we haven't already (not last iteration)
                    if not is_last_iteration:
                        self._write_final_outputs(
                            prev_alphas,
                            frames,
                            frame_files,
                            current_masks,
                            flow_results,
                            matte_dir,
                            comp_dir,
                            plate_dir,
                        )
                    current_masks = prev_alphas
                    break

            current_masks = prev_alphas

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frames(
        input_dir: Path, frame_files: list[str]
    ) -> list[np.ndarray]:
        """Load all frames as [H, W, 3] float32 in [0, 1] linear color space."""
        frames: list[np.ndarray] = []
        for frame_file in frame_files:
            frame_path = str(input_dir / frame_file)
            ext = os.path.splitext(frame_file)[1].lower()

            raw = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")

            if ext == ".exr":
                # EXR: already float32, read directly
                frame = raw.astype(np.float32)
            else:
                # PNG/JPG/TIFF: uint8 → float32 / 255.0 → srgb_to_linear
                if raw.dtype == np.uint8:
                    frame = raw.astype(np.float32) / 255.0
                else:
                    frame = raw.astype(np.float32)
                frame = srgb_to_linear(frame).astype(np.float32)

            # BGR → RGB (OpenCV reads as BGR)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ensure 3-channel
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            elif frame.shape[2] > 3:
                frame = frame[:, :, :3]

            frames.append(frame)
        return frames

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------

    def _write_final_outputs(
        self,
        alphas: list[np.ndarray],
        frames: list[np.ndarray],
        frame_files: list[str],
        current_masks: list[np.ndarray],
        flow_results: list,
        matte_dir: Path,
        comp_dir: Path,
        plate_dir: Path | None,
    ) -> None:
        """Write outputs for all frames after convergence."""
        for t in range(len(frames)):
            try:
                plate, _ = self.synthesizer.synthesize(
                    t, frames, current_masks, flow_results
                )
                stem = os.path.splitext(frame_files[t])[0]
                self._write_outputs(
                    stem,
                    alphas[t],
                    frames[t],
                    plate,
                    matte_dir,
                    comp_dir,
                    plate_dir,
                )
            except Exception:
                logger.exception(
                    "Error writing outputs for frame %d", t
                )

    @staticmethod
    def _write_outputs(
        stem: str,
        alpha: np.ndarray,
        frame: np.ndarray,
        plate: np.ndarray,
        matte_dir: Path,
        comp_dir: Path,
        plate_dir: Path | None,
    ) -> None:
        """Write matte EXR, comp PNG, and optionally clean plate EXR."""
        # PlateMatte: EXR alpha matte
        write_depth_map(str(matte_dir / f"{stem}.exr"), alpha)

        # Comp: PNG preview composite
        h, w = frame.shape[:2]
        checkerboard = create_checkerboard(w, h)
        alpha_3ch = alpha[..., np.newaxis]
        comp = composite_straight(frame, checkerboard, alpha_3ch)
        comp_srgb = linear_to_srgb(comp)
        comp_uint8 = np.clip(comp_srgb * 255.0, 0, 255).astype(np.uint8)
        comp_bgr = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(comp_dir / f"{stem}.png"), comp_bgr)

        # CleanPlate: optional EXR
        if plate_dir is not None:
            write_rgb_exr(str(plate_dir / f"{stem}.exr"), plate)
