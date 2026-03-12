"""Motion Blur Refiner — top-level orchestrator for motion blur alpha refinement.

Chains Blur_Region_Detector → Clean_Plate_Provider → Alpha_Solver →
Temporal_Coherence_Filter into a complete post-processing pipeline.
Reads frames from ``clip_dir/Input/``, reads alpha from ``Alpha/`` or
``DepthMatte/``, and writes refined mattes to ``RefinedMatte/``,
optional foreground to ``RefinedFG/``, and comp previews to ``Comp/``.

Follows the same orchestration pattern as ``DepthKeyingEngine.process_clip()``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

# Enable OpenEXR codec in OpenCV before any cv2 usage.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from CorridorKeyModule.core.color_utils import (
    composite_straight,
    create_checkerboard,
    linear_to_srgb,
    srgb_to_linear,
)
from CorridorKeyModule.depth.alpha_solver import AlphaSolver
from CorridorKeyModule.depth.blur_region_detector import BlurRegionDetector
from CorridorKeyModule.depth.clean_plate_provider import CleanPlateProvider
from CorridorKeyModule.depth.data_models import MotionBlurConfig
from CorridorKeyModule.depth.exr_io import read_depth_map, read_flow_field, write_depth_map
from CorridorKeyModule.depth.optical_flow import OpticalFlowEngine
from CorridorKeyModule.depth.temporal_coherence import TemporalCoherenceFilter

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset(
    (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")
)

# EXR write flags for 3-channel float output.
_EXR_FLOAT32_FLAGS = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]


def _is_image_file(filename: str) -> bool:
    """Return True if *filename* has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTENSIONS


class MotionBlurRefiner:
    """Top-level orchestrator for motion blur alpha refinement.

    Follows the same pattern as ``DepthKeyingEngine.process_clip()``.

    Parameters
    ----------
    config : MotionBlurConfig
        All user-configurable parameters for the pipeline.
    device : str
        Device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(self, config: MotionBlurConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device

        # Instantiate pipeline components from config.
        self.blur_detector = BlurRegionDetector(
            blur_threshold=config.blur_threshold,
            dilation_radius=config.blur_dilation,
        )
        self.clean_plate_provider = CleanPlateProvider(
            search_radius=config.plate_search_radius,
            alpha_threshold=config.plate_alpha_threshold,
        )
        self.alpha_solver = AlphaSolver(
            division_epsilon=config.division_epsilon,
            kernel_profile=config.kernel_profile,
        )
        self.temporal_filter = TemporalCoherenceFilter(
            temporal_smoothing=config.temporal_smoothing,
        )

        # Flow engine for computing flow when FlowField/ is absent.
        self.flow_engine = OpticalFlowEngine(method="farneback", device=device)

    def process_clip(
        self,
        clip_dir: str,
        *,
        clean_plate_path: str | None = None,
        save_refined_fg: bool = False,
        on_frame_complete: Callable[[int, int], None] | None = None,
    ) -> None:
        """Process all frames in a clip directory.

        Parameters
        ----------
        clip_dir : str
            Path to clip directory with ``Input/``, ``Alpha/`` or ``DepthMatte/``.
        clean_plate_path : str or None
            Optional explicit clean plate file.
        save_refined_fg : bool
            Write recovered foreground colour to ``RefinedFG/``.
        on_frame_complete : callable, optional
            Progress callback ``(frame_index, total_frames) -> None``.

        Raises
        ------
        ValueError
            If no input frames or no alpha mattes are found.
        """
        clip_path = Path(clip_dir)
        input_dir = clip_path / "Input"

        # --- Scan and sort input frames ---
        if not input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")

        frame_files = sorted(
            f for f in os.listdir(input_dir) if _is_image_file(f)
        )
        if len(frame_files) < 1:
            raise ValueError(
                f"Motion blur refinement requires at least 1 input frame, "
                f"found 0 in {input_dir}"
            )

        # --- Locate alpha directory ---
        alpha_dir = clip_path / "Alpha"
        if not alpha_dir.is_dir():
            alpha_dir = clip_path / "DepthMatte"
        if not alpha_dir.is_dir():
            raise ValueError(
                f"No Alpha/ or DepthMatte/ directory found in {clip_dir}"
            )

        # Validate at least 1 alpha matte exists.
        alpha_files = sorted(os.listdir(alpha_dir))
        if len(alpha_files) < 1:
            raise ValueError(
                f"Motion blur refinement requires at least 1 alpha matte, "
                f"found 0 in {alpha_dir}"
            )

        total_frames = len(frame_files)

        # --- Create output directories ---
        matte_dir = clip_path / "RefinedMatte"
        comp_dir = clip_path / "Comp"
        matte_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)

        fg_dir: Path | None = None
        if save_refined_fg:
            fg_dir = clip_path / "RefinedFG"
            fg_dir.mkdir(parents=True, exist_ok=True)

        # --- Check for pre-computed flow fields ---
        flow_dir = clip_path / "FlowField"
        has_flow_dir = flow_dir.is_dir()

        # --- Reset temporal filter for fresh clip ---
        self.temporal_filter.reset()

        # --- Pre-load all frames and alphas for clean plate synthesis ---
        all_frames: list[np.ndarray] = []
        all_alphas: list[np.ndarray] = []
        all_flows: list[np.ndarray | None] = []

        for frame_file in frame_files:
            frame_path = str(input_dir / frame_file)
            stem = os.path.splitext(frame_file)[0]

            # Read frame.
            raw = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                all_frames.append(np.zeros((1, 1, 3), dtype=np.float32))
                all_alphas.append(np.zeros((1, 1), dtype=np.float32))
                continue

            # Convert to float32 [0, 1] RGB.
            if raw.dtype == np.uint8:
                frame = raw.astype(np.float32) / 255.0
            else:
                frame = raw.astype(np.float32)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert sRGB input to linear.
            frame_linear = srgb_to_linear(frame).astype(np.float32)
            all_frames.append(frame_linear)

            # Read alpha matte.
            alpha_path = str(alpha_dir / f"{stem}.exr")
            if os.path.isfile(alpha_path):
                alpha = read_depth_map(alpha_path)
            else:
                alpha = np.zeros(frame_linear.shape[:2], dtype=np.float32)
            all_alphas.append(alpha)

        # --- Pre-compute or load flow fields ---
        for i in range(len(frame_files) - 1):
            stem = os.path.splitext(frame_files[i])[0]
            flow_path = str(flow_dir / f"{stem}.exr") if has_flow_dir else ""

            if has_flow_dir and os.path.isfile(flow_path):
                try:
                    flow = read_flow_field(flow_path)
                    all_flows.append(flow)
                    continue
                except Exception:
                    pass

            # Compute flow between consecutive frames.
            try:
                flow_result = self.flow_engine.compute(
                    all_frames[i], all_frames[i + 1]
                )
                all_flows.append(flow_result.forward_flow)
            except Exception:
                all_flows.append(None)

        # --- Synthesize static clean plate once if needed ---
        static_plate: np.ndarray | None = None
        if clean_plate_path is None and self.config.static_clean_plate:
            static_plate = self.clean_plate_provider.synthesize_static(
                all_frames, all_alphas
            )

        # --- Process each frame ---
        for idx in range(total_frames):
            frame_file = frame_files[idx]
            stem = os.path.splitext(frame_file)[0]

            try:
                frame_linear = all_frames[idx]
                alpha = all_alphas[idx]
                h, w = alpha.shape

                # Get flow field for this frame (use forward flow from idx-1→idx
                # or idx→idx+1).
                if idx < len(all_flows) and all_flows[idx] is not None:
                    flow = all_flows[idx]
                elif idx > 0 and all_flows[idx - 1] is not None:
                    flow = all_flows[idx - 1]
                else:
                    flow = np.zeros((h, w, 2), dtype=np.float32)

                # Step 1: Detect blur regions.
                blur_mask = self.blur_detector.detect(flow, alpha)

                # Step 2: Obtain clean plate.
                if clean_plate_path is not None:
                    clean_plate = self.clean_plate_provider.load(
                        clean_plate_path, h, w
                    )
                elif static_plate is not None:
                    clean_plate = static_plate
                else:
                    clean_plate = self.clean_plate_provider.synthesize_dynamic(
                        idx, all_frames, all_alphas, all_flows
                    )

                # Step 3: Solve alpha + foreground.
                refined_alpha, fg_color = self.alpha_solver.solve(
                    frame_linear, clean_plate, alpha, flow, blur_mask
                )

                # Step 4: Temporal smoothing.
                final_alpha = self.temporal_filter.smooth(refined_alpha, blur_mask)

                # Step 5: Write refined alpha to RefinedMatte/ as EXR.
                matte_path = str(matte_dir / f"{stem}.exr")
                write_depth_map(matte_path, final_alpha)

                # Step 6: Optionally write foreground to RefinedFG/ as 3-channel EXR.
                if fg_dir is not None:
                    fg_path = str(fg_dir / f"{stem}.exr")
                    bgr = cv2.cvtColor(fg_color, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        fg_path,
                        bgr.astype(np.float32),
                        _EXR_FLOAT32_FLAGS,
                    )

                # Step 7: Generate comp preview PNG.
                checkerboard = create_checkerboard(w, h)
                alpha_3ch = final_alpha[..., np.newaxis]
                comp = composite_straight(frame_linear, checkerboard, alpha_3ch)
                comp_srgb = linear_to_srgb(comp)
                comp_uint8 = np.clip(comp_srgb * 255.0, 0, 255).astype(np.uint8)
                comp_bgr = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2BGR)
                comp_path = str(comp_dir / f"{stem}.png")
                cv2.imwrite(comp_path, comp_bgr)

            except Exception:
                logger.exception(
                    "Error processing frame %d (%s), skipping.", idx, frame_file
                )
                continue

            # Step 8: Progress callback.
            if on_frame_complete is not None:
                on_frame_complete(idx, total_frames)
