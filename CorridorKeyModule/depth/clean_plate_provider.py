"""Clean plate provision for motion blur alpha refinement.

Supplies a per-pixel background color estimate — static or per-frame dynamic.
Three modes of operation:

1. **Explicit**: User provides a clean plate image file (EXR or PNG).
2. **Dynamic (default)**: Per-frame synthesis by warping low-alpha pixels from
   neighboring frames to the current frame's coordinate space using optical flow,
   then blending with alpha-inverse weighting.
3. **Static (legacy)**: Single plate for the whole sequence by selecting the
   lowest-alpha frame per pixel without flow warping.

References
----------
Sengupta et al. (CVPR 2020) "Background Matting: The World is Your Green Screen"
— known-background-plate matting approach.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from CorridorKeyModule.core.color_utils import srgb_to_linear
from CorridorKeyModule.depth.exr_io import read_depth_map


class CleanPlateProvider:
    """Supplies a per-pixel background color estimate.

    Parameters
    ----------
    search_radius : int
        Number of neighboring frames to search for background donors.
        Default 10.
    alpha_threshold : float
        Maximum alpha for a donor pixel to be considered reliable background.
        Default 0.1.
    """

    def __init__(
        self,
        search_radius: int = 10,
        alpha_threshold: float = 0.1,
    ) -> None:
        self.search_radius = search_radius
        self.alpha_threshold = alpha_threshold

    def load(self, path: str, target_h: int, target_w: int) -> np.ndarray:
        """Load an explicit clean plate from an EXR or PNG file.

        Parameters
        ----------
        path : str
            Path to clean plate image (EXR or PNG).
        target_h, target_w : int
            Expected spatial resolution.

        Returns
        -------
        np.ndarray
            [H, W, 3] float32 in linear color space.

        Raises
        ------
        ValueError
            If spatial resolution does not match (target_h, target_w).
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".exr":
            # For 3-channel EXR, use cv2 directly (read_depth_map is single-channel only)
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read EXR file: {path}")
            # OpenCV loads as BGR — convert to RGB
            if img.ndim == 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 2:
                # Single-channel EXR — expand to 3 channels
                img = np.stack([img, img, img], axis=-1)
            plate = img[:, :, :3].astype(np.float32, copy=False)
        else:
            # PNG (or other image formats): load, normalize to [0,1], apply srgb_to_linear
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image file: {path}")
            # Convert BGR to RGB
            if img.ndim == 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            # Normalize to [0, 1] float32
            plate = img[:, :, :3].astype(np.float32) / 255.0
            # Convert sRGB to linear
            plate = srgb_to_linear(plate)

        h, w = plate.shape[:2]
        if h != target_h or w != target_w:
            raise ValueError(
                f"Clean plate resolution ({h}, {w}) does not match "
                f"target resolution ({target_h}, {target_w})"
            )

        return plate.astype(np.float32, copy=False)

    def synthesize_dynamic(
        self,
        frame_idx: int,
        frames: list[np.ndarray],
        alphas: list[np.ndarray],
        flows: list[np.ndarray | None],
    ) -> np.ndarray:
        """Synthesize a per-frame dynamic clean plate for a specific frame.

        For each pixel in frame ``frame_idx``:
        1. Search neighboring frames within ``search_radius`` for donor pixels
           with alpha < ``alpha_threshold``.
        2. Warp each donor pixel to the current frame's coordinate space using
           the cumulative optical flow between the donor frame and frame_idx.
        3. Blend warped donors using alpha-inverse weighting:
           weight = (1.0 - donor_alpha).
        4. If no donor pixel below ``alpha_threshold`` is found, fall back to
           the donor with the lowest alpha for that pixel.

        Parameters
        ----------
        frame_idx : int
            Index of the target frame.
        frames : list[np.ndarray]
            Full sequence of [H, W, 3] float32 frames in linear color space.
        alphas : list[np.ndarray]
            Corresponding [H, W] float32 alpha mattes.
        flows : list[np.ndarray | None]
            Forward flow fields [H, W, 2] between consecutive frames.
            flows[i] maps frame i -> frame i+1. May be None if unavailable.

        Returns
        -------
        np.ndarray
            [H, W, 3] float32 dynamic clean plate in linear color space.
        """
        n_frames = len(frames)
        h, w = frames[frame_idx].shape[:2]

        # Accumulators for weighted blending
        weight_sum = np.zeros((h, w), dtype=np.float32)
        color_sum = np.zeros((h, w, 3), dtype=np.float32)

        # Track lowest-alpha donor per pixel for fallback
        best_alpha = np.full((h, w), np.inf, dtype=np.float32)
        best_color = np.zeros((h, w, 3), dtype=np.float32)

        start = max(0, frame_idx - self.search_radius)
        end = min(n_frames, frame_idx + self.search_radius + 1)

        for d in range(start, end):
            if d == frame_idx:
                # Current frame is also a candidate
                warped_frame = frames[d]
                warped_alpha = alphas[d]
            else:
                # Compute cumulative flow from donor d to frame_idx
                cumulative_flow = self._accumulate_flow(d, frame_idx, flows, h, w)
                if cumulative_flow is None:
                    continue

                # Build remap coordinates
                grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
                map_x = grid_x + cumulative_flow[:, :, 0]
                map_y = grid_y + cumulative_flow[:, :, 1]

                # Warp donor frame and alpha to current frame's coordinate space
                warped_frame = cv2.remap(
                    frames[d], map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                warped_alpha = cv2.remap(
                    alphas[d], map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

            # Update fallback: track lowest-alpha donor per pixel
            lower_mask = warped_alpha < best_alpha
            best_alpha = np.where(lower_mask, warped_alpha, best_alpha)
            best_color = np.where(
                lower_mask[:, :, np.newaxis],
                warped_frame,
                best_color,
            )

            # Accumulate donors below alpha threshold
            good_mask = warped_alpha < self.alpha_threshold
            donor_weight = (1.0 - warped_alpha) * good_mask.astype(np.float32)
            weight_sum += donor_weight
            color_sum += donor_weight[:, :, np.newaxis] * warped_frame

        # Blend where we have valid donors
        valid = weight_sum > 0.0
        plate = np.zeros((h, w, 3), dtype=np.float32)
        plate[valid] = color_sum[valid] / weight_sum[valid, np.newaxis]

        # Fallback: use lowest-alpha donor where no valid donors found
        plate[~valid] = best_color[~valid]

        return plate

    def synthesize_static(
        self,
        frames: list[np.ndarray],
        alphas: list[np.ndarray],
    ) -> np.ndarray:
        """Synthesize a single static clean plate for the entire sequence.

        For each pixel, selects the frame where that pixel has the lowest
        alpha value. No flow warping.

        Parameters
        ----------
        frames : list[np.ndarray]
            List of [H, W, 3] float32 frames in linear color space.
        alphas : list[np.ndarray]
            Corresponding [H, W] float32 alpha mattes.

        Returns
        -------
        np.ndarray
            [H, W, 3] float32 synthesized clean plate in linear color space.
        """
        # Stack alphas to [N, H, W] and find argmin per pixel
        alpha_stack = np.stack(alphas, axis=0)  # [N, H, W]
        min_indices = np.argmin(alpha_stack, axis=0)  # [H, W]

        # Stack frames to [N, H, W, 3]
        frame_stack = np.stack(frames, axis=0)  # [N, H, W, 3]

        # Gather: select frame with lowest alpha per pixel
        h, w = min_indices.shape
        y_idx, x_idx = np.mgrid[0:h, 0:w]
        plate = frame_stack[min_indices, y_idx, x_idx]  # [H, W, 3]

        return plate.astype(np.float32, copy=False)

    @staticmethod
    def _accumulate_flow(
        src: int,
        dst: int,
        flows: list[np.ndarray | None],
        h: int,
        w: int,
    ) -> np.ndarray | None:
        """Compute cumulative optical flow from frame ``src`` to frame ``dst``.

        Parameters
        ----------
        src, dst : int
            Source and destination frame indices.
        flows : list
            Forward flow fields. flows[i] maps frame i -> frame i+1.
        h, w : int
            Spatial dimensions.

        Returns
        -------
        np.ndarray or None
            [H, W, 2] cumulative flow, or None if any intermediate flow is missing.
        """
        cumulative = np.zeros((h, w, 2), dtype=np.float32)

        if src < dst:
            # Forward: chain flows[src] + flows[src+1] + ... + flows[dst-1]
            for i in range(src, dst):
                if i >= len(flows) or flows[i] is None:
                    return None
                cumulative += flows[i]
        elif src > dst:
            # Backward: negate and chain flows[dst] + ... + flows[src-1]
            for i in range(dst, src):
                if i >= len(flows) or flows[i] is None:
                    return None
                cumulative -= flows[i]

        return cumulative
