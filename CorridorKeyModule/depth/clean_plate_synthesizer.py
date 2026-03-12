"""Clean plate synthesis via flow-warped temporal median.

For each target frame, warps background-classified pixels from neighboring
frames into the target's coordinate space using cumulative optical flow,
then combines them via per-channel temporal median.
"""

from __future__ import annotations

import cv2
import numpy as np

from CorridorKeyModule.depth.data_models import FlowResult
from CorridorKeyModule.depth.optical_flow import accumulate_flow


class CleanPlateSynthesizer:
    """Synthesize per-frame clean plates via flow-warped temporal median.

    For each target frame, warps background-classified pixels from neighboring
    frames into the target's coordinate space using cumulative optical flow,
    then combines them via per-channel temporal median.

    Parameters
    ----------
    plate_search_radius : int
        Number of neighboring frames to search in each direction. Default 15.
    donor_threshold : float
        Maximum Bootstrap_Mask value for a pixel to qualify as a donor.
        Default 0.3 (pixels with mask < 0.3 are considered background).
    """

    def __init__(
        self,
        plate_search_radius: int = 15,
        donor_threshold: float = 0.3,
    ) -> None:
        self.plate_search_radius = plate_search_radius
        self.donor_threshold = donor_threshold

    def synthesize(
        self,
        frame_idx: int,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        flow_results: list[FlowResult | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize a clean plate and confidence map for a single frame.

        Parameters
        ----------
        frame_idx : int
            Index of the target frame.
        frames : list[np.ndarray]
            Full sequence of [H, W, 3] float32 frames in linear color space.
        masks : list[np.ndarray]
            Per-frame masks [H, W] float32 in [0.0, 1.0].
            0.0 = confident background, 1.0 = confident foreground.
        flow_results : list[FlowResult | None]
            Flow results between consecutive frames.
            flow_results[i] contains flow from frame i to frame i+1.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (clean_plate [H, W, 3] float32, plate_confidence [H, W] float32)
        """
        n_frames = len(frames)
        h, w = frames[frame_idx].shape[:2]
        r = self.plate_search_radius

        # Donor window bounds
        win_start = max(0, frame_idx - r)
        win_end = min(n_frames, frame_idx + r + 1)

        # Collect donor data: list of (warped_rgb, donor_weight, is_valid)
        donors: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for d in range(win_start, win_end):
            if d == frame_idx:
                continue

            # Compute cumulative flow from donor d to target frame_idx
            cumulative_flow = accumulate_flow(d, frame_idx, flow_results, h, w)
            if cumulative_flow is None:
                continue

            # Compute flow consistency from forward-backward error
            flow_consistency = self._compute_flow_consistency(
                d, frame_idx, flow_results, cumulative_flow, h, w,
            )

            # Build remap coordinates
            grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
            map_x = grid_x + cumulative_flow[:, :, 0]
            map_y = grid_y + cumulative_flow[:, :, 1]

            # Warp donor frame and mask to target coordinate space
            warped_rgb = cv2.remap(
                frames[d], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_mask = cv2.remap(
                masks[d], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Donor weight = (1 - warped_mask) * flow_consistency
            donor_weight = (1.0 - warped_mask) * flow_consistency

            # Valid donors: pixels where warped_mask < donor_threshold
            is_valid = warped_mask < self.donor_threshold

            donors.append((warped_rgb, donor_weight, is_valid))

        total_possible = len(donors)

        # Initialize outputs
        clean_plate = np.zeros((h, w, 3), dtype=np.float32)
        plate_confidence = np.zeros((h, w), dtype=np.float32)

        if total_possible == 0:
            # No donors at all — use observed frame pixel, confidence = 0.0
            clean_plate[:] = frames[frame_idx]
            # plate_confidence stays 0.0
            return (
                np.clip(clean_plate, 0.0, 1.0),
                plate_confidence,
            )

        # Stack donor data for vectorized computation
        # valid_count[y, x] = number of valid donors at each pixel
        valid_count = np.zeros((h, w), dtype=np.float32)
        for _, _, is_valid in donors:
            valid_count += is_valid.astype(np.float32)

        # Mask for pixels with >= 2 valid donors (median path)
        median_mask = valid_count >= 2.0
        # Mask for pixels with < 2 valid donors (weighted mean fallback)
        fallback_mask = ~median_mask

        # --- Median path: stack valid donors and compute per-channel median ---
        if np.any(median_mask):
            # For efficiency, stack all donor RGB values and use masked median
            # We process per-channel to handle variable valid counts
            rgb_stack = np.stack([rgb for rgb, _, _ in donors], axis=0)  # [D, H, W, 3]
            valid_stack = np.stack(
                [v for _, _, v in donors], axis=0,
            )  # [D, H, W] bool

            for c in range(3):
                channel_vals = rgb_stack[:, :, :, c]  # [D, H, W]
                # At median_mask pixels, compute median of valid donors
                # Use masked array for correct median computation
                masked_vals = np.ma.array(
                    channel_vals,
                    mask=~valid_stack,
                )
                median_result = np.ma.median(masked_vals, axis=0)  # [H, W]
                clean_plate[:, :, c] = np.where(
                    median_mask,
                    median_result.filled(0.0),
                    clean_plate[:, :, c],
                )

        # --- Fallback path: weighted mean of ALL donors ---
        if np.any(fallback_mask):
            weight_sum = np.zeros((h, w), dtype=np.float32)
            color_sum = np.zeros((h, w, 3), dtype=np.float32)

            for warped_rgb, donor_weight, _ in donors:
                weight_sum += donor_weight
                color_sum += donor_weight[:, :, np.newaxis] * warped_rgb

            # Where weight_sum > 0, compute weighted mean
            has_weight = (weight_sum > 0.0) & fallback_mask
            if np.any(has_weight):
                clean_plate[has_weight] = (
                    color_sum[has_weight] / weight_sum[has_weight, np.newaxis]
                )

            # Where weight_sum == 0 (no donors at all), use observed frame pixel
            no_weight = (weight_sum <= 0.0) & fallback_mask
            if np.any(no_weight):
                clean_plate[no_weight] = frames[frame_idx][no_weight]

        # --- Plate confidence ---
        # confidence = valid_count / total_possible
        plate_confidence = valid_count / total_possible
        # Force confidence to 0.0 where fallback was used
        plate_confidence[fallback_mask] = 0.0

        return (
            np.clip(clean_plate, 0.0, 1.0).astype(np.float32),
            np.clip(plate_confidence, 0.0, 1.0).astype(np.float32),
        )

    @staticmethod
    def _compute_flow_consistency(
        donor_idx: int,
        target_idx: int,
        flow_results: list[FlowResult | None],
        cumulative_flow: np.ndarray,
        h: int,
        w: int,
        consistency_scale: float = 2.0,
    ) -> np.ndarray:
        """Compute flow consistency weight from forward-backward error.

        Uses the FlowResult at the boundary between donor and target to
        estimate forward-backward consistency. The consistency decays
        exponentially with the magnitude of the error.

        Parameters
        ----------
        donor_idx : int
            Index of the donor frame.
        target_idx : int
            Index of the target frame.
        flow_results : list[FlowResult | None]
            Flow results between consecutive frames.
        cumulative_flow : np.ndarray
            [H, W, 2] cumulative flow from donor to target.
        h, w : int
            Spatial dimensions.
        consistency_scale : float
            Scale factor for the exponential decay. Default 2.0.

        Returns
        -------
        np.ndarray
            [H, W] float32 consistency weights in [0.0, 1.0].
        """
        # Use the flow result closest to the target for consistency check
        if donor_idx < target_idx:
            # Forward direction: use flow at target_idx - 1
            check_idx = target_idx - 1
        else:
            # Backward direction: use flow at target_idx
            check_idx = target_idx

        if check_idx < 0 or check_idx >= len(flow_results):
            return np.ones((h, w), dtype=np.float32)

        fr = flow_results[check_idx]
        if fr is None:
            return np.ones((h, w), dtype=np.float32)

        # Forward-backward consistency: warp backward flow to forward position
        # and check if forward + backward ≈ 0
        forward = fr.forward_flow  # [H, W, 2]
        backward = fr.backward_flow  # [H, W, 2]

        # Warp backward flow to the forward-warped position
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped_x = grid_x + forward[:, :, 0]
        warped_y = grid_y + forward[:, :, 1]

        warped_backward = cv2.remap(
            backward, warped_x, warped_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # fb_error = magnitude of (forward + warped_backward)
        fb_sum = forward + warped_backward
        fb_error = np.sqrt(fb_sum[:, :, 0] ** 2 + fb_sum[:, :, 1] ** 2)

        # consistency = exp(-fb_error / consistency_scale)
        consistency = np.exp(-fb_error / consistency_scale)

        return consistency.astype(np.float32)
