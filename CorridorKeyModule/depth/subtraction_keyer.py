"""Subtraction Keyer — foreground alpha from observed-vs-plate color difference.

Computes a per-pixel difference between the observed frame and the synthesized
clean plate, then converts the difference magnitude into an alpha matte via
cosine-interpolated soft thresholding.  Plate confidence modulates the final
alpha so that uncertain plate regions fall back to ``low_confidence_alpha``.
"""

from __future__ import annotations

import numpy as np


class SubtractionKeyer:
    """Compute foreground alpha from observed-vs-plate color difference.

    Parameters
    ----------
    difference_threshold:
        Difference magnitude above which a pixel is fully foreground.
        Default 0.05.
    difference_falloff:
        Width of the soft transition zone below the threshold.
        Default 0.03.
    low_confidence_alpha:
        Alpha value for pixels where plate confidence is low.
        Default 1.0 (treat uncertain regions as foreground).
    color_space_mode:
        ``"max_channel"`` (default) or ``"luminance"``.
    """

    _LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    def __init__(
        self,
        difference_threshold: float = 0.05,
        difference_falloff: float = 0.03,
        low_confidence_alpha: float = 1.0,
        color_space_mode: str = "max_channel",
    ) -> None:
        self.difference_threshold = difference_threshold
        self.difference_falloff = difference_falloff
        self.low_confidence_alpha = low_confidence_alpha
        self.color_space_mode = color_space_mode

    def compute(
        self,
        observed: np.ndarray,
        clean_plate: np.ndarray,
        plate_confidence: np.ndarray,
    ) -> np.ndarray:
        """Compute alpha matte from frame-vs-plate difference.

        Parameters
        ----------
        observed:
            [H, W, 3] float32 observed frame in linear color space.
        clean_plate:
            [H, W, 3] float32 synthesized clean plate in linear color space.
        plate_confidence:
            [H, W] float32 plate confidence in [0.0, 1.0].

        Returns
        -------
        np.ndarray
            [H, W] float32 alpha matte in [0.0, 1.0].
        """
        # --- Step 1: Per-pixel difference ---
        pixel_diff = observed - clean_plate  # [H, W, 3]

        if self.color_space_mode == "max_channel":
            diff = np.max(np.abs(pixel_diff), axis=2)  # [H, W]
        else:  # luminance
            diff = np.abs(np.dot(pixel_diff, self._LUMINANCE_WEIGHTS))  # [H, W]

        # --- Step 2: Cosine-interpolated soft threshold ---
        threshold = self.difference_threshold
        falloff = self.difference_falloff
        zone_bottom = threshold - falloff

        alpha = np.empty_like(diff, dtype=np.float32)

        # Region 1: diff >= threshold → alpha 1.0
        fg_mask = diff >= threshold
        alpha[fg_mask] = 1.0

        # Region 2: diff < zone_bottom → alpha 0.0
        bg_mask = diff < zone_bottom
        alpha[bg_mask] = 0.0

        # Region 3: transition zone [zone_bottom, threshold)
        transition_mask = ~fg_mask & ~bg_mask
        if np.any(transition_mask):
            if falloff > 0:
                d_in_zone = diff[transition_mask]
                # Cosine interpolation matching DepthThresholder formula:
                # alpha = 0.5 * (1 + cos(π * (threshold - d) / falloff))
                # At zone_bottom (d = threshold - falloff): cos(π) = -1 → alpha = 0.0
                # At threshold (d = threshold):             cos(0) = 1  → alpha = 1.0
                t = (threshold - d_in_zone) / falloff
                alpha[transition_mask] = (
                    0.5 * (1.0 + np.cos(np.pi * t))
                ).astype(np.float32)
            else:
                # Zero falloff → hard threshold; anything in zone is foreground
                alpha[transition_mask] = 1.0

        # --- Step 3: Plate confidence modulation ---
        final_alpha = (
            alpha * plate_confidence
            + self.low_confidence_alpha * (1.0 - plate_confidence)
        )

        # --- Step 4: Clamp and return ---
        return np.clip(final_alpha, 0.0, 1.0).astype(np.float32)
