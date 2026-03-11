"""Depth Thresholder — converts Background_Score to alpha matte.

High Background_Score → alpha 0.0 (background).
Low Background_Score  → alpha 1.0 (foreground).
Transition zone uses cosine interpolation.
Confidence modulates the final alpha toward ``low_confidence_alpha``.
"""

from __future__ import annotations

import numpy as np

from .data_models import CubeResult


class DepthThresholder:
    """Convert a :class:`CubeResult` Background_Score into a compositing-ready
    single-channel alpha matte.

    Parameters
    ----------
    depth_threshold:
        Background_Score cutoff in [0.0, 1.0].  Pixels at or above this
        value are treated as background (alpha 0.0).
    depth_falloff:
        Width of the soft transition zone in [0.0, 0.5].
    low_confidence_alpha:
        Alpha value that low-confidence pixels are biased toward.
        Default 0.0 treats uncertain pixels as background.
    """

    def __init__(
        self,
        depth_threshold: float = 0.5,
        depth_falloff: float = 0.05,
        low_confidence_alpha: float = 0.0,
    ) -> None:
        self.depth_threshold = depth_threshold
        self.depth_falloff = depth_falloff
        self.low_confidence_alpha = low_confidence_alpha

    def apply(self, cube_result: CubeResult) -> np.ndarray:
        """Convert Background_Score to alpha matte.

        Returns
        -------
        np.ndarray
            [H, W] float32 in [0.0, 1.0] at original input resolution.
        """
        bg_score = cube_result.background_score
        confidence = cube_result.confidence_map

        threshold = self.depth_threshold
        falloff = self.depth_falloff
        zone_bottom = threshold - falloff

        # --- Piecewise threshold function ---
        alpha = np.empty_like(bg_score, dtype=np.float32)

        # Region 1: background_score >= threshold → alpha 0.0
        bg_mask = bg_score >= threshold
        alpha[bg_mask] = 0.0

        # Region 2: background_score < threshold - falloff → alpha 1.0
        fg_mask = bg_score < zone_bottom
        alpha[fg_mask] = 1.0

        # Region 3: transition zone [zone_bottom, threshold)
        transition_mask = ~bg_mask & ~fg_mask
        if np.any(transition_mask):
            if falloff > 0:
                scores_in_zone = bg_score[transition_mask]
                # Cosine interpolation: alpha = 0.5 * (1 + cos(π * (score - zone_bottom) / falloff))
                # At zone_bottom: cos(0) = 1 → alpha = 1.0
                # At threshold:   cos(π) = -1 → alpha = 0.0
                t = (scores_in_zone - zone_bottom) / falloff
                alpha[transition_mask] = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
            else:
                # Zero falloff → hard threshold; anything below threshold is foreground
                alpha[transition_mask] = 1.0

        # --- Confidence modulation ---
        # final_alpha = raw_alpha * confidence + low_confidence_alpha * (1 - confidence)
        final_alpha = alpha * confidence + self.low_confidence_alpha * (1.0 - confidence)

        return np.clip(final_alpha, 0.0, 1.0).astype(np.float32)
