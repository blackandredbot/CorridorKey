"""Mask refinement for depth-derived alpha mattes.

Applies edge-aware refinement, despeckle, hole fill, and resolution-scaled
matte tightening to produce compositing-ready alpha mattes.
"""

from __future__ import annotations

import cv2
import numpy as np


class MaskRefiner:
    """Refine raw depth-derived alpha mattes using RGB guide signal.

    Refinement steps (applied in order when ``refinement_strength > 0``):
    1. Guided filter — aligns matte edges with visible object boundaries.
    2. Despeckle — remove small connected foreground regions.
    3. Hole fill — fill small background holes inside the foreground.
    4. Resolution-scaled matte tightening — morphological erosion + Gaussian
       re-feather, with correction magnitude proportional to the resolution
       gap between processing and original resolution.

    When ``refinement_strength == 0.0`` the input is returned unchanged.
    Intermediate values blend between raw and refined output.
    """

    def __init__(
        self,
        refinement_strength: float = 1.0,
        despeckle_size: int = 50,
        hole_fill_size: int = 50,
        processing_resolution: int | None = None,
        original_resolution: int | None = None,
    ) -> None:
        self.refinement_strength = refinement_strength
        self.despeckle_size = despeckle_size
        self.hole_fill_size = hole_fill_size
        self.processing_resolution = processing_resolution
        self.original_resolution = original_resolution

    def refine(self, raw_alpha: np.ndarray, rgb_guide: np.ndarray) -> np.ndarray:
        """Refine raw alpha matte using RGB guide signal.

        Parameters
        ----------
        raw_alpha : np.ndarray
            [H, W] float32 alpha matte in [0.0, 1.0].
        rgb_guide : np.ndarray
            [H, W, 3] float32 RGB frame in [0.0, 1.0].

        Returns
        -------
        np.ndarray
            [H, W] float32 refined alpha in [0.0, 1.0].
        """
        # Ensure float32
        alpha = raw_alpha.astype(np.float32, copy=True)

        if self.refinement_strength == 0.0:
            return alpha

        # Apply full refinement pipeline
        refined = self._apply_guided_filter(alpha, rgb_guide)
        refined = self._despeckle(refined)
        refined = self._hole_fill(refined)
        refined = self._resolution_tighten(refined)

        # Clamp to [0, 1]
        refined = np.clip(refined, 0.0, 1.0).astype(np.float32)

        # Blend between raw and refined based on refinement_strength
        if self.refinement_strength < 1.0:
            result = (
                alpha * (1.0 - self.refinement_strength)
                + refined * self.refinement_strength
            )
            return np.clip(result, 0.0, 1.0).astype(np.float32)

        return refined

    # ------------------------------------------------------------------
    # Internal refinement steps
    # ------------------------------------------------------------------

    def _apply_guided_filter(
        self, alpha: np.ndarray, rgb_guide: np.ndarray
    ) -> np.ndarray:
        """Edge-aware guided filter using the RGB frame as guide."""
        radius = 8
        eps = 1e-4
        guide = rgb_guide.astype(np.float32)
        return cv2.ximgproc.guidedFilter(guide, alpha, radius, eps)

    def _despeckle(self, alpha: np.ndarray) -> np.ndarray:
        """Remove connected foreground regions smaller than despeckle_size."""
        if self.despeckle_size <= 0:
            return alpha

        # Binarize at 0.5 threshold
        mask_fg = (alpha > 0.5).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_fg, connectivity=8
        )

        # Build a mask keeping only large-enough foreground components
        keep_mask = np.zeros_like(mask_fg)
        for i in range(1, num_labels):  # skip background label 0
            if stats[i, cv2.CC_STAT_AREA] >= self.despeckle_size:
                keep_mask[labels == i] = 255

        # Zero out alpha where small foreground specks were removed
        keep_float = keep_mask.astype(np.float32) / 255.0
        return alpha * keep_float

    def _hole_fill(self, alpha: np.ndarray) -> np.ndarray:
        """Fill small background holes inside the foreground region."""
        if self.hole_fill_size <= 0:
            return alpha

        # Invert: background holes become foreground components
        inv_mask = (alpha <= 0.5).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inv_mask, connectivity=8
        )

        # Identify small hole components (skip label 0 = the main background)
        fill_mask = np.zeros_like(inv_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.hole_fill_size:
                fill_mask[labels == i] = 255

        # Fill holes: set alpha to 1.0 where small holes were found
        fill_float = fill_mask.astype(np.float32) / 255.0
        return np.maximum(alpha, fill_float)

    def _resolution_tighten(self, alpha: np.ndarray) -> np.ndarray:
        """Morphological erosion + Gaussian re-feather scaled by resolution gap."""
        if self.processing_resolution is None or self.original_resolution is None:
            return alpha

        if self.original_resolution <= 0:
            return alpha

        correction = max(0.0, 1.0 - self.processing_resolution / self.original_resolution)
        if correction <= 0.0:
            return alpha

        # Scale erosion kernel size by correction magnitude (1-5 pixels)
        erode_size = max(1, int(round(correction * 5)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size * 2 + 1, erode_size * 2 + 1)
        )
        eroded = cv2.erode(alpha, kernel)

        # Gaussian re-feather — blur size proportional to correction
        blur_radius = max(1, int(round(correction * 7)))
        blur_size = blur_radius * 2 + 1
        result = cv2.GaussianBlur(eroded, (blur_size, blur_size), 0)

        return result.astype(np.float32)
