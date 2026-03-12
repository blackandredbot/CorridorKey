"""Blur region detection via optical flow magnitude thresholding.

Identifies pixels affected by motion blur by analyzing optical flow
magnitude and comparing it against a configurable threshold. Only
partially transparent pixels (alpha in the open interval (0.0, 1.0))
are considered — fully opaque and fully transparent pixels are left
untouched.
"""

from __future__ import annotations

import cv2
import numpy as np


class BlurRegionDetector:
    """Identifies motion-blurred pixels from optical flow magnitude.

    Parameters
    ----------
    blur_threshold : float
        Minimum flow magnitude (pixels) to classify as motion-blurred.
        Default 2.0.
    dilation_radius : int
        Morphological dilation radius for the blur mask. Default 3.
        Set to 0 to disable dilation.
    """

    def __init__(
        self,
        blur_threshold: float = 2.0,
        dilation_radius: int = 3,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dilation_radius = dilation_radius

    def detect(self, flow_field: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute binary blur mask.

        Parameters
        ----------
        flow_field : np.ndarray
            [H, W, 2] float32 forward flow in pixel units.
        alpha : np.ndarray
            [H, W] float32 alpha matte in [0.0, 1.0].

        Returns
        -------
        np.ndarray
            [H, W] float32 binary mask: 1.0 = motion-blurred, 0.0 = clean.
        """
        # Compute per-pixel blur magnitude: sqrt(dx^2 + dy^2)
        dx = flow_field[:, :, 0]
        dy = flow_field[:, :, 1]
        blur_magnitude = np.sqrt(dx * dx + dy * dy)

        # Threshold: mark pixels where magnitude exceeds blur_threshold
        magnitude_mask = blur_magnitude > self.blur_threshold

        # Restrict to partially transparent pixels: 0.0 < alpha < 1.0
        partial_mask = (alpha > 0.0) & (alpha < 1.0)

        # Combine both conditions
        mask = (magnitude_mask & partial_mask).astype(np.float32)

        # Apply morphological dilation if radius > 0
        if self.dilation_radius > 0:
            kernel_size = 2 * self.dilation_radius + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask.astype(np.float32)
