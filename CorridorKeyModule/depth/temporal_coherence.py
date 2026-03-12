"""Temporal coherence filtering for refined alpha mattes.

Applies exponential moving average (EMA) smoothing across consecutive frames
to suppress flickering and ensure temporal stability in motion-blurred regions.
"""

from __future__ import annotations

import numpy as np


class TemporalCoherenceFilter:
    """EMA smoothing of alpha values across consecutive frames.

    Parameters
    ----------
    temporal_smoothing : float
        EMA weight for current frame. Default 0.3.
        smoothed = temporal_smoothing * current + (1 - temporal_smoothing) * previous
    """

    def __init__(self, temporal_smoothing: float = 0.3) -> None:
        self._w = temporal_smoothing
        self._prev_smoothed: np.ndarray | None = None

    def reset(self) -> None:
        """Reset internal state for a new sequence."""
        self._prev_smoothed = None

    def smooth(self, alpha: np.ndarray, blur_mask: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to a refined alpha matte.

        Parameters
        ----------
        alpha : np.ndarray
            [H, W] float32 refined alpha for current frame.
        blur_mask : np.ndarray
            [H, W] float32 binary blur mask. Only blur-masked pixels are smoothed.

        Returns
        -------
        np.ndarray
            [H, W] float32 temporally smoothed alpha.
        """
        if self._prev_smoothed is None:
            # First frame: store as previous and return as-is.
            self._prev_smoothed = alpha.copy()
            return alpha.copy()

        # Start with the current alpha (non-masked pixels pass through unchanged).
        smoothed = alpha.copy()

        # Apply EMA only where blur_mask == 1.0.
        mask = blur_mask == 1.0
        smoothed[mask] = (
            self._w * alpha[mask]
            + (1.0 - self._w) * self._prev_smoothed[mask]
        )

        self._prev_smoothed = smoothed.copy()
        return smoothed
