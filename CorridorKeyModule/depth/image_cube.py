"""Image Cube Builder — three-signal background score estimation.

Fuses three complementary signals across a rolling buffer of optical flow
results to produce a per-pixel Background_Score:

1. **Parallax channel** — residual flow magnitude after RANSAC-based global
   motion compensation.  High score = far/background, low = near/foreground.
2. **Persistence channel** — temporal stability of local patch descriptors
   across frames.  High persistence = background, low = foreground.
3. **Positional stability channel** — cumulative displacement from first-seen
   position.  Zero drift = 1.0, large drift → 0.0.

Signal fusion supports three modes: ``"blend"`` (weighted average),
``"max"`` (highest signal wins), ``"min"`` (all must agree).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .data_models import CubeResult, FlowResult

_VALID_FUSION_MODES = {"blend", "max", "min"}
_EPS = 1e-6


@dataclass
class _FrameEstimate:
    """Per-frame channel estimates stored in the rolling buffer."""

    parallax: np.ndarray       # [H, W] float32 in [0, 1]
    persistence: np.ndarray    # [H, W] float32 in [0, 1]
    stability: np.ndarray      # [H, W] float32 in [0, 1]
    confidence: np.ndarray     # [H, W] float32 in [0, 1]
    occlusion_mask: np.ndarray # [H, W] float32, 1.0 = occluded
    global_motion_mag: float   # median magnitude of global motion vector


class ImageCubeBuilder:
    """Build a per-pixel Background_Score by fusing three signals.

    Parameters
    ----------
    buffer_size : int
        Rolling buffer depth (number of recent frames to keep).
    stationary_threshold : float
        When median global motion magnitude across the buffer falls below
        this value (in pixels), parallax weight is reduced and persistence
        + stability weights are increased proportionally.
    parallax_weight, persistence_weight, stability_weight : float
        Default channel weights for ``"blend"`` fusion.  Must sum to 1.0.
    fusion_mode : str
        ``"blend"`` (weighted average), ``"max"`` (highest wins),
        ``"min"`` (all must agree).
    """

    def __init__(
        self,
        buffer_size: int = 10,
        stationary_threshold: float = 2.0,
        parallax_weight: float = 0.4,
        persistence_weight: float = 0.3,
        stability_weight: float = 0.3,
        fusion_mode: str = "blend",
    ) -> None:
        if fusion_mode not in _VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {_VALID_FUSION_MODES}, "
                f"got {fusion_mode!r}"
            )
        weight_sum = parallax_weight + persistence_weight + stability_weight
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum}"
            )

        self.buffer_size = buffer_size
        self.stationary_threshold = stationary_threshold
        self.parallax_weight = parallax_weight
        self.persistence_weight = persistence_weight
        self.stability_weight = stability_weight
        self.fusion_mode = fusion_mode

        # Rolling buffer of per-frame estimates
        self._buffer: deque[_FrameEstimate] = deque(maxlen=buffer_size)

        # Accumulated displacement from first-seen position (for stability)
        self._cumulative_displacement: np.ndarray | None = None

        # Patch descriptor history for persistence channel
        self._patch_history: deque[np.ndarray] = deque(maxlen=buffer_size)

        # Frame history for persistence (grayscale patches)
        self._frame_history: deque[np.ndarray] = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        flow_result: FlowResult,
        emit_channel_maps: bool = False,
        frame: np.ndarray | None = None,
    ) -> CubeResult:
        """Incorporate a new flow result and recompute the fused score.

        Parameters
        ----------
        flow_result : FlowResult
            Bidirectional flow + occlusion mask from the optical flow engine.
        emit_channel_maps : bool
            If ``True``, include individual channel maps in the output.
        frame : np.ndarray | None
            Optional RGB frame ``[H, W, 3]`` float32 for persistence tracking.
            If not provided, persistence is estimated from flow statistics.

        Returns
        -------
        CubeResult
        """
        h, w = flow_result.forward_flow.shape[:2]

        # --- Global motion compensation (RANSAC affine) ---
        residual_flow, global_motion_mag = self._compensate_global_motion(
            flow_result.forward_flow
        )

        # --- Parallax channel ---
        parallax = self._compute_parallax(residual_flow)

        # --- Persistence channel ---
        persistence = self._compute_persistence(flow_result, frame)

        # --- Positional stability channel ---
        stability = self._compute_stability(flow_result.forward_flow, h, w)

        # --- Per-frame confidence ---
        frame_confidence = self._compute_frame_confidence(
            parallax, persistence, stability,
            residual_flow, flow_result.occlusion_mask,
        )

        # --- Store in rolling buffer ---
        estimate = _FrameEstimate(
            parallax=parallax,
            persistence=persistence,
            stability=stability,
            confidence=frame_confidence,
            occlusion_mask=flow_result.occlusion_mask,
            global_motion_mag=global_motion_mag,
        )
        self._buffer.append(estimate)

        # --- Confidence-weighted averaging across buffer ---
        avg_parallax, avg_persistence, avg_stability, avg_confidence = (
            self._aggregate_buffer(h, w)
        )

        # --- Determine effective weights (stationary camera handling) ---
        eff_pw, eff_per_w, eff_sw = self._effective_weights()

        # --- Signal fusion ---
        background_score = self._fuse_signals(
            avg_parallax, avg_persistence, avg_stability,
            eff_pw, eff_per_w, eff_sw,
        )

        return CubeResult(
            background_score=background_score,
            confidence_map=avg_confidence,
            parallax_map=avg_parallax if emit_channel_maps else None,
            persistence_map=avg_persistence if emit_channel_maps else None,
            stability_map=avg_stability if emit_channel_maps else None,
        )

    def reset(self) -> None:
        """Clear the rolling buffer (e.g., on scene cut)."""
        self._buffer.clear()
        self._cumulative_displacement = None
        self._patch_history.clear()
        self._frame_history.clear()

    # ------------------------------------------------------------------
    # Global motion compensation
    # ------------------------------------------------------------------

    @staticmethod
    def _compensate_global_motion(
        forward_flow: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Estimate global affine motion via RANSAC and return residual flow.

        Returns
        -------
        residual_flow : np.ndarray
            ``[H, W, 2]`` float32 — flow after subtracting global motion.
        global_motion_mag : float
            Median magnitude of the estimated global motion field.
        """
        h, w = forward_flow.shape[:2]

        # Build source and destination point arrays from flow
        ys, xs = np.mgrid[:h, :w].astype(np.float32)
        src_pts = np.stack([xs.ravel(), ys.ravel()], axis=1)  # [N, 2]
        dst_pts = src_pts + forward_flow.reshape(-1, 2)       # [N, 2]

        # Subsample for RANSAC speed (use ~2000 points max)
        n_pts = src_pts.shape[0]
        max_pts = 2000
        if n_pts > max_pts:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_pts, max_pts, replace=False)
            src_sample = src_pts[idx]
            dst_sample = dst_pts[idx]
        else:
            src_sample = src_pts
            dst_sample = dst_pts

        # Estimate affine transform using RANSAC
        # estimateAffine2D returns (M, inliers) where M is 2x3
        affine_mat, _inliers = cv2.estimateAffine2D(
            src_sample, dst_sample,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=500,
            confidence=0.99,
        )

        if affine_mat is None:
            # Fallback: no valid affine found — treat all flow as residual
            global_flow = np.zeros_like(forward_flow)
        else:
            # Compute global motion field from affine: dst = M * [x, y, 1]^T
            # Global displacement = M * [x, y, 1]^T - [x, y]^T
            ones = np.ones((n_pts, 1), dtype=np.float32)
            src_h = np.hstack([src_pts, ones])  # [N, 3]
            global_dst = src_h @ affine_mat.T   # [N, 2]
            global_disp = (global_dst - src_pts).reshape(h, w, 2)
            global_flow = global_disp.astype(np.float32)

        residual_flow = forward_flow - global_flow

        # Median magnitude of global motion
        global_mag = np.sqrt(
            global_flow[..., 0] ** 2 + global_flow[..., 1] ** 2
        )
        global_motion_mag = float(np.median(global_mag))

        return residual_flow, global_motion_mag

    # ------------------------------------------------------------------
    # Parallax channel
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_parallax(residual_flow: np.ndarray) -> np.ndarray:
        """Compute parallax score from residual flow magnitude.

        ``parallax_score ∝ 1 / (magnitude + ε)`` — high score = background
        (small residual), low score = foreground (large residual).

        Normalized using robust 2nd/98th percentile anchors.
        """
        magnitude = np.sqrt(
            residual_flow[..., 0] ** 2 + residual_flow[..., 1] ** 2
        )

        # If the magnitude range is negligible, treat as uniform field
        mag_range = float(np.percentile(magnitude, 98) - np.percentile(magnitude, 2))
        if mag_range < 1e-4:
            return np.full(magnitude.shape, 0.5, dtype=np.float32)

        # Inverse: small motion → high score
        raw_score = 1.0 / (magnitude + _EPS)

        # Robust percentile normalization
        p2 = np.percentile(raw_score, 2)
        p98 = np.percentile(raw_score, 98)

        if p98 - p2 < _EPS:
            # Uniform field — everything is equally "background"
            return np.full(raw_score.shape, 0.5, dtype=np.float32)

        normalized = (raw_score - p2) / (p98 - p2)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence channel
    # ------------------------------------------------------------------

    def _compute_persistence(
        self,
        flow_result: FlowResult,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        """Compute persistence score from patch descriptor stability.

        If an RGB frame is provided, uses grayscale patch descriptors
        (local mean + variance in a small window).  Otherwise, falls back
        to flow-field statistics as a proxy for neighborhood stability.
        """
        h, w = flow_result.forward_flow.shape[:2]

        if frame is not None:
            # Convert to grayscale descriptor
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = np.mean(frame, axis=2).astype(np.float32)
            else:
                gray = frame.astype(np.float32)
            descriptor = self._compute_patch_descriptor(gray)
        else:
            # Fallback: use flow magnitude as a proxy descriptor
            mag = np.sqrt(
                flow_result.forward_flow[..., 0] ** 2
                + flow_result.forward_flow[..., 1] ** 2
            )
            descriptor = self._compute_patch_descriptor(mag)

        self._patch_history.append(descriptor)

        if len(self._patch_history) < 2:
            # Not enough history — assume moderate persistence
            return np.full((h, w), 0.5, dtype=np.float32)

        # Measure temporal stability: variance of descriptors across frames
        descriptors = np.stack(list(self._patch_history), axis=0)  # [T, H, W]
        temporal_var = np.var(descriptors, axis=0)  # [H, W]

        # Low variance → high persistence (background)
        # Use exponential decay: persistence = exp(-k * variance)
        k = 5.0
        persistence = np.exp(-k * temporal_var)

        return np.clip(persistence, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _compute_patch_descriptor(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Compute a local patch descriptor (blurred local mean).

        Uses a box filter to capture neighborhood structure.
        """
        return cv2.blur(image, (ksize, ksize)).astype(np.float32)

    # ------------------------------------------------------------------
    # Positional stability channel
    # ------------------------------------------------------------------

    def _compute_stability(
        self,
        forward_flow: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Compute positional stability from cumulative displacement.

        Score decays with displacement from first-seen position:
        ``score = exp(-displacement / scale)``.
        Computed at neighborhood level via Gaussian blur.
        """
        if self._cumulative_displacement is None:
            self._cumulative_displacement = np.zeros((h, w, 2), dtype=np.float32)

        # Accumulate displacement
        self._cumulative_displacement = (
            self._cumulative_displacement + forward_flow
        )

        # Magnitude of cumulative displacement
        disp_mag = np.sqrt(
            self._cumulative_displacement[..., 0] ** 2
            + self._cumulative_displacement[..., 1] ** 2
        )

        # Decay function: zero drift → 1.0, large drift → 0.0
        scale = max(float(np.percentile(disp_mag, 95)) + _EPS, 1.0)
        raw_stability = np.exp(-disp_mag / scale)

        # Neighborhood-level: blur to incorporate surrounding patch coherence
        stability = cv2.GaussianBlur(
            raw_stability.astype(np.float32), (7, 7), sigmaX=2.0
        )

        return np.clip(stability, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Per-frame confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_frame_confidence(
        parallax: np.ndarray,
        persistence: np.ndarray,
        stability: np.ndarray,
        residual_flow: np.ndarray,
        occlusion_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute per-pixel confidence for this frame.

        Based on:
        - Cross-channel consistency (low variance → higher confidence)
        - Parallax signal strength (higher residual magnitude → higher confidence)
        - Absence of occlusion (non-occluded → higher confidence)
        """
        # Cross-channel consistency: stack and measure variance
        channels = np.stack([parallax, persistence, stability], axis=0)  # [3, H, W]
        channel_var = np.var(channels, axis=0)  # [H, W]
        consistency = np.exp(-3.0 * channel_var)  # low var → high consistency

        # Parallax strength: magnitude of residual flow
        res_mag = np.sqrt(
            residual_flow[..., 0] ** 2 + residual_flow[..., 1] ** 2
        )
        # Normalize to [0, 1] using a soft saturation
        parallax_strength = 1.0 - np.exp(-0.5 * res_mag)

        # Occlusion penalty
        visibility = 1.0 - occlusion_mask

        # Combine
        confidence = consistency * 0.4 + parallax_strength * 0.3 + visibility * 0.3

        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Buffer aggregation
    # ------------------------------------------------------------------

    def _aggregate_buffer(
        self, h: int, w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Confidence-weighted average of channel estimates across the buffer.

        Occluded pixels in a given frame are excluded from the average for
        that pixel across all three channels.
        """
        if not self._buffer:
            empty = np.full((h, w), 0.5, dtype=np.float32)
            return empty, empty.copy(), empty.copy(), empty.copy()

        # Accumulate weighted sums
        par_sum = np.zeros((h, w), dtype=np.float64)
        per_sum = np.zeros((h, w), dtype=np.float64)
        stb_sum = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)
        conf_sum = np.zeros((h, w), dtype=np.float64)
        conf_count = np.zeros((h, w), dtype=np.float64)

        for est in self._buffer:
            # Exclude occluded pixels
            visible = 1.0 - est.occlusion_mask
            w_pixel = est.confidence * visible  # per-pixel weight

            par_sum += est.parallax * w_pixel
            per_sum += est.persistence * w_pixel
            stb_sum += est.stability * w_pixel
            weight_sum += w_pixel

            conf_sum += est.confidence * visible
            conf_count += visible

        # Avoid division by zero
        safe_weight = np.maximum(weight_sum, _EPS)
        safe_count = np.maximum(conf_count, _EPS)

        avg_parallax = np.clip(par_sum / safe_weight, 0.0, 1.0).astype(np.float32)
        avg_persistence = np.clip(per_sum / safe_weight, 0.0, 1.0).astype(np.float32)
        avg_stability = np.clip(stb_sum / safe_weight, 0.0, 1.0).astype(np.float32)
        avg_confidence = np.clip(conf_sum / safe_count, 0.0, 1.0).astype(np.float32)

        return avg_parallax, avg_persistence, avg_stability, avg_confidence

    # ------------------------------------------------------------------
    # Stationary camera handling
    # ------------------------------------------------------------------

    def _effective_weights(self) -> tuple[float, float, float]:
        """Compute effective channel weights, adjusting for stationary camera.

        When median global motion across the buffer is below
        ``stationary_threshold``, parallax weight is reduced and persistence
        + stability weights are increased proportionally.
        """
        if not self._buffer:
            return self.parallax_weight, self.persistence_weight, self.stability_weight

        median_motion = float(np.median(
            [est.global_motion_mag for est in self._buffer]
        ))

        if median_motion >= self.stationary_threshold:
            return self.parallax_weight, self.persistence_weight, self.stability_weight

        # Reduce parallax weight proportionally to how stationary the camera is
        # ratio = 0 when no motion, 1 when at threshold
        ratio = median_motion / self.stationary_threshold
        eff_parallax = self.parallax_weight * ratio

        # Redistribute the freed weight to persistence and stability
        freed = self.parallax_weight - eff_parallax
        other_total = self.persistence_weight + self.stability_weight
        if other_total < _EPS:
            # Edge case: all weight was on parallax
            eff_persistence = freed / 2.0
            eff_stability = freed / 2.0
        else:
            eff_persistence = self.persistence_weight + freed * (
                self.persistence_weight / other_total
            )
            eff_stability = self.stability_weight + freed * (
                self.stability_weight / other_total
            )

        return eff_parallax, eff_persistence, eff_stability

    # ------------------------------------------------------------------
    # Signal fusion
    # ------------------------------------------------------------------

    def _fuse_signals(
        self,
        parallax: np.ndarray,
        persistence: np.ndarray,
        stability: np.ndarray,
        pw: float,
        per_w: float,
        sw: float,
    ) -> np.ndarray:
        """Fuse three channel scores into a single Background_Score.

        Parameters
        ----------
        parallax, persistence, stability : np.ndarray
            Per-channel scores in [0, 1].
        pw, per_w, sw : float
            Effective weights for each channel.

        Returns
        -------
        np.ndarray
            ``[H, W]`` float32 Background_Score in [0.0, 1.0].
        """
        if self.fusion_mode == "blend":
            score = pw * parallax + per_w * persistence + sw * stability
        elif self.fusion_mode == "max":
            score = np.maximum(np.maximum(parallax, persistence), stability)
        elif self.fusion_mode == "min":
            score = np.minimum(np.minimum(parallax, persistence), stability)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode!r}")

        return np.clip(score, 0.0, 1.0).astype(np.float32)
