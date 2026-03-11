"""Optical Flow Engine — dense pixel correspondence between consecutive frames.

Supports two methods:
- ``"farneback"``: CPU-based classical optical flow via ``cv2.calcOpticalFlowFarneback``.
- ``"raft"``: GPU-accelerated learning-based flow via ``torchvision.models.optical_flow.raft_small``.

Occlusion detection uses forward-backward consistency: a pixel is marked
occluded when ``||forward(p) + backward(warp(p))|| > consistency_threshold``.
"""

from __future__ import annotations

import numpy as np
import cv2

from .data_models import FlowResult


_VALID_METHODS = {"farneback", "raft"}


class OpticalFlowEngine:
    """Compute bidirectional optical flow between two RGB frames.

    Parameters
    ----------
    method : str
        ``"farneback"`` (CPU, no model weights) or ``"raft"`` (GPU-accelerated).
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda"``). Only used by RAFT.
    consistency_threshold : float
        Forward-backward consistency error threshold in pixels.  Pixels whose
        error exceeds this value are flagged as occluded.  Default ``1.5``.
    """

    def __init__(
        self,
        method: str = "farneback",
        device: str = "cpu",
        consistency_threshold: float = 1.5,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {method!r}"
            )
        if consistency_threshold <= 0:
            raise ValueError(
                f"consistency_threshold must be > 0, got {consistency_threshold}"
            )

        self.method = method
        self.device = device
        self.consistency_threshold = consistency_threshold

        # Lazy-load RAFT model on first use
        self._raft_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, frame_a: np.ndarray, frame_b: np.ndarray) -> FlowResult:
        """Compute bidirectional flow between two RGB frames.

        Parameters
        ----------
        frame_a, frame_b : np.ndarray
            RGB frames of shape ``[H, W, 3]``, dtype ``float32``, values in
            ``[0, 1]``.  Both frames must have the same spatial resolution.

        Returns
        -------
        FlowResult
            ``forward_flow`` and ``backward_flow`` as ``[H, W, 2]`` float32 in
            pixel units, plus an ``occlusion_mask`` of ``[H, W]`` float32 with
            values in ``{0.0, 1.0}``.
        """
        self._validate_frame(frame_a, "frame_a")
        self._validate_frame(frame_b, "frame_b")
        if frame_a.shape != frame_b.shape:
            raise ValueError(
                f"Frame resolution mismatch: frame_a {frame_a.shape} vs "
                f"frame_b {frame_b.shape}"
            )

        if self.method == "farneback":
            forward_flow, backward_flow = self._compute_farneback(frame_a, frame_b)
        else:
            forward_flow, backward_flow = self._compute_raft(frame_a, frame_b)

        occlusion_mask = self._compute_occlusion_mask(
            forward_flow, backward_flow
        )

        return FlowResult(
            forward_flow=forward_flow,
            backward_flow=backward_flow,
            occlusion_mask=occlusion_mask,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame: np.ndarray, name: str) -> None:
        """Ensure *frame* is [H, W, 3] float32 in [0, 1]."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"{name} must have shape [H, W, 3], got {frame.shape}"
            )
        if frame.dtype != np.float32:
            raise ValueError(
                f"{name} must be float32, got {frame.dtype}"
            )
        if frame.min() < 0.0 or frame.max() > 1.0:
            raise ValueError(
                f"{name} values must be in [0, 1], got "
                f"[{frame.min()}, {frame.max()}]"
            )

    # ------------------------------------------------------------------
    # Farnebäck path (CPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_farneback(
        frame_a: np.ndarray, frame_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward and backward flow using Farnebäck."""
        gray_a = cv2.cvtColor(
            (frame_a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        gray_b = cv2.cvtColor(
            (frame_b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )

        # Farnebäck parameters tuned for general-purpose use
        fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        forward_flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None, **fb_params
        ).astype(np.float32)

        backward_flow = cv2.calcOpticalFlowFarneback(
            gray_b, gray_a, None, **fb_params
        ).astype(np.float32)

        return forward_flow, backward_flow

    # ------------------------------------------------------------------
    # RAFT path (GPU)
    # ------------------------------------------------------------------

    def _ensure_raft_model(self):
        """Lazy-load the RAFT-Small model."""
        if self._raft_model is not None:
            return

        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
        model = model.to(self.device).eval()
        self._raft_model = model
        self._raft_transforms = weights.transforms()

    def _compute_raft(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward and backward flow using RAFT-Small."""
        import torch

        self._ensure_raft_model()

        def _to_tensor(frame: np.ndarray) -> torch.Tensor:
            # [H, W, 3] float32 [0,1] → [1, 3, H, W] uint8 for RAFT transforms
            img_uint8 = (frame * 255).astype(np.uint8)
            t = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0)
            return t.to(self.device)

        t_a = _to_tensor(frame_a)
        t_b = _to_tensor(frame_b)

        # Apply RAFT preprocessing transforms
        t_a_prep, t_b_prep = self._raft_transforms(t_a, t_b)

        with torch.no_grad():
            # raft_small returns a list of flow predictions; last is finest
            forward_flows = self._raft_model(t_a_prep, t_b_prep)
            backward_flows = self._raft_model(t_b_prep, t_a_prep)

        # [1, 2, H, W] → [H, W, 2] float32 numpy
        forward_flow = (
            forward_flows[-1].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )
        backward_flow = (
            backward_flows[-1].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )

        return forward_flow, backward_flow

    # ------------------------------------------------------------------
    # Occlusion detection via forward-backward consistency
    # ------------------------------------------------------------------

    def _compute_occlusion_mask(
        self,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
    ) -> np.ndarray:
        """Compute occlusion mask from forward-backward consistency error.

        For each pixel *p*, warp the backward flow by the forward flow and
        check whether ``||forward(p) + backward(warp(p))|| > threshold``.

        Returns
        -------
        np.ndarray
            ``[H, W]`` float32 with values in ``{0.0, 1.0}``.
            ``1.0`` = occluded, ``0.0`` = visible.
        """
        h, w = forward_flow.shape[:2]

        # Build coordinate grids
        ys, xs = np.mgrid[:h, :w].astype(np.float32)

        # Warped coordinates: where forward flow sends each pixel
        wx = xs + forward_flow[..., 0]
        wy = ys + forward_flow[..., 1]

        # Sample backward flow at warped locations using bilinear interpolation
        backward_at_warp = np.empty_like(forward_flow)
        backward_at_warp[..., 0] = cv2.remap(
            backward_flow[..., 0], wx, wy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        backward_at_warp[..., 1] = cv2.remap(
            backward_flow[..., 1], wx, wy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )

        # Consistency error: ||forward(p) + backward(warp(p))||
        error = np.sqrt(
            (forward_flow[..., 0] + backward_at_warp[..., 0]) ** 2
            + (forward_flow[..., 1] + backward_at_warp[..., 1]) ** 2
        )

        occlusion_mask = (error > self.consistency_threshold).astype(np.float32)
        return occlusion_mask
