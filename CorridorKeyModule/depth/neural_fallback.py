"""Optional neural depth estimation fallback using Depth Anything V2.

This module provides a monocular depth estimation adapter that can be used
as a fallback when the classical parallax-based pipeline reports low
confidence (e.g., static camera, insufficient motion).

Disabled by default — requires explicit opt-in via ``--depth-fallback``.
"""

from __future__ import annotations

import logging

import numpy as np

from .data_models import CubeResult

logger = logging.getLogger(__name__)


class NeuralDepthFallback:
    """Adapter for Depth Anything V2 Small monocular depth estimation.

    Parameters
    ----------
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).

    Raises
    ------
    RuntimeError
        If the model cannot be loaded at construction time (fail fast).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self._transform = None

        try:
            import torch
            from torchvision import transforms

            self._torch = torch

            # Attempt to load Depth Anything V2 Small via torch hub
            logger.info("Loading Depth Anything V2 Small model on device=%s", device)
            model = torch.hub.load(
                "depth-anything/Depth-Anything-V2",
                "depth_anything_v2_vits",
                pretrained=True,
                trust_repo=True,
            )
            model = model.to(device).eval()
            self._model = model

            # Standard transform for the model input
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(518),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            logger.info("Depth Anything V2 Small model loaded successfully.")

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Depth Anything V2 model: {exc}. "
                "Ensure 'torch' and 'torchvision' are installed and the model "
                "is accessible. The neural fallback cannot be used."
            ) from exc

    def estimate(self, frame: np.ndarray) -> CubeResult:
        """Run monocular depth estimation on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame as [H, W, 3] float32 in [0, 1].

        Returns
        -------
        CubeResult
            ``background_score`` derived from model depth (normalized to
            [0, 1], far = high score = background), ``confidence_map`` set
            to 1.0 everywhere, channel maps = None.
        """
        import torch

        h, w = frame.shape[:2]

        # Convert float32 [0,1] RGB to uint8 for the transform pipeline
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        # Apply transform and add batch dimension
        input_tensor = self._transform(frame_uint8).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            depth = self._model(input_tensor)

        # depth shape: [1, 1, H', W'] or [1, H', W'] depending on model
        if depth.dim() == 4:
            depth = depth.squeeze(0).squeeze(0)
        elif depth.dim() == 3:
            depth = depth.squeeze(0)

        depth_np = depth.cpu().numpy().astype(np.float32)

        # Resize to original frame resolution
        if depth_np.shape != (h, w):
            import cv2
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]: far = high score = background
        # Depth Anything outputs relative depth where higher = closer,
        # so we invert: background_score = 1 - normalized_depth
        d_min = depth_np.min()
        d_max = depth_np.max()
        if d_max - d_min > 1e-8:
            normalized = (depth_np - d_min) / (d_max - d_min)
        else:
            normalized = np.zeros_like(depth_np)

        # Invert: model's "far" (low depth) → high background score
        background_score = 1.0 - normalized

        # Confidence is 1.0 everywhere (model produces complete depth map)
        confidence_map = np.ones((h, w), dtype=np.float32)

        return CubeResult(
            background_score=background_score,
            confidence_map=confidence_map,
            parallax_map=None,
            persistence_map=None,
            stability_map=None,
        )
