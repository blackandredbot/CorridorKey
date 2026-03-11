"""EXR I/O utilities for depth maps, confidence maps, channel maps, and flow fields.

All functions use 32-bit float EXR via OpenCV.  The ``OPENCV_IO_ENABLE_OPENEXR``
environment variable is set before importing ``cv2`` to ensure the OpenEXR
codec is available.

Typical usage::

    from CorridorKeyModule.depth.exr_io import write_depth_map, read_depth_map

    write_depth_map("/tmp/score.exr", background_score)
    loaded = read_depth_map("/tmp/score.exr")
    assert np.allclose(background_score, loaded, atol=1e-7)
"""

from __future__ import annotations

import os

# Enable OpenEXR codec in OpenCV — must happen before cv2 import.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

# EXR write flags: 32-bit float output
_EXR_FLOAT32_FLAGS = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]


# ------------------------------------------------------------------
# Single-channel maps (depth, background_score, confidence, etc.)
# ------------------------------------------------------------------

def write_depth_map(path: str, data: np.ndarray) -> None:
    """Write a single-channel float32 map to a 32-bit float EXR file.

    Works for Background_Score maps, confidence maps, parallax maps,
    persistence maps, stability maps, and alpha mattes — any ``[H, W]``
    float32 array.

    Parameters
    ----------
    path : str
        Destination file path (should end in ``.exr``).
    data : np.ndarray
        Single-channel image of shape ``[H, W]`` with dtype ``float32``.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D array [H, W], got shape {data.shape}"
        )
    arr = data.astype(np.float32, copy=False)
    cv2.imwrite(path, arr, _EXR_FLOAT32_FLAGS)


def read_depth_map(path: str) -> np.ndarray:
    """Read a single-channel float32 map from a 32-bit float EXR file.

    Parameters
    ----------
    path : str
        Source EXR file path.

    Returns
    -------
    np.ndarray
        ``[H, W]`` float32 array.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If OpenCV cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read EXR file: {path}")

    # OpenCV may load single-channel EXR as [H, W, 1] or [H, W, 3].
    # Collapse to [H, W].
    if img.ndim == 3:
        img = img[:, :, 0]

    return img.astype(np.float32, copy=False)


# ------------------------------------------------------------------
# Two-channel flow fields
# ------------------------------------------------------------------

def write_flow_field(path: str, flow: np.ndarray) -> None:
    """Write a two-channel float32 flow field to a 32-bit float EXR file.

    Parameters
    ----------
    path : str
        Destination file path (should end in ``.exr``).
    flow : np.ndarray
        Flow field of shape ``[H, W, 2]`` with dtype ``float32``
        representing ``(dx, dy)`` displacement in pixel units.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"Expected 3-D array [H, W, 2], got shape {flow.shape}"
        )
    arr = flow.astype(np.float32, copy=False)
    cv2.imwrite(path, arr, _EXR_FLOAT32_FLAGS)


def read_flow_field(path: str) -> np.ndarray:
    """Read a two-channel float32 flow field from a 32-bit float EXR file.

    Parameters
    ----------
    path : str
        Source EXR file path.

    Returns
    -------
    np.ndarray
        ``[H, W, 2]`` float32 array.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If OpenCV cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read EXR file: {path}")

    # OpenCV loads multi-channel EXR as [H, W, C].
    # A 2-channel EXR may be loaded as [H, W, 3] with the third channel
    # zeroed out, or as [H, W, 2].  Extract the first two channels.
    if img.ndim == 2:
        raise RuntimeError(
            f"Expected multi-channel EXR, got single-channel: {path}"
        )

    return img[:, :, :2].astype(np.float32, copy=False)
