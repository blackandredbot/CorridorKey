"""Depth keying pipeline — classical geometry-based foreground/background separation."""

from __future__ import annotations

from .data_models import CubeResult as CubeResult
from .data_models import DepthKeyingConfig as DepthKeyingConfig
from .data_models import FlowResult as FlowResult
from .depth_keying_engine import DepthKeyingEngine as DepthKeyingEngine
from .depth_thresholder import DepthThresholder as DepthThresholder
from .exr_io import read_depth_map as read_depth_map
from .exr_io import read_flow_field as read_flow_field
from .exr_io import write_depth_map as write_depth_map
from .exr_io import write_flow_field as write_flow_field
from .image_cube import ImageCubeBuilder as ImageCubeBuilder
from .mask_refiner import MaskRefiner as MaskRefiner
from .neural_fallback import NeuralDepthFallback as NeuralDepthFallback
from .optical_flow import OpticalFlowEngine as OpticalFlowEngine
