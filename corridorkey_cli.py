"""CorridorKey command-line interface and interactive wizard.

This module handles CLI subcommands, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage:
    uv run corridorkey wizard "V:\\..."
    uv run corridorkey run-inference
    uv run corridorkey generate-alphas
    uv run corridorkey list-clips
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import sys
import warnings
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from clip_manager import (
    LINUX_MOUNT_ROOT,
    ClipEntry,
    InferenceSettings,
    generate_alphas,
    is_video_file,
    map_path,
    organize_target,
    run_inference,
    run_videomama,
    scan_clips,
)
from device_utils import resolve_device
from backend.ffmpeg_tools import extract_frames, probe_video, stitch_video
from CorridorKeyModule.depth import DepthKeyingConfig, DepthKeyingEngine
from CorridorKeyModule.depth.data_models import MotionBlurConfig, PlateSubtractionConfig
from CorridorKeyModule.depth.motion_blur_refiner import MotionBlurRefiner
from CorridorKeyModule.depth.plate_subtraction_engine import PlateSubtractionEngine

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="corridorkey",
    help="Neural network green screen keying for professional VFX pipelines.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _configure_environment() -> None:
    """Set up logging and warnings for interactive CLI use."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# Progress helpers (callback protocol → rich.progress)
# ---------------------------------------------------------------------------


class ProgressContext:
    """Context manager bridging clip_manager callbacks to Rich progress bars.

    clip_manager's callback protocol doesn't know about Rich, so this class
    owns the Progress instance and exposes bound methods as callbacks.
    ``__exit__`` always cleans up, even if inference raises.
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self._frame_task_id: TaskID | None = None

    def __enter__(self) -> "ProgressContext":
        self._progress.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.__exit__(*exc)

    def on_clip_start(self, clip_name: str, num_frames: int) -> None:
        """Callback: reset the progress bar for a new clip."""
        if self._frame_task_id is not None:
            self._progress.remove_task(self._frame_task_id)
        self._frame_task_id = self._progress.add_task(f"[cyan]{clip_name}", total=num_frames)

    def on_frame_complete(self, frame_idx: int, num_frames: int) -> None:
        """Callback: advance the progress bar by one frame."""
        if self._frame_task_id is not None:
            self._progress.advance(self._frame_task_id)


def _on_clip_start_log_only(clip_name: str, total_clips: int) -> None:
    """Clip-level callback for generate-alphas.

    Unlike ProgressContext.on_clip_start (frame-level granularity with a Rich
    task per clip), GVM has no per-frame progress so we just log.
    """
    console.print(f"  Processing [bold]{clip_name}[/bold] ({total_clips} total)")


# ---------------------------------------------------------------------------
# Inference settings prompt (rich.prompt — CLI layer only)
# ---------------------------------------------------------------------------


def _prompt_inference_settings(
    *,
    default_linear: bool | None = None,
    default_despill: int | None = None,
    default_despeckle: bool | None = None,
    default_despeckle_size: int | None = None,
    default_refiner: float | None = None,
) -> InferenceSettings:
    """Interactively prompt for inference settings, skipping any pre-filled values."""
    console.print(Panel("Inference Settings", style="bold cyan"))

    if default_linear is not None:
        input_is_linear = default_linear
    else:
        gamma_choice = Prompt.ask(
            "Input colorspace",
            choices=["linear", "srgb"],
            default="srgb",
        )
        input_is_linear = gamma_choice == "linear"

    if default_despill is not None:
        despill_int = max(0, min(10, default_despill))
    else:
        despill_int = IntPrompt.ask(
            "Despill strength (0–10, 10 = max despill)",
            default=5,
        )
        despill_int = max(0, min(10, despill_int))
    despill_strength = despill_int / 10.0

    if default_despeckle is not None:
        auto_despeckle = default_despeckle
    else:
        auto_despeckle = Confirm.ask(
            "Enable auto-despeckle (removes tracking dots)?",
            default=True,
        )

    despeckle_size = default_despeckle_size if default_despeckle_size is not None else 400
    if auto_despeckle and default_despeckle_size is None and default_despeckle is None:
        despeckle_size = IntPrompt.ask(
            "Despeckle size (min pixels for a spot)",
            default=400,
        )
        despeckle_size = max(0, despeckle_size)

    if default_refiner is not None:
        refiner_scale = default_refiner
    else:
        refiner_val = Prompt.ask(
            "Refiner strength multiplier [dim](experimental)[/dim]",
            default="1.0",
        )
        try:
            refiner_scale = float(refiner_val)
        except ValueError:
            refiner_scale = 1.0

    return InferenceSettings(
        input_is_linear=input_is_linear,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
    )


# ---------------------------------------------------------------------------
# Typer callback (shared options)
# ---------------------------------------------------------------------------


@app.callback()
def app_callback(
    ctx: typer.Context,
    device: Annotated[
        str,
        typer.Option(help="Compute device: auto, cuda, mps, cpu"),
    ] = "auto",
) -> None:
    """Neural network green screen keying for professional VFX pipelines."""
    _configure_environment()
    ctx.ensure_object(dict)
    ctx.obj["device"] = resolve_device(device)
    logger.info("Using device: %s", ctx.obj["device"])


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("list-clips")
def list_clips_cmd(ctx: typer.Context) -> None:
    """List all clips in ClipsForInference and their status."""
    scan_clips()


@app.command("generate-alphas")
def generate_alphas_cmd(ctx: typer.Context) -> None:
    """Generate coarse alpha hints via GVM for clips missing them."""
    clips = scan_clips()
    with console.status("[bold green]Loading GVM model..."):
        generate_alphas(clips, device=ctx.obj["device"], on_clip_start=_on_clip_start_log_only)
    console.print("[bold green]Alpha generation complete.")

def _prompt_depth_settings() -> dict:
    """Interactively prompt for depth keying parameters.

    Returns a dict of keyword arguments suitable for passing
    to ``_run_depth_inference()``.
    """
    console.print(Panel("Depth Keying Settings", style="bold cyan"))

    # --- Basic settings (always asked) ---
    flow_method = Prompt.ask(
        "Flow method",
        choices=["farneback", "raft"],
        default="farneback",
    )

    # depth_threshold with range validation
    while True:
        raw = Prompt.ask("Depth threshold [dim](0.0–1.0)[/dim]", default="0.5")
        try:
            depth_threshold = float(raw)
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
            continue
        if 0.0 <= depth_threshold <= 1.0:
            break
        console.print("[red]Must be in [0.0, 1.0].[/red]")

    # depth_falloff with range validation
    while True:
        raw = Prompt.ask("Depth falloff [dim](0.0–0.5)[/dim]", default="0.05")
        try:
            depth_falloff = float(raw)
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
            continue
        if 0.0 <= depth_falloff <= 0.5:
            break
        console.print("[red]Must be in [0.0, 0.5].[/red]")

    fusion_mode = Prompt.ask(
        "Fusion mode",
        choices=["blend", "max", "min"],
        default="blend",
    )

    depth_fallback = Confirm.ask("Enable neural depth fallback?", default=False)

    # --- Advanced settings (behind gate) ---
    parallax_weight = 0.4
    persistence_weight = 0.3
    stability_weight = 0.3
    cube_buffer_size = 10
    refinement_strength = 1.0
    save_depth_maps = False
    save_flow = False

    if Confirm.ask("Advanced settings?", default=False):
        # Weight validation loop
        while True:
            raw_pw = Prompt.ask("Parallax weight [dim](0.0–1.0)[/dim]", default="0.4")
            raw_per = Prompt.ask("Persistence weight [dim](0.0–1.0)[/dim]", default="0.3")
            raw_sw = Prompt.ask("Stability weight [dim](0.0–1.0)[/dim]", default="0.3")
            try:
                parallax_weight = float(raw_pw)
                persistence_weight = float(raw_per)
                stability_weight = float(raw_sw)
            except ValueError:
                console.print("[red]Please enter valid numbers for all weights.[/red]")
                continue
            if any(not 0.0 <= w <= 1.0 for w in (parallax_weight, persistence_weight, stability_weight)):
                console.print("[red]Each weight must be in [0.0, 1.0].[/red]")
                continue
            if abs(parallax_weight + persistence_weight + stability_weight - 1.0) > 1e-9:
                console.print("[yellow]Weights must sum to 1.0. Please re-enter.[/yellow]")
                continue
            break

        raw_buf = Prompt.ask("Cube buffer size [dim](≥ 2)[/dim]", default="10")
        try:
            cube_buffer_size = max(2, int(raw_buf))
        except ValueError:
            cube_buffer_size = 10

        raw_ref = Prompt.ask("Refinement strength [dim](0.0–1.0)[/dim]", default="1.0")
        try:
            refinement_strength = max(0.0, min(1.0, float(raw_ref)))
        except ValueError:
            refinement_strength = 1.0

        save_depth_maps = Confirm.ask("Save depth maps?", default=False)
        save_flow = Confirm.ask("Save flow fields?", default=False)

    return {
        "depth_threshold": depth_threshold,
        "depth_falloff": depth_falloff,
        "refinement_strength": refinement_strength,
        "flow_method": flow_method,
        "cube_buffer_size": cube_buffer_size,
        "parallax_weight": parallax_weight,
        "persistence_weight": persistence_weight,
        "stability_weight": stability_weight,
        "fusion_mode": fusion_mode,
        "save_depth_maps": save_depth_maps,
        "save_flow": save_flow,
        "depth_fallback": depth_fallback,
    }


def _has_input_frames(clip: ClipEntry) -> bool:
    """Check if a clip has an Input/ folder containing at least 2 image frames."""
    input_dir = os.path.join(clip.root_path, "Input")
    if not os.path.isdir(input_dir):
        return False
    exts = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    frames = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    return len(frames) >= 2


def _extract_video_frames(clip: ClipEntry) -> float | None:
    """Extract frames from a video input into an ``Input/`` directory.

    Returns the video fps for later stitching use, or ``None`` if extraction
    was not needed (non-video input or frames already exist).
    """
    if clip.input_asset is None or clip.input_asset.type != "video":
        return None

    input_dir = os.path.join(clip.root_path, "Input")

    # Idempotent: skip if Input/ already has ≥2 image frames
    if os.path.isdir(input_dir):
        exts = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        existing = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
        if len(existing) >= 2:
            logger.info(f"Clip '{clip.name}': Input/ already has {len(existing)} frames, skipping extraction.")
            return None

    video_path = clip.input_asset.path

    # Try ffmpeg-based extraction first
    try:
        info = probe_video(video_path)
        fps = info["fps"]
        extract_frames(video_path, input_dir, pattern="frame_%06d.png")
        logger.info(f"Clip '{clip.name}': Extracted frames via ffmpeg at {fps} fps.")
        return fps
    except RuntimeError:
        logger.warning(f"Clip '{clip.name}': ffmpeg not available, falling back to cv2 for frame extraction.")

    # cv2 fallback
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Clip '{clip.name}': Could not open video '{video_path}' with cv2.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    os.makedirs(input_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(input_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        logger.error(f"Clip '{clip.name}': cv2 extracted 0 frames from '{video_path}'.")
        return None

    logger.info(f"Clip '{clip.name}': Extracted {frame_idx} frames via cv2 at {fps} fps.")
    return fps

def _stitch_comp_video(clip: ClipEntry, fps: float) -> None:
    """Stitch ``Comp/`` PNG frames back into a ``Comp.mp4`` video.

    This is called after depth inference when the original input was a video
    file, so the user gets a video output matching the original format.

    If ffmpeg is not available a warning is logged and the function returns
    without failing — the user still has the individual Comp frames.
    """
    comp_dir = os.path.join(clip.root_path, "Comp")

    if not os.path.isdir(comp_dir):
        return

    png_files = [f for f in os.listdir(comp_dir) if f.lower().endswith(".png")]
    if not png_files:
        return

    comp_video_path = os.path.join(clip.root_path, "Comp.mp4")

    try:
        stitch_video(comp_dir, comp_video_path, fps=fps)
        logger.info(f"Clip '{clip.name}': Stitched {len(png_files)} frames into '{comp_video_path}'.")
    except RuntimeError:
        logger.warning(
            f"Clip '{clip.name}': Could not stitch Comp video (ffmpeg not found). "
            f"Comp frames are still available in '{comp_dir}'."
        )





def _run_depth_inference(
    clips: list[ClipEntry],
    device: str,
    *,
    depth_threshold: float,
    depth_falloff: float,
    refinement_strength: float,
    flow_method: str,
    cube_buffer_size: int,
    parallax_weight: float,
    persistence_weight: float,
    stability_weight: float,
    fusion_mode: str,
    save_depth_maps: bool,
    save_flow: bool,
    depth_fallback: bool,
) -> None:
    """Validate depth parameters, build engine, and process all clips."""
    # Construct DepthKeyingConfig — validation happens in __post_init__
    try:
        config = DepthKeyingConfig(
            flow_method=flow_method,
            depth_threshold=depth_threshold,
            depth_falloff=depth_falloff,
            cube_buffer_size=cube_buffer_size,
            refinement_strength=refinement_strength,
            parallax_weight=parallax_weight,
            persistence_weight=persistence_weight,
            stability_weight=stability_weight,
            fusion_mode=fusion_mode,
            depth_fallback=depth_fallback,
            save_depth_maps=save_depth_maps,
            save_flow=save_flow,
        )
    except ValueError as exc:
        console.print(f"[bold red]Invalid depth parameter:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    engine = DepthKeyingEngine(
        device=device,
        flow_method=config.flow_method,
        depth_threshold=config.depth_threshold,
        depth_falloff=config.depth_falloff,
        cube_buffer_size=config.cube_buffer_size,
        refinement_strength=config.refinement_strength,
        parallax_weight=config.parallax_weight,
        persistence_weight=config.persistence_weight,
        stability_weight=config.stability_weight,
        fusion_mode=config.fusion_mode,
        depth_fallback=config.depth_fallback,
        save_depth_maps=config.save_depth_maps,
        save_flow=config.save_flow,
    )

    with ProgressContext() as ctx_progress:
        for clip in clips:
            ctx_progress.on_clip_start(clip.name, 0)
            engine.process_clip(
                clip.root_path,
                on_frame_complete=ctx_progress.on_frame_complete,
            )

    console.print("[bold green]Depth inference complete.")


def _run_plate_subtraction_inference(
    clips: list[ClipEntry],
    device: str,
    *,
    difference_threshold: float,
    difference_falloff: float,
    color_space_mode: str,
    low_confidence_alpha: float,
    plate_search_radius: int,
    donor_threshold: float,
    max_iterations: int,
    convergence_threshold: float,
    flow_method: str,
    cube_buffer_size: int,
    fusion_mode: str,
    parallax_weight: float,
    persistence_weight: float,
    stability_weight: float,
    depth_threshold: float,
    depth_falloff: float,
    refinement_strength: float,
    save_clean_plates: bool,
    save_bootstrap: bool,
    save_flow: bool,
) -> None:
    """Validate plate-subtraction parameters, build engine, and process all clips."""
    try:
        config = PlateSubtractionConfig(
            difference_threshold=difference_threshold,
            difference_falloff=difference_falloff,
            color_space_mode=color_space_mode,
            low_confidence_alpha=low_confidence_alpha,
            plate_search_radius=plate_search_radius,
            donor_threshold=donor_threshold,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            flow_method=flow_method,
            cube_buffer_size=cube_buffer_size,
            fusion_mode=fusion_mode,
            parallax_weight=parallax_weight,
            persistence_weight=persistence_weight,
            stability_weight=stability_weight,
            depth_threshold=depth_threshold,
            depth_falloff=depth_falloff,
            refinement_strength=refinement_strength,
            save_clean_plates=save_clean_plates,
            save_bootstrap=save_bootstrap,
            save_flow=save_flow,
        )
    except ValueError as exc:
        console.print(f"[bold red]Invalid plate-subtraction parameter:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    engine = PlateSubtractionEngine(config=config, device=device)

    with ProgressContext() as ctx_progress:
        for clip in clips:
            ctx_progress.on_clip_start(clip.name, 0)
            engine.process_clip(
                clip.root_path,
                on_frame_complete=ctx_progress.on_frame_complete,
            )

    console.print("[bold green]Plate subtraction inference complete.")


@app.command("run-inference")
def run_inference_cmd(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option(help="Inference backend: auto, torch, mlx"),
    ] = "auto",
    max_frames: Annotated[
        Optional[int],
        typer.Option("--max-frames", help="Limit frames per clip"),
    ] = None,
    linear: Annotated[
        Optional[bool],
        typer.Option("--linear/--srgb", help="Input colorspace (default: prompt)"),
    ] = None,
    despill: Annotated[
        Optional[int],
        typer.Option("--despill", help="Despill strength 0–10 (default: prompt)"),
    ] = None,
    despeckle: Annotated[
        Optional[bool],
        typer.Option("--despeckle/--no-despeckle", help="Auto-despeckle toggle (default: prompt)"),
    ] = None,
    despeckle_size: Annotated[
        Optional[int],
        typer.Option("--despeckle-size", help="Min pixel size for despeckle (default: prompt)"),
    ] = None,
    refiner: Annotated[
        Optional[float],
        typer.Option("--refiner", help="Refiner strength multiplier (default: prompt)"),
    ] = None,
    # --- Mode selection ---
    mode: Annotated[
        str,
        typer.Option(help="Pipeline mode: greenscreen (default), depth, or plate-subtraction"),
    ] = "greenscreen",
    # --- Depth-specific options ---
    depth_threshold: Annotated[
        float,
        typer.Option("--depth-threshold", help="Background score cutoff [0.0–1.0]"),
    ] = 0.5,
    depth_falloff: Annotated[
        float,
        typer.Option("--depth-falloff", help="Soft transition zone width [0.0–0.5]"),
    ] = 0.05,
    refinement_strength: Annotated[
        float,
        typer.Option("--refinement-strength", help="Mask refinement strength [0.0–1.0]"),
    ] = 1.0,
    flow_method: Annotated[
        str,
        typer.Option("--flow-method", help="Optical flow algorithm: farneback or raft"),
    ] = "farneback",
    cube_buffer_size: Annotated[
        int,
        typer.Option("--cube-buffer-size", help="Image cube rolling buffer depth (>= 2)"),
    ] = 10,
    parallax_weight: Annotated[
        float,
        typer.Option("--parallax-weight", help="Parallax channel weight [0.0–1.0]"),
    ] = 0.4,
    persistence_weight: Annotated[
        float,
        typer.Option("--persistence-weight", help="Persistence channel weight [0.0–1.0]"),
    ] = 0.3,
    stability_weight: Annotated[
        float,
        typer.Option("--stability-weight", help="Positional stability channel weight [0.0–1.0]"),
    ] = 0.3,
    fusion_mode: Annotated[
        str,
        typer.Option("--fusion-mode", help="Signal fusion mode: blend, max, or min"),
    ] = "blend",
    save_depth_maps: Annotated[
        bool,
        typer.Option("--save-depth-maps/--no-save-depth-maps", help="Write intermediate depth maps"),
    ] = False,
    save_flow: Annotated[
        bool,
        typer.Option("--save-flow/--no-save-flow", help="Write intermediate flow fields"),
    ] = False,
    depth_fallback: Annotated[
        bool,
        typer.Option("--depth-fallback/--no-depth-fallback", help="Enable neural depth fallback"),
    ] = False,
    # --- Motion blur refinement options ---
    refine_motion_blur: Annotated[
        bool,
        typer.Option("--refine-motion-blur/--no-refine-motion-blur", help="Enable motion blur alpha refinement post-processing"),
    ] = False,
    blur_threshold: Annotated[
        float,
        typer.Option("--blur-threshold", help="Min flow magnitude to classify as motion-blurred (> 0)"),
    ] = 2.0,
    kernel_profile: Annotated[
        str,
        typer.Option("--kernel-profile", help="Blur kernel profile: linear, cosine, or gaussian"),
    ] = "linear",
    temporal_smoothing: Annotated[
        float,
        typer.Option("--temporal-smoothing", help="EMA weight for temporal coherence (0.0, 1.0]"),
    ] = 0.3,
    division_epsilon: Annotated[
        float,
        typer.Option("--division-epsilon", help="Epsilon guard for division safety (> 0)"),
    ] = 1e-4,
    blur_dilation: Annotated[
        int,
        typer.Option("--blur-dilation", help="Morphological dilation radius for blur mask (>= 0)"),
    ] = 3,
    clean_plate: Annotated[
        Optional[str],
        typer.Option("--clean-plate", help="Explicit clean plate file path (EXR or PNG)"),
    ] = None,
    save_refined_fg: Annotated[
        bool,
        typer.Option("--save-refined-fg/--no-save-refined-fg", help="Write recovered foreground color to RefinedFG/"),
    ] = False,
    plate_search_radius: Annotated[
        int,
        typer.Option("--plate-search-radius", help="Neighboring frames to search for background donors (>= 1)"),
    ] = 10,
    plate_alpha_threshold: Annotated[
        float,
        typer.Option("--plate-alpha-threshold", help="Max alpha for reliable background donor (0.0, 1.0]"),
    ] = 0.1,
    static_clean_plate: Annotated[
        bool,
        typer.Option("--static-clean-plate/--no-static-clean-plate", help="Force legacy single-plate synthesis"),
    ] = False,
    # --- Plate-subtraction-specific options ---
    difference_threshold: Annotated[
        float,
        typer.Option("--difference-threshold", help="Subtraction difference cutoff (0.0, 1.0]"),
    ] = 0.05,
    difference_falloff: Annotated[
        float,
        typer.Option("--difference-falloff", help="Soft transition zone width [0.0, 0.5]"),
    ] = 0.03,
    donor_threshold: Annotated[
        float,
        typer.Option("--donor-threshold", help="Max bootstrap mask value for background donor (0.0, 1.0]"),
    ] = 0.3,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Iterative refinement passes [1, 5]"),
    ] = 2,
    convergence_threshold: Annotated[
        float,
        typer.Option("--convergence-threshold", help="Mean alpha diff to stop iterating (> 0)"),
    ] = 0.001,
    color_space_mode: Annotated[
        str,
        typer.Option("--color-space-mode", help="Difference mode: max_channel or luminance"),
    ] = "max_channel",
    low_confidence_alpha: Annotated[
        float,
        typer.Option("--low-confidence-alpha", help="Alpha for low-confidence plate regions [0.0, 1.0]"),
    ] = 1.0,
    save_clean_plates: Annotated[
        bool,
        typer.Option("--save-clean-plates/--no-save-clean-plates", help="Write synthesized clean plates to CleanPlate/"),
    ] = False,
    save_bootstrap: Annotated[
        bool,
        typer.Option("--save-bootstrap/--no-save-bootstrap", help="Write bootstrap masks to Bootstrap/"),
    ] = False,
) -> None:
    """Run CorridorKey inference on clips with Input + AlphaHint.

    Settings can be passed as flags for non-interactive use, or omitted to
    prompt interactively.

    Use --mode depth to run the depth keying pipeline instead of the default
    green-screen pipeline. Use --mode plate-subtraction for clean-plate
    subtraction keying.
    """
    clips = scan_clips()

    # --- Depth mode routing ---
    if mode == "depth":
        _run_depth_inference(
            clips,
            device=ctx.obj["device"],
            depth_threshold=depth_threshold,
            depth_falloff=depth_falloff,
            refinement_strength=refinement_strength,
            flow_method=flow_method,
            cube_buffer_size=cube_buffer_size,
            parallax_weight=parallax_weight,
            persistence_weight=persistence_weight,
            stability_weight=stability_weight,
            fusion_mode=fusion_mode,
            save_depth_maps=save_depth_maps,
            save_flow=save_flow,
            depth_fallback=depth_fallback,
        )
        return

    if mode == "plate-subtraction":
        _run_plate_subtraction_inference(
            clips,
            device=ctx.obj["device"],
            difference_threshold=difference_threshold,
            difference_falloff=difference_falloff,
            color_space_mode=color_space_mode,
            low_confidence_alpha=low_confidence_alpha,
            plate_search_radius=plate_search_radius,
            donor_threshold=donor_threshold,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            flow_method=flow_method,
            cube_buffer_size=cube_buffer_size,
            fusion_mode=fusion_mode,
            parallax_weight=parallax_weight,
            persistence_weight=persistence_weight,
            stability_weight=stability_weight,
            depth_threshold=depth_threshold,
            depth_falloff=depth_falloff,
            refinement_strength=refinement_strength,
            save_clean_plates=save_clean_plates,
            save_bootstrap=save_bootstrap,
            save_flow=save_flow,
        )
        return

    if mode != "greenscreen":
        console.print(f"[bold red]Unknown mode:[/bold red] {mode!r}. Use 'greenscreen', 'depth', or 'plate-subtraction'.")
        raise typer.Exit(code=1)

    # --- Green-screen mode (existing behavior) ---

    # despeckle_size excluded — sensible default even in headless mode
    required_flags_set = all(v is not None for v in [linear, despill, despeckle, refiner])
    if required_flags_set:
        assert linear is not None and despill is not None and despeckle is not None and refiner is not None
        despill_clamped = max(0, min(10, despill))
        settings = InferenceSettings(
            input_is_linear=linear,
            despill_strength=despill_clamped / 10.0,
            auto_despeckle=despeckle,
            despeckle_size=despeckle_size if despeckle_size is not None else 400,
            refiner_scale=refiner,
        )
    else:
        settings = _prompt_inference_settings(
            default_linear=linear,
            default_despill=despill,
            default_despeckle=despeckle,
            default_despeckle_size=despeckle_size,
            default_refiner=refiner,
        )

    with ProgressContext() as ctx_progress:
        run_inference(
            clips,
            device=ctx.obj["device"],
            backend=backend,
            max_frames=max_frames,
            settings=settings,
            on_clip_start=ctx_progress.on_clip_start,
            on_frame_complete=ctx_progress.on_frame_complete,
        )

    console.print("[bold green]Inference complete.")

    # --- Motion blur refinement post-processing ---
    if refine_motion_blur:
        try:
            config = MotionBlurConfig(
                blur_threshold=blur_threshold,
                kernel_profile=kernel_profile,
                temporal_smoothing=temporal_smoothing,
                division_epsilon=division_epsilon,
                blur_dilation=blur_dilation,
                plate_search_radius=plate_search_radius,
                plate_alpha_threshold=plate_alpha_threshold,
                static_clean_plate=static_clean_plate,
            )
        except ValueError as exc:
            console.print(f"[bold red]Invalid motion blur parameter:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

        refiner = MotionBlurRefiner(config, device=ctx.obj["device"])

        with ProgressContext() as ctx_progress:
            for clip in clips:
                ctx_progress.on_clip_start(clip.name, 0)
                refiner.process_clip(
                    clip.root_path,
                    clean_plate_path=clean_plate,
                    save_refined_fg=save_refined_fg,
                    on_frame_complete=ctx_progress.on_frame_complete,
                )

        console.print("[bold green]Motion blur refinement complete.")


@app.command()
def wizard(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path (Windows or local)")],
) -> None:
    """Interactive wizard for organizing clips and running the pipeline."""
    interactive_wizard(path, device=ctx.obj["device"])


# ---------------------------------------------------------------------------
# Wizard (rich-styled)
# ---------------------------------------------------------------------------


def interactive_wizard(win_path: str, device: str | None = None) -> None:
    console.print(Panel("[bold]CORRIDOR KEY — SMART WIZARD[/bold]", style="cyan"))

    # 1. Resolve Path
    console.print(f"Windows Path: {win_path}")

    if os.path.exists(win_path):
        process_path = win_path
        console.print(f"Running locally: [bold]{process_path}[/bold]")
    else:
        process_path = map_path(win_path)
        console.print(f"Linux/Remote Path: [bold]{process_path}[/bold]")

        if not os.path.exists(process_path):
            console.print(
                f"\n[bold red]ERROR:[/bold red] Path does not exist locally OR on Linux mount!\n"
                f"Expected Linux Mount Root: {LINUX_MOUNT_ROOT}"
            )
            raise typer.Exit(code=1)

    # 2. Analyze — shot or project?
    target_is_shot = False
    if os.path.exists(os.path.join(process_path, "Input")) or glob.glob(os.path.join(process_path, "Input.*")):
        target_is_shot = True

    work_dirs: list[str] = []
    # Pipeline output dirs, not clip sources
    excluded_dirs = {"Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"}
    if target_is_shot:
        work_dirs = [process_path]
    else:
        work_dirs = [
            os.path.join(process_path, d)
            for d in os.listdir(process_path)
            if os.path.isdir(os.path.join(process_path, d)) and d not in excluded_dirs
        ]

    console.print(f"\nFound [bold]{len(work_dirs)}[/bold] potential clip folders.")

    # Files already named Input/AlphaHint/etc are organized, not "loose"
    known_names = {"input", "alphahint", "videomamamaskhint"}
    loose_videos = [
        f
        for f in os.listdir(process_path)
        if is_video_file(f)
        and os.path.isfile(os.path.join(process_path, f))
        and os.path.splitext(f)[0].lower() not in known_names
    ]

    dirs_needing_org = []
    for d in work_dirs:
        has_input = os.path.exists(os.path.join(d, "Input")) or glob.glob(os.path.join(d, "Input.*"))
        has_alpha = os.path.exists(os.path.join(d, "AlphaHint"))
        has_mask = os.path.exists(os.path.join(d, "VideoMamaMaskHint"))
        if not has_input or not has_alpha or not has_mask:
            dirs_needing_org.append(d)

    if loose_videos or dirs_needing_org:
        if loose_videos:
            console.print(f"Found [yellow]{len(loose_videos)}[/yellow] loose video files:")
            for v in loose_videos:
                console.print(f"  • {v}")

        if dirs_needing_org:
            console.print(f"Found [yellow]{len(dirs_needing_org)}[/yellow] folders needing setup:")
            display_limit = 10
            for d in dirs_needing_org[:display_limit]:
                console.print(f"  • {os.path.basename(d)}")
            if len(dirs_needing_org) > display_limit:
                console.print(f"  …and {len(dirs_needing_org) - display_limit} others.")

        # 3. Organize
        if Confirm.ask("\nOrganize clips & create hint folders?", default=False):
            for v in loose_videos:
                clip_name = os.path.splitext(v)[0]
                ext = os.path.splitext(v)[1]
                target_folder = os.path.join(process_path, clip_name)

                if os.path.exists(target_folder):
                    logger.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
                    continue

                try:
                    os.makedirs(target_folder)
                    target_file = os.path.join(target_folder, f"Input{ext}")
                    shutil.move(os.path.join(process_path, v), target_file)
                    logger.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")
                    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                        os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to organize video '{v}': {e}")

            for d in work_dirs:
                organize_target(d)
            console.print("[green]Organization complete.[/green]")

            if not target_is_shot:
                work_dirs = [
                    os.path.join(process_path, d)
                    for d in os.listdir(process_path)
                    if os.path.isdir(os.path.join(process_path, d)) and d not in excluded_dirs
                ]

    # 4. Status Check Loop
    while True:
        ready: list[ClipEntry] = []
        masked: list[ClipEntry] = []
        raw: list[ClipEntry] = []

        for d in work_dirs:
            entry = ClipEntry(os.path.basename(d), d)
            try:
                entry.find_assets()
            except (FileNotFoundError, ValueError, OSError):
                pass

            has_mask = False
            mask_dir = os.path.join(d, "VideoMamaMaskHint")
            if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                has_mask = True
            if not has_mask:
                for f in os.listdir(d):
                    stem, _ = os.path.splitext(f)
                    if stem.lower() == "videomamamaskhint" and is_video_file(f):
                        has_mask = True
                        break

            if entry.alpha_asset:
                ready.append(entry)
            elif has_mask:
                masked.append(entry)
            else:
                raw.append(entry)

        table = Table(title="Status Report", show_lines=True)
        table.add_column("Category", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Clips")

        table.add_row(
            "[green]Ready[/green] (AlphaHint)",
            str(len(ready)),
            ", ".join(c.name for c in ready) or "—",
        )
        table.add_row(
            "[yellow]Masked[/yellow] (VideoMaMaMaskHint)",
            str(len(masked)),
            ", ".join(c.name for c in masked) or "—",
        )
        table.add_row(
            "[red]Raw[/red] (Input only)",
            str(len(raw)),
            ", ".join(c.name for c in raw) or "—",
        )
        console.print(table)

        missing_alpha = masked + raw
        actions: list[str] = []

        if missing_alpha:
            actions.append(f"[bold]v[/bold] — Run VideoMaMa ({len(masked)} with masks)")
            actions.append(f"[bold]g[/bold] — Run GVM (auto-matte {len(raw)} clips)")
        if ready:
            actions.append(f"[bold]i[/bold] — Run Inference [dim](greenscreen / depth)[/dim] ({len(ready)} ready)")
        actions.append("[bold]r[/bold] — Re-scan folders")
        actions.append("[bold]q[/bold] — Quit")

        console.print(Panel("\n".join(actions), title="Actions", style="blue"))

        choice = Prompt.ask("Select action", choices=["v", "g", "i", "r", "q"], default="q")

        if choice == "v":
            console.print(Panel("VideoMaMa", style="magenta"))
            run_videomama(missing_alpha, chunk_size=50, device=device)
            Prompt.ask("VideoMaMa batch complete. Press Enter to re-scan")

        elif choice == "g":
            console.print(Panel("GVM Auto-Matte", style="magenta"))
            console.print(f"Will generate alphas for {len(raw)} clips without mask hints.")
            if Confirm.ask("Proceed with GVM?", default=False):
                generate_alphas(raw, device=device)
                Prompt.ask("GVM batch complete. Press Enter to re-scan")

        elif choice == "i":
            console.print(Panel("Corridor Key Inference", style="magenta"))
            mode = Prompt.ask(
                "Inference mode",
                choices=["greenscreen", "depth"],
                default="greenscreen",
            )
            try:
                if mode == "greenscreen":
                    settings = _prompt_inference_settings()
                    with ProgressContext() as ctx_progress:
                        run_inference(
                            ready,
                            device=device,
                            settings=settings,
                            on_clip_start=ctx_progress.on_clip_start,
                            on_frame_complete=ctx_progress.on_frame_complete,
                        )
                else:
                    depth_params = _prompt_depth_settings()

                    # Extract video frames before checking eligibility
                    video_fps_map: dict[str, float] = {}
                    for c in ready + masked + raw:
                        fps = _extract_video_frames(c)
                        if fps is not None:
                            video_fps_map[c.name] = fps

                    depth_eligible = [c for c in ready + masked + raw if _has_input_frames(c)]
                    if not depth_eligible:
                        console.print("[yellow]No clips with Input frames found.[/yellow]")
                    else:
                        console.print(f"Running depth inference on {len(depth_eligible)} clips…")
                        _run_depth_inference(depth_eligible, device=device, **depth_params)

                        # Stitch comp videos for clips that had video inputs
                        for c in depth_eligible:
                            if c.name in video_fps_map:
                                _stitch_comp_video(c, video_fps_map[c.name])
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                console.print(f"[bold red]Inference failed:[/bold red] {e}")
            Prompt.ask("Inference batch complete. Press Enter to re-scan")

        elif choice == "r":
            console.print("Re-scanning…")

        elif choice == "q":
            break

    console.print("[bold green]Wizard complete. Goodbye![/bold green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point called by the `corridorkey` console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
