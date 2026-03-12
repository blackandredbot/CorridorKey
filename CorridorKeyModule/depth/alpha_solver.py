"""Per-pixel alpha and foreground color solving for motion-blurred regions.

Decomposes observed motion-blurred pixels into foreground and background
contributions using the standard compositing equation:

    C_observed = alpha * C_foreground + (1 - alpha) * C_background

With C_background known from a clean plate, the solver computes refined
per-pixel alpha and recovers the foreground color along the blur trail.
Alpha is further shaped by a 1-D blur kernel that tapers from the object
center (fully opaque) to the blur trail edges (nearly transparent).

References
----------
Giusti, A. & Caglioti, V. (IJCV 2008) "On the Apparent Transparency of a
Motion Blurred Object" — interprets motion blur as apparent transparency
where alpha corresponds to the temporal coverage of the foreground object
at each pixel during the exposure interval.

Lin, H., Tai, Y.-W. & Brown, M. S. (TPAMI 2011) "Motion Regularization
for Matting Motion Blurred Objects" — introduces motion-aware
regularization that constrains alpha estimation along the blur kernel
direction, applied here to the kernel shaping step.
"""

from __future__ import annotations

import numpy as np


class AlphaSolver:
    """Solves the compositing equation for per-pixel alpha and foreground color.

    Parameters
    ----------
    division_epsilon : float
        Minimum absolute denominator value per channel before falling back
        to the original alpha. Default 1e-4.
    kernel_profile : str
        Blur kernel taper profile. One of ``"linear"``, ``"cosine"``,
        ``"gaussian"``. Default ``"linear"``.
    """

    def __init__(
        self,
        division_epsilon: float = 1e-4,
        kernel_profile: str = "linear",
    ) -> None:
        self.division_epsilon = division_epsilon
        self.kernel_profile = kernel_profile

    def _build_kernel(self, length: int) -> np.ndarray:
        """Build a 1-D blur kernel of the configured profile.

        The kernel has *length* elements.  Normalised position ``t`` runs
        from 0.0 (index 0, object centre) to 1.0 (index ``length - 1``,
        blur-trail tip).

        Parameters
        ----------
        length : int
            Kernel length in pixels (must be >= 1).

        Returns
        -------
        np.ndarray
            1-D float32 array of weights from 1.0 (centre) to ~0.0 (tip).
        """
        if length <= 1:
            return np.ones(1, dtype=np.float32)

        t = np.linspace(0.0, 1.0, length, dtype=np.float32)

        if self.kernel_profile == "linear":
            kernel = 1.0 - t
        elif self.kernel_profile == "cosine":
            kernel = 0.5 * (1.0 + np.cos(np.pi * t))
        elif self.kernel_profile == "gaussian":
            kernel = np.exp(-3.0 * t * t)
        else:
            raise ValueError(
                f"Unknown kernel_profile {self.kernel_profile!r}; "
                f"expected 'linear', 'cosine', or 'gaussian'"
            )

        return kernel.astype(np.float32)

    def _estimate_foreground(
        self,
        observed: np.ndarray,
        alpha: np.ndarray,
        flow_field: np.ndarray,
        py: int,
        px: int,
    ) -> np.ndarray:
        """Estimate foreground colour from nearest fully opaque pixel along
        the motion vector direction.

        Walks pixel-by-pixel along the motion vector starting from
        ``(py, px)`` until a pixel with ``alpha >= 1.0`` is found.  If the
        walk goes out of bounds without finding one, the observed colour at
        ``(py, px)`` is returned as the fallback estimate.

        Parameters
        ----------
        observed : np.ndarray
            [H, W, 3] float32 observed frame.
        alpha : np.ndarray
            [H, W] float32 alpha matte.
        flow_field : np.ndarray
            [H, W, 2] float32 forward flow.
        py, px : int
            Pixel coordinates to start from.

        Returns
        -------
        np.ndarray
            [3] float32 RGB foreground estimate.
        """
        h, w = alpha.shape
        dx = flow_field[py, px, 0]
        dy = flow_field[py, px, 1]
        mag = np.sqrt(dx * dx + dy * dy)

        if mag < 1e-6:
            return observed[py, px].copy()

        # Normalise direction
        step_x = dx / mag
        step_y = dy / mag

        # Walk along the motion vector direction
        max_steps = int(np.ceil(mag)) + 1
        for i in range(1, max_steps + 1):
            ny = int(round(py + step_y * i))
            nx = int(round(px + step_x * i))
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                break
            if alpha[ny, nx] >= 1.0:
                return observed[ny, nx].copy()

        # Also try the opposite direction
        for i in range(1, max_steps + 1):
            ny = int(round(py - step_y * i))
            nx = int(round(px - step_x * i))
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                break
            if alpha[ny, nx] >= 1.0:
                return observed[ny, nx].copy()

        # Fallback: return observed pixel colour
        return observed[py, px].copy()

    def solve(
        self,
        observed: np.ndarray,
        clean_plate: np.ndarray,
        alpha: np.ndarray,
        flow_field: np.ndarray,
        blur_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute refined alpha and foreground colour for one frame.

        Parameters
        ----------
        observed : np.ndarray
            [H, W, 3] float32 observed frame in linear colour space.
        clean_plate : np.ndarray
            [H, W, 3] float32 background in linear colour space.
        alpha : np.ndarray
            [H, W] float32 original alpha matte.
        flow_field : np.ndarray
            [H, W, 2] float32 forward flow.
        blur_mask : np.ndarray
            [H, W] float32 binary blur mask.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (refined_alpha [H, W] float32, foreground_colour [H, W, 3] float32)
        """
        h, w = alpha.shape
        refined_alpha = alpha.copy().astype(np.float32)
        foreground = observed.copy().astype(np.float32)

        # Pre-compute blur magnitudes for kernel position lookup
        dx_all = flow_field[:, :, 0]
        dy_all = flow_field[:, :, 1]
        blur_mag = np.sqrt(dx_all * dx_all + dy_all * dy_all)

        # Find blur-masked pixel coordinates
        ys, xs = np.where(blur_mask > 0.5)

        for idx in range(len(ys)):
            py = int(ys[idx])
            px = int(xs[idx])

            c_obs = observed[py, px]  # [3]
            c_bg = clean_plate[py, px]  # [3]

            # Step 1: Estimate foreground colour
            c_fg = self._estimate_foreground(observed, alpha, flow_field, py, px)

            # Step 2: Compute denominator per channel
            denom = c_fg - c_bg  # [3]
            abs_denom = np.abs(denom)

            # Step 3: Epsilon guard — if ALL channels below epsilon, preserve
            if np.all(abs_denom < self.division_epsilon):
                # Keep original alpha and observed colour
                continue

            # Step 4: Per-channel alpha, weighted average
            # Only use channels where |denom| >= epsilon
            valid = abs_denom >= self.division_epsilon
            alpha_ch = np.where(valid, (c_obs - c_bg) / np.where(valid, denom, 1.0), 0.0)
            weights = np.where(valid, abs_denom, 0.0)
            w_sum = weights.sum()
            if w_sum < 1e-12:
                continue
            solved_alpha = float(np.sum(alpha_ch * weights) / w_sum)

            # Step 5: Clamp
            solved_alpha = max(0.0, min(1.0, solved_alpha))

            # Step 6: Kernel shaping
            pixel_mag = float(blur_mag[py, px])
            if pixel_mag >= 2.0:
                kernel_len = max(2, int(round(pixel_mag)))
                kernel = self._build_kernel(kernel_len)

                # Determine position along motion vector for this pixel
                # Use fractional position based on alpha pattern:
                # pixels closer to opaque region → lower t, pixels at tip → higher t
                # Approximate position as: how far along the blur trail this pixel is
                # relative to the blur magnitude
                dx_px = float(flow_field[py, px, 0])
                dy_px = float(flow_field[py, px, 1])
                # Normalise position: use original alpha as proxy for position
                # (higher alpha = closer to centre, lower alpha = closer to tip)
                # Map to kernel index
                t_pos = 1.0 - alpha[py, px]  # 0 at centre (alpha=1), 1 at tip (alpha=0)
                t_pos = max(0.0, min(1.0, float(t_pos)))
                k_idx = min(int(t_pos * (kernel_len - 1)), kernel_len - 1)
                kernel_weight = float(kernel[k_idx])
                solved_alpha *= kernel_weight
            else:
                # Short blur — no kernel shaping needed
                pass

            # Clamp again after shaping
            solved_alpha = max(0.0, min(1.0, solved_alpha))
            refined_alpha[py, px] = np.float32(solved_alpha)

            # Step 7: Recover foreground colour
            a = max(solved_alpha, self.division_epsilon)
            fg_recovered = (c_obs - (1.0 - solved_alpha) * c_bg) / a
            fg_recovered = np.clip(fg_recovered, 0.0, 1.0)
            foreground[py, px] = fg_recovered.astype(np.float32)

        return refined_alpha, foreground
