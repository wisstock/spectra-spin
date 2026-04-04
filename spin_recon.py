"""
Spectral image reconstruction from spinning-disk spectral microscopy data.

This module provides a complete, Numba-accelerated pipeline for
reconstructing hyperspectral images from raw camera frames captured through
a spinning-disk spectral module.  All computationally intensive routines
(Dynamic-Programming path extraction, spectral band extraction,
sparse-to-dense expansion, interpolation, and spatial binning) are
JIT-compiled via Numba for multi-threaded execution.

Main public API
---------------
* ``SpectralRecon``  - reconstruction engine (constructor + pipeline methods).
* ``SingleResult``   - result of processing a single image.
* ``BatchResult``    - result of batch processing with streaming accumulation.

Quick start
-----------
::

    from spin_recon import SpectralRecon

    recon = SpectralRecon(
        crop=(500, 2500, 1000, 2500),
        spectral_band_width=100,
        custom_lines_num=True,
        lines_num=22,
        output_binning=2,
    )
    result = recon.process_batch(image_paths, method="max")
    result.interpolate_missing_zeros(method="2d")
"""

__all__ = ["SpectralRecon", "SingleResult", "BatchResult"]

import gc
import json
import yaml
import logging
import os
from dataclasses import dataclass, field
from typing import Optional


import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi
from scipy import signal
from scipy.interpolate import UnivariateSpline
from skimage import io
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

### Module-level constants

_GAUSSIAN_SIGMA: tuple[float, float] = (1.5, 2.5)
"""Gaussian smoothing sigma ``(row, col)`` applied to the edge image
before DP path extraction."""

_GAP_WIDTH_PX: float = 6.0
"""Pixel gap between neighbouring band boundaries in
``regularize_structures``."""

_AUTOCORR_MIN_PEAK_DISTANCE: int = 10
"""Minimum distance (in pixels) between autocorrelation peaks used for
spatial-period estimation."""

_MASK_OFFSET: float = 1000.0
"""Intensity offset added/subtracted to mask previously detected lines
during DP path extraction."""


### Numba kernels - module-level private functions

@njit(parallel=True, cache=True)
def _extract_paths_numba(img_work: np.ndarray, num_lines: int,
                         is_dark: bool, mask_width: int):
    """Dynamic-programming path extraction (Numba-compiled).

    Parameters
    ----------
    img_work : np.ndarray
        2D array (rows, cols) to extract paths from.
    num_lines : int
        Number of lines to extract.
    is_dark : bool
        Whether the image is dark.
    mask_width : int
        Width of the mask to apply to the extracted lines.

    Returns
    -------
    np.ndarray
        Array of extracted lines.

    """
    rows, cols = img_work.shape
    lines = np.zeros((num_lines, cols), dtype=np.int32)

    mask_val = (np.max(img_work) + _MASK_OFFSET if is_dark
                else np.min(img_work) - _MASK_OFFSET)
    pad_val = np.inf if is_dark else -np.inf

    for line_idx in range(num_lines):
        acc = np.zeros_like(img_work)
        for i in range(rows):
            acc[i, 0] = img_work[i, 0]

        ptr = np.zeros((rows, cols), dtype=np.int32)

        for j in range(1, cols):
            for i in range(rows):
                v_up = acc[i - 1, j - 1] if i > 0 else pad_val
                v_st = acc[i, j - 1]
                v_dn = acc[i + 1, j - 1] if i < rows - 1 else pad_val

                if is_dark:
                    best = v_up
                    best_ptr = -1
                    if v_st < best:
                        best = v_st
                        best_ptr = 0
                    if v_dn < best:
                        best = v_dn
                        best_ptr = 1
                else:
                    best = v_up
                    best_ptr = -1
                    if v_st > best:
                        best = v_st
                        best_ptr = 0
                    if v_dn > best:
                        best = v_dn
                        best_ptr = 1

                acc[i, j] = img_work[i, j] + best
                ptr[i, j] = best_ptr

        path = np.zeros(cols, dtype=np.int32)

        best_idx = 0
        best_val = acc[0, cols - 1]
        if is_dark:
            for i in range(1, rows):
                if acc[i, cols - 1] < best_val:
                    best_val = acc[i, cols - 1]
                    best_idx = i
        else:
            for i in range(1, rows):
                if acc[i, cols - 1] > best_val:
                    best_val = acc[i, cols - 1]
                    best_idx = i
        path[cols - 1] = best_idx

        for j in range(cols - 1, 0, -1):
            cur = path[j]
            nxt = cur + ptr[cur, j]
            if nxt < 0:
                nxt = 0
            if nxt >= rows:
                nxt = rows - 1
            path[j - 1] = nxt

        lines[line_idx] = path

        # Mask found line
        for j in range(cols):
            r = path[j]
            r_lo = max(0, r - mask_width)
            r_hi = min(rows, r + mask_width + 1)
            for i in range(r_lo, r_hi):
                img_work[i, j] = mask_val

    return lines


@njit(parallel=True, cache=True)
def _extract_bands_numba(image: np.ndarray, up_lim: np.ndarray,
                         down_lim: np.ndarray, spectral_width: int):
    """Parallel spectral band extraction (Numba-compiled).
    
    """
    num_cols, num_bands = up_lim.shape
    lambda_img = np.zeros((num_bands, num_cols, spectral_width),
                          dtype=np.float32)

    for band in prange(num_bands):
        for col in range(num_cols):
            d = down_lim[col, band]
            u = up_lim[col, band]
            h = u - d
            if h <= 0:
                continue
            limit = min(h, spectral_width)
            for k in range(limit):
                lambda_img[band, col, k] = image[d + k, col]

    return lambda_img


@njit(parallel=True, cache=True)
def _expand_full_1d_numba(h: int, w: int, s: int,
                          row_indices: np.ndarray,
                          spectral_bands: np.ndarray):
    """Sparse → dense expansion with 1D row indices (Numba-compiled).
    
    """
    out = np.zeros((h, w, s), dtype=np.float32)
    num_bands = row_indices.shape[0]
    for i in prange(num_bands):
        r = row_indices[i]
        if 0 <= r < h:
            for col in range(w):
                for k in range(s):
                    out[r, col, k] = spectral_bands[i, col, k]
    return out


@njit(parallel=True, cache=True)
def _expand_full_2d_numba(h: int, w: int, s: int,
                          row_indices: np.ndarray,
                          spectral_bands: np.ndarray):
    """Sparse → dense expansion with 2D row indices (Numba-compiled).
    
    """
    out = np.zeros((h, w, s), dtype=np.float32)
    num_bands, cols = row_indices.shape
    for i in prange(num_bands):
        for col in range(min(cols, w)):
            r = row_indices[i, col]
            if 0 <= r < h:
                for k in range(s):
                    out[r, col, k] = spectral_bands[i, col, k]
    return out


# Interpolation kernels
@njit(parallel=True, cache=True)
def _interp_1d_fill(profile: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation over non-zero values in a 1-D profile.

    Parameters
    ----------
    profile : np.ndarray
        1-D array where zeros mark missing data.
    x : np.ndarray
        Coordinate array to interpolate onto (same length as *profile*).

    Returns
    -------
    np.ndarray
        Interpolated profile (float64), or zeros if no data.

    """
    n = profile.shape[0]
    nnz = 0
    for i in range(n):
        if profile[i] != 0:
            nnz += 1
    if nnz == 0:
        return np.zeros(n, dtype=np.float64)
    if nnz == 1:
        val = 0.0
        for i in range(n):
            if profile[i] != 0:
                val = profile[i]
                break
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = val
        return out

    xp = np.zeros(nnz, dtype=np.float64)
    fp = np.zeros(nnz, dtype=np.float64)
    idx = 0
    for i in range(n):
        if profile[i] != 0:
            xp[idx] = i
            fp[idx] = profile[i]
            idx += 1
    return np.interp(x, xp, fp)


@njit(parallel=True, cache=True)
def _interpolate_zeros_1d(img3d: np.ndarray) -> np.ndarray:
    """1-D column-wise linear interpolation of zero-valued pixels.

    For every (column, channel) pair the vertical profile is extracted,
    non-zero values are used as knots, and ``np.interp`` fills the gaps.

    Delegates to ``_interp_1d_fill`` for the actual interpolation.

    """
    rows, cols, channels = img3d.shape
    out = np.zeros_like(img3d)
    x = np.arange(rows).astype(np.float64)

    for ch in prange(channels):
        for c in range(cols):
            profile = np.zeros(rows, dtype=np.float64)
            for r in range(rows):
                profile[r] = img3d[r, c, ch]
            filled = _interp_1d_fill(profile, x)
            for r in range(rows):
                out[r, c, ch] = filled[r]

    return out


@njit(parallel=True, cache=True)
def _interpolate_zeros_2d(img3d: np.ndarray) -> np.ndarray:
    """Two-pass linear 2-D interpolation for missing (zero-valued) pixels.

    Pass 1 — interpolate along rows (horizontal) for each (row, channel).
    Pass 2 — interpolate along columns (vertical) for each (col, channel).
    The final value is the average of both passes where both provide data;
    otherwise the single available pass is used. Pixels that cannot be
    interpolated by either pass remain zero.

    """
    rows, cols, channels = img3d.shape
    out = img3d.copy()
    x_row = np.arange(cols).astype(np.float64)
    x_col = np.arange(rows).astype(np.float64)

    for ch in prange(channels):
        # --- Pass 1: interpolate along rows (horizontal) ---
        row_interp = np.zeros((rows, cols), dtype=np.float64)
        row_valid = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            profile = np.zeros(cols, dtype=np.float64)
            nnz = 0
            for c in range(cols):
                profile[c] = img3d[r, c, ch]
                if profile[c] != 0:
                    nnz += 1
            if nnz > 0:
                filled = _interp_1d_fill(profile, x_row)
                for c in range(cols):
                    row_interp[r, c] = filled[c]
                    row_valid[r, c] = 1.0

        # --- Pass 2: interpolate along columns (vertical) ---
        col_interp = np.zeros((rows, cols), dtype=np.float64)
        col_valid = np.zeros((rows, cols), dtype=np.float64)
        for c in range(cols):
            profile = np.zeros(rows, dtype=np.float64)
            nnz = 0
            for r in range(rows):
                profile[r] = img3d[r, c, ch]
                if profile[r] != 0:
                    nnz += 1
            if nnz > 0:
                filled = _interp_1d_fill(profile, x_col)
                for r in range(rows):
                    col_interp[r, c] = filled[r]
                    col_valid[r, c] = 1.0

        # --- Combine: average when both passes valid ---
        for r in range(rows):
            for c in range(cols):
                if img3d[r, c, ch] != 0:
                    continue  # original data — already in out
                rv = row_valid[r, c]
                cv = col_valid[r, c]
                if rv > 0 and cv > 0:
                    out[r, c, ch] = (row_interp[r, c]
                                     + col_interp[r, c]) / 2.0
                elif rv > 0:
                    out[r, c, ch] = row_interp[r, c]
                elif cv > 0:
                    out[r, c, ch] = col_interp[r, c]
                # else: stays 0

    return out

# Binning kernel
@njit(parallel=True, cache=True)
def _binning_2d(img3d: np.ndarray, bin_size: int) -> np.ndarray:
    """Down-sample a 3-D image ``(H, W, S)`` by averaging non-zero pixels
    in each ``(bin_size × bin_size)`` spatial block.

    Parameters
    ----------
    img3d : np.ndarray
        Input image of shape ``(rows, cols, channels)``.
    bin_size : int
        Side length of the square binning block.

    Returns
    -------
    np.ndarray
        Down-sampled image, shape
        ``(rows // bin_size, cols // bin_size, channels)``.

    """
    rows, cols, channels = img3d.shape
    out_rows = rows // bin_size
    out_cols = cols // bin_size
    out = np.zeros((out_rows, out_cols, channels), dtype=img3d.dtype)

    for ch in prange(channels):
        for br in range(out_rows):
            r_start = br * bin_size
            r_end = r_start + bin_size
            for bc in range(out_cols):
                c_start = bc * bin_size
                c_end = c_start + bin_size
                total = 0.0
                count = 0
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        val = img3d[r, c, ch]
                        if val != 0.0:
                            total += val
                            count += 1
                if count > 0:
                    out[br, bc, ch] = total / count
    return out


### Data classes

@dataclass
class SingleResult:
    """Result of processing a single image.

    Attributes
    ----------
    spectral_img : np.ndarray
        Full reconstructed spectral image ``(H, W, S)``, dtype float32.
    raw_image : np.ndarray
        Original input image (after cropping), 2-D.
    spectral_bands : np.ndarray
        Compact spectral data, shape ``(num_bands, width, spectral_width)``.
    row_indices : np.ndarray
        Row positions where each band is placed.
    image_shape : tuple[int, int]
        ``(height, width)`` of the source image (after cropping).
    spectral_width : int
        Number of spectral channels.
    regularized_lines : dict
        Full regularized-line dictionary (kept for diagnostics).

    """
    spectral_img: np.ndarray
    raw_image: np.ndarray
    spectral_bands: np.ndarray
    row_indices: np.ndarray
    image_shape: tuple[int, int]
    spectral_width: int
    regularized_lines: dict

    def plot_reg_structure(self, figsize: tuple[int, int] = (12, 8),
                           cmap: str = "jet",
                           linewidth: float = 0.8) -> None:
        """Overlay regularized lines on the raw input image."""
        rl = self.regularized_lines
        if not rl:
            logger.warning("No regularized lines to plot.")
            return

        cols = self.raw_image.shape[1]
        x = np.arange(cols)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(self.raw_image, cmap=cmap, aspect="auto")

        for i in range(rl["light"].shape[0]):
            ax.plot(x, rl["light"][i], color="red",
                    linewidth=linewidth * 2,
                    label="light" if i == 0 else None)

        for i in range(rl["dark_up"].shape[0]):
            ax.plot(x, rl["dark_up"][i], color="blue",
                    linewidth=linewidth,
                    label="dark_up" if i == 0 else None)

        for i in range(rl["dark_down"].shape[0]):
            ax.plot(x, rl["dark_down"][i], color="green",
                    linewidth=linewidth,
                    label="dark_down" if i == 0 else None)

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Regularized structure overlay")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.tight_layout()
        plt.show()


@dataclass
class BatchResult:
    """Result of batch processing.

    Attributes
    ----------
    spectral_img : np.ndarray
        Accumulated spectral image, shape ``(H, W, S)``, dtype float32.
    spectral_img_interpolated : np.ndarray or None
        Interpolated version of ``spectral_img`` where zero-valued
        pixels are filled via Numba-accelerated linear interpolation.
        ``None`` until ``interpolate_missing_zeros`` is called.
    row_indices : dict[str, np.ndarray]
        Mapping *image_path → row-index array* for every image.
    num_images : int
        Number of images that contributed to the result.
    spectral_width : int
        Spectral band width used during reconstruction.
    accumulation_method : str
        ``'sum'`` or ``'max'``.

    """
    spectral_img: np.ndarray
    spectral_img_interpolated: Optional[np.ndarray] = None
    row_indices: dict[str, np.ndarray] = field(default_factory=dict)
    num_images: int = 0
    spectral_width: int = 0
    accumulation_method: str = "max"

    def interpolate_missing_zeros(self, method: str = "2d") -> None:
        """Interpolate the accumulated spectral image, treating zero-valued
        pixels as missing.

        Parameters
        ----------
        method : ``'1d'`` | ``'2d'``
            ``'1d'`` — column-wise linear interpolation (fast).
            ``'2d'`` — two-pass row + column linear interpolation
            (better spatial coverage).

        """
        if self.spectral_img is None:
            logger.warning("No spectral image to interpolate.")
            return

        if method == "1d":
            logger.info("Starting parallel Numba 1D linear interpolation...")
            self.spectral_img_interpolated = _interpolate_zeros_1d(
                self.spectral_img)
            logger.info("Interpolation complete.")
        elif method == "2d":
            logger.info("Starting parallel Numba 2D linear interpolation...")
            self.spectral_img_interpolated = _interpolate_zeros_2d(
                self.spectral_img)
            logger.info("Interpolation complete.")
        else:
            raise ValueError("Unknown method. Use '1d' or '2d'.")

    def apply_output_binning(self, bin_size: int) -> None:
        """Apply spatial binning (down-sampling) to spectral images.

        Averages non-zero pixel intensities within each
        ``(bin_size × bin_size)`` block.  Updates ``spectral_img`` and,
        if present, ``spectral_img_interpolated`` in-place.

        Parameters
        ----------
        bin_size : int
            Side length of the binning block.  A value ≤ 1 is a no-op.

        """
        if bin_size <= 1:
            return
        if self.spectral_img is not None:
            logger.info("Applying %d×%d output binning (ignoring zeros)...",
                        bin_size, bin_size)
            self.spectral_img = _binning_2d(self.spectral_img, bin_size)
        if self.spectral_img_interpolated is not None:
            logger.info("Applying %d×%d binning to interpolated image...",
                        bin_size, bin_size)
            self.spectral_img_interpolated = _binning_2d(
                self.spectral_img_interpolated, bin_size)
        logger.info("Binning complete.")

    def save_metadata(self, path: str) -> None:
        """Save row-index metadata to disk.

        * ``.npz``  - NumPy compressed archive.
        * ``.json`` - human-readable JSON.
        * ``.yaml`` / ``.yml`` - YAML.

        The format is chosen automatically based on the file extension.

        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            np.savez_compressed(path, **self.row_indices)
        elif ext == ".json":
            serialisable = {
                k: v.tolist() for k, v in self.row_indices.items()
            }
            with open(path, "w") as fh:
                json.dump(serialisable, fh, indent=2)
        elif ext in (".yaml", ".yml"):
            serialisable = {
                k: v.tolist() for k, v in self.row_indices.items()
            }
            with open(path, "w") as fh:
                yaml.dump(serialisable, fh, default_flow_style=False,
                          allow_unicode=True)
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. Use .npz, .json, or .yaml."
            )

    @staticmethod
    def load_metadata(path: str) -> dict[str, np.ndarray]:
        """Load row-index metadata previously saved by ``save_metadata``."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            data = np.load(path)
            return {k: data[k] for k in data.files}
        elif ext == ".json":
            with open(path) as fh:
                raw = json.load(fh)
            return {k: np.asarray(v, dtype=np.int32) for k, v in raw.items()}
        elif ext in (".yaml", ".yml"):
            with open(path) as fh:
                raw = yaml.safe_load(fh)
            return {k: np.asarray(v, dtype=np.int32) for k, v in raw.items()}
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. Use .npz, .json, or .yaml."
            )


### Main reconstruction class

class SpectralRecon:
    """Numba-accelerated spectral image reconstruction.

    This class implements the full pipeline for extracting per-pixel
    spectral information from raw spinning-disk microscopy frames:
    edge detection → structure detection (DP) → regularization →
    band extraction → pixel allocation → (optional) binning.

    Parameters
    ----------
    crop : tuple of int, optional
        ``(row_start, row_end, col_start, col_end)`` ROI.
    spectral_band_width : int
        Target spectral width for each band.
    custom_lines_num : bool
        If ``True``, use ``lines_num`` instead of auto-estimation.
    lines_num : int
        Number of periodic structures (used when ``custom_lines_num=True``).
    mask_width : int
        Half-width of the depletion mask in the DP stage.
    lines_smooth_factor : float
        B-spline smoothing factor *s*.
    dist_offset : float
        Global offset added to ``dist_up`` / ``dist_down``.
    custom_dist : bool
        Whether to use ``dist_up``/``dist_down`` directly.
    dist_up, dist_down : float or None
        Fixed distances from a light line to the dark boundaries.
    precise_allocation : bool
        If ``True``, use per-column row indices (2-D) instead of
        a single row index per band.
    output_binning : int
        Spatial down-sampling factor for the final reconstructed image.
        A value of 0 or 1 means no binning.  For example,
        ``output_binning=2`` averages each 2×2 block of pixels (ignoring
        zeros) producing an image half the size in each spatial dimension.

    Raises
    ------
    ValueError
        If *output_binning* is negative or *spectral_band_width* ≤ 0.

    """

    def __init__(self,
                 crop: Optional[tuple[int, int, int, int]] = None,
                 spectral_band_width: int = 100,
                 custom_lines_num: bool = False,
                 lines_num: float = 20,
                 mask_width: float = 80,
                 lines_smooth_factor: float = 1e5,
                 dist_offset: float = 0,
                 custom_dist: bool = True,
                 dist_up: Optional[float] = None,
                 dist_down: Optional[float] = None,
                 precise_allocation: bool = False,
                 output_binning: int = 0) -> None:
        if spectral_band_width <= 0:
            raise ValueError("spectral_band_width must be > 0")
        if output_binning < 0:
            raise ValueError("output_binning must be >= 0")

        self.crop = crop
        self.spectral_band_width = spectral_band_width
        self.custom_lines_num = custom_lines_num
        self.lines_num = lines_num
        self.mask_width = mask_width
        self.smooth_factor = lines_smooth_factor
        self.dist_offset = dist_offset
        self.custom_dist = custom_dist
        self.dist_up = dist_up
        self.dist_down = dist_down
        self.precise_allocation = precise_allocation
        self.output_binning = output_binning

    # Static computational methods

    @staticmethod
    def _estimate_num_lines(img_work: np.ndarray) -> int:
        """Estimate the number of periodic structures from the vertical
        autocorrelation of the median intensity profile.

        """
        rows, _ = img_work.shape
        profile = np.median(img_work, axis=1)
        profile -= np.mean(profile)

        autocorr = signal.correlate(profile, profile, mode="full")
        autocorr = autocorr[autocorr.size // 2:]

        peaks, _ = signal.find_peaks(
            autocorr, distance=_AUTOCORR_MIN_PEAK_DISTANCE)
        period = peaks[0] if len(peaks) > 0 else rows
        return max(1, rows // period)

    @staticmethod
    def _extract_paths(img_work: np.ndarray, num_lines: int,
                       is_dark: bool, mask_width: int = 20) -> np.ndarray:
        """Dynamic-programming path extraction (Numba-accelerated).

        Parameters
        ----------
        img_work : np.ndarray
            Pre-smoothed copy of the edge image (modified **in-place**).
        num_lines : int
            Number of lines to extract.
        is_dark : bool
            ``True`` → minimise cost (dark lines);
            ``False`` → maximise cost (light lines).
        mask_width : int
            Half-width of the depletion mask.

        Returns
        -------
        np.ndarray
            Shape ``(num_lines, cols)`` — sorted row-index coordinates.

        """
        lines = _extract_paths_numba(img_work, num_lines, is_dark, mask_width)
        return np.sort(lines, axis=0)

    @staticmethod
    def extract_spectral_bands(image: np.ndarray,
                               regularized_lines: dict,
                               spectral_width: int) -> np.ndarray:
        """Extract spectral bands (Numba-accelerated).

        Parameters
        ----------
        image : np.ndarray
            Original 2-D image.
        regularized_lines : dict
            Output of ``regularize_structures``.
        spectral_width : int
            Number of spectral channels per band.

        Returns
        -------
        np.ndarray
            Shape ``(num_bands, image_width, spectral_width)``, float32.

        """
        up_lim = regularized_lines["dark_up"].T
        down_lim = regularized_lines["dark_down"].T
        return _extract_bands_numba(
            image.astype(np.float32), up_lim, down_lim, spectral_width)

    @staticmethod
    def allocate_spectral_pixels(
        image: np.ndarray,
        regularized_lines: dict,
        spectral_bands: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine row positions without expanding to a full 3-D array.

        Returns
        -------
        row_indices : np.ndarray
            1-D int32 array giving the row for each band.
        used_bands : np.ndarray
            Subset of ``spectral_bands`` that gets placed.

        """
        row_idx = regularized_lines["light"].T[0]
        n = min(len(row_idx), spectral_bands.shape[0])
        return row_idx[:n].copy(), spectral_bands[:n].copy()

    @staticmethod
    def allocate_spectral_pixels_precise(
        image: np.ndarray,
        regularized_lines: dict,
        spectral_bands: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine precise per-column row positions.

        Returns a 2-D array of shape ``(num_bands, width)``.

        """
        row_idx = regularized_lines["light"]
        n = min(len(row_idx), spectral_bands.shape[0])
        return row_idx[:n].copy(), spectral_bands[:n].copy()

    @staticmethod
    def expand_to_full(image_shape: tuple[int, int],
                       row_indices: np.ndarray,
                       spectral_bands: np.ndarray) -> np.ndarray:
        """Expand compact (bands + row_indices) into ``(H, W, S)``
        (Numba-accelerated).

        Parameters
        ----------
        image_shape : tuple[int, int]
            ``(height, width)`` of the original/cropped image.
        row_indices : np.ndarray
            Row positions — 1-D ``(num_bands,)`` or
            2-D ``(num_bands, cols)``.
        spectral_bands : np.ndarray
            Shape ``(num_bands, width, spectral_width)``.

        Returns
        -------
        np.ndarray
            Shape ``(height, width, spectral_width)``, dtype float32.

        """
        h, w = image_shape
        s = spectral_bands.shape[2]
        if row_indices.ndim == 1:
            return _expand_full_1d_numba(
                h, w, s, row_indices, spectral_bands)
        else:
            return _expand_full_2d_numba(
                h, w, s, row_indices, spectral_bands)

    # Instance pipeline methods

    def load_image(self, path: str) -> np.ndarray:
        """Load an image and apply the optional crop."""
        img = io.imread(path)
        if self.crop is not None:
            r0, r1, c0, c1 = self.crop
            img = img[r0:r1, c0:c1]
        return img

    def detect_structures(self, edge_image: np.ndarray) -> dict:
        """Detect periodic light and dark lines via dynamic programming."""
        img_work = ndi.gaussian_filter(
            edge_image.astype(np.float64), sigma=_GAUSSIAN_SIGMA)

        if self.custom_lines_num:
            num_lines = self.lines_num
            logger.info("Custom lines number: %d", num_lines)
        else:
            num_lines = self._estimate_num_lines(img_work)
            logger.info("Estimated lines number: %d", num_lines)

        return {
            "light": self._extract_paths(
                img_work.copy(), num_lines,
                is_dark=False, mask_width=self.mask_width),
            "dark": self._extract_paths(
                img_work.copy(), num_lines,
                is_dark=True, mask_width=self.mask_width),
        }

    def regularize_structures_light(self, detected_lines: dict) -> dict:
        """Smooth detected light lines and generate regularized dark
        boundaries at fixed distances.

        Uses **light lines** as the reference for boundary placement.
        See ``regularize_structures`` for the dark-line-based alternative.

        .. note::
           Dict keys ``"dark_up"`` / ``"dark_down"`` refer to the upper /
           lower extraction boundaries (higher / lower row indices), which
           is the inverse of the spatial direction of the offset from the
           light line.

        """
        light_lines = detected_lines.get("light", np.empty((0, 0)))
        dark_lines = detected_lines.get("dark", np.empty((0, 0)))
        if light_lines.size == 0:
            return {}

        cols = light_lines.shape[1]
        x_vals = np.arange(cols)
        splined_light = np.zeros_like(light_lines, dtype=np.float64)

        for i, line in enumerate(light_lines):
            spl = UnivariateSpline(x_vals, line, s=self.smooth_factor)
            splined_light[i] = spl(x_vals)

        dist_up = self.dist_up
        dist_down = self.dist_down

        if dist_up is None or dist_down is None or self.custom_dist is False:
            est_up, est_down = [], []
            for l_line in splined_light:
                dists = np.array(
                    [d_line - l_line for d_line in dark_lines])
                dists_mean = np.mean(dists, axis=1)

                above = dists_mean[dists_mean < -5]
                below = dists_mean[dists_mean > 5]

                if len(above) > 0:
                    est_up.append(np.abs(np.max(above)))
                if len(below) > 0:
                    est_down.append(np.min(below))

            if dist_up is None or self.custom_dist is False:
                dist_up = float(np.median(est_up)) if est_up else 10.0
            if dist_down is None or self.custom_dist is False:
                dist_down = (float(np.median(est_down))
                             if est_down else 10.0)

        dist_up = dist_up + self.dist_offset
        dist_down = dist_down + self.dist_offset

        num_light = len(splined_light)
        dark_up = np.zeros((num_light, cols), dtype=np.float64)
        dark_down = np.zeros((num_light, cols), dtype=np.float64)

        for i in range(num_light):
            dark_up[i] = splined_light[i] - dist_up
            dark_down[i] = splined_light[i] + dist_down

        return {"light":     np.trunc(splined_light).astype(np.int32),
                "dark_up":   np.trunc(dark_down).astype(np.int32),
                "dark_down": np.trunc(dark_up).astype(np.int32),
                "params":    {"dist_up": dist_up,
                            "dist_down": dist_down,
                            "offset": self.dist_offset}}

    def regularize_structures(self, detected_lines: dict) -> dict:
        """Smooth detected dark lines and generate regularized boundaries.

        Uses **dark lines** as the reference: light lines are placed at
        midpoints between consecutive darks, and boundaries are set at
        ``half_gap - gap_width`` on each side.

        .. note::
           Dict keys ``"dark_up"`` / ``"dark_down"`` refer to the upper /
           lower extraction boundaries (higher / lower row indices), which
           is the inverse of the spatial direction of the offset from the
           light line.

        Returns the same dictionary layout as
        ``regularize_structures_light``.

        """
        dark_lines = detected_lines.get("dark", np.empty((0, 0)))
        if dark_lines.size == 0:
            return {}

        cols = dark_lines.shape[1]
        x_vals = np.arange(cols)

        # smooth dark lines with B-splines
        splined_dark = np.zeros_like(dark_lines, dtype=np.float64)
        for i, line in enumerate(dark_lines):
            spl = UnivariateSpline(x_vals, line, s=self.smooth_factor)
            splined_dark[i] = spl(x_vals)

        # sort dark lines top-to-bottom
        order = np.argsort(np.mean(splined_dark, axis=1))
        splined_dark = splined_dark[order]

        # average distance between neighbouring dark lines
        gaps = np.diff(splined_dark, axis=0)
        avg_gap = float(np.mean(gaps))

        # place light lines at midpoints
        num_light = splined_dark.shape[0] - 1
        splined_light = np.zeros((num_light, cols), dtype=np.float64)
        for i in range(num_light):
            splined_light[i] = (
                (splined_dark[i] + splined_dark[i + 1]) / 2.0)

        half_gap = avg_gap / 2.0
        dist_up = half_gap - _GAP_WIDTH_PX
        dist_down = half_gap - _GAP_WIDTH_PX

        light_arr = np.zeros((num_light, cols), dtype=np.float64)
        dark_up_arr = np.zeros((num_light, cols), dtype=np.float64)
        dark_down_arr = np.zeros((num_light, cols), dtype=np.float64)
        for i in range(num_light):
            light_arr[i] = splined_light[i] + self.dist_offset
            dark_up_arr[i] = (splined_light[i] - dist_up
                              + self.dist_offset)
            dark_down_arr[i] = (splined_light[i] + dist_down
                                + self.dist_offset)

        return {"light":     np.trunc(light_arr).astype(np.int32),
                "dark_up":   np.trunc(dark_down_arr).astype(np.int32),
                "dark_down": np.trunc(dark_up_arr).astype(np.int32),
                "params":    {"dist_up": dist_up,
                            "dist_down": dist_down,
                            "offset": self.dist_offset}}

    def process_single(self, image_path: str) -> SingleResult:
        """Run the full pipeline on one image.

        Returns a ``SingleResult`` containing both the compact
        representation and the full ``(H, W, S)`` spectral image.
        If ``output_binning > 1``, the spectral image is down-sampled.

        """
        image = self.load_image(image_path)
        edges = ndi.prewitt(image, axis=0)
        structures = self.detect_structures(edges)
        del edges
        reg_lines = self.regularize_structures(structures)
        del structures

        bands = self.extract_spectral_bands(
            image, reg_lines, self.spectral_band_width)

        if self.precise_allocation:
            row_idx, used_bands = self.allocate_spectral_pixels_precise(
                image, reg_lines, bands)
        else:
            row_idx, used_bands = self.allocate_spectral_pixels(
                image, reg_lines, bands)
        del bands

        spectral_img = self.expand_to_full(
            image.shape[:2], row_idx, used_bands)

        if self.output_binning > 1:
            logger.info("Applying %d×%d output binning to single result...",
                        self.output_binning, self.output_binning)
            spectral_img = _binning_2d(spectral_img, self.output_binning)
            logger.info("Binning complete.")

        return SingleResult(
            spectral_img=spectral_img,
            raw_image=image,
            spectral_bands=used_bands,
            row_indices=row_idx,
            image_shape=image.shape[:2],
            spectral_width=self.spectral_band_width,
            regularized_lines=reg_lines,
        )

    def process_batch(self, image_paths: list[str],
                      method: str = "max") -> BatchResult:
        """Process multiple images with streaming accumulation.

        Only one reconstructed spectral image is held in memory at a time.
        Each result is immediately accumulated and then discarded.

        Parameters
        ----------
        image_paths : list of str
            Paths to the input images.
        method : ``'sum'`` | ``'max'``
            Accumulation strategy.

        Returns
        -------
        BatchResult
            Accumulated spectral image together with per-image metadata.

        """
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list")

        accumulator = None
        all_row_indices: dict[str, np.ndarray] = {}
        ref_shape = None
        sw_values: list[int] = []

        for idx, path in enumerate(image_paths, 1):
            logger.info("[%d/%d] Processing %s ...",
                        idx, len(image_paths), os.path.basename(path))

            single = self.process_single(path)
            sw_values.append(single.spectral_width)
            all_row_indices[path] = single.row_indices.copy()

            if ref_shape is None:
                ref_shape = single.image_shape

            full_img = self.expand_to_full(
                single.image_shape, single.row_indices,
                single.spectral_bands)

            if accumulator is None:
                accumulator = full_img
            else:
                if method == "max":
                    np.maximum(accumulator, full_img, out=accumulator)
                else:
                    accumulator += full_img

            del single, full_img
            gc.collect()

        sw_arr = np.asarray(sw_values, dtype=int)
        logger.info("Batch spectral width: min %d, max %d",
                    np.min(sw_arr), np.max(sw_arr))

        result = BatchResult(
            spectral_img=accumulator,
            spectral_img_interpolated=None,
            row_indices=all_row_indices,
            num_images=len(image_paths),
            spectral_width=int(sw_arr[0]),
            accumulation_method=method,
        )

        # Apply output binning if configured
        if self.output_binning > 1:
            result.apply_output_binning(self.output_binning)

        return result
