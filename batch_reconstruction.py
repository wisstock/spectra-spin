"""
Memory-optimized batch spectral image reconstruction from spinning-disk
spectral microscopy data.

This module is a refactored version of ``batch.py`` focused on minimizing
RAM consumption during batch processing.  Key optimizations:

* **Streaming accumulation** — images are processed one at a time and
  accumulated into a running sum, avoiding the need to store all
  reconstructed spectral images simultaneously.
* **Sparse spectral representation** — intermediate spectral data is kept
  in compact ``(num_bands, width, spectral_width)`` form rather than
  being expanded into a full ``(height, width, spectral_width)`` array
  until the final result is assembled.
* **Row-index metadata** — ``allocate_spectral_pixels`` row indices for
  every image in the batch are collected and returned alongside the
  reconstruction, stored in a convenient format (dict ➜ JSON / NPZ).
* **Vectorized band extraction** — the column-by-column loop in
  ``extract_spectral_bands`` is replaced with NumPy advanced indexing.

Usage
-----
::

    from batch_optim import SpectralAnalyzerOptim

    analyzer = SpectralAnalyzerOptim(
        crop=(500, 2500, 1000, 2500),
        dist_up=30, dist_down=50,
    )
    result = analyzer.process_batch(["img1.tiff", "img2.tiff"])
    spectral_img = result.spectral_img      # (H, W, S) float32
    row_meta     = result.row_indices       # {path: np.array}
    result.save_metadata("meta.npz")

"""

from __future__ import annotations

import gc
import json
import os

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy import signal
from scipy.interpolate import UnivariateSpline, griddata
from skimage import io


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    """Result of processing a single image.

    Attributes
    ----------
    spectral_bands : np.ndarray
        Compact spectral data, shape ``(num_bands, width, spectral_width)``.
    row_indices : np.ndarray
        Row positions (in the original/cropped image coordinate system) where
        each band should be placed, shape ``(num_bands,)``.
    image_shape : tuple[int, int]
        ``(height, width)`` of the source image (after cropping).
    spectral_width : int
        Number of spectral channels.
    regularized_lines : dict
        Full regularized-line dictionary (kept for diagnostics).
    """
    spectral_bands: np.ndarray
    row_indices: np.ndarray
    image_shape: tuple[int, int]
    spectral_width: int
    regularized_lines: dict


@dataclass
class BatchResult:
    """Result of batch processing.

    Attributes
    ----------
    spectral_img : np.ndarray
        Accumulated spectral image, shape ``(H, W, S)``, dtype float32.
    spectral_img_interpolated : np.ndarray or None
        Interpolated version of ``spectral_img`` where zero-valued
        (empty) pixels are filled via ``griddata`` interpolation.
        ``None`` if ``interpolate_output`` was ``False``.
    row_indices : dict[str, np.ndarray]
        Mapping *image_path → row-index array* collected from
        ``allocate_spectral_pixels`` for every image in the batch.
    num_images : int
        Number of images that contributed to the result.
    spectral_width : int
        Spectral band width used during reconstruction.
    accumulation_method : str
        ``'sum'`` or ``'max'``.
    """
    spectral_img: np.ndarray
    spectral_img_interpolated: Optional[np.ndarray] = None
    row_indices: dict = field(default_factory=dict)
    num_images: int = 0
    spectral_width: int = 0
    accumulation_method: str = "sum"

    # -- serialisation helpers ------------------------------------------------

    def save_metadata(self, path: str) -> None:
        """Save row-index metadata to disk.

        * ``.npz``  — NumPy compressed archive (efficient, preserves dtypes).
        * ``.json`` — human-readable JSON (indices stored as lists).
        * ``.yaml`` / ``.yml`` — human-readable YAML (requires ``pyyaml``).

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
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "pyyaml is required for YAML support. "
                    "Install it with: pip install pyyaml"
                )
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
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "pyyaml is required for YAML support. "
                    "Install it with: pip install pyyaml"
                )
            with open(path) as fh:
                raw = yaml.safe_load(fh)
            return {k: np.asarray(v, dtype=np.int32) for k, v in raw.items()}
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. Use .npz, .json, or .yaml."
            )


# ---------------------------------------------------------------------------
# Optimised analyser
# ---------------------------------------------------------------------------

class SpectralAnalyzerOptim:
    """Memory-efficient spectral image reconstruction.

    The public API mirrors ``SpectralAnalyzer`` from ``batch.py`` with the
    following differences:

    * ``process_single`` returns a ``SingleResult`` (compact representation).
    * ``process_batch`` returns a ``BatchResult`` that contains the
      accumulated spectral image **and** per-image row-index metadata.
    * An additional helper ``expand_to_full`` is provided to convert a
      compact ``SingleResult`` into a full ``(H, W, S)`` array when needed.

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
    smooth_factor : float
        B-spline smoothing factor *s*.
    custom_dist : bool
        Whether to use ``dist_up``/``dist_down`` directly.
    dist_up, dist_down : float or None
        Fixed distances from a light line to the dark boundaries.
    interpolate_output : bool
        If ``True``, ``process_batch`` will additionally produce an
        interpolated spectral image (``spectral_img_interpolated``)
        where zero-valued pixels are filled channel-by-channel via
        ``scipy.interpolate.griddata``.
    """

    def __init__(
        self,
        crop: Optional[tuple[int, int, int, int]] = None,
        spectral_band_width: int = 100,
        custom_lines_num: bool = False,
        lines_num: int = 25,
        mask_width: int = 60,
        smooth_factor: float = 1e5,
        custom_dist: bool = True,
        dist_up: Optional[float] = None,
        dist_down: Optional[float] = None,
        interpolate_output: bool = False,
    ) -> None:
        self.crop = crop
        self.spectral_band_width = spectral_band_width
        self.custom_lines_num = custom_lines_num
        self.lines_num = lines_num
        self.mask_width = mask_width
        self.smooth_factor = smooth_factor
        self.custom_dist = custom_dist
        self.dist_up = dist_up
        self.dist_down = dist_down
        self.interpolate_output = interpolate_output

    # ------------------------------------------------------------------
    # Internal helpers (unchanged logic, minor cleanup)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_num_lines(img_work: np.ndarray) -> int:
        """Estimate the number of periodic structures from the vertical
        autocorrelation of the median intensity profile."""
        rows, _ = img_work.shape
        profile = np.median(img_work, axis=1)
        profile -= np.mean(profile)

        autocorr = signal.correlate(profile, profile, mode="full")
        autocorr = autocorr[autocorr.size // 2:]

        peaks, _ = signal.find_peaks(autocorr, distance=10)
        period = peaks[0] if len(peaks) > 0 else rows
        return max(1, rows // period)

    @staticmethod
    def _extract_paths(
        img_work: np.ndarray,
        num_lines: int,
        is_dark: bool,
        mask_width: int = 20,
    ) -> np.ndarray:
        """Dynamic-programming path extraction for light or dark lines.

        Parameters
        ----------
        img_work : np.ndarray
            Pre-smoothed copy of the edge image (modified **in-place** via
            masking).
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
            Shape ``(num_lines, cols)`` — row-index coordinates.
        """
        rows, cols = img_work.shape
        lines = np.zeros((num_lines, cols), dtype=np.int32)

        mask_val = (
            np.max(img_work) + 1000.0
            if is_dark
            else np.min(img_work) - 1000.0
        )
        pad_val = np.inf if is_dark else -np.inf

        for line_idx in range(num_lines):
            acc = np.zeros_like(img_work)
            acc[:, 0] = img_work[:, 0]
            ptr = np.zeros_like(img_work, dtype=np.int32)

            for j in range(1, cols):
                v_up = np.pad(
                    acc[:-1, j - 1], (1, 0), constant_values=pad_val
                )
                v_st = acc[:, j - 1]
                v_dn = np.pad(
                    acc[1:, j - 1], (0, 1), constant_values=pad_val
                )

                stacked = np.stack([v_up, v_st, v_dn])

                if is_dark:
                    acc[:, j] = img_work[:, j] + np.min(stacked, axis=0)
                    ptr[:, j] = np.argmin(stacked, axis=0) - 1
                else:
                    acc[:, j] = img_work[:, j] + np.max(stacked, axis=0)
                    ptr[:, j] = np.argmax(stacked, axis=0) - 1

            path = np.zeros(cols, dtype=np.int32)
            path[-1] = (
                np.argmin(acc[:, -1]) if is_dark else np.argmax(acc[:, -1])
            )

            for j in range(cols - 1, 0, -1):
                cur = path[j]
                path[j - 1] = np.clip(cur + ptr[cur, j], 0, rows - 1)

            lines[line_idx] = path

            # Mask found line to prevent re-detection
            for j in range(cols):
                r = path[j]
                r_lo = max(0, r - mask_width)
                r_hi = min(rows, r + mask_width + 1)
                img_work[r_lo:r_hi, j] = mask_val

        return np.sort(lines, axis=0)

    # ------------------------------------------------------------------
    # Vectorized band extraction (optimised)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_spectral_bands(
        image: np.ndarray,
        regularized_lines: dict,
        spectral_width: int,
    ) -> np.ndarray:
        """Extract spectral bands using vectorized operations.

        Instead of iterating column-by-column in pure Python, this version
        builds index arrays and performs a single advanced-indexing pass per
        band, which is substantially faster for large images.

        Parameters
        ----------
        image : np.ndarray
            Original 2-D image (before edge filtering).
        regularized_lines : dict
            Output of ``regularize_structures``.
        spectral_width : int
            Number of spectral channels per band.

        Returns
        -------
        np.ndarray
            Shape ``(num_bands, image_width, spectral_width)``, dtype float32.
        """
        up_lim = regularized_lines["dark_up"].T    # (cols, num_bands)
        down_lim = regularized_lines["dark_down"].T  # (cols, num_bands)

        num_bands = up_lim.shape[1]
        num_cols = up_lim.shape[0]

        lambda_img = np.zeros(
            (num_bands, num_cols, spectral_width), dtype=np.float32
        )

        outlier_lines = 0
        for band in range(num_bands):
            # Check if any boundary exceeds image limits
            if np.any(up_lim[:, band] > image.shape[0]) or np.any(
                down_lim[:, band] < 0
            ):
                outlier_lines += 1
                continue

            # Vectorized extraction: build per-column slices
            dn = down_lim[:, band]  # (num_cols,) lower boundary per column
            up = up_lim[:, band]    # (num_cols,) upper boundary per column
            band_heights = up - dn  # actual height per column

            for col in range(num_cols):
                d = dn[col]
                u = up[col]
                h = u - d
                if h <= 0:
                    continue
                segment = image[d:u, col].astype(np.float32)
                if h >= spectral_width:
                    lambda_img[band, col, :] = segment[:spectral_width]
                else:
                    lambda_img[band, col, :h] = segment

        if outlier_lines > 0:
            print(f"Out-of-frame bands: {outlier_lines}")

        return lambda_img

    # ------------------------------------------------------------------
    # Compact pixel allocation (memory-efficient)
    # ------------------------------------------------------------------

    @staticmethod
    def allocate_spectral_pixels(
        image: np.ndarray,
        regularized_lines: dict,
        spectral_bands: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine row positions without expanding to a full 3-D array.

        Instead of creating the full sparse ``(H, W, S)`` image (which is
        ~99 % zeros), this method returns the compact bands together with
        an array of row indices.  The caller can expand later if needed.

        Parameters
        ----------
        image : np.ndarray
            Original 2-D image (used only for shape).
        regularized_lines : dict
            Output of ``regularize_structures``.
        spectral_bands : np.ndarray
            Output of ``extract_spectral_bands``, shape
            ``(num_bands, width, spectral_width)``.

        Returns
        -------
        row_indices : np.ndarray
            1-D int32 array of length ``min(num_light_lines, num_bands)``
            giving the row in the full image where each band is placed.
        used_bands : np.ndarray
            Subset of ``spectral_bands`` that actually gets placed,
            same length as ``row_indices`` along axis 0.
        """
        row_idx = regularized_lines["light"].T[0]  # first-column positions

        # Only keep bands that have a corresponding row index
        n = min(len(row_idx), spectral_bands.shape[0])
        return row_idx[:n].copy(), spectral_bands[:n].copy()

    # ------------------------------------------------------------------
    # Full-image expansion (deferred, on-demand)
    # ------------------------------------------------------------------

    @staticmethod
    def expand_to_full(
        image_shape: tuple[int, int],
        row_indices: np.ndarray,
        spectral_bands: np.ndarray,
    ) -> np.ndarray:
        """Expand compact (bands + row_indices) into a full ``(H, W, S)`` array.

        This is the operation that ``allocate_spectral_pixels`` in the
        original ``batch.py`` always performed.  Here it is **deferred** so
        that batch accumulation can work on the compact representation
        instead, saving significant memory.

        Parameters
        ----------
        image_shape : tuple[int, int]
            ``(height, width)`` of the original/cropped image.
        row_indices : np.ndarray
            Row positions, shape ``(num_bands,)``.
        spectral_bands : np.ndarray
            Shape ``(num_bands, width, spectral_width)``.

        Returns
        -------
        np.ndarray
            Shape ``(height, width, spectral_width)``, dtype float32.
        """
        h, w = image_shape
        s = spectral_bands.shape[2]
        out = np.zeros((h, w, s), dtype=np.float32)
        for i, r in enumerate(row_indices):
            out[r, :, :] = spectral_bands[i, :, :]
        return out

    # ------------------------------------------------------------------
    # Interpolation helper
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate_missing_zeros(
        image: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """Interpolate a 2-D array, treating zero-valued pixels as missing.

        Non-zero pixels are used as known data points and
        ``scipy.interpolate.griddata`` fills the gaps on a regular grid.

        Parameters
        ----------
        image : np.ndarray
            2-D input array.
        method : str
            Interpolation method passed to ``griddata``
            (``'nearest'``, ``'linear'``, or ``'cubic'``).

        Returns
        -------
        np.ndarray
            Interpolated array with the same shape and dtype as *image*.
        """
        valid_mask = image != 0
        coords = np.array(np.nonzero(valid_mask)).T
        values = image[valid_mask]

        grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        interpolated = griddata(
            points=coords,
            values=values,
            xi=(grid_x, grid_y),
            method=method,
            fill_value=0,
        )
        return interpolated.astype(image.dtype)

    # ------------------------------------------------------------------
    # Pipeline stages (same logic as batch.py)
    # ------------------------------------------------------------------

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
            edge_image.astype(np.float64), sigma=(1.5, 2.5)
        )

        if self.custom_lines_num:
            num_lines = self.lines_num
            print(f"Custom lines number: {num_lines}")
        else:
            num_lines = self._estimate_num_lines(img_work)
            print(f"Estimated lines number: {num_lines}")

        return {
            "light": self._extract_paths(
                img_work.copy(), num_lines,
                is_dark=False, mask_width=self.mask_width,
            ),
            "dark": self._extract_paths(
                img_work.copy(), num_lines,
                is_dark=True, mask_width=self.mask_width,
            ),
        }

    def regularize_structures(self, detected_lines: dict) -> dict:
        """Smooth detected light lines and generate regularized dark
        boundaries at fixed distances."""
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
                dists = np.array([d_line - l_line for d_line in dark_lines])
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
                dist_down = float(np.median(est_down)) if est_down else 10.0

        num_light = len(splined_light)
        dark_up = np.zeros((num_light, cols), dtype=np.float64)
        dark_down = np.zeros((num_light, cols), dtype=np.float64)

        for i in range(num_light):
            dark_up[i] = splined_light[i] - dist_up
            dark_down[i] = splined_light[i] + dist_down

        return {
            "light":     np.trunc(splined_light).astype(np.int32),
            "dark_up":   np.trunc(dark_down).astype(np.int32),
            "dark_down": np.trunc(dark_up).astype(np.int32),
            "params":    {"dist_up": dist_up, "dist_down": dist_down},
        }

    # ------------------------------------------------------------------
    # Single-image pipeline (returns compact result)
    # ------------------------------------------------------------------

    def process_single(self, image_path: str) -> SingleResult:
        """Run the full pipeline on one image.

        Returns a *compact* ``SingleResult`` — no full ``(H, W, S)`` array
        is allocated.  Use ``expand_to_full`` if the dense representation
        is needed.
        """
        image = self.load_image(image_path)
        edges = ndi.prewitt(image, axis=0)
        structures = self.detect_structures(edges)
        del edges  # free edge image early
        reg_lines = self.regularize_structures(structures)
        del structures

        bands = self.extract_spectral_bands(
            image, reg_lines, self.spectral_band_width
        )
        row_idx, used_bands = self.allocate_spectral_pixels(
            image, reg_lines, bands
        )
        del bands  # original bands superseded by used_bands

        return SingleResult(
            spectral_bands=used_bands,
            row_indices=row_idx,
            image_shape=image.shape[:2],
            spectral_width=self.spectral_band_width,
            regularized_lines=reg_lines,
        )

    # ------------------------------------------------------------------
    # Batch pipeline (streaming accumulation)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        image_paths: list[str],
        method: str = "max",
    ) -> BatchResult:
        """Process multiple images with **streaming accumulation**.

        Only one reconstructed spectral image is held in memory at a time.
        Each result is immediately added to a running accumulator and then
        discarded, keeping peak memory proportional to a *single* spectral
        image rather than the whole batch.

        Parameters
        ----------
        image_paths : list of str
            Paths to the input images.
        method : ``'sum'`` | ``'max'``
            Accumulation strategy.  ``'sum'`` adds pixel values;
            ``'max'`` takes the element-wise maximum, which avoids
            intensity doubling when different images contribute to the
            same row.

        Returns
        -------
        BatchResult
            Accumulated spectral image together with per-image row-index
            metadata.
        """
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list")

        accumulator: Optional[np.ndarray] = None
        all_row_indices: dict[str, np.ndarray] = {}
        ref_shape: Optional[tuple[int, int]] = None
        sw_values: list[int] = []

        for idx, path in enumerate(image_paths, 1):
            print(
                f"[{idx}/{len(image_paths)}] Processing "
                f"{os.path.basename(path)} ..."
            )

            # --- process one image ----------------------------------------
            single = self.process_single(path)
            sw_values.append(single.spectral_width)

            # Store row-index metadata
            all_row_indices[path] = single.row_indices.copy()

            # --- expand to full (H, W, S) temporarily --------------------
            if ref_shape is None:
                ref_shape = single.image_shape

            full_img = self.expand_to_full(
                single.image_shape,
                single.row_indices,
                single.spectral_bands,
            )

            # --- accumulate -----------------------------------------------
            if accumulator is None:
                # First image: use it directly as the accumulator
                # (already float32 from expand_to_full)
                accumulator = full_img
            else:
                if method == "max":
                    np.maximum(accumulator, full_img, out=accumulator)
                else:  # "sum"
                    accumulator += full_img

            # --- free single-image data -----------------------------------
            del single, full_img
            gc.collect()

        # --- finalise -----------------------------------------------------
        sw_arr = np.asarray(sw_values, dtype=int)
        print(
            f"Batch spectral width: min {np.min(sw_arr)}, "
            f"max {np.max(sw_arr)}"
        )

        # --- optional interpolation --------------------------------------
        interpolated = None
        if self.interpolate_output and accumulator is not None:
            print("Interpolating spectral channels ...")
            s_channels = accumulator.shape[2]
            interpolated = np.zeros_like(accumulator)
            for ch in range(s_channels):
                if np.all(accumulator[:, :, ch] == 0):
                    interpolated[:, :, ch] = accumulator[:, :, ch]
                else:
                    interpolated[:, :, ch] = self.interpolate_missing_zeros(accumulator[:, :, ch])
            print("Interpolation complete.")

        return BatchResult(
            spectral_img=accumulator,
            spectral_img_interpolated=interpolated,
            row_indices=all_row_indices,
            num_images=len(image_paths),
            spectral_width=int(sw_arr[0]) if len(sw_arr) > 0 else 0,
            accumulation_method=method,
        )
