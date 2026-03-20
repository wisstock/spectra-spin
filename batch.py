"""
Batch spectral image reconstruction from spinning-disk spectral microscopy data.

This module encapsulates the spectral reconstruction pipeline
(periodic structure detection via dynamic programming, B-spline
regularization, spectral band extraction, and pixel allocation)
into the ``SpectralAnalyzer`` class and provides a convenience
wrapper ``reconstruct_spectral_image()`` for processing multiple
input images into a single accumulated spectral image.

Usage
-----
As a package::

    from batch import SpectralAnalyzer

    analyzer = SpectralAnalyzer(crop=(500, 2500, 1000, 2500),
                                dist_up=30, dist_down=50)
    spectral_img = analyzer.process_single("image.tiff")

    # or batch:
    spectral_img = analyzer.process_batch(["img1.tiff", "img2.tiff"])

"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy import signal
from scipy.interpolate import UnivariateSpline
from skimage import io


class SpectralAnalyzer:
    """Spectral image reconstruction from spinning-disk spectral microscopy data.

    Parameters
    ----------
    crop : tuple of int (row_start, row_end, col_start, col_end), optional
        Region of interest applied to every loaded image.  ``None`` means
        no cropping.
    mask_width : int
        Half-width of the depletion mask used after detecting each line in
        the dynamic-programming stage (prevents re-detection of the same
        structure).
    smooth_factor : float
        Smoothing factor *s* passed to ``UnivariateSpline`` when
        approximating detected light lines with B-splines.
    dist_up : float or None
        Fixed distance (in pixels) from a light line **upward** to the
        nearest dark boundary.  If ``None`` the distance is estimated
        automatically from detected dark lines.
    dist_down : float or None
        Fixed distance (in pixels) from a light line **downward** to the
        nearest dark boundary.  If ``None`` the distance is estimated
        automatically.
    """

    def __init__(self,
                 crop: Optional[tuple[int, int, int, int]] = None,
                 spectral_band_width:int = 100,
                 custom_lines_num: bool = False,
                 lines_num: int = 25,
                 mask_width: int = 60,
                 smooth_factor: float = 1e5,
                 custom_dist: bool = True,
                 dist_up: Optional[float] = None,
                 dist_down: Optional[float] = None) -> None:
        self.crop = crop
        self.spectral_band_width = spectral_band_width
        self.custom_lines_num = custom_lines_num
        self.lines_num = lines_num
        self.mask_width = mask_width
        self.smooth_factor = smooth_factor
        self.custom_dist = custom_dist
        self.dist_up = dist_up
        self.dist_down = dist_down


    @staticmethod
    def _estimate_num_lines(img_work: np.ndarray) -> int:
        """Estimate the number of periodic structures from the vertical
        autocorrelation of the median intensity profile."""
        rows, _ = img_work.shape
        profile = np.median(img_work, axis=1)
        profile -= np.mean(profile)

        autocorr = signal.correlate(profile, profile, mode='full')
        autocorr = autocorr[autocorr.size // 2:]

        peaks, _ = signal.find_peaks(autocorr, distance=10)
        period = peaks[0] if len(peaks) > 0 else rows
        return max(1, rows // period)


    @staticmethod
    def _extract_paths(img_work: np.ndarray, num_lines: int,
                       is_dark: bool, mask_width: int = 20) -> np.ndarray:
        """Dynamic-programming path extraction for light or dark lines.

        Parameters
        ----------
        img_work : np.ndarray
            Pre-smoothed copy of the edge image (will be modified in-place
            via masking).
        num_lines : int
            Number of lines to extract.
        is_dark : bool
            ``True`` for dark-line extraction (minimise cost),
            ``False`` for light (maximise cost).

        Returns
        -------
        np.ndarray
            Array of shape ``(num_lines, cols)`` with row-index coordinates.
        """
        rows, cols = img_work.shape
        lines = np.zeros((num_lines, cols), dtype=np.int32)

        mask_val = (np.max(img_work) + 1000.0 if is_dark
                    else np.min(img_work) - 1000.0)
        pad_val = np.inf if is_dark else -np.inf

        for line_idx in range(num_lines):
            acc = np.zeros_like(img_work)
            acc[:, 0] = img_work[:, 0]
            ptr = np.zeros_like(img_work, dtype=np.int32)

            for j in range(1, cols):
                v_up = np.pad(acc[:-1, j-1], (1, 0),
                              constant_values=pad_val)
                v_st = acc[:, j-1]
                v_dn = np.pad(acc[1:, j-1], (0, 1),
                              constant_values=pad_val)

                stacked = np.stack([v_up, v_st, v_dn])

                if is_dark:
                    acc[:,j] = img_work[:, j] + np.min(stacked, axis=0)
                    ptr[:,j] = np.argmin(stacked, axis=0)-1
                else:
                    acc[:,j] = img_work[:, j] + np.max(stacked, axis=0)
                    ptr[:,j] = np.argmax(stacked, axis=0)-1

            path = np.zeros(cols, dtype=np.int32)
            path[-1] = (np.argmin(acc[:, -1]) if is_dark
                        else np.argmax(acc[:, -1]))

            for j in range(cols-1, 0, -1):
                cur = path[j]
                path[j-1] = np.clip(cur+ptr[cur, j], 0, rows-1)

            lines[line_idx] = path

            # mask found line to prevent re-detection
            for j in range(cols):
                r = path[j]
                r_lo = max(0, r - mask_width)
                r_hi = min(rows, r + mask_width + 1)
                img_work[r_lo:r_hi, j] = mask_val

        return np.sort(lines, axis=0)


    @staticmethod
    def extract_spectral_bands(image: np.ndarray,
                               regularized_lines: dict,
                               spectral_width) -> np.ndarray:
        """Extract spectral bands from the raw image using regularized
        boundaries.

        Parameters
        ----------
        image : np.ndarray
            Original 2-D image (before edge filtering).
        regularized_lines : dict
            Output of ``regularize_structures``.

        Returns
        -------
        np.ndarray
            3-D array of shape ``(num_bands, image_width, spectral_width)``
            containing the collapsed spectral data.
        """
        up_lim = regularized_lines['dark_up'].T
        down_lim = regularized_lines['dark_down'].T

        # spectral_width = int(up_lim[0, 0] - down_lim[0, 0])

        lambda_img = np.zeros((up_lim.shape[1], up_lim.shape[0], spectral_width),
                            dtype=np.float32) # shape: (num_bands, img_cols, spectral_width)

        outligher_lines = 0
        bad_lines = 0
        for band in range(lambda_img.shape[0]):
            if np.any(up_lim[:, band]>image.shape[0]) or np.any(down_lim[:, band]<0):
                outligher_lines += 1
                continue
            else:
                for col in range(image.shape[1]):
                    up_idx = up_lim[col, band]
                    dn_idx = down_lim[col, band]
                    # band_row = image[dn_idx:up_idx, col]
                    # if band_row.shape[0] > spectral_width:
                    #     band_row = band_row[:spectral_width]
                    #     bad_lines += 1   
                    # elif band_row.shape[0] < spectral_width:
                    #     band_row = np.pad(band_row, pad_width=(0,spectral_width-band_row.shape[0]),
                    #                     mode='constant', constant_values=(0))
                    #     bad_lines += 1
                    lambda_img[band, col] = np.pad(image[dn_idx:up_idx, col],
                                                   pad_width=(0,spectral_width-(up_idx-dn_idx)),
                                                   mode='constant', constant_values=(0))
        print(outligher_lines, bad_lines)
        print(f'Out the frame bands number: {outligher_lines}')
        return lambda_img


    @staticmethod
    def allocate_spectral_pixels(image: np.ndarray, regularized_lines: dict, spectral_bands: np.ndarray) -> np.ndarray:
        """Place extracted spectral bands at their correct spatial positions
        to form the final reconstructed spectral image.

        Parameters
        ----------
        image : np.ndarray
            Original 2-D image (used only for its shape).
        regularized_lines : dict
            Output of ``regularize_structures``.
        spectral_bands : np.ndarray
            Output of ``extract_spectral_bands``.

        Returns
        -------
        np.ndarray
            3-D array of shape ``(image_height, image_width, spectral_width)``
            — the reconstructed spectral image.
        """
        spectral_width = spectral_bands.shape[2]
        row_idx = regularized_lines['light'].T[0]  # first column positions

        spectral_img = np.zeros(
            (image.shape[0], image.shape[1], spectral_width),
            dtype=np.float32,
        )
        for i in range(len(row_idx)):
            if i < spectral_bands.shape[0]:
                spectral_img[row_idx[i], :, :] = spectral_bands[i, :, :]

        return spectral_img


    def load_image(self, path: str) -> np.ndarray:
        """Load an image from *path* and apply the optional crop.

        Parameters
        ----------
        path : str
            Filesystem path to the image file (any format supported by
            ``skimage.io.imread``).

        Returns
        -------
        np.ndarray
            2-D image array.
        """
        img = io.imread(path)
        if self.crop is not None:
            r0, r1, c0, c1 = self.crop
            img = img[r0:r1, c0:c1]
        return img


    def detect_structures(self, edge_image: np.ndarray) -> dict:
        """Detect periodic light and dark lines via dynamic programming.

        The algorithm automatically estimates the number of structures
        from the vertical autocorrelation of the median intensity profile.

        Parameters
        ----------
        edge_image : np.ndarray
            2-D edge-filtered image (e.g. Sobel or Prewitt filter).

        Returns
        -------
        dict
            ``{'light': np.ndarray, 'dark': np.ndarray}`` where each value
            is an array of shape ``(num_lines, cols)`` containing row
            indices of detected structures.
        """
        img_work = ndi.gaussian_filter(edge_image.astype(np.float64),
                                       sigma=(1.5, 2.5))
        
        if self.custom_lines_num:
            num_lines = self.lines_num
            print(f'Custom lines number: {num_lines}')
        else:
            num_lines = self._estimate_num_lines(img_work)
            print(f'Estimated lines number: {num_lines}')

        return {'light': self._extract_paths(img_work.copy(), num_lines,
                                             is_dark=False,
                                             mask_width=self.mask_width),
                'dark':  self._extract_paths(img_work.copy(), num_lines,
                                             is_dark=True,
                                             mask_width=self.mask_width)}


    def regularize_structures(self, detected_lines: dict) -> dict:
        """Smooth detected light lines with B-splines and generate
        regularized dark boundaries at fixed distances.

        Parameters
        ----------
        detected_lines : dict
            Output of ``detect_structures`` (must contain ``'light'`` and
            ``'dark'`` keys).

        Returns
        -------
        dict
            ``{'light': ..., 'dark_up': ..., 'dark_down': ...,
            'params': {'dist_up': ..., 'dist_down': ...}}``
        """
        light_lines = detected_lines.get('light', np.empty((0, 0)))
        dark_lines = detected_lines.get('dark', np.empty((0, 0)))
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
            'light':     np.trunc(splined_light).astype(np.int32),
            'dark_up':   np.trunc(dark_down).astype(np.int32),
            'dark_down': np.trunc(dark_up).astype(np.int32),
            'params':    {'dist_up': dist_up, 'dist_down': dist_down},
        }


    def process_single(self, image_path: str) -> np.ndarray:
        """Run the complete reconstruction pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image file.

        Returns
        -------
        np.ndarray
            Reconstructed spectral image of shape
            ``(height, width, spectral_width)``.
        """
        image = self.load_image(image_path)
        edges =  ndi.prewitt(image, axis=0)
        structures = self.detect_structures(edges)
        reg_lines = self.regularize_structures(structures)
        bands = self.extract_spectral_bands(image, reg_lines, self.spectral_band_width)
        spectral_img = self.allocate_spectral_pixels(image, reg_lines, bands)
        return spectral_img


    def process_batch(self, image_paths: list[str]) -> np.ndarray:
        """Process multiple images and accumulate them into a single
        reconstructed spectral image.

        Parameters
        ----------
        image_paths : list of str
            Paths to the input images.
        method : ``'mean'`` | ``'sum'`` | ``'median'``
            Accumulation strategy applied pixel-wise across all processed
            images.

        Returns
        -------
        np.ndarray
            Final accumulated spectral image of shape
            ``(height, width, spectral_width)``.
        """
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list")

        results: list[np.ndarray] = []
        s_w = []
        for idx, path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] Processing {os.path.basename(path)} ...")
            single_result = self.process_single(path)
            s_w.append(single_result.shape[-1]) 
            results.append(single_result)

        s_w = np.asarray(s_w, dtype=int)
        print(f'Batch spectral width: min {np.min(s_w)}, max {np.max(s_w)}')

        stack = np.stack(results, axis=0)  # (N, H, W, S)

        return np.sum(stack, axis=0).astype(np.float32)
