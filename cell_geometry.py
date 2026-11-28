"""Per-cell geometry features: sin/cos positional embedding of an object's COM.

Shared by every dataset that feeds the SSL-family collate (training
``PointCloudObjects``, validation ``ProbeDataset``, inference
``_SuperMasterObjectDataset``) so the constructed ``point.feat`` is identical
across train / val / inference. The sin/cos formula is the per-axis block of
``GlobalTextureOrganoids.get_positional_embed`` (which now reuses
:func:`sincos_embed` so there is a single source of truth).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .SampleTypes import CellGeometry, Data
from skimage.measure import marching_cubes

logger = logging.getLogger(__name__)

# Columns written by scripts/add_com_to_supermaster.py (and, in future, by
# LMTextureNucleiDataset.volume2DF): per-object COM and its source-volume shape,
# both in [x, y, z] order to match the point-cloud frame.
COM_COLS: tuple[str, str, str] = ("com_x", "com_y", "com_z")
SHAPE_COLS: tuple[str, str, str] = ("shape_x", "shape_y", "shape_z")


def mask_to_surface_points(
    mask: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
    sigma: float = 1.0,
    pad: int = 2,
) -> np.ndarray:
    """``(P, 3)`` float32 ``[x, y, z]`` surface points for one isolated instance.

    The instance is cropped to its bounding box, surrounded by at least *pad* voxels
    of zeros, blurred with a gaussian of width *sigma*, and the iso-surface is taken
    from that continuous field rather than from the binary mask.  Both steps matter:

    * The zero padding closes the mesh.  Callers hand this a bbox-tight crop
      (:meth:`PointCloudObjects.get_hr_mask`), so without it the instance touches
      every face of the array and marching cubes leaves the surface open there.
    * Marching cubes on a binary array can only place vertices at voxel-edge
      midpoints, which reproduces the voxel staircase.  On a blurred field they
      interpolate instead: for a sphere of radius 12 the radial spread of the
      surface falls from 0.21 voxels to 0.07.

    The mask is deliberately *not* passed as ``marching_cubes(mask=...)``: that keeps
    only the cubes whose corner voxel is already inside the instance, which drops
    roughly half the surface and biases what survives inward.

    The level is ``0.5`` except for instances thin enough that the blur never reaches
    it -- a 3-voxel-wide structure peaks at 0.479 for ``sigma=1.5`` -- where it falls
    back to half the field's maximum, so a thin instance still yields a surface
    instead of dropping out of the dataset.

    *spacing* defaults to isotropic because callers feed this from an N5 that
    :meth:`BaseVolumeDataset.prepareSampleVolume` already resampled to isotropic voxels;
    an anisotropic default would apply the correction a second time.
    """
    from scipy.ndimage import find_objects, gaussian_filter

    assert mask.any(), "The mask is empty"

    # Padded with np.pad rather than by widening the crop: the instance may already
    # touch the array border (callers crop to its bbox), and clipping the widened
    # slice back to the array would silently leave that side unpadded.  The padding
    # must also outlast the kernel, or the blur reads the boundary instead of zeros.
    sl = find_objects(mask.astype(np.uint8))[0]
    pad = max(pad, int(np.ceil(3 * sigma)))
    field = gaussian_filter(np.pad(mask[sl].astype(np.float32), pad),
                            sigma=sigma, mode="constant")

    verts, *_ = marching_cubes(field, level=min(0.5, 0.5 * float(field.max())),
                               spacing=spacing)

    verts += np.multiply([s.start - pad for s in sl], spacing)   # back into the crop's frame
    return verts[:, ::-1].astype(np.float32)                     # zyx -> POINTCLOUD_COLUMNS' xyz


def compute_volume_coms(
    mask: np.ndarray,
) -> Tuple[Dict[int, Tuple[float, float, float]], Tuple[int, int, int]]:
    """Per-label centroid (uniform weight) of a 3-D label mask.

    Returns ``({label_id: (x, y, z)}, (shape_x, shape_y, shape_z))``. The mask is
    assumed ``[z, y, x]`` (tifffile/skimage page order); the COM and shape are
    returned in ``[x, y, z]`` order to match the point-cloud frame. Shared by the
    retrospective ``add_com_to_supermaster`` script and ``LMTextureNucleiDataset``
    so both compute COM identically. ``scipy`` is imported lazily to keep this
    module light for the dataset/collate import path.
    """
    import scipy.ndimage as ndi

    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got shape {mask.shape}")
    shape_xyz = (int(mask.shape[2]), int(mask.shape[1]), int(mask.shape[0]))

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return {}, shape_xyz

    # Uniform weights → geometric centroid of each label's voxels. center_of_mass
    # returns (z, y, x); reorder to (x, y, z).
    coms = np.asarray(
        ndi.center_of_mass(np.ones_like(mask, dtype=np.uint8), labels=mask, index=labels)
    ).reshape(-1, 3)
    out = {
        int(lab): (float(c[2]), float(c[1]), float(c[0]))
        for lab, c in zip(labels.tolist(), coms)
    }
    return out, shape_xyz


def add_com_columns(
    df: pd.DataFrame,
    mask_col: str = "mask_volume_path",
    label_col: str = "label_id",
) -> pd.DataFrame:
    """Populate ``com_*``/``shape_*`` columns on a per-object frame, in place.

    Groups rows by their segmentation mask (``mask_col``) — matching the organoid
    first — loads each mask once, and assigns each object's mask-weighted COM and
    source-volume shape by ``label_col``. Shared by ``PointCloudObjects._build_supermaster``
    (during construction) and ``scripts/add_com_to_supermaster.py`` (retrospective)
    so both write COM identically. ``tifffile`` is imported lazily to keep this
    module light for the dataset/collate import path.
    """
    import tifffile

    for col in (*COM_COLS, *SHAPE_COLS):
        if col not in df.columns:
            df[col] = np.nan

    n_objects = n_missing_mask = n_missing_label = 0
    for mask_path, group in df.groupby(mask_col):
        mask_path = str(mask_path)
        if not os.path.exists(mask_path):
            logger.warning("Mask not found, leaving COM as NaN: %s", mask_path)
            n_missing_mask += len(group)
            continue

        coms, shape_xyz = compute_volume_coms(tifffile.imread(mask_path))
        logger.info("%-60s  %d objects, shape(xyz)=%s", os.path.basename(mask_path),
                    len(coms), shape_xyz)

        for idx in group.index:
            com = coms.get(int(df.at[idx, label_col]))
            if com is None:
                n_missing_label += 1
                continue
            df.at[idx, "com_x"], df.at[idx, "com_y"], df.at[idx, "com_z"] = com
            df.at[idx, "shape_x"], df.at[idx, "shape_y"], df.at[idx, "shape_z"] = shape_xyz
            n_objects += 1

    logger.info("COM written for %d objects (%d masks missing, %d labels absent from mask)",
                n_objects, n_missing_mask, n_missing_label)
    return df


def sincos_embed(values: np.ndarray, num_sin_cos: int) -> np.ndarray:
    """Per-axis sin/cos positional code for a 1-D array of coordinates.

    Extracted verbatim from the ``if num_sin_cos > 0`` block of
    ``GlobalTextureOrganoids.get_positional_embed`` (without the trailing
    ``dist_embed``). Returns ``(M, 2 * (num_sin_cos // 6))`` — an empty
    ``(M, 0)`` when ``num_sin_cos < 6``, matching the original.
    """
    values = np.asarray(values, dtype=float).ravel()
    omega = np.arange(num_sin_cos // 6, dtype="float")
    omega /= num_sin_cos // 6.0          # 1.0 for num_sin_cos == 6 -> omega == [0.0]
    omega = 1.0 / 10000 ** omega          # [1.0] for num_sin_cos == 6
    out = np.einsum("m,d->md", values, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def com_sincos(com_norm: Sequence[float], num_sin_cos: int = 6) -> np.ndarray:
    """Flatten the per-axis sin/cos of a single normalized COM into one vector.

    ``com_norm``: length-3 ``[x, y, z]`` in roughly ``[0, 1]``. With the default
    ``num_sin_cos=6`` this yields the 6 values
    ``[sin x, cos x, sin y, cos y, sin z, cos z]`` — the same per-axis ordering as
    ``get_positional_embed``.
    """
    com_norm = np.asarray(com_norm, dtype=float).ravel()
    per_axis = [sincos_embed(np.array([c]), num_sin_cos) for c in com_norm]  # each (1, k)
    return np.concatenate(per_axis, axis=1).ravel().astype(np.float32)


def _read_triplet(row: "pd.Series", cols: Sequence[str]) -> Optional[np.ndarray]:
    """Read three numeric columns from *row*; ``None`` if absent/non-numeric/NaN."""
    try:
        vals = np.array([float(row[c]) for c in cols], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return None if np.any(np.isnan(vals)) else vals


def cell_geometry_from_row(
    row: "pd.Series",
    data: Data,
    file_path: str,
    num_sin_cos: int = 6,
) -> CellGeometry:
    """Build a :class:`CellGeometry` from a super-master row + its point cloud.

    Reads the per-object COM (``com_x/y/z``) and source-volume shape
    (``shape_x/y/z``), normalizes ``com / shape`` to ``[0, 1]``, and stores the
    sin/cos embedding as ``feat``. If either triplet is missing/NaN (or shape has
    a zero axis), ``feat=None`` so the collate falls back to
    ``[coord, color, normal]``.
    """
    com = _read_triplet(row, COM_COLS)
    shape = _read_triplet(row, SHAPE_COLS)
    assert not (com is None or shape is None or np.any(shape == 0)), "COM is NONE"
    if com is None or shape is None or np.any(shape == 0):
        return CellGeometry(data=data, filePath=file_path, feat=None)
    return CellGeometry(
        data=data,
        filePath=file_path,
        feat=com_sincos(com / shape, num_sin_cos),
    )
