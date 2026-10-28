"""Coordinate/label column resolution and vertex extraction for SMLM point clouds.

Pure helpers extracted from ``dataset.py`` so the column-candidate lists and the
resolve/extract logic have a single home (they were previously duplicated across
``SMLMDataset`` and ``dir2HeterogeneousCSV``). ``SMLMDataset`` now delegates to these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate
from torchvision.transforms.v2.functional import get_num_frames

from constants import MOBIE_LABEL_KEY
from SampleLoader import read_localization_file
from SampleTypes import Vertices

X_COLUMN_CANDIDATES = ("xnm", "x [nm]", "x[nm]", "x[pix]", "x", "X")
Y_COLUMN_CANDIDATES = ("ynm", "y [nm]", "y[nm]", "y[pix]", "y", "Y")
Z_COLUMN_CANDIDATES = ("znm", "z [nm]", "z[nm]", "z[pix]", "z", "Z")
LABEL_COLUMN_CANDIDATES = (
    "cluster_id",
    "clusterIndex",
    "labels",
    "label",
    "index",
    "particle_id",
)

_AXIS_CANDIDATES = (
    ("x", X_COLUMN_CANDIDATES),
    ("y", Y_COLUMN_CANDIDATES),
    ("z", Z_COLUMN_CANDIDATES),
)


FRAME_COLUMN_CANDIDATES = ("frame", "Frame", "frame [#]", "frame_ix", "t", "time")
PHOTON_COLUMN_CANDIDATES = ("photons", "phot", "intensity [photon]", "intensity", "Intensity", "amplitude", "I")


def _first_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> np.ndarray | None:
    """Return the first candidate column present in *dataframe* as a 1D array, else None."""
    name = next((c for c in candidates if c in dataframe.columns), None)
    return None if name is None else np.atleast_1d(dataframe[name].to_numpy().squeeze())


def get_frames(dataframe: pd.DataFrame) -> np.ndarray | None:
    """Frame index per localization (ThunderSTORM / Picasso / SMAP / DECODE naming)."""
    return _first_column(dataframe, FRAME_COLUMN_CANDIDATES)


def get_photons(dataframe: pd.DataFrame) -> np.ndarray | None:
    """Photon count / intensity per localization (ThunderSTORM / Picasso / SMAP / DECODE naming)."""
    return _first_column(dataframe, PHOTON_COLUMN_CANDIDATES)


def resolve_vertices_col(
        columns, input_dim: int = 3, strict: bool = False
) -> list[str] | None:
    """Default coordinate columns for a given model ``input_dim``.

    Selects the first ``input_dim`` spatial axes — ``input_dim == 2`` -> [x, y],
    ``input_dim == 3`` -> [x, y, z] — using the module-level candidate lists with a
    first-match convention. ``strict=True`` raises when an axis is absent; otherwise
    returns ``None`` so callers can skip files that don't match ``input_dim``.
    """
    if input_dim not in (2, 3):
        raise ValueError(f"input_dim must be 2 or 3, got {input_dim}")

    header = list(columns)
    present = set(header)

    resolved = []
    for axis, candidates in _AXIS_CANDIDATES[:input_dim]:
        col = next((c for c in candidates if c in present), None)
        if col is None:
            if strict:
                raise ValueError(
                    f"Could not resolve '{axis}' coordinate column for "
                    f"input_dim={input_dim} from {header}; tried {list(candidates)}"
                )
            else:
                continue
        resolved.append(col)
    return resolved


def get_vertices(
        item_df: pd.DataFrame,
        vertices_col: list[str] = None,
        input_dim: int = 2,
) -> "np.ndarray[Any, np.dtype[np.float32]]":

    if isinstance(vertices_col, str):
        vertices_col = [vertices_col]
    # An explicit column list wins when this file has it; otherwise (none given, or a
    # heterogeneous dataset whose files name their axes differently) resolve from the
    # candidate lists, as before.
    if not vertices_col or not set(vertices_col).issubset(item_df.columns):
        vertices_col = resolve_vertices_col(item_df.columns, input_dim)

    # (N, D): no squeeze(), which collapsed a single-row (1, D) or single-column (N, 1)
    # selection to 1-D and broke 2-D indexing downstream (e.g. xy[:, 0]).
    return item_df[list(vertices_col)].to_numpy(dtype=np.float32)


def _for_file(spec, file, allow_missing: bool = False):
    """Resolve a per-file column mapping to this file's entry.

    ``spec`` is either a plain column name (or list of names) applied to every file,
    or a DataFrame/Series keyed by ``sample_path`` -- the ``standardized_*`` /
    ``standardizedLabels`` columns of an index CSV. Resolving here, where ``file`` is
    in scope, is what keeps every caller from having to do it; the ones that forgot
    used to hand pandas a whole Series as a column key, which ``DataFrame.get``
    swallowed into an all-zero fallback.

    A NaN entry raises, unless ``allow_missing`` (then ``None``): an empty label entry
    in the mapping is how an unlabelled file is declared.
    """
    if not isinstance(spec, (pd.DataFrame, pd.Series)):
        return spec
    try:
        value = spec.loc[str(file)]
    except KeyError:
        raise KeyError(f"no sample-mapping entry for {file}") from None
    if isinstance(value, pd.Series):
        value = value.tolist()
    if np.any(pd.isna(value)):
        if allow_missing:
            return None
        raise KeyError(f"unresolved column mapping for {file}: {value!r}")
    return value
def file_to_vertices(
        file,
        vertices_col: Union[list[str], pd.DataFrame, None] = None,
        vertices_instance_label_col: Union[str, pd.Series] = "",
        input_dim: int = 3,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Read one localization file into ``(coords, instance_labels)``.

    Both column arguments accept a plain name or a per-file mapping; see
    :func:`_for_file`. An empty label name means "this file is unlabelled" and yields
    zeros, which is the inference path.

    A **named but absent** column raises. It used to fall back to zeros via
    ``DataFrame.get(col, default)`` -- silently, per file. That turned a one-word typo in
    a sample-mapping CSV (``labels`` where the files hold ``cluster_id``) into 98% of a
    2076-sample dataset arriving fully unlabelled, with no error and no warning: training
    ran, every prompt targeted the empty mask, and the only symptom was a metric pinned at
    zero.
    """
    vertices_col = _for_file(vertices_col, file)
    # No label entry for this file in the mapping -> unlabelled, same as an empty name.
    wanted = _for_file(vertices_instance_label_col, file, allow_missing=True) or ""

    dataframe = read_localization_file(file)
    localizations: np.ndarray = get_vertices(dataframe, vertices_col, input_dim)

    assert localizations.size > 1, f"The numpy array is empty: {localizations.size}"

    if not isinstance(wanted, str):
        raise KeyError(f"unresolved instance-label column for {file}: {wanted!r}")
    if wanted and wanted not in dataframe.columns:
        # Named column absent in THIS file: fall back to the candidate names, so a
        # heterogeneous dataset (`labels` here, `cluster_id` there) still loads its labels.
        fallback = next((c for c in LABEL_COLUMN_CANDIDATES if c in dataframe.columns), None)
        if fallback is None:
            raise KeyError(
                f"instance-label column {wanted!r} not found in {file}; tried "
                f"{LABEL_COLUMN_CANDIDATES}; available columns: {list(dataframe.columns)}. "
                "Fix the sample-mapping CSV's 'standardizedLabels' entry, or leave it "
                "empty to read this file as unlabelled."
            )
        wanted = fallback

    # Integer zeros, so an unlabelled sample cannot make a batch mix float and int.
    column = (
        dataframe[wanted] if wanted else pd.Series(np.zeros(len(dataframe), dtype=int))
    )

    frame : np.ndarray = get_frames(dataframe)
    photons : np.ndarray = get_photons(dataframe)

    return localizations, np.atleast_1d(column.to_numpy().squeeze()), frame, photons


def collate_localizations(
    batch: list[Vertices],
    *,
    grid_size: Optional[float],
    forward_transform,
    batch_collate_fn: Optional[Callable] = None,
):
    assert grid_size is not None, "grid_size must always be float between 0 - 1"

    points: list = []
    for localizations in batch:  # type: Vertices
        xy = localizations.data.vertices  # (N, 3) float32
        label = localizations.data.label

        coord = pad_to_3d(xy)

        extras = {k: getattr(localizations, k) for k in getattr(localizations, "EXTRA", ())}

        # Label goes through the transform so SphereCrop crops it with coord; the
        # forward transform's Collect must list LABEL_KEY in its keys to keep it.
        # Store it as a tensor (not raw numpy) to preserve the "all tensors now"
        # invariant: the vendored sonata.data.collate_fn cats tensors per-key (variable
        # length OK) but routes numpy arrays through default_collate -> torch.stack,
        # which fails for K>1 objects with different point counts. Mirrors upstream
        # sonata, where ToTensor converts every per-point field before collation.
        point = {
            "coord": coord,
            "grid_size": grid_size,
            MOBIE_LABEL_KEY: (
                label if torch.is_tensor(label) else torch.as_tensor(np.asarray(label))
            ),
            **extras,
        }
        # Frame index as an optional (N, 1) feature channel, per-sample scaled to [0, 1]
        # like NormalizeCoord scales coord; raw frame numbers (up to ~1e5) would swamp the
        # unit-sphere coords. Absent from the file -> no "frame" key (Collect names it).
        if localizations.frame is not None:
            frame = np.asarray(localizations.frame, dtype=np.float32).reshape(-1, 1)  # (N, 1)
            frame = frame - np.nanmin(frame, initial=np.inf)                          # NaN-safe min
            frame = frame / max(float(np.nanmax(frame, initial=0.0)), 1.0)
            point["frame"] = np.nan_to_num(frame, nan=0.0)  # a NaN frame must not poison the cloud

        point = forward_transform(point)  # per-sample; all tensors now
        point["filePath"] = localizations.filePath
        points.append(point)

    collate = batch_collate_fn if batch_collate_fn is not None else default_collate
    collated = collate(points)

    return collated


@dataclass
class LocalizationCollator:
    """Picklable collate callable — carries only transforms, never a model.

    Safe to ship to DataLoader workers (spawn/fork): it holds the forward-transform
    pipeline and the batch-collate function (both picklable module-level objects), and
    delegates to :func:`collate_localizations`. Return one of these from a model's
    ``build_collate_fn`` to customise collation without capturing ``self``.
    """

    grid_size: float
    forward_transform: Any
    batch_collate_fn: Optional[Callable] = None

    def __call__(self, batch):
        return collate_localizations(
            batch,
            grid_size=self.grid_size,
            forward_transform=self.forward_transform,
            batch_collate_fn=self.batch_collate_fn,
        )


def pad_to_3d(xy: np.ndarray) -> np.ndarray:
    """Pad an (N, d) coordinate array to (N, 3) with zeros for d < 3."""
    N, d = xy.shape
    if d >= 3:
        return xy.astype(np.float32)
    #print(f"We are not in 3D as dim {d}")
    return np.concatenate(
        [xy.astype(np.float32), np.zeros((N, 3 - d), dtype=np.float32)], axis=1
    )
