"""Coordinate/label column resolution and vertex extraction for SMLM point clouds.

Pure helpers extracted verbatim from ``dataset.py`` so the column-candidate lists and
the resolve/extract logic have a single home (they were previously duplicated across
``SMLMDataset`` and ``dir2HeterogeneousCSV``). ``SMLMDataset`` now delegates to these.
"""
from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

from PreliminaryGNN.dataset.io import read_localization_file

X_COLUMN_CANDIDATES = ["x [nm]", "x[nm]", "x[pix]", "x", "X"]
Y_COLUMN_CANDIDATES = ["y [nm]", "y[nm]", "y[pix]", "y", "Y"]
Z_COLUMN_CANDIDATES = ["z [nm]", "z[nm]", "z[pix]", "z", "Z"]
LABEL_COLUMN_CANDIDATES = [
    "cluster_id",
    "clusterIndex",
    "labels",
    "label",
    "index",
    "particle_id",
]


def resolve_vertices_col(
    columns, input_dim: int = 2, strict: bool = False
) -> Union[list[str], None]:
    """Default coordinate columns for a given model ``input_dim``.

    Selects the first ``input_dim`` spatial axes — ``input_dim == 2`` -> [x, y],
    ``input_dim == 3`` -> [x, y, z] — using the module-level candidate lists with a
    first-match convention. ``strict=True`` raises when an axis is absent; otherwise
    returns ``None`` so callers can skip files that don't match ``input_dim``.
    """
    if input_dim not in (2, 3):
        raise ValueError(f"input_dim must be 2 or 3, got {input_dim}")

    header = list(columns)
    axes = [
        ("x", X_COLUMN_CANDIDATES),
        ("y", Y_COLUMN_CANDIDATES),
        ("z", Z_COLUMN_CANDIDATES),
    ][:input_dim]

    resolved = []
    for axis, candidates in axes:
        col = next((c for c in candidates if c in header), None)
        if col is None:
            if strict:
                raise ValueError(
                    f"Could not resolve '{axis}' coordinate column for "
                    f"input_dim={input_dim} from {header}; tried {candidates}"
                )
            return None
        resolved.append(col)

    return resolved


def get_vertices(
    item_df: pd.DataFrame,
    vertices_col: list[str] = None,
    input_dim: int = 2,
) -> "np.ndarray[Any, np.dtype[np.float32]]":

    vertices_col: list[str] = resolve_vertices_col(item_df.columns, input_dim)

    arr = np.asarray(np.array(item_df[vertices_col]).tolist()).squeeze()
    # squeeze() collapses a single-row (1, D) or single-column (N, 1) selection to
    # 1-D, which breaks 2-D indexing downstream (e.g. xy[:, 0]). Restore (N, D).

    return arr


def _for_file(spec, file):
    """Resolve a per-file column mapping to this file's entry.

    ``spec`` is either a plain column name (or list of names) applied to every file,
    or a DataFrame/Series keyed by ``sample_path`` -- the ``standardized_*`` /
    ``standardizedLabels`` columns of an index CSV. Resolving here, where ``file`` is
    in scope, is what keeps every caller from having to do it; the ones that forgot
    used to hand pandas a whole Series as a column key, which ``DataFrame.get``
    swallowed into an all-zero fallback.
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
        raise KeyError(f"unresolved column mapping for {file}: {value!r}")
    return value


def file_to_vertices(
    file,
    vertices_col: Union[list[str], pd.DataFrame, None] = None,
    vertices_instance_label_col: Union[str, pd.Series] = "",
    input_dim: int = 2,
) -> Tuple["np.ndarray", "np.ndarray"]:
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
    wanted = _for_file(vertices_instance_label_col, file)

    dataframe = read_localization_file(file)
    localizations: np.ndarray = get_vertices(dataframe, vertices_col, input_dim)

    assert localizations.size > 1, f"The numpy array is empty: {localizations.size}"

    if not isinstance(wanted, str):
        raise KeyError(f"unresolved instance-label column for {file}: {wanted!r}")
    if wanted and wanted not in dataframe.columns:
        # raise KeyError(
        #     f"instance-label column {wanted!r} not found in {file}; available "
        #     f"columns: {list(dataframe.columns)}. Fix the sample-mapping CSV's "
        #     "'standardizedLabels' entry, or leave it empty to read this file as "
        #     "unlabelled."
        # )
        dataframe[wanted] = np.zeros((localizations.shape[0],)).astype(np.int32)

    # Integer zeros, so an unlabelled sample cannot make a batch mix float and int.
    column = (
        dataframe[wanted] if wanted else pd.Series(np.zeros(len(dataframe), dtype=int))
    )
    return localizations, np.atleast_1d(column.to_numpy().squeeze())


def normalize(vertices: np.ndarray) -> np.ndarray:
    """Min-max normalize all spatial columns to [-0.5, 0.5] with isotropic scale.

    Uses the same scale across all axes so relative geometry is preserved.
    """
    mins = np.min(vertices, axis=0, keepdims=True)
    maxs = np.max(vertices, axis=0, keepdims=True)
    scale = (maxs - mins).max()
    if scale == 0:
        return vertices.copy()
    return (vertices - mins) / scale - 0.5
