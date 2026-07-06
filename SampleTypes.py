from typing import NamedTuple, Any, Optional
import numpy as np
import torch


class TrainingSampleLM(NamedTuple):
    features: np.ndarray
    positional_embedding: np.ndarray
    feature_mask: np.ndarray
    rand_ids: np.ndarray
    label: int
    map_ids: np.ndarray


class TrainingSampleLMTensor(NamedTuple):
    """Post-collation version of TrainingSampleLM — numpy arrays become torch.Tensor."""

    features: torch.Tensor
    positional_embedding: torch.Tensor
    feature_mask: torch.Tensor
    rand_ids: torch.Tensor
    label: torch.Tensor
    map_ids: torch.Tensor


class Data(NamedTuple):
    vertices: np.ndarray[Any, np.dtype[np.float32]]
    label: np.ndarray[Any, np.dtype[np.int32]]


class Vertices(NamedTuple):
    """Container for localization data with vertices, labels, and file paths."""

    data: Data
    filePath: str


class PromptData(Data):
    pass


class SAMVertices(Vertices):

    promptData: PromptData


class CellGeometry(NamedTuple):
    """A :class:`Vertices` plus an optional per-object feature bundle.

    Mirrors :class:`Vertices` (``data`` + ``filePath``) and adds ``feat`` — the
    ``(6,)`` normalized sin/cos centre-of-mass embedding built by
    :func:`dl_utils.cell_geometry.cell_geometry_from_row`. When present, the
    SONATA-family collate maps ``feat`` into ``point.feat[:, 3:9]`` (replacing the
    color+normal supplements while keeping ``coord``); when ``None`` the collate
    falls back to the computed ``[coord, color, normal]`` features.
    """

    data: Data
    filePath: str
    feat: Optional[np.ndarray] = None
