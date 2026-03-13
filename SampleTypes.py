from typing import NamedTuple, Any, Union
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
