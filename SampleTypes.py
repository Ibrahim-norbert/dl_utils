from typing import NamedTuple, Any, Tuple, List, Optional, Literal, Union
import numpy as np


class TrainingSampleLM(NamedTuple):
    features: np.ndarray
    positional_embedding: np.ndarray
    feature_mask: np.ndarray
    rand_ids: np.ndarray
    label: int
    map_ids: np.ndarray


class Data(NamedTuple):
    vertices: np.ndarray[Any, np.dtype[np.float32]]
    label: np.ndarray[Any, np.dtype[np.int32]]


class Localizations(NamedTuple):
    """Container for localization data with vertices, labels, and file paths."""

    data: Data
    filePath: str
