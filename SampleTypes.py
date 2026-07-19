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

def _to_numpy(value):
    """Tensor -> CPU ndarray; everything else passes through unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


class EmbeddingsNumpy(NamedTuple):
    """Numpy twin of :class:`Embeddings` (see :meth:`Embeddings.numpy`).

    ``labels``/``pred_labels`` are optional because some forwards never see
    labels (SSL contrastive batches, coordinate-only inputs) or predict none.
    """

    data: np.ndarray
    labels: Optional[np.ndarray] = None
    pred_labels: Optional[np.ndarray] = None
    misc: Optional[dict] = None


class Embeddings(NamedTuple):
    """Typed output of top-level model forwards.

    ``labels``/``pred_labels`` are optional because some forwards never see
    labels (SSL contrastive batches, coordinate-only inputs) or predict none.
    Access fields by attribute (``emb.data``), never by string key: this is a
    NamedTuple, so ``emb[...]`` only takes integer indices.
    """

    data: torch.Tensor
    labels: Optional[torch.Tensor] = None
    pred_labels: Optional[torch.Tensor] = None
    misc: Optional[dict] = None

    def numpy(self) -> EmbeddingsNumpy:
        """Return an :class:`EmbeddingsNumpy` with every tensor moved to CPU numpy.

        Non-tensor values (floats, strings, nested Points) pass through; ``misc``
        is shallow-converted (one level of tensor values).
        """
        return EmbeddingsNumpy(
            data=_to_numpy(self.data),
            labels=_to_numpy(self.labels),
            pred_labels=_to_numpy(self.pred_labels),
            misc=(
                {k: _to_numpy(v) for k, v in self.misc.items()}
                if self.misc is not None
                else None
            ),
        )


class EmbeddingsTraining(NamedTuple):
    """Loss + embeddings bundle returned by SSL ``shared_step`` implementations.

    ``loss`` is ``None`` when ``shared_step`` runs a loss-free forward (e.g. the
    embedding path used by prediction).
    """

    loss: Optional[torch.Tensor]
    data: Embeddings