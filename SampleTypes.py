from dataclasses import dataclass, replace
from typing import NamedTuple, Any, Optional, ClassVar
import numpy as np
import torch
from huggingface_hub import dataclasses
from constants import LOSS_KEY 

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

def _t_detach(x):
    return x.detach() if isinstance(x, torch.Tensor) else x

def _t_detach_cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x

# Recurse into tensors, nested Embeddings, lists/tuples, and dicts; pass
# everything else (floats, strings, None) through untouched. This is what makes
# prev_iters (a list of Embeddings) actually get processed.
def _apply_recursive(v, fn):
    if isinstance(v, torch.Tensor):  return fn(v)
    if isinstance(v, Embeddings):    return v._apply(fn)
    if isinstance(v, (list, tuple)): return type(v)(_apply_recursive(x, fn) for x in v)
    if isinstance(v, dict):          return {k: _apply_recursive(x, fn) for k, x in v.items()}
    return v

def _map_misc(misc, fn):
    return None if misc is None else {k: fn(v) for k, v in misc.items()}


@dataclass(frozen=True, eq=False)
class EmbeddingsNumpy:
    data: object
    labels: Optional[object] = None
    pred_labels: Optional[object] = None
    misc: Optional[dict] = None








@dataclass(frozen=True, eq=False)
class Embeddings:
    data: torch.Tensor
    labels: Optional[torch.Tensor] = None
    pred_labels: Optional[torch.Tensor] = None
    misc: Optional[dict] = None

    def _apply(self, fn, skip_misc_keys: tuple = ()) -> "Embeddings":
        """Map ``fn`` over every tensor (recursing into nested bundles/lists),
        leaving ``misc`` keys named in ``skip_misc_keys`` exactly as-is."""
        updates = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "misc" and v is not None:
                updates[f.name] = {
                    k: (val if k in skip_misc_keys else _apply_recursive(val, fn))
                    for k, val in v.items()
                }
            else:
                updates[f.name] = _apply_recursive(v, fn)
        return replace(self, **updates)

    def detach(self) -> "Embeddings":
        return self._apply(_t_detach)

    def detach_cpu(self) -> "Embeddings":
        return self._apply(_t_detach_cpu)







@dataclass(frozen=True, eq=False)
class EmbeddingsPrompts(Embeddings):
    """Loss + embeddings bundle returned by SSL ``shared_step`` implementations.

    ``misc['loss']`` is ``None`` when ``shared_step`` runs a loss-free forward
    (e.g. the embedding path used by prediction).
    """
    prev_iters: Optional[list["Embeddings"]] = None

    def detach(self) -> "EmbeddingsPrompts":
        """Detach every tensor from the graph EXCEPT ``misc['loss']``, which stays
        attached so ``backward()`` can still run on it."""
        return self._apply(_t_detach, skip_misc_keys=(LOSS_KEY,))

    def detach_cpu(self) -> "EmbeddingsPrompts":
        """New bundle with every tensor detached and moved to CPU EXCEPT
        ``misc['loss']``, kept attached and on-device for ``backward()``.
        ``prev_iters`` are recursed, so their GPU tensors are freed."""
        return self._apply(_t_detach_cpu, skip_misc_keys=(LOSS_KEY,))


class EmbeddingsTraining(NamedTuple):
    """Loss + embeddings bundle returned by SSL ``shared_step`` implementations.

    ``loss`` is ``None`` when ``shared_step`` runs a loss-free forward (e.g. the
    embedding path used by prediction).
    """

    loss: Optional[torch.Tensor]
    data: Embeddings