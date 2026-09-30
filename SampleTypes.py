from dataclasses import dataclass, replace, fields, field
from pathlib import Path
from typing import NamedTuple, Any, Optional, ClassVar
import numpy as np
import torch
from huggingface_hub import dataclasses
from .constants import LOSS_KEY

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


class Volume(NamedTuple):
    """Dense 3-D image sample — ``(C, Z, Y, X)`` voxels and the object's label.

    The volumetric counterpart of :class:`Data`. ``default_collate`` rebuilds a
    NamedTuple field-wise, so a collated ``Volume`` carries ``data`` of shape
    ``(B, C, Z, Y, X)`` and ``label`` of ``(B,)`` without a custom collate.

    ``label`` must be a real int — ``default_collate`` cannot batch ``None`` — and
    ``data`` must be shape-stable across the batch, which the fixed
    ``CNNVolumeDatasetObject.TARGET_SHAPE`` window guarantees.
    """

    data: torch.Tensor

    label: torch.Tensor

class VolumeSIMCLR(NamedTuple):
    """Two augmented views of one object — the sample the SimCLR training step contrasts.

    The two-view counterpart of :class:`Volume`: ``data1`` and ``data2`` are independent
    augmentations of the *same* object's window, drawn per sample by
    ``SimCLR3DModel.traincollate_fn``, so a collated instance carries two
    ``(B, C, Z, Y, X)`` tensors and one ``(B,)`` label.

    One ``label``, not one per view, because both views are the same object: it is that
    object's row position in ``objectMapperDF``, and a copy per view would only invite the
    two to disagree. Validation and prediction never see this type — they consume a plain
    :class:`Volume`, which is what stops the un-augmented path from reaching the
    contrastive loss by accident.

    ``default_collate`` rebuilds a NamedTuple field-wise, so no custom collate is needed
    beyond the per-sample augmentation itself, and the same shape-stability rule as
    :class:`Volume` applies to both view tensors.
    """

    data1: torch.Tensor

    data2: torch.Tensor

    label: torch.Tensor


@dataclass(frozen=True)   # frozen=True keeps it immutable like a NamedTuple
class Vertices:
    data: Data
    filePath: str

    EXTRA = ()

@dataclass(frozen=True)
class SonataVertices(Vertices):
    centroid: np.ndarray
    num_pixels: np.ndarray
    dataIndex: int

    EXTRA = ("centroid", "num_pixels", "dataIndex")


class PromptData(Data):
    pass


class SAMVertices(Vertices):

    promptData: PromptData


class CellGeometry(NamedTuple):
    """A :class:`Vertices` plus an optional per-object feature bundle.

    Mirrors :class:`Vertices` (``data`` + ``BaseDataset.SAMPLE_PATH_COLUMN``) and adds ``feat`` — the
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


# One entry per SCENE, not per row — indexed, never split. Mirrors
# BioimagePoint._PER_SCENE (PreliminaryGNN/module/pointTV3/structure.py:280) and
# sidesteps the shape-sniff ambiguity when B == n_total.
_PER_SCENE = ("filePath", "dataIndex")


@dataclass(frozen=True, eq=False)
class EmbeddingsNumpy:
    data: np.ndarray
    labels: Optional[np.ndarray] = None
    pred_labels: Optional[np.ndarray] = None
    misc: Optional[dict] = None

    def batch_split_points(self, offset) -> list["EmbeddingsNumpy"]:
        """Split the batched rows into one bundle per scene, by ``offset``.

        ``offset`` is the cumulative per-scene row count (the collate's ``offset``).
        Per-row arrays (``data``, ``labels``, ``misc["coord"]``, …) are sliced;
        ``_PER_SCENE`` keys are indexed one entry per scene; everything else is
        shared by reference.
        """
        counts = np.diff(np.asarray(offset, dtype=np.int64).ravel(), prepend=0)
        n_total = self.data.shape[0]
        assert counts.sum() == n_total, (
            f"offset totals {counts.sum()} rows but data has {n_total}"
        )
        bounds = np.cumsum(counts)[:-1]

        def parts(v):
            if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n_total:
                return np.split(v, bounds)
            return [v] * len(counts)

        misc = self.misc or {}
        per_scene = {k: misc[k] for k in _PER_SCENE if k in misc}
        split = {f.name: parts(getattr(self, f.name))
                 for f in dataclasses.fields(self) if f.name != "misc"}
        split_misc = {k: parts(v) for k, v in misc.items() if k not in per_scene}

        return [
            replace(self,
                    **{k: p[i] for k, p in split.items()},
                    misc={**{k: p[i] for k, p in split_misc.items()},
                          **{k: v[i] for k, v in per_scene.items()}})
            for i in range(len(counts))
        ]










@dataclass(frozen=False, eq=False)
class Embeddings:
    data: torch.Tensor
    labels: Optional[torch.Tensor] = None
    pred_labels: Optional[torch.Tensor] = None
    misc: Optional[dict] = None

    def _mapped(self, fn, skip_misc_keys: tuple = ()) -> dict:
        """Field->value dict with ``fn`` mapped over every tensor (recursing
        into nested bundles/lists), skipping the named ``misc`` keys."""
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "misc" and v is not None:
                out[f.name] = {k: (val if k in skip_misc_keys else _apply_recursive(val, fn))
                               for k, val in v.items()}
            else:
                out[f.name] = _apply_recursive(v, fn)
        return out

    def _apply(self, fn, skip_misc_keys: tuple = ()) -> "Embeddings":
        return replace(self, **self._mapped(fn, skip_misc_keys))

    def numpy(self) -> "EmbeddingsNumpy":
        return EmbeddingsNumpy(**self._mapped(_to_numpy))

    def detach(self) -> "Embeddings":
        return self._apply(_t_detach)

    def detach_cpu(self) -> "Embeddings":
        return self._apply(_t_detach_cpu)


@dataclass(frozen=False, eq=False)
class EmbeddingsPrompts(Embeddings):
    """
    Must contain prompt_target and prompt_indices
    """
    prompt_target : Optional[torch.Tensor] = None
    prompt_indices : Optional[torch.Tensor] = None

    def detach(self) -> "EmbeddingsPrompts":
        """Detach every tensor from the graph EXCEPT ``misc['loss']``, which stays
        attached so ``backward()`` can still run on it."""
        return self._apply(_t_detach, skip_misc_keys=(LOSS_KEY,))

    def detach_cpu(self) -> "EmbeddingsPrompts":
        """New bundle with every tensor detached and moved to CPU EXCEPT
        ``misc['loss']``, kept attached and on-device for ``backward()``.
        ``prev_iters`` are recursed, so their GPU tensors are freed."""
        return self._apply(_t_detach_cpu, skip_misc_keys=(LOSS_KEY,))

@dataclass(frozen=False, eq=False)
class EmbeddingsPromptsFeat(EmbeddingsPrompts):
    feat: torch.Tensor = None
    grid_size: torch.Tensor = None
    offset: torch.Tensor = None

    @classmethod
    def from_embeddings(
        cls,
        emb: EmbeddingsPrompts,
        *,
        feat: torch.Tensor,
        grid_size: torch.Tensor,
        offset: torch.Tensor,
    ) -> "EmbeddingsPromptsFeat":
        return cls(
            **{f.name: getattr(emb, f.name) for f in fields(EmbeddingsPrompts)},
            feat=feat,
            grid_size=grid_size,
            offset=offset,
        )

    def detach(self) -> "EmbeddingsPromptsFeat":
        """Detach every tensor from the graph EXCEPT ``misc['loss']``, which stays
        attached so ``backward()`` can still run on it."""
        return self._apply(_t_detach, skip_misc_keys=(LOSS_KEY,))

    def detach_cpu(self) -> "EmbeddingsPromptsFeat":
        """New bundle with every tensor detached and moved to CPU EXCEPT
        ``misc['loss']``, kept attached and on-device for ``backward()``.
        ``prev_iters`` are recursed, so their GPU tensors are freed."""
        return self._apply(_t_detach_cpu, skip_misc_keys=(LOSS_KEY,))


@dataclass(frozen=False, eq=False)
class EmbeddingsSAM(Embeddings):
    """Loss + embeddings bundle returned by SSL ``shared_step`` implementations.

    ``misc['loss']`` is ``None`` when ``shared_step`` runs a loss-free forward
    (e.g. the embedding path used by prediction).
    """
    prev_iters: Optional[list["Embeddings"]] = None

    def detach(self) -> "EmbeddingsSAM":
        """Detach every tensor from the graph EXCEPT ``misc['loss']``, which stays
        attached so ``backward()`` can still run on it."""
        return self._apply(_t_detach, skip_misc_keys=(LOSS_KEY,))

    def detach_cpu(self) -> "EmbeddingsSAM":
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

@dataclass(frozen=True, eq=True)
class EmbeddingOrganoidObjectsPC(EmbeddingsNumpy):

    coord : np.ndarray = field(kw_only=True)
    sample_path : Path = field(kw_only=True)


