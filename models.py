from __future__ import annotations

import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
from torch.utils.data.dataloader import default_collate

from .constants import LOSS_KEY
from .util import save_model as _save_model, load_model as _load_model, OneCycleLR, CosineScheduler
import re
import torch
from torch.optim.lr_scheduler import OneCycleLR

logger = logging.getLogger(__name__)

# Module-path pieces that mark a "layer" in common architectures:
#   PTv3/Sonata: enc.enc2.block1   ViT: blocks.3   Swin/ConvNeXt: stages.0.blocks.2
#   ResNet: layer3   HF BERT: encoder.layer.5   GPT-2: h.7   CLIP: resblocks.4
# Numbered submodules *inside* a block (fc1, norm1, conv1) are not matched.
_BLOCK_RE = re.compile(
    r"(?:^|\.)(?:enc|dec|stages?|blocks?|layers?|levels?|resblocks|h)\.?\d+(?=\.)"
)


class BaseModelClass(pl.LightningModule, metaclass=ABCMeta):
    """Masked Autoencoder with VisionTransformer backbone"""

    # SHould
    _LAYER_DECAY_SCOPE = "backbone."

    def __init__(
        self,
        backbone : Optional[Union[torch.nn, pl.LightningModule]],
        version_name: str,
        accelerator: str,
        lr=0.001,
        batch_size=1,
        input_dim=256,
        layer_decay=0.2,
        lr_final_div_factor=None,
        warmup_ratio=None,
        lr_div_factor=None,
        maxweight_decay=None,
        minweight_decay=None,
        no_weight_decay_keywords=None,
        **kwargs
    ):
        # pl.LightningModule.__init__ takes no args; extra model-config kwargs are
        # captured below by save_hyperparameters(), not forwarded to the base.
        super().__init__()

        self.input_dim = input_dim
        self.layer_decay = layer_decay
        self.backbone = backbone
        self.lr_final_div_factor = lr_final_div_factor
        self.warmup_ratio = warmup_ratio
        self.weight_decay_scheduler: Optional[CosineScheduler] = None
        self.lr_div_factor = lr_div_factor
        self.maxweight_decay = maxweight_decay
        self.lr = lr
        self.minweight_decay = minweight_decay
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.version_name = version_name
        self.no_weight_decay_keywords = tuple(no_weight_decay_keywords or ())
        # logger=False: the trainer logs its full config (incl. modelConfig) as the
        # run's hparams; checkpoints still store these.
        self.save_hyperparameters(logger=False)


        torch.set_float32_matmul_precision("medium")

        self.initialize_weights()

        # Per-batch validation outputs accumulated across one validation epoch.
        # validation_step appends a dict here; on_validation_epoch_end reduces it.
        self.validation_step_outputs: list[dict] = []


    def log(self, *args, **kwargs):
        super().log(*args, **kwargs)

    @staticmethod
    def collate_fn(batch, *args, **kwargs):
        batch = [x for x in batch if x is not None]
        if not batch:
            raise ValueError("collate_fn received a batch of all-None items — check __getitem__ for errors")
        return default_collate(batch, *args, **kwargs)

    # ------------------------------------------------------------------
    # Validation contract
    # ------------------------------------------------------------------
    # The only hard contract is the *type* validation_step returns/accumulates:
    # a per-batch dict whose (all-optional) keys are
    #   LOSS_KEY, EMBED_DICT_EMBED, "coord", LABEL_KEY, "metrics dict".
    # on_validation_epoch_end is deliberately NOT a fixed pipeline — subclasses
    # may override it entirely, extend it via super(), or just call
    # reduce_validation_outputs() to fold the per-batch dicts into one. This keeps
    # each model free to define its own epoch-end while sharing the buffer and the
    # reduction utility.

    def validation_step(self, batch, batch_idx) -> dict:
        """Default validation step: run shared_step, log, accumulate, return.

        Subclasses typically override ``shared_step`` (or this method) so the
        returned dict carries whatever the model's ``on_validation_epoch_end``
        needs (e.g. ``EMBED_DICT_EMBED`` / ``coord`` / ``LABEL_KEY``).
        """
        output: dict = self.shared_step(batch)

        if output.get(LOSS_KEY) is not None:
            self.log(
                "Validation LOSS", output[LOSS_KEY],
                on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=1,
            )
        for k, v in (output.get("metrics dict") or {}).items():
            self.log(
                f"Validation {k}", v,
                on_step=True, on_epoch=True, logger=True, batch_size=1,
            )

        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs.clear()

    def reduce_validation_outputs(self, outputs: Union[list, None] = None) -> dict:
        """Fold the per-batch validation dicts into a single dict.

        Tensors / ndarrays under a shared key are concatenated along axis 0;
        scalar entries (and the values inside ``"metrics dict"``) are averaged.
        Provided as an opt-in utility so a subclass can build its own
        ``on_validation_epoch_end`` without being forced through ``super()``.
        """
        outputs = self.validation_step_outputs if outputs is None else outputs
        if not outputs:
            return {}

        keys = set().union(*(o.keys() for o in outputs))
        reduced: dict = {}
        for key in keys:
            values = [o[key] for o in outputs if o.get(key) is not None]
            if not values:
                continue
            first = values[0]
            if isinstance(first, torch.Tensor):
                # Scalar (0-dim / single-element) tensors are per-batch metrics:
                # average them. Multi-row tensors are per-point: concatenate.
                if first.ndim == 0 or first.numel() == 1:
                    reduced[key] = torch.stack([v.detach().cpu().reshape(()) for v in values]).mean()
                else:
                    reduced[key] = torch.cat([v.detach().cpu() for v in values], dim=0)
            elif isinstance(first, np.ndarray):
                if first.ndim == 0 or first.size == 1:
                    reduced[key] = float(np.mean([np.asarray(v).reshape(()) for v in values]))
                else:
                    reduced[key] = np.concatenate(values, axis=0)
            elif isinstance(first, dict):  # e.g. "metrics dict": mean each scalar
                sub_keys = set().union(*(v.keys() for v in values))
                reduced[key] = {
                    sk: float(np.mean([v[sk] for v in values if sk in v]))
                    for sk in sub_keys
                }
            elif isinstance(first, (int, float)):
                reduced[key] = float(np.mean(values))
            else:
                reduced[key] = values  # leave heterogeneous payloads as a list
        return reduced

    def on_validation_epoch_end(self) -> dict:
        """Default epoch-end: reduce the accumulated outputs and clear the buffer.

        Returns the reduced dict so callers/overrides can reuse it. Subclasses
        that need bespoke logging override this method (and may call
        ``reduce_validation_outputs()`` themselves).
        """
        reduced = self.reduce_validation_outputs()
        self.validation_step_outputs.clear()
        return reduced

    def get_device(self):
        return next(self.parameters()).device

    @staticmethod
    def getConfig(path):
        with open(path, "r") as f:
            hparams = yaml.safe_load(f)
        return hparams

    def tensor2Numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            # .detach().cpu() is a no-op when already detached / on CPU, so guard-free.
            return tensor.detach().cpu().squeeze().numpy()
        return tensor

    def save_model(self, args, epoch, model, model_without_ddp, optimizer, loss_scaler, wb_run):
        _save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, wb_run)

    def load_model(self, args, model_without_ddp, optimizer, loss_scaler):
        _load_model(args, model_without_ddp, optimizer, loss_scaler)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_init_params(self):
        """Return all __init__ parameter names and their current values."""
        import inspect
        sig = inspect.signature(self.__class__.__init__)
        params = {}
        for name in sig.parameters:
            if name == "self":
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
            elif name in self.hparams:
                params[name] = self.hparams[name]
        return params

    def compute_model_size(self):
        total_size_bytes = 0
        for param in self.parameters():
            total_size_bytes += param.nelement() * param.element_size()
        for buffer in self.buffers():
            total_size_bytes += buffer.nelement() * buffer.element_size()
        total_size_gb = total_size_bytes / (1024**3)
        logger.info("Model memory size: %.4f GB", total_size_gb)

    def compute_batch_memory_usage(self, batch_shape: tuple[int, ...]) -> float:
        """GB of one training pass on a dummy batch of shape ``batch_shape``.

        Lightning's ``ModelSummary`` runs ``forward`` on ``example_input_array`` (a zero
        batch of that shape) and records the output size of every layer. A training pass
        holds the parameters (``model_size``), their gradients (one per trainable
        parameter) and the output of every leaf module, kept for backward. Optimizer state
        and allocator overhead are not counted, so it is a lower bound. ``forward`` must
        take one tensor.
        """
        from pytorch_lightning.utilities.model_summary import ModelSummary

        previous = self.example_input_array
        self.example_input_array = torch.zeros(batch_shape, device=self.device)
        try:
            summary = ModelSummary(self, max_depth=-1)
        finally:
            self.example_input_array = previous

        # leaf outputs only (parents re-report their children's); "?" = size unknown
        leaves = {name for name, module in self.named_modules() if next(module.children(), None) is None}
        outputs = [
            size for name, size in zip(summary.layer_names, summary.out_sizes)
            if name in leaves and size != "?"
        ]
        n_activations = sum(
            sum(map(np.prod, size)) if size and isinstance(size[0], list) else np.prod(size)
            for size in outputs
        )
        bytes_per_value = next(self.parameters()).element_size()
        total_gb = (
            summary.model_size * 1e6  # MB of parameters
            + (summary.trainable_parameters + n_activations) * bytes_per_value
        ) / 1e9
        logger.info("One training pass on a %s batch: %.4f GB", tuple(batch_shape), total_gb)
        return total_gb

    @staticmethod
    def add_model_specific_args(parent_parser):

        parent_parser.add_argument(
            "--version_name", default="default_name_experiment", type=str
        )

        # layer_decay: 0.9
        # lr: 0.004
        # lr_div_factor: 10.0
        # lr_final_div_factor: 1000.0
        # maxweight_decay: 0.2
        # minweight_decay: 0.04
        # warmup_ratio: 0.05
        # val_analysis_label_col: labels
        # val_analysis_vertices_col:
        # - x
        # - y
        # val_num_prompts: 16
        # val_probe: E:\\Project
        # 1\\data\\Collection\\RealData - Florian - 30.10
        # .2025\\npc_jonas_labelled
        # crop\\patched\\annotations\\specifiedCrop\\npc_jonaspatch2_specialCrop.csv

        return parent_parser

    @staticmethod
    def _block_decay_entries(scope: str, depths, lr: float, decay: float) -> list[dict]:
        """``{scope, keyword, lr}`` per block, deepest at ``lr``.

        The keyword ``enc{e}.block{b}.`` is matched as a *substring* of the full
        parameter name (the backbone's blocks sit at ``backbone._model.enc.``), with the
        leading scope carried separately so the two stacks — which share the naming —
        cannot capture each other's parameters. The exponent counts blocks backwards
        from the last one: the deepest block gets ``lr * decay**0`` and the first
        gets the most-decayed LR, the standard "later layers move more" schedule from
        ``SMLMSonata.configure_optimizers`` / Pointcept.
        """
        total = sum(depths)
        return [
            dict(
                scope=scope,
                keyword=f"enc{e}.block{b}.",
                lr=lr * decay ** (total - sum(depths[:e]) - b - 1),
            )
            for e in range(len(depths))
            for b in range(depths[e])
        ]






    def layer_decay_param_groups(self,
                                 model : torch.nn.Module,
                                 base_lr,
                                 weight_decay,              # = min weight decay
                                 layer_decay,
                                 max_weight_decay=None,     # new
                                 block_regex=_BLOCK_RE,
                                 no_weight_decay_keys=(),
                                 verbose=True,

    ):
        """Model-agnostic layer-wise LR decay (MAE/BEiT style).

        - Each unique deepest block prefix (e.g. 'enc.enc2.block1') becomes one
          layer, numbered in module registration order (normally the forward order).
        - Unmatched params seen before the first block (embeddings/stem) -> layer 0.
        - Unmatched params seen after it (final norm, heads)            -> top layer.
        - lr = base_lr * layer_decay ** (num_layers - layer_id)
        - Biases and 1-D params (norms) get no weight decay.
        """
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        skip = set(no_weight_decay_keys)
        if hasattr(model, "no_weight_decay"):  # timm convention
            skip |= set(model.no_weight_decay())

        # 1) deepest block-like prefix per param, ordered by first appearance
        prefix_of, layer_of_prefix = {}, {}
        for n, _ in named:
            matches = list(block_regex.finditer(n))
            if matches:
                pre = n[: matches[-1].end()]
                prefix_of[n] = pre
                layer_of_prefix.setdefault(pre, len(layer_of_prefix) + 1)
        num_layers = len(layer_of_prefix) + 1  # +1 = head bucket

        # 2) bucket params by (layer_id, apply_wd)
        groups, seen_block = {}, False
        for n, p in named:
            if n in prefix_of:
                lid, seen_block = layer_of_prefix[prefix_of[n]], True
            else:
                lid = num_layers if seen_block else 0
            no_wd = p.ndim <= 1 or n.endswith(".bias") or any(k in n for k in skip)
            key = (lid, no_wd)
            if key not in groups:
                scale = layer_decay ** (num_layers - lid)
                wd_min = 0.0 if no_wd else weight_decay
                wd_max = 0.0 if no_wd else (
                    weight_decay if max_weight_decay is None else max_weight_decay
                )
                groups[key] = {
                    "params": [],
                    "lr": base_lr * scale,
                    "lr_scale": scale,
                    "weight_decay": wd_min,
                    "minweight_decay": wd_min,
                    "maxweight_decay": wd_max,
                    "layer_id": lid,
                }

            groups[key]["params"].append(p)

        out = [groups[k] for k in sorted(groups)]
        if verbose:
            inv = {v: k for k, v in layer_of_prefix.items()}
            for g in out:
                name = {0: "<embed/stem>", num_layers: "<head/norm>"}.get(
                    g["layer_id"], inv.get(g["layer_id"])
                )
                print(f"layer {g['layer_id']:>3} {name:<28} lr_scale={g['lr_scale']:.4f} "
                      f"wd={g['weight_decay']:<6} n={len(g['params'])}")
        return out

    def _is_no_decay(self, name: str, param: torch.Tensor) -> bool:
        """Whether ``param`` should be exempt from weight decay.

        ``ndim <= 1`` covers every bias, LayerNorm weight and LayerScale vector in one
        rule (as ``dl_utils.lr_decay.param_groups_lrd`` does); the keyword list adds the
        2-D ``nn.Embedding`` tokens, which that rule misses.
        """
        return param.ndim <= 1 or any(
            kw in name for kw in self.no_weight_decay_keywords
        )

    def configure_optimizers(self, layer_decay_kwargs={},
                             optimizer_kwargs={},
                             OneCyclerLR_kwargs={},
                             model_str_temp="student"):
        self._total_steps = int(self.trainer.estimated_stepping_batches)

        # TODO: This class should not contain the "student" attribute but instead backbone
        layer_groups = self.layer_decay_param_groups(
            getattr(self, model_str_temp),
            base_lr=self.lr,
            weight_decay=self.minweight_decay,
            max_weight_decay=self.maxweight_decay,
            layer_decay=self.layer_decay,
            **layer_decay_kwargs
        )

        optimizer = torch.optim.AdamW(layer_groups, lr=self.lr, **optimizer_kwargs)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in layer_groups],
            pct_start=self.warmup_ratio,
            anneal_strategy="cos",
            div_factor=self.lr_div_factor,
            final_div_factor=self.lr_final_div_factor,
            total_steps=self._total_steps,
            **OneCyclerLR_kwargs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_start(self) -> None:
        """Build the cosine weight-decay ramp, resumable mid-run."""
        super().on_train_start()
        total_steps = int(self.trainer.estimated_stepping_batches)
        self.weight_decay_scheduler = CosineScheduler(
            base_value=self.minweight_decay,
            final_value=self.maxweight_decay,
            total_iters=total_steps,
        )
        # Fast-forward on resume so the ramp picks up where the checkpoint left off.
        self.weight_decay_scheduler.iter = self.trainer.global_step
        self.minweight_decay = self.minweight_decay


    def on_before_optimizer_step(self, optimizer):
        """Advance the WD ramp and write it to every non-exempt group.

        Uniform overwrite across the layer-wise LR groups, matching Pointcept's hook —
        all decaying groups share one WD by design (AdamW's decoupled decay still scales
        per group via each group's lr). Groups flagged ``apply_wd_schedule=False`` by
        :meth:`configure_optimizers` (biases, norms, learned tokens) are skipped;
        overwriting them here would silently undo the no-decay split on the first step.
        Groups built by some other code path carry no flag and default to decaying, which
        preserves the previous behaviour.
        """
        if self.weight_decay_scheduler is None:
            return
        # CosineScheduler.step() returns a numpy scalar; cast so self.log and the
        # optimizer state both see a plain Python float.
        self.minweight_decay = float(self.weight_decay_scheduler.step())
        for group in optimizer.param_groups:
            if group.get("apply_wd_schedule", True):
                group["minweight_decay"] = self.minweight_decay

    def _is_global_zero(self):
        return self._trainer is None or self._trainer.is_global_zero

    def build_collate_fn(self, train=False):
        return BaseModelClass.collate_fn  # default_collate, picklable

    def _collate_fn(self, batch, *args, **kwargs):
        return BaseModelClass.collate_fn(batch)

    @abstractmethod
    def shared_step(self, sample, **kwargs):
        """Override so that :meth:`forward` receives a ``point_dict`` directly.

        The parent :meth:`shared_step` expects a :class:`Vertices` object and
        calls ``self.preprocess`` / ``self.transformer``.  Here the collate_fn
        already produces a ready-to-use dict of CPU tensors, so we just move
        them to the model device and call :meth:`forward`.
        """
        pass

    @abstractmethod
    def logAnalysis(self, df, emb_np, titel, save_dir, pred_labels, scalars, classMapping):
        pass

    @abstractmethod
    def _linearProbe(self, emb_np, gt_labels_bin):
        """Phase 4 — fit a :class:`SegHead` linear probe on the single-batch
        embeddings and return the predicted labels ``(N,)``.
        """
        pass

    @abstractmethod
    def _F1_Accuracy(self, pred_labels: torch.Tensor, gt_labels_bin: np.ndarray,
                     tag : str ="Validation"):
        """Phase 5 — compute metrics, log F1, and run the full visual analysis."""
        pass

    @abstractmethod
    def _validation_log_linear_probe(self, emb, save_dir):
        """Fit + log the SegHead linear probe on the configured npc_jonas crop.

        Loads the fixed probe crop from :attr:`_VAL_PROBE` (falling back to
        :attr:`_VAL_ANALYSIS_CSV`) — settable per instance / from the training YAML
        via ``modelConfig: val_probe`` —
        embeds it with the backbone through the same collate + :meth:`embed_step`
        path used by :meth:`validation_step`, fits a :class:`SegHead` linear probe on
        the per-point embeddings, and logs ``LinearProbe/val/f1`` plus the PCA/UMAP
        feature-space figures. Independent of the training set's validation split.

        Shared entrypoint so both :class:`SMLMTransformer` and the SSL
        :class:`SMLMSonata` log ``LinearProbe/val/f1`` from their respective
        ``on_validation_epoch_end`` hooks. Self-contained + fully guarded, so it is
        safe to call alongside a model's other epoch-end logging.
        """
        pass


    @abstractmethod
    def predict_step(self, sample, batch_idx):
        """Save each batch element's voxel-level embeddings to a separate JSON.

        Overrides :meth:`SMLMSegmentation.predict_step`. When the predict
        ``DataLoader`` runs with ``batch_size > 1`` the collated batch holds
        several objects concatenated along dim 0, with per-object boundaries in
        ``offset``. We partition them back apart with :meth:`Point.batch_split`
        and write **one** ``dataframe_*.json`` per object.

        The saved coordinates/embeddings are at **voxel** (GridSample) resolution:
        we use the pre-inverse-expansion embeddings (``EMBED_DICT_EMBED`` +
        ``"_NonSerialized"``) and ``gridsampled_coord``, both of which are aligned
        row-for-row with the input voxel ``offset``. GT / predicted labels are not
        included here — they only exist at original-point resolution, whose
        per-object boundaries are not recoverable from the collated batch (the
        vendored ``collate_fn`` keeps ``inverse`` per-sample-local).
        """
        pass

    @abstractmethod
    def embed_forward(self, sample):
        """Run the backbone and return per-point embeddings (the embedding path).

        This is the overridable seam for the embedding forward used by validation /
        probes. Split out from :meth:`forward` so a subclass whose ``forward`` is a
        different computation (SSL loss) does not also lose the ability to customise
        embedding extraction.

        The input ``point`` is not mutated; all forward outputs live on the
        returned :class:`Embeddings`.

        When the collate's ``GridSample`` merged points, the ``inverse`` index maps
        voxels back to the original points: backbone features and coords are
        expanded through it so ``data``/``misc["coord"]`` stay row-aligned with
        ``labels`` (which are attached at original-point resolution).
        """
        pass

    @abstractmethod
    def forward(self, sample):
        """Eval/embedding forward. Delegates to the overridable :meth:`embed_forward`.

        Kept as ``forward`` so ``SMLMSegmentation``/``shared_step`` callers are
        unchanged; subclasses whose ``forward`` means something else (e.g.
        :class:`SMLMSonata`'s SSL forward) can still reach the embedding path via
        ``self.embed_forward`` (used by :meth:`embed_step`).
        """
        pass


class PretrainedBackbone(nn.Module, metaclass=ABCMeta):
    """Backbone wrapper: optional HuggingFace checkpoint config + ``backboneConfig`` overrides.

    Subclasses provide the checkpoint loader (:meth:`load_checkpoint`) and the model class
    (:meth:`build_model`); every extra keyword is forwarded verbatim to that model, so an
    unknown key still fails loudly at construction. ``init_model`` is called by the concrete
    subclass once its own attributes are set.
    """

    def __init__(
        self,
        block="Block",
        patchSize=1024,
        repo_id: str = "facebook/sonata",
        name: str = "sonata",
        **kwargs,
    ):
        super().__init__()
        print(f"Init with PatchSize: {patchSize}")
        repo_id, name = (None if v == "None" else v for v in (repo_id, name))

        if (repo_id is None) != (name is None):
            raise ValueError(
                f"[{type(self).__name__}] repo_id={repo_id!r}, name={name!r}: set both to load a "
                "checkpoint, or both to None to train from scratch."
            )

        # Not self.name: __getattr__ falls back to the wrapped model.
        self.repo_id = repo_id
        self.ckpt_name = name
        self.pretrained = repo_id is not None

        self.block = block
        self.kwargs: dict = kwargs
        self.patchSize = patchSize

    @abstractmethod
    def load_checkpoint(self, custom_config: dict) -> dict:
        """``{"config": ..., "state_dict": ...}`` of ``self.ckpt_name`` in ``self.repo_id``."""

    @abstractmethod
    def build_model(self, config: dict) -> nn.Module:
        """The wrapped model, built from the merged config."""

    def custom_config(self) -> dict:
        """Overrides applied on top of the checkpoint config."""
        return dict(
            enable_flash=False,  # reduce patch size if necessary
            block=self.block,
            **self.kwargs,
        )

    def init_model(self):

        if self.repo_id is not None and "Utonia" in self.repo_id:
            self.block = "BlockRoPE"
        else:
            print(f"I hope the {self.block} you chose is appropriate for the model")

        custom_config = self.custom_config()

        # NOTE: Must always have cuda: assert indices.is_cuda, "implicit gemm only support cuda"
        #
        # Serialize the HuggingFace hub download across processes: under a
        # distributed launcher every rank constructs the model *before* any
        # process group exists (no dist.barrier available), so concurrent
        # hf_hub_download calls into the shared ~/.cache/sonata/ckpt would race.
        # filelock ships as a hard dependency of huggingface_hub. Rank 0 (per
        # node) downloads; the rest block, then hit the cache. Single-process:
        # the lock is uncontended and free.
        if self.pretrained:
            from filelock import FileLock

            cache_root = os.path.expanduser("~/.cache")
            os.makedirs(cache_root, exist_ok=True)
            with FileLock(os.path.join(cache_root, "ckpt.download.lock")):
                config = self.load_checkpoint(custom_config)
        else:
            # No checkpoint: the model class fills every other argument with its defaults.
            config = {"config": custom_config}
        print(f"{type(self).__name__}: ")
        for k, v in config["config"].items():
            print(f"k: {k}; v: {v}")

        self._model = self.build_model(config["config"])

        if config.get("state_dict"):
            self._load_pretrained(config["state_dict"])

        self.feat_dim = self._model.feat_dim


    def _load_pretrained(self, state_dict: dict) -> None:
        """Load the pretrained tensors that still fit, and report what was dropped.

        Overriding an architecture key (enc_channels, enc_depths, ...) changes tensor
        shapes, so a strict load raises. Filter by shape and say plainly how much of
        the checkpoint survived -- a scratch run must not look like a fine-tune.
        """
        own = self._model.state_dict()
        keep = {
            k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape
        }
        missing, _ = self._model.load_state_dict(keep, strict=False)
        message = (
            f"[SONATACustom] pretrained tensors loaded: {len(keep)}/{len(state_dict)}"
            f" (dropped: {len(state_dict) - len(keep)},"
            f" randomly initialised: {len(missing)})"
        )
        if keep:
            logger.info(message)
        else:
            message += " -- training from scratch"
            logger.warning(message)
        print(message)

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ already checks _parameters/_buffers/_modules.
        # Fall back to the wrapped model for everything else.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

    def forward(self, *args, **kwargs):

        return self._model(*args, **kwargs)
