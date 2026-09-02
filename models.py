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
from yamlfix import fix_files
from torch.utils.data.dataloader import default_collate

from .constants import LOSS_KEY
from .util import save_model as _save_model, load_model as _load_model, OneCycleLR, CosineScheduler

logger = logging.getLogger(__name__)

def _install_print_tee(save_dir: str) -> None:
    """Mirror all print() calls to *log_path*, following the same closure
    pattern as util.print_for_distributed."""
    import builtins
    import functools
    builtin_print = builtins.print
    log_path = os.path.join(save_dir, "print.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)

    @functools.wraps(builtin_print)
    def _tee_print(*args, **kwargs):
        builtin_print(*args, **kwargs)
        builtin_print(*args, **{**kwargs, "file": log_file})

    builtins.print = _tee_print
    
class BaseModelClass(pl.LightningModule, metaclass=ABCMeta):
    """Masked Autoencoder with VisionTransformer backbone"""

    # SHould
    _LAYER_DECAY_SCOPE = "backbone."

    def __init__(
        self,
        save_dir: str,
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
        self.save_dir = save_dir
        self.no_weight_decay_keywords = tuple(no_weight_decay_keywords or ())
        self.save_hyperparameters()


        torch.set_float32_matmul_precision("medium")

        self.initialize_weights()

        # Per-batch validation outputs accumulated across one validation epoch.
        # validation_step appends a dict here; on_validation_epoch_end reduces it.
        self.validation_step_outputs: list[dict] = []

        #_install_print_tee(save_dir)

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

    def set_save_dir(self):

        dirName = (self.__class__.__name__,)

        if isinstance(self.version_name, str) and self.version_name != "":
            dirName = dirName + (f"{self.version_name}",)

        self.save_dir = os.path.join(self.save_dir, *dirName)

        os.makedirs(self.save_dir, exist_ok=True)

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

    def saveConfig(self):
        path = os.path.join(self.save_dir, self.__class__.__name__)
        filpath = f"{path}.yaml"
        with open(filpath, "w") as file:
            yaml.dump(self.config, file)
        fix_files([filpath])

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

    def compute_batch_memory_usage(self, sample):
        """
        Compute the approximate memory usage of a single batch during training.

        Parameters:
        - sample: Input batch tensors.

        Returns:
        - Total memory usage in gigabytes (GB).
        """
        input_memory = sum(x.element_size() * x.nelement() for x in sample)
        param_memory = sum(p.element_size() * p.nelement() for p in self.parameters())
        activation_memory = input_memory * 2
        total_memory = input_memory + param_memory + activation_memory
        logger.info("Batch in memory during training: %.4f GB", total_memory / (1024 ** 3))

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

    def _layer_decay_param_dicts(self) -> list[dict]:
        """Layer-decay entries for the backbone, plus the decoder when enabled.

        The decoder entries only exist when ``decoder_layer_decay`` is set. Without
        them — and with the default ``freeze_encoder=True``, which empties every backbone
        group — ``configure_optimizers`` ends up with a single group and layer-wise decay
        is a no-op for the whole model.
        """
        entries = self._block_decay_entries(
            self._LAYER_DECAY_SCOPE,
            self.backbone.enc_depths,
            self.lr,
            self.layer_decay,
        )

        return entries

    def _is_no_decay(self, name: str, param: torch.Tensor) -> bool:
        """Whether ``param`` should be exempt from weight decay.

        ``ndim <= 1`` covers every bias, LayerNorm weight and LayerScale vector in one
        rule (as ``dl_utils.lr_decay.param_groups_lrd`` does); the keyword list adds the
        2-D ``nn.Embedding`` tokens, which that rule misses.
        """
        return param.ndim <= 1 or any(
            kw in name for kw in self.no_weight_decay_keywords
        )

    def configure_optimizers(self):

        self._total_steps = int(self.trainer.estimated_stepping_batches)

        param_dicts = self._layer_decay_param_dicts()

        # Each LR level owns TWO groups: one that weight-decays and one that does not.
        # `apply_wd_schedule` is a private marker read back by on_before_optimizer_step,
        # which would otherwise re-apply the ramped WD to the exempt group every step and
        # undo the split. torch.optim keeps unknown param_group keys untouched.
        def _pair(lr: float) -> tuple[dict, dict]:
            return (
                {
                    "params": [],
                    "lr": lr,
                    "minweight_decay": self.minweight_decay,
                    "apply_wd_schedule": True,
                },
                {
                    "params": [],
                    "lr": lr,
                    "minweight_decay": 0.0,
                    "apply_wd_schedule": False,
                },
            )

        # Level 0 = default (unmatched): the freshly-initialised SAM stack (feat_proj,
        # prompt_encoder, mask_decoder) plus any non-block params, all at full lr.
        # Levels 1..N = one per param_dict entry.
        levels: list[tuple[dict, dict]] = [_pair(self.lr)]
        levels += [_pair(d["lr"]) for d in param_dicts]

        for n, p in self.named_parameters():
            # freeze_encoder sets requires_grad=False on the whole backbone, so this is
            # also what keeps the frozen encoder out of the optimizer entirely.
            if not p.requires_grad:
                continue
            level = 0
            for i, d in enumerate(param_dicts):
                if n.startswith(str(d["scope"])) and str(d["keyword"]) in n:
                    level = i + 1
                    break
            levels[level][1 if self._is_no_decay(n, p) else 0]["params"].append(p)

        # Flatten and drop empty groups so max_lr length stays consistent (every block
        # group is empty when freeze_encoder is on, as is any no-decay group when
        # no_weight_decay_keywords is empty and the level holds only weight matrices).
        layer_groups = [g for pair in levels for g in pair if g["params"]]
        assert (
            layer_groups
        ), "no trainable parameters — freeze_encoder froze everything?"
        lrs = [g["lr"] for g in layer_groups]
        n_no_decay = sum(
            len(g["params"]) for g in layer_groups if not g["apply_wd_schedule"]
        )
        print(
            f"{type(self).__name__}: {len(layer_groups)} param group(s), "
            f"lr {min(lrs):.3g}..{max(lrs):.3g} (base {self.lr}, "
            f"layer decay {self.layer_decay}, decoder layer decay "
            f"{self.decoder_layer_decay}), {n_no_decay} param(s) exempt from weight "
            f"decay, freeze_encoder={self.freeze_encoder}"
        )

        optimizer = torch.optim.AdamW(
            layer_groups, lr=self.lr, weight_decay=self.minweight_decay
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lrs,
            pct_start=self.warmup_ratio,
            anneal_strategy="cos",
            div_factor=self.lr_div_factor,
            final_div_factor=self.lr_final_div_factor,
            total_steps=self._total_steps,
        )

        # interval="step": OneCycleLR is sized in optimizer steps
        # (estimated_stepping_batches); Lightning's default "epoch" interval would
        # advance it once per epoch and pin the LR near the warmup value.
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
        # Guarded: the hook must stay callable with a bare optimizer and no Trainer
        # (unit tests, manual stepping). self.log needs Lightning's result collection.
        if getattr(self.trainer, "loggers", None):
            self.log(
                "params/minweight_decay",
                self.minweight_decay,
                on_step=True,
                on_epoch=False,
                logger=True,
                batch_size=self.batch_size,
            )

    @abstractmethod
    def _is_global_zero(self):
        """True on rank 0 or when detached from any Trainer (manual/offline use).

        Guards output-only work (figures, file writes) in multi-rank runs. Reads
        ``_trainer`` directly because the ``LightningModule.trainer`` property
        raises ``RuntimeError`` on a detached module.
        """
        pass

    @abstractmethod
    def build_collate_fn(self, train):
        """Build a **picklable** collate callable for a ``DataLoader``.

        This is the collate override seam. Override *this* (not the ``collate_fn`` /
        ``traincollate_fn`` properties): it runs in the *parent* process, so it may
        read any instance state, but it MUST return a picklable callable that does
        **not** capture ``self`` — DataLoader workers (``num_workers > 0``; spawn on
        Windows) pickle the returned object, and capturing ``self`` would drag the
        whole model (backbones, CUDA storages) into every worker. Returning a
        :class:`~smlm_sonata.data.collate.LocalizationCollator` satisfies that.

        ``SMLMTransformer`` has no train-time augmentation, so ``train`` is ignored
        here; :class:`~smlm_sonata.models.sonata.SMLMSonata` overrides this to select
        its multi-view SSL pipeline when ``train=True``.

        Uses the vendored sonata batch collate (not ``default_collate``): it
        concatenates the per-sample point dicts and builds a 1-D ``offset`` by cumsum
        of each sample's point count (per-object boundaries), which is what
        Point/BioimagePoint's ``offset2batch`` expects.
        """
        pass

    @abstractmethod
    def _collate_fn(self, batch, grid_size, forward_transform, batch_collate_fn):
        """Single-process collate convenience — instance method, overridable.

        Not used by DataLoader workers (those get the picklable callable from
        :meth:`build_collate_fn`); it exists for manual, in-process collation such as
        the epoch-end linear probe and :class:`EmbeddingPredictor`. Kept as an
        instance method so a subclass can wrap it via ``super()._collate_fn(...)``.
        Defaults to the instance's ``grid_size`` / ``forwardtransform`` when not given.
        """
        pass

    @abstractmethod
    def shared_step(self, point_dict, kwargs):
        """Override so that :meth:`forward` receives a ``point_dict`` directly.

        The parent :meth:`shared_step` expects a :class:`Vertices` object and
        calls ``self.preprocess`` / ``self.transformer``.  Here the collate_fn
        already produces a ready-to-use dict of CPU tensors, so we just move
        them to the model device and call :meth:`forward`.
        """
        pass

    @abstractmethod
    def embed_step(self, point_dict):
        """Run the *eval* forward and return per-point embeddings + GT.

        Unlike :meth:`shared_step`, this calls :meth:`embed_forward` — the overridable
        embedding seam — so the per-point embedding path is used even for subclasses
        (e.g. ``SMLMSonata``) whose ``self.forward`` is the SSL forward returning a loss
        dict, while still letting a subclass customise embedding extraction. This
        replaces the previous class-bound ``SMLMTransformer.forward(self, ...)`` call and
        the older ``self.forward`` monkey-patch.

        Returns a per-batch dict (the validation contract type) with
        ``EMBED_DICT_EMBED`` (inverse-expanded to the input point count),
        ``"coord"`` and ``LABEL_KEY`` (the input coords / labels), all CPU numpy.
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
    def _log_probe_results(self, df, emb_np, pred_labels, gt_labels_bin, save_dir):
        """Phase 5 — compute metrics, log F1, and run the full visual analysis."""
        pass

    @abstractmethod
    def _VAL_PROBE2Localization(self):
        pass

    @abstractmethod
    def _VAL_PROBE2embdding(self):
        pass

    @abstractmethod
    def _validate_linear_probe(self, emb, save_dir):
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
    def run_validation_epoch(self):
        """Overridable epoch-end template. Subclasses may extend via ``super()``.

        The base pass logs the npc_jonas linear probe; :class:`SMLMSonata` overrides
        this to also log the feature space and the SSL crops.
        """
        pass

    @abstractmethod
    def predict_step(self, point_dict, batch_idx):
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
    def embed_forward(self, point):
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
    def forward(self, point):
        """Eval/embedding forward. Delegates to the overridable :meth:`embed_forward`.

        Kept as ``forward`` so ``SMLMSegmentation``/``shared_step`` callers are
        unchanged; subclasses whose ``forward`` means something else (e.g.
        :class:`SMLMSonata`'s SSL forward) can still reach the embedding path via
        ``self.embed_forward`` (used by :meth:`embed_step`).
        """
        pass
