import logging
import os
from typing import Union, Any

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
from yamlfix import fix_files
from torch.utils.data.dataloader import default_collate
from .constants import LOSS_KEY, EMBED_DICT_EMBED, LABEL_KEY
from .util import save_model as _save_model, load_model as _load_model

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
    
class BaseModelClass(pl.LightningModule):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        save_dir: str,
        version_name: str,
        accelerator: str,
        bnm_clip=1e-2,
        lr_clip=1e-5,
        lr=0.001,
        lr_decay=0.5,
        batch_size=1,
        decay_step=3e5,
        bn_momentum=0.5,
        bnm_decay=0.5,
        weight_decay=0.0,
        space_threshold=0.5,
        **kwargs
    ):
        # pl.LightningModule.__init__ takes no args; extra model-config kwargs are
        # captured below by save_hyperparameters(), not forwarded to the base.
        super().__init__()

        self.save_hyperparameters()
        self.__dict__.update(self.hparams)

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
            "--wandb_log_project",
            default="mae_training_nuclei_texture",
            type=str,
            help="wandb project name",
        )
        parent_parser.add_argument("--stop_logging", action="store_true")
        parent_parser.add_argument(
            "--experiment_name", default="default_name_experiment", type=str
        )

        parent_parser.add_argument(
            "--mask_ratio",
            default=0.33,
            type=float,
            help="Masking ratio (percentage of removed patches).",
        )
        parent_parser.add_argument(
            "--embed_dim",
            default=768,
            type=int,
            help="Dimensionality of the input embeddings",
        )
        parent_parser.add_argument(
            "--encoder_embed_dim",
            default=768,
            type=int,
            help="Number of features in encoder (if None, no projection layer is used)",
        )
        parent_parser.add_argument(
            "--encoder_depth", default=12, type=int, help="Number of transformer blocks"
        )
        parent_parser.add_argument(
            "--encoder_num_heads",
            default=12,
            type=int,
            help="Number of attention heads",
        )
        parent_parser.add_argument(
            "--mae_encoder", action="store_false", help="MAE ViT-B encoder"
        )
        parent_parser.add_argument(
            "--pretrained", default="ViT", type=str, help="ViT pretrained ?"
        )

        return parent_parser
