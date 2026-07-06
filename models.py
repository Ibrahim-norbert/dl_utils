import os
from typing import Union, Any

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
from yamlfix import fix_files
from torch.utils.data.dataloader import default_collate
from .util import save_model as _save_model, load_model as _load_model

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
    
        #_install_print_tee(save_dir)

    def log(self, *args, **kwargs):
        # v = args[1]
        # if v is not None:
        #     if isinstance(v, torch.Tensor):
        #         if v.ndim > 1 or v.size(0) > 1:
        #             kwargs["batch_size"] = v.size(0)
        #             args= (args[0], v.mean(dim=0))

        super().log(*args, **kwargs)

    @staticmethod
    def collate_fn(batch, *args, **kwargs):
        batch = [x for x in batch if x is not None]
        if not batch:
            raise ValueError("collate_fn received a batch of all-None items — check __getitem__ for errors")
        return default_collate(batch, *args, **kwargs)

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

            if self.device != "cpu":
                tensor = tensor.cpu()

            return tensor.detach().squeeze().numpy()
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
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        print(f"Model memory size: {total_size_gb} GB")

    def compute_batch_memory_usage(self, sample):
        """
        Compute the approximate memory usage of a single batch during training.

        Parameters:
        - sample: Input batch tensors.

        Returns:
        - Total memory usage in gigabytes (GB).
        """
        input_memory = sum([x.element_size() * x.nelement() for x in sample])
        param_memory = sum(p.element_size() * p.nelement() for p in self.parameters())
        activation_memory = input_memory * 2
        total_memory = input_memory + param_memory + activation_memory
        print(f"Batch in memory during training: {total_memory / (1024 ** 3)}GB")

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
