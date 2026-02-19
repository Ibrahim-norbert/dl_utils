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
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.__dict__.update(vars(self.hparams))

        self.initialize_weights()
        
    def collate_fn(self, **kwargs):
        return default_collate(**kwargs)

    def whatDevice(self):
        return next(self.parameters()).device

    def setsave_dir(self):

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

        if self.device != "cpu":
            tensor = tensor.cpu()

        return tensor.detach().squeeze().numpy()

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
