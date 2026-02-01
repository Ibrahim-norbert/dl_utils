from sqlite3.dbapi2 import Timestamp
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from typing import Union, Any
import numpy as np
import torch
import torch.nn as nn
from utils import datasets
import os
import sys
from yamlfix import fix_files
import yaml


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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.__dict__.update(vars(self.hparams))

        self.initialize_weights()

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

        # # TODO: I put this in place to ensure old configs are compatible. Maybe, transfer to trainer baseclass ???

        # modelConfig = hparams.get("modelConfig", {})
        # datasetConfig = hparams.get("datasetConfig", {})

        # keys = list(datasetConfig.keys()) + list(modelConfig.keys())

        # [hparams.pop(k, None) for k in keys]

        return hparams

    def tensor2Numpy(self, tensor: torch.tensor) -> np.ndarray:

        if self.device != "cpu":
            tensor = tensor.cpu()

        return tensor.detach().squeeze().numpy()

    # TODO: Use the two methods below to save model with config to checkpoint so we ahve reduced ambiguity when loading models.
    def save_model(
        self, args, epoch, model, model_without_ddp, optimizer, loss_scaler, wb_run
    ):

        # TODO: Implement this: https://pytorch-lightning.readthedocs.io/en/1.6.5/common/hyperparameters.html
        output_dir = args.output_dir
        epoch_name = str(epoch)
        if loss_scaler is not None:
            checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
            for checkpoint_path in checkpoint_paths:
                to_save = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                }
                if wb_run is not None:
                    to_save["wb_run_id"] = wb_run.id
                    to_save["wb_run_url"] = wb_run.url

                torch.save(to_save, checkpoint_path)
        else:
            client_state = {"epoch": epoch}
            model.save_checkpoint(
                save_dir=args.output_dir,
                tag="checkpoint-%s" % epoch_name,
                client_state=client_state,
            )

    def load_model(self, args, model_without_ddp, optimizer, loss_scaler):
        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            print("Resume checkpoint %s" % args.resume)
            if (
                "optimizer" in checkpoint
                and "epoch" in checkpoint
                and not (hasattr(args, "eval") and args.eval)
            ):
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"] + 1
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                print("With optim & sched!")

    def saveConfig(self):
        # Writing the data to a YAML file

        # Save to yaml file
        path = os.path.join(self.save_dir, self.__class__.__name__)

        # Test if each value is acceptable to yaml
        filpath = f"{path}.yaml"
        with open(filpath, "w") as file:
            yaml.dump(self.config, file)

        fix_files([filpath])

    def initialize_weights(self):
        # initialization

        # # _timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_model_size(self):
        total_size_bytes = 0
        for param in self.parameters():
            total_size_bytes += param.nelement() * param.element_size()
        for buffer in self.buffers():
            total_size_bytes += buffer.nelement() * buffer.element_size()
        total_size_gb = total_size_bytes / (1024**3)  # Convert bytes to gigabytes
        print(f"Model memory size: {total_size_gb} GB")

    def compute_batch_memory_usage(self, sample):
        """
        Compute the approximate memory usage of a single batch during training.

        Parameters:
        - model: The PyTorch model (nn.Module).
        - input_size: Tuple representing the size of the input batch (e.g., (batch_size, channels, height, width)).
        - dtype: Data type of the input tensors (default: torch.float32).

        Returns:
        - Total memory usage in gigabytes (GB).
        """

        # Calculate the size of the input data
        input_memory = sum([x.element_size() * x.nelement() for x in sample])

        # Calculate the size of the model parameters
        param_memory = sum(p.element_size() * p.nelement() for p in self.parameters())

        # Estimate the size of intermediate activations
        # This is a rough estimate; actual usage may vary depending on the model architecture
        activation_memory = (
            input_memory * 2
        )  # Assuming activations are roughly twice the input size

        # Total memory usage
        total_memory = input_memory + param_memory + activation_memory

        print(f"Batch in memory during training: {total_memory / (1024 ** 3)}GB")

    @staticmethod
    def add_model_specific_args(parent_parser):

        # TODO Requires wandb account, setting output directory

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

        # Default are ViT-Base encoder
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
