from __future__ import annotations

import argparse
from argparse import Namespace
import inspect
import types
from pathlib import Path
import time
import os
import sys
import torch
from pytorch_lightning.utilities import parsing
import pytorch_lightning as pl
import typing
from typing import Any, TYPE_CHECKING
import yaml
from lightning.pytorch import seed_everything
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger as logger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import typing
import dl_utils.datasets as datasets

if TYPE_CHECKING:
    import PreliminaryGNN.dataset


def full_stack() -> str:
    import traceback
    import sys
    from traceback import FrameSummary

    exc: type[BaseException] | None = sys.exc_info()[0]
    # last one would be full_stack()
    stack: list[FrameSummary] = traceback.extract_stack()[:-1]
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr: str = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr

def _get_init_args(
    frame: types.FrameType,
) -> tuple[typing.Optional[typing.Any], dict[str, typing.Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return None, {}
    cls = local_vars["__class__"]
    init_parameters: types.MappingProxyType[str, inspect.Parameter] = inspect.signature(
        cls.__init__
    ).parameters
    self_var, args_var, kwargs_var = parsing.parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature
    local_args: dict[str, Any] = {k: local_vars[k] for k in init_parameters}

    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))

    local_args: dict[str, Any] = {
        k: v for k, v in local_args.items() if k not in exclude_argnames
    }
    self_arg: Any | types.NoneType = local_vars.get(self_var, None)

    return self_arg, local_args


class PeakGPUMemoryCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        if not torch.cuda.is_available():
            return
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
        print(
            f"Epoch {trainer.current_epoch} — "
            f"peak GPU memory: {peak_mb:.1f} MB allocated / {reserved_mb:.1f} MB reserved"
        )
        pl_module.log("gpu_peak_memory_MB", peak_mb, on_epoch=True, logger=True)


class BaseClassTrainerAndPredictor(pl.Trainer):
    def __init__(
        self,
        max_epochs=4,
        ckptPath=None,
        modelName="SMLMSegmentation",
        reproducibility_seed=43,
        num_workers=1,
        fast_dev_run=False,
        dataType: str = "SMLM",
        trainFrac: float = 0.8,
        batch_size=1,
        shuffle=True,
        modelConfig: typing.Union[None, dict] = None,
        datasetConfig: typing.Union[None, dict] = None,
        limit_val_batches=0,
        args={},
    ) -> types.NoneType:

        self.__dict__.update(locals())

        # Saving twice as attribute -> overcome Pylance typing error "Attribute < > is unknown"

        self.modelConfig = modelConfig
        self.datasetConfig = datasetConfig
        # Print entries of modelConfig:
        self.device = self.modelConfig.accelerator

        if self.hparams.fast_dev_run is True or self.hparams.fast_dev_run > 0:
            # Ideally, you shoud not save config. As it causes problems for reusing
            self.device = "cpu"
            self.num_workers = 1

        # TODO: Currently, only using https://lightning.ai/docs/pytorch/stable/common/trainer.html#testing
        # Fix the seed for reproducibility
        seed_everything(self.hparams.reproducibility_seed, workers=True)
        # https://docs.pytorch.org/docs/2.9/notes/randomness.html#cuda-convolution-benchmarking
        # cudnn.benchmark = True
        # torch.use_deterministic_algorithms(True,
        #     warn_only=True)

        print("The following config was parsed:")
        for key, value in self.config.items():
            print(f"{key}: {value}")

        if ckptPath is not None and os.path.exists(ckptPath):
            self.ckpt = BaseClassTrainerAndPredictor.getcheckpointinfoandargs(ckptPath)
            self.epoch = self.ckpt["epoch"]
            max_epochs += self.epoch
            self.epoch_start = max_epochs - self.epoch
            self.epoch_end = max_epochs

        torch.set_float32_matmul_precision('medium')
        
        super().__init__(
            accelerator=self.device,
            fast_dev_run=self.hparams.fast_dev_run,
            deterministic=True,
            max_epochs=max_epochs,
            log_every_n_steps=batch_size * 2,
            limit_val_batches=limit_val_batches,
            **args,
        )

    def save_hyperparameters(self) -> Namespace | types.NoneType:
        # From pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
        current_frame: sys.FrameType | types.NoneType = inspect.currentframe()
        if current_frame:
            frame: sys.FrameType | types.NoneType = current_frame.f_back
            # Walk up to the outermost (leaf) __init__ so that calling
            # save_hyperparameters() from a base-class __init__ always captures
            # the full parameter set of the most-derived subclass.
            while frame and frame.f_back and frame.f_back.f_code.co_name == "__init__":
                frame = frame.f_back
            hparams: dict[str, Any] = _get_init_args(frame=frame)[-1]

            # Merge config file values into hparams (config file overrides defaults)
            config_file = hparams.pop("config_file", None)
            if config_file is not None:
                with open(config_file, "r") as _f:
                    hparams.update(yaml.safe_load(_f) or {})

            # Clean config
            modelConfig = hparams.get("modelConfig", {})
            datasetConfig = hparams.get("datasetConfig", {})

            if isinstance(datasetConfig, argparse.Namespace):
                datasetConfig = vars(datasetConfig)

            if isinstance(modelConfig, argparse.Namespace):
                modelConfig = vars(modelConfig)

            keys: list[Any] = list(datasetConfig.keys()) + list(modelConfig.keys())

            [hparams.pop(k, None) for k in keys]

            self.config = self.__dict__.get("config", {})

            self.config.update(hparams)

            # Check if saved kwargs
            self.kwargs = self.__dict__.get("kwargs", {})
            [self.kwargs.pop(k, None) for k in keys]

            self.__dict__.update(self.config)
            # After Namespace init, modelConfig and DatasetConfig are removed
            self.hparams = Namespace(**self.config)
            self.modelConfig = Namespace(**modelConfig)
            self.datasetConfig = Namespace(**datasetConfig)
            return self.hparams

    def getModel(self):
        # Lazy import to avoid circular dependency with PreliminaryGNN
        from PreliminaryGNN import models
        modelConfig = vars(self.modelConfig)
        modelConfig["save_dir"] = self.save_dir
        return models.__dict__[self.modelName](
            **modelConfig, batch_size=self.hparams.batch_size
        )

    def getValDataloader(self, val_ds: PreliminaryGNN.dataset.SMLMDataset, **kwargs):

        val_ds = datasets.BaseDataset.getDataloader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )
        return val_ds

    def splitDataset(self, dataset: PreliminaryGNN.dataset.SMLMDataset, **kwargs):
        n_total: int = len(dataset)
        n_train = int(self.trainFrac * n_total)
        n_val: int = n_total - n_train

        train_ds, val_ds = random_split(dataset, [n_train, n_val], **kwargs)

        return train_ds, val_ds

    def getTrainDataloader(self, ds: PreliminaryGNN.dataset.SMLMDataset, **kwargs):
        dsdl = datasets.BaseDataset.getDataloader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )
        return dsdl

    def getTrainValDataloader(
        self, dataset: PreliminaryGNN.dataset.SMLMDataset, segmentor
    ):
        train_ds, val_ds = self.splitDataset(dataset=dataset)

        train_ds = self.getTrainDataloader(train_ds, collate_fn=segmentor.collate_fn)

        val_ds = self.getValDataloader(val_ds=val_ds, collate_fn=segmentor.collate_fn)

        return train_ds, val_ds

    @staticmethod
    def getcheckpointinfoandargs(pnt: typing.Union[str, Path]):
        print("Predicting features for", pnt)
        checkpoint_info = torch.load(pnt, map_location="cpu")
        return checkpoint_info

    def getConfig(self):

        path: str = os.path.join(self.save_dir, self.__class__.__name__)

        # Test if each value is acceptable to yaml
        filpath: str = f"{path}.yaml"

        with open(filpath, "r") as f:
            hparams = yaml.safe_load(f)

        return hparams

    def saveConfig(self) -> types.NoneType:
        # Writing the data to a YAML file

        # Save to yaml file
        path: str = os.path.join(self.save_dir, self.__class__.__name__)

        # Any argparse Namespaces in config are converted to dicts
        for key, value in self.config.items():
            if isinstance(value, argparse.Namespace):
                self.config[key] = vars(value)

        # TODO: Read through following for improvement:
        # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html#run-using-a-config-file

        # Test if each value is acceptable to yaml
        filpath: str = f"{path}.yaml"
        print(f"Saving config here: {filpath}")
        with open(filpath, "w") as file:
            yaml.dump(self.config, file)

    def timeStampsave_dir(self) -> types.NoneType:

        # Add time stamp
        timestamp: float = time.time()
        time_struct: time.struct_time = time.localtime(timestamp)
        time_string: str = time.strftime("%Y-%m-%d_%H-%M-%S", time_struct)
        print("Timestamp:", time_string)
        self.time_string: str = time_string

        self.config["time_string"] = self.time_string

        # Time stamped save_dir = training_dir
        self.save_dir: str = os.path.join(self.save_dir, self.time_string)

        os.makedirs(self.save_dir, exist_ok=True)

    def getDataset(self, dataType: str, **kwargs):
        # Lazy import to avoid circular dependency with PreliminaryGNN
        import dataset as dataDS
        return dataDS.__dict__[dataType](**kwargs)

    @torch.no_grad()
    def compute_peak_gpu_memory(self, model, dataloader):
        """Compute peak GPU memory usage for a single forward pass.

        Grabs the first batch from the dataloader, moves it to the model's
        device, runs a forward pass, and reports memory stats.

        Args:
            model: The Lightning module (must be on a CUDA device).
            dataloader: A DataLoader to draw one batch from.

        Returns:
            Peak memory usage in GB, or None if not on CUDA.
        """
        device = next(model.parameters()).device
        if device.type != "cuda":
            print("Skipping peak GPU memory profiling (model not on CUDA)")
            return None

        sample = next(iter(dataloader))
        if isinstance(sample, (list, tuple)):
            sample = [
                s.to(device) if isinstance(s, torch.Tensor) else s for s in sample
            ]
        elif isinstance(sample, dict):
            sample = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
        elif isinstance(sample, torch.Tensor):
            sample = sample.to(device)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        mem_before = torch.cuda.memory_allocated(device)

        model.eval()
        _ = model(sample)

        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)

        peak_gb = peak_mem / (1024**3)
        allocated_gb = mem_before / (1024**3)
        forward_gb = (peak_mem - mem_before) / (1024**3)

        print(f"Peak GPU memory: {peak_gb:.3f} GB")
        print(f"  Baseline (model + state): {allocated_gb:.3f} GB")
        print(f"  Forward pass overhead:    {forward_gb:.3f} GB")

        model.train()
        return peak_gb


class DummyWandbLogger(pl.loggers.Logger):
    """Drop-in replacement for WandbLogger that requires no W&B account or connection.

    Implements the full pytorch_lightning Logger interface but silently discards
    every metric and hyperparameter. Accepts the same constructor kwargs as
    WandbLogger so it can be swapped in without touching call sites.
    """

    def __init__(self, *, name: str | None = None, version: str = "0", save_dir: str = ".", **kwargs):
        super().__init__()
        self._name = name or "dummy"
        self._version = version
        self._save_dir = save_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def experiment(self):
        """Return a no-op object so any attribute access on the experiment is safe."""
        return self

    def __getattr__(self, item: str):
        """Absorb any attribute access (e.g. logger.experiment.log(...))."""
        return lambda *args, **kwargs: None

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    def finalize(self, status: str) -> None:
        pass


class SafeEarlyStopping(EarlyStopping):
    def _validate_condition_metric(self, logs):
        if self.monitor not in logs:
            available = list(logs.keys())
            if available:
                print(
                    f"[EarlyStopping] '{self.monitor}' not found. "
                    f"Falling back to '{available[0]}'"
                )
                self.monitor = available[0]
            else:
                print("[EarlyStopping] No metrics available. Skipping.")
                return False
        return True

class BaseClassTrainer(BaseClassTrainerAndPredictor):
    def __init__(
        self,
        modelName="SMLMSegmentation",
        max_epochs=4,
        ckptPath=None,
        save_dir="",
        reproducibility_seed=43,
        num_workers=1,
        dataType: str = "SMLM",
        trainFrac: float = 0.8,
        batch_size=1,
        shuffle=True,
        fast_dev_run=False,
        ModelCheckpoint_save_top_k=3,
        ModelCheckpoint_monitor="Validation F1",
        ModelCheckpoint_mode="max",
        EarlyStopping_monitor="Validation LOSS",
        EarlyStopping_mode="min",
        EarlyStopping_patience=20,
        datasetConfig: typing.Union[dict, None] = None,
        modelConfig=None,
        wandbProjectName="",
        limit_val_batches=0,
        config_file: typing.Union[str, None] = None,
        **kwargs,
    ) -> types.NoneType:

        self.save_hyperparameters()

        self.kwargs = kwargs

        # Allow subclasses to customise the root save directory before the
        # timestamp is appended.  The result is also written into self.config
        # so it ends up in the saved YAML.
        self.save_dir = self._compute_save_dir()
        self.config["save_dir"] = self.save_dir

        # save_dir needs to be declared here because of the callback classes
        # Need to timestamp in initializer as logger requires save_dir
        self.timeStampsave_dir()

        self.saveConfig()

        # TODO: Ensure LOSS is always specified as log key
        # TODO: Figure out what the key strings mean
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir,
            save_top_k=self.hparams.ModelCheckpoint_save_top_k,
            monitor=self.hparams.ModelCheckpoint_monitor,
            mode=self.hparams.ModelCheckpoint_mode,
            save_last=True,
        )

        wandb_logger = logger(
            save_dir=self.save_dir,
            name=self.hparams.version_name if hasattr(self.hparams, "version_name") else None,
            version='')
        
        early_stopping_callback = SafeEarlyStopping(
            monitor=self.hparams.EarlyStopping_monitor,
            mode=self.hparams.EarlyStopping_mode,
            patience=self.hparams.EarlyStopping_patience,
        )

        # TODO: Hack for now until smarter config parsing
        args = {
            "callbacks": [checkpoint_callback, early_stopping_callback],
            "logger": wandb_logger,
        }

        super().__init__(
            modelName=modelName,
            max_epochs=max_epochs,
            ckptPath=self.hparams.ckptPath,
            reproducibility_seed=self.hparams.reproducibility_seed,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            trainFrac=self.hparams.trainFrac,
            shuffle=self.hparams.shuffle,
            datasetConfig=self.hparams.datasetConfig,
            modelConfig=self.modelConfig,
            dataType=self.hparams.dataType,
            fast_dev_run=self.hparams.fast_dev_run,
            limit_val_batches=limit_val_batches,
            args=args,
        )

    def _compute_save_dir(self) -> str:
        """Hook for subclasses to define the root save directory.

        Called inside __init__ after save_hyperparameters(), so self.modelConfig
        and self.hparams are already available.  The default falls back to
        hparams.save_dir (when BaseClassTrainer is used directly) and then to
        modelConfig.save_dir.  Subclasses override this instead of computing
        save_dir manually before calling super().__init__().
        """
        return getattr(self, "save_dir", "") or getattr(self.modelConfig, "save_dir", "")

    def fit(self, segmentor, train, val, **kwargs):

        self.config = self.getConfig()

        try:
            super().fit(segmentor, train, val, **kwargs)
            self.config["ckptPath"] = self.checkpoint_callback.best_model_path

        except:
            print(f"Exited prematurely with exception: {full_stack()}")
            self.config["ckptPath"] = self.checkpoint_callback.last_model_path

        finally:
            self.saveConfig()