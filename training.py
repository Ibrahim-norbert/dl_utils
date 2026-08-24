from __future__ import annotations

import argparse
from abc import ABCMeta, abstractmethod
from argparse import Namespace
import configargparse
import inspect
import types
from pathlib import Path
import time
import os
import sys
import torch
from pytorch_lightning.utilities import parsing
import pytorch_lightning as pl
from typing import Any
import yaml
from pytorch_lightning import seed_everything
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as logger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import typing
import dl_utils.datasets as datasets
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import LearningRateMonitor
import gc
import logging
import shutil
from datetime import timedelta

_console_logger = logging.getLogger(__name__)


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


class AverageMeter:
    """Running average of a scalar (Pointcept's meter, trimmed)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def _log_scalar(experiment: Any, tag: str, value: float, step: int) -> None:
    """Duck-typed scalar dispatch: TensorBoard (`add_scalar`) or WandB (`log`)."""
    if hasattr(experiment, "add_scalar"):
        experiment.add_scalar(tag, value, step)
    elif hasattr(experiment, "log"):
        experiment.log({tag: value}, step=step)


class IterationInfoCallback(pl.Callback):
    """Periodic one-line training status: loss, LR, data/batch time, ETA.

    Port of Pointcept's ``InformationWriter`` + ``IterationTimer`` hooks (no
    Pointcept dependency: its EventStorage/comm machinery is replaced by
    :class:`AverageMeter`, ``trainer.is_global_zero`` and the trainer's own
    logger). Every ``log_interval`` train batches (rank zero only) one INFO
    line is emitted::

        Train [ep/max][step/total] loss 1.2345 (last 1.1000) lr 4.000e-03 data 0.120s batch 0.480s eta 1:23:45

    and, when ``log_scalars`` is set, ``time/data`` / ``time/batch`` scalars go
    to the trainer's logger. Data-time measures the gap between one batch's end
    and the next batch's start (dataloader stall); batch-time the full step.
    The ETA extrapolates the mean batch-time over the remaining optimizer steps
    and is approximate across partial gradient-accumulation windows.

    Timers carry no checkpointed state: on resume they simply refill, and the
    ETA is correct because ``trainer.global_step`` is restored by Lightning.
    """

    def __init__(self, log_interval: int = 10, log_scalars: bool = True) -> None:
        self.log_interval = max(1, int(log_interval))
        self.log_scalars = log_scalars
        self._data_time = AverageMeter()
        self._batch_time = AverageMeter()
        self._loss = AverageMeter()
        self._batch_start: typing.Optional[float] = None
        self._last_batch_end: typing.Optional[float] = None
        self._total_opt_steps: typing.Optional[int] = None
        self._accum = 1

    def on_train_start(self, trainer, pl_module) -> None:
        self._total_opt_steps = int(trainer.estimated_stepping_batches)
        self._accum = max(1, int(trainer.accumulate_grad_batches))
        for meter in (self._data_time, self._batch_time, self._loss):
            meter.reset()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # Per-epoch reset (as Pointcept does) and forget the previous batch end
        # so validation/epoch-boundary time is not counted as data-time.
        self._data_time.reset()
        self._batch_time.reset()
        self._loss.reset()
        self._last_batch_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._last_batch_end is not None:
            self._data_time.update(now - self._last_batch_end)
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._batch_start is not None:
            self._batch_time.update(now - self._batch_start)
        self._last_batch_end = now

        loss = self._extract_loss(outputs, trainer)
        if loss is not None:
            self._loss.update(loss)

        if (batch_idx + 1) % self.log_interval or not trainer.is_global_zero:
            return

        _console_logger.info(self._format_line(trainer))
        if self.log_scalars:
            experiment = getattr(trainer.logger, "experiment", None)
            if experiment is not None:
                _log_scalar(experiment, "time/data", self._data_time.avg, trainer.global_step)
                _log_scalar(experiment, "time/batch", self._batch_time.avg, trainer.global_step)

    @staticmethod
    def _extract_loss(outputs, trainer) -> typing.Optional[float]:
        value = None
        if isinstance(outputs, dict):
            value = outputs.get("loss")
        elif outputs is not None:
            value = outputs
        if value is None:
            value = trainer.callback_metrics.get("Train LOSS")
        try:
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    def _format_line(self, trainer) -> str:
        max_epochs = trainer.max_epochs if trainer.max_epochs is not None else "?"
        total = self._total_opt_steps or 0
        remaining = max(total - trainer.global_step, 0)
        eta = timedelta(seconds=int(self._batch_time.avg * self._accum * remaining))

        lr = float("nan")
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]

        return (
            f"Train [{trainer.current_epoch + 1}/{max_epochs}]"
            f"[{trainer.global_step}/{total}] "
            f"loss {self._loss.avg:.4f} (last {self._loss.val:.4f}) "
            f"lr {lr:.3e} "
            f"data {self._data_time.avg:.3f}s batch {self._batch_time.avg:.3f}s "
            f"eta {eta}"
        )


class GarbageCollectionCallback(pl.Callback):
    """Deterministic garbage collection (port of Pointcept's ``GarbageHandler``).

    Automatic GC triggers at unpredictable points and pauses every process
    independently; disabling it and collecting manually every ``interval``
    batches removes those stalls (per-process, so it runs on every rank).
    Also collects after each validation run (allocation-heavy: figures,
    clustering, probes). Unlike Pointcept, ``teardown`` re-enables automatic
    GC — the training process keeps living after ``fit()`` and ``teardown``
    also fires on exceptions.
    """

    def __init__(
        self,
        interval: int = 150,
        disable_auto: bool = True,
        empty_cache: bool = False,
    ) -> None:
        self.interval = max(1, int(interval))
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self._disabled_here = False

    def on_train_start(self, trainer, pl_module) -> None:
        if self.disable_auto and gc.isenabled():
            gc.disable()
            self._disabled_here = True
            _console_logger.info(
                "Automatic GC disabled; collecting manually every %d batches",
                self.interval,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (batch_idx + 1) % self.interval == 0:
            gc.collect()
            if self.empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def on_validation_end(self, trainer, pl_module) -> None:
        gc.collect()

    def teardown(self, trainer, pl_module, stage) -> None:
        if self._disabled_here:
            gc.collect()
            gc.enable()
            self._disabled_here = False


class BaseClassTrainerAndPredictor(pl.Trainer):
    def __init__(
        self,
        max_epochs=4,
        ckptPath=None,
        modelName="SMLMSegmentation",
        reproducibility_seed=43,
        num_workers=1,
        fast_dev_run=False,
        dataset: str = "SMLM",
        trainFrac: float = 0.8,
        batch_size=1,
        shuffle=True,
        modelConfig: typing.Union[None, dict] = None,
        datasetConfig: typing.Union[None, dict] = None,
        limit_val_batches=0,
        accumulate_grad_batches=10,
        use_fsdp: bool = False,
        devices="auto",
        num_nodes: int = 1,
        precision=None,
        fsdpConfig: typing.Union[None, dict] = None,
        args={},
    ) -> types.NoneType:

        _locals = locals()
        # pl.Trainer exposes precision/num_nodes as read-only properties (data
        # descriptors, which win over instance-dict entries on read), so storing
        # them here would only add dead, confusing entries — they are forwarded
        # explicitly to super().__init__ below instead.
        for _prop in ("precision", "num_nodes", "devices"):
            _locals.pop(_prop, None)
        self.__dict__.update(_locals)

        # Saving twice as attribute -> overcome Pylance typing error "Attribute < > is unknown"

        self.modelConfig = modelConfig
        self.datasetConfig = datasetConfig
        # Print entries of modelConfig:
        self.device = self.modelConfig.accelerator

        if self.hparams.fast_dev_run is True or self.hparams.fast_dev_run > 0:
            # Ideally, you shoud not save config. As it causes problems for reusing
            #self.device = "cpu"
            self.num_workers = 1

        # TODO: Currently, only using https://lightning.ai/docs/pytorch/stable/common/trainer.html#testing
        # Fix the seed for reproducibility
        seed_everything(self.hparams.reproducibility_seed, workers=True)
        # https://docs.pytorch.org/docs/2.9/notes/randomness.html#cuda-convolution-benchmarking
        # cudnn.benchmark = True
        # torch.use_deterministic_algorithms(True,
        #     warn_only=True)

        print("The following config was parsed:")
        for key, value in vars(self.hparams).items():
            print(f"{key}: {value}")

        # if ckptPath is not None and os.path.exists(ckptPath):
        #     self.ckpt = BaseClassTrainerAndPredictor.getcheckpointinfoandargs(ckptPath)
        #     self.epoch = self.ckpt["epoch"]
        #     max_epochs += self.epoch
        #     self.epoch_start = max_epochs - self.epoch
        #     self.epoch_end = max_epochs
        
        # Resolve the boolean FSDP switch into a Lightning strategy. Kept as an
        # overridable hook: this generic base cannot build a model-aware FSDP
        # wrap policy, so subclasses (e.g. SMLMSegmentationTraining) override
        # _resolve_strategy to construct the real FSDPStrategy from the model class.
        strategy = self._resolve_strategy(use_fsdp, fsdpConfig)

        super().__init__(
            accelerator=self.device,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,  # None -> Lightning default ("32-true")
            fast_dev_run=self.hparams.fast_dev_run,
            deterministic="warn",
            max_epochs=max_epochs,
            log_every_n_steps=batch_size,
            limit_val_batches=limit_val_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            # Skip the pre-training sanity-check validation pass: a validation
            # error there would otherwise abort the run before any training
            # metric is flushed, leaving TensorBoard empty.
            num_sanity_val_steps=1,
            **args,
        )

    def _resolve_strategy(
        self,
        use_fsdp: bool = False,
        fsdpConfig: typing.Union[None, dict] = None,
    ):
        """Map the boolean FSDP switch to a Lightning ``strategy`` argument.

        Returns ``"auto"`` (the pl.Trainer default) when ``use_fsdp`` is False.
        The generic base deliberately refuses ``use_fsdp=True``: building an
        FSDPStrategy without a model-aware auto-wrap policy would flatten the
        whole LightningModule into a single FSDP unit (one giant flat parameter),
        which breaks models relying on aligned submodule sharding (e.g. the
        SMLMSonata teacher/student EMA). Trainer subclasses that know their model
        family override this to construct the real strategy — see
        ``SMLMSegmentationTraining._resolve_strategy``.
        """
        if use_fsdp:
            raise NotImplementedError(
                "use_fsdp=True needs a model-aware FSDP wrap policy; use a trainer "
                "subclass that overrides _resolve_strategy (e.g. "
                "SMLMSegmentationTraining)."
            )
        return "auto"


    
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

            # Merge config file values with correct priority:
            #   explicitly-passed args  >  config file  >  signature defaults
            # We detect "was this param explicitly passed?" by comparing its
            # current value against the __init__ signature default.  If the
            # value equals the default (or the param has no default), the
            # config file is allowed to override it.
            config_file = hparams.pop("config_file", None)
            if config_file is not None and os.path.isfile(str(config_file)):
                _cls = frame.f_locals.get("__class__") if frame else None
                _sig_defaults: dict[str, Any] = {}
                if _cls is not None:
                    _sig_defaults = {
                        name: param.default
                        for name, param in inspect.signature(_cls.__init__).parameters.items()
                        if param.default is not inspect.Parameter.empty
                    }
                with open(config_file, "r") as _f:
                    yaml_data: dict[str, Any] = yaml.safe_load(_f) or {}
                _sentinel = object()
                for k, v in yaml_data.items():
                    if k not in hparams:
                        # Key not declared in __init__ — add it from config.
                        hparams[k] = v
                    elif hparams[k] == _sig_defaults.get(k, _sentinel):
                        # Still at its default — let config file override.
                        hparams[k] = v
                    elif isinstance(v, dict) and isinstance(hparams.get(k), dict):
                        # Nested dict (e.g. modelConfig/datasetConfig):
                        # config file provides base keys; explicit dict wins per-key.
                        hparams[k] = {**v, **hparams[k]}

            # Clean config
            modelConfig = hparams.get("modelConfig", {})
            datasetConfig = hparams.get("datasetConfig", {})

            if isinstance(datasetConfig, argparse.Namespace):
                datasetConfig = vars(datasetConfig)

            if isinstance(modelConfig, argparse.Namespace):
                modelConfig = vars(modelConfig)

            keys: list[Any] = list(datasetConfig.keys()) + list(modelConfig.keys())

            [hparams.pop(k, None) for k in keys]

            existing = vars(self.hparams) if hasattr(self, "hparams") and isinstance(self.hparams, Namespace) else {}
            merged = {**existing, **hparams}

            # Check if saved kwargs
            self.kwargs = self.__dict__.get("kwargs", {})
            [self.kwargs.pop(k, None) for k in keys]

            self.__dict__.update(merged)
            # After Namespace init, modelConfig and DatasetConfig are removed
            self.hparams = Namespace(**merged)
            self.modelConfig = Namespace(**modelConfig)
            self.datasetConfig = Namespace(**datasetConfig)
            return self.hparams


    def loadWeights(self, ckptPath: str, module: pl.LightningModule) -> pl.LightningModule:
        checkpoint = torch.load(ckptPath, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Drop keys whose shapes don't match the module before loading, so an
        # architecture change in the tokenizer/decoder head doesn't abort the load
        # (load_state_dict raises on a shape mismatch even with strict=False). These
        # are reported as `shape_skipped`; the backbone weights still load.
        module_state = module.state_dict()
        shape_skipped = [
            k for k, v in state_dict.items()
            if k in module_state and hasattr(v, "shape") and v.shape != module_state[k].shape
        ]
        if shape_skipped:
            state_dict = {k: v for k, v in state_dict.items() if k not in shape_skipped}
        missing, unexpected = module.load_state_dict(state_dict=state_dict, strict=False)
        print(f"[getModel] Loaded weights from: {ckptPath}")
        if shape_skipped:
            print(f"  Shape-mismatched keys skipped ({len(shape_skipped)}): "
                  f"{shape_skipped[:5]}{'...' if len(shape_skipped) > 5 else ''}")
        if missing:
            print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        return module
    
    def _load_weights_if_specified(self, module: pl.LightningModule) -> pl.LightningModule:
        """Load weights-only from weightsCkptPath if set, leaving optimizer/scheduler state untouched.

        Handles both raw state dicts and Lightning checkpoints (nested under "state_dict").
        Filters out shape-mismatched keys before loading so architecture changes in the
        tokenizer or decoder head don't block the backbone weights from loading.
        Raises FileNotFoundError if the path is set but does not exist.
        """
        weights_path: str = getattr(self.hparams, "weightsCkptPath", "") or ""
        if not weights_path:
            return module
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[getModel] weightsCkptPath not found: {weights_path}")
        return self.loadWeights(weights_path, module)

    def getValDataloader(self, val_ds: datasets.BaseDataset, **kwargs):

        val_ds = datasets.BaseDataset.getDataloader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )
        return val_ds

    def splitDataset(self, dataset: datasets.BaseDataset, **kwargs):
        n_total: int = len(dataset)
        n_train = int(self.trainFrac * n_total)
        n_val: int = n_total - n_train

        train_ds, val_ds = random_split(dataset, [n_train, n_val], **kwargs)

        return train_ds, val_ds

    def getTrainDataloader(self, ds: datasets.BaseDataset, **kwargs):
        dsdl = datasets.BaseDataset.getDataloader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            **kwargs,
        )
        return dsdl

    def getTrainValDataloader(
        self, dataset: datasets.BaseDataset, segmentor
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

        if os.path.exists(self.save_dir):
            # Save to yaml file
            path: str = os.path.join(self.save_dir, self.__class__.__name__)

            # Any argparse Namespaces in hparams are converted to dicts
            for key, value in vars(self.hparams).items():
                if isinstance(value, argparse.Namespace):
                    setattr(self.hparams, key, vars(value))

            # TODO: Read through following for improvement:
            # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html#run-using-a-config-file

            # Test if each value is acceptable to yaml
            filpath: str = f"{path}.yaml"
            print(f"Saving config here: {filpath}")
            with open(filpath, "w") as file:
                yaml.dump(vars(self.hparams), file)

    def timeStampsave_dir(self) -> types.NoneType:

        # Add time stamp
        timestamp: float = time.time()
        time_struct: time.struct_time = time.localtime(timestamp)
        time_string: str = time.strftime("%Y-%m-%d_%H-%M-%S", time_struct)
        print("Timestamp:", time_string)
        self.time_string: str = time_string

        self.hparams.time_string = self.time_string

        # Time stamped save_dir = training_dir
        self.save_dir: str = os.path.join(self.save_dir, self.time_string)

        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, *args, **kwargs):
        

        try:
            super().fit(*args, **kwargs)

        except Exception:
            save_dir = getattr(self, "save_dir", None)
            if save_dir and os.path.isdir(save_dir):
                # Keep the run dir if any checkpoint (.ckpt) or visual output was
                # saved before the crash; checkpoints may sit in a subfolder, so
                # scan recursively. Only an output-less dir is cleaned up.
                keep_exts = (".ckpt", ".pth", ".png", ".svg")
                has_saved_output = any(
                    f.endswith(keep_exts)
                    for _root, _dirs, files in os.walk(save_dir)
                    for f in files
                )
                if not has_saved_output:
                    shutil.rmtree(save_dir)
            raise


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
    


class BaseClassTrainer(BaseClassTrainerAndPredictor, metaclass=ABCMeta):
    _MODEL_METRIC_PREFIXES = {
        "PointClassifier": ("train/", "val/no_promptlabels/"),
        "SAMSerialized": ("train/", "val/no_promptlabels/"),
    }

    def __init__(
        self,
        modelName="SMLMSegmentation",
        max_epochs=200,
        ckptPath=None,
        weightsCkptPath: str = "",
        save_dir="",
        reproducibility_seed=43,
        num_workers=1,
        dataset: str = "SMLMDataset",
        trainFrac: float = 0.9,
        batch_size=1,
        shuffle=True,
        fast_dev_run=1,
        use_fsdp: bool = False,
        devices="auto",
        num_nodes: int = 1,
        precision=None,
        fsdpConfig: typing.Union[dict, str, None] = None,
        profiler: str = "advanced",
        ModelCheckpoint_save_top_k=3,
        ModelCheckpoint_monitor="Train LOSS",
        ModelCheckpoint_mode="min",
        EarlyStopping_monitor="Train LOSS",
        EarlyStopping_mode="min",
        EarlyStopping_patience=50,
        datasetConfig: typing.Union[dict, None] = None,
        modelConfig=None,
        wandbProjectName="",
        limit_val_batches=1.0,
        config_file: typing.Union[str, None] = None,
        accumulate_grad_batches=10,
        gradient_clip_val = 1.,
        **kwargs,
    ) -> types.NoneType:

        self.save_hyperparameters()

        self.kwargs = kwargs

        # Metric-name prefixes each model family actually logs. Only families whose tags
        # diverge from the trainer's defaults need an entry; anything unlisted is skipped
        # rather than guessed at.
        _MODEL_METRIC_PREFIXES = {
            "PointClassifier": ("train/", "val/no_promptlabels/"),
            "SAMSerialized": ("train/", "val/no_promptlabels/"),
        }

        # Allow subclasses to customise the root save directory before versioning.
        # The result is also written into self.hparams so it ends up in the saved YAML.
        root_save_dir = self._compute_save_dir()

        # Checkpoints, TB logs and config all go directly into the specified
        # save_dir. name="" + version="" stop TensorBoardLogger from appending
        # name/version_N, so its log_dir == root_save_dir (re-runs reuse/overwrite
        # this dir; Lightning's ModelCheckpoint manages top-k/last files).
        self.save_dir = root_save_dir
        self.hparams.save_dir = self.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        wandb_logger = logger(save_dir=root_save_dir, name="", version="")

        print(
            f"\n[TensorBoard] logging to: {self.save_dir}\n"
            f"[TensorBoard] view with:  tensorboard --logdir \"{self.save_dir}\"\n",
            flush=True,
        )

        self.saveConfig()

        # Top-k best checkpoints, ranked by the monitored metric.
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir,
            save_top_k=self.hparams.ModelCheckpoint_save_top_k,
            monitor=self.hparams.ModelCheckpoint_monitor,
            mode=self.hparams.ModelCheckpoint_mode,
            save_on_train_epoch_end=True,
            # save_last intentionally omitted: in Lightning 2.6 a monitor-coupled
            # callback only writes last.ckpt on epochs where a new top-k file is
            # saved (see ModelCheckpoint.on_train_epoch_end guard), so it freezes
            # once the metric stops improving. latest_checkpoint_callback below
            # captures the true final-epoch model instead.
        )
        # NB: do not store this as self.checkpoint_callback - pl.Trainer exposes
        # checkpoint_callback as a read-only property (the first ModelCheckpoint in
        # the callbacks list, i.e. this top-k callback), which fit() reads directly.

        # Genuine final-epoch checkpoint. monitor=None routes through
        # _save_none_monitor_checkpoint, which saves every epoch (rolling, keeps
        # the most recent 1) independent of the monitored metric.
        latest_checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir,
            monitor=None,
            save_top_k=1,
            save_on_train_epoch_end=True,
            filename="last-{epoch:03d}-{step}",
        )
        self.latest_checkpoint_callback = latest_checkpoint_callback
        
        early_stopping_callback = SafeEarlyStopping(
            monitor=self.hparams.EarlyStopping_monitor,
            mode=self.hparams.EarlyStopping_mode,
            patience=self.hparams.EarlyStopping_patience,
        )

        # TODO: Hack for now until smarter config parsing
        callbacks = [
            checkpoint_callback,
            latest_checkpoint_callback,
            early_stopping_callback,
            DeviceStatsMonitor(cpu_stats=False),
            # log_weight_decay also tracks per-group WD (cross-checks any
            # weight-decay schedule the model applies to its param groups).
            LearningRateMonitor(logging_interval="step", log_weight_decay=True),
            IterationInfoCallback(
                log_interval=int(self.kwargs.get("info_log_interval", 10))
            ),
        ]
        gc_interval = int(self.kwargs.get("gc_collect_interval", 0))
        if gc_interval > 0:  # opt-in: manual GC instead of automatic pauses
            callbacks.append(GarbageCollectionCallback(interval=gc_interval))
        args = {
            "callbacks": callbacks,
            "logger": wandb_logger,
            "gradient_clip_val": gradient_clip_val,
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
            dataset=self.hparams.dataset,
            fast_dev_run=self.hparams.fast_dev_run,
            limit_val_batches=limit_val_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            # Distributed knobs: read from hparams so a config-file YAML (merged
            # by save_hyperparameters above) can flip them, e.g. `use_fsdp: true`.
            use_fsdp=getattr(self.hparams, "use_fsdp", False),
            devices=getattr(self.hparams, "devices", "auto"),
            num_nodes=getattr(self.hparams, "num_nodes", 1),
            precision=getattr(self.hparams, "precision", None),
            fsdpConfig=getattr(self.hparams, "fsdpConfig", None),
            args=args,
        )

    def _warn_on_unlogged_monitors(self) -> None:
        """Flag a ModelCheckpoint/EarlyStopping monitor the chosen model never logs.

        ``ModelCheckpoint`` raises ``MisconfigurationException`` at the first epoch end
        when its monitor was never logged, which surfaces as a crash minutes into a run
        rather than at startup. (``SafeEarlyStopping`` is forgiving and only prints, so
        an unlogged EarlyStopping monitor silently disables early stopping instead —
        also worth knowing.)

        The trainer's own defaults — ``LinearProbe/val/f1`` and ``Validation LOSS`` —
        are the SMLMTransformer probe/base-class tags; the prompt-conditioned models log
        under ``train/`` and ``val/no_promptlabels/`` and match neither. See
        ``PreliminaryGNN/configs/pointclassifier.yaml`` for a config that lines up.
        """
        hp = self.hparams
        prefixes = self._MODEL_METRIC_PREFIXES.get(getattr(hp, "modelName", ""))
        if not prefixes:
            return
        for setting, fatal in (
            ("ModelCheckpoint_monitor", True),
            ("EarlyStopping_monitor", False),
        ):
            monitor = str(getattr(hp, setting, "") or "")
            if monitor and not monitor.startswith(prefixes):
                logger.warning(
                    "%s=%r is never logged by %s, which logs under %s. %s Set a "
                    "matching monitor (see PreliminaryGNN/configs/pointclassifier.yaml).",
                    setting,
                    monitor,
                    getattr(hp, "modelName", ""),
                    " / ".join(repr(p) for p in prefixes),
                    (
                        "ModelCheckpoint will raise at the first validation epoch end."
                        if fatal
                        else "Early stopping will be skipped."
                    ),
                )


    @classmethod
    def get_class_name(cls):
        return cls.__name__

    @classmethod
    def fromConfig(cls, configPath):
        import yaml

        with open(configPath, encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f)
        # Extract top-level scalar overrides so they are passed as explicit
        # kwargs rather than relying on config_file parsing (which only works
        # when the trainer is launched via the CLI argument parser).
        _scalar_keys = {
            k: v
            for k, v in _cfg.items()
            if not isinstance(v, (dict, list)) and k != "config_file"
        }
        return cls(config_file=configPath, **_scalar_keys)


    @classmethod
    def get_args(
        cls,
        default_config_files: list[str] | None = None,
        description: str = "Training",
        **parser_kwargs,
    ) -> configargparse.ArgumentParser:
        """Return a configargparse parser pre-populated with all BaseClassTrainer params.

        Subclasses should override this, call ``super().get_args(...)``, and
        extend the returned parser with their own arguments::

            @classmethod
            def get_args(cls, default_config_files=None, **kw):
                parser = super().get_args(default_config_files, **kw)
                parser.add_argument("--my_param", type=int, default=42)
                return parser

        Parameters
        ----------
        default_config_files:
            Paths probed automatically when ``--config_file`` is not supplied
            on the CLI (forwarded to ``configargparse.ArgumentParser``).
        description:
            Parser description string.
        **parser_kwargs:
            Additional keyword arguments forwarded to
            ``configargparse.ArgumentParser``.

        Returns
        -------
        configargparse.ArgumentParser
            Parser with ``--config_file`` (``is_config_file=True``) and one
            argument per ``BaseClassTrainer.__init__`` parameter.
        """
        parser = configargparse.ArgumentParser(
            description=description,
            default_config_files=default_config_files or [],
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            allow_abbrev=False,
            **parser_kwargs,
        )
        parser.add_argument(
            "--config_file", is_config_file=True,
            help="Path to a YAML config file. CLI args take priority over config file values.",
        )
        # --- identifiers ---
        parser.add_argument("--modelName", default="Model")
        parser.add_argument("--save_dir", default="")
        parser.add_argument("--ckptPath", default=None)
        parser.add_argument("--weightsCkptPath", default="")
        # --- training loop ---
        parser.add_argument("--max_epochs", type=int, default=200)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=None)
        parser.add_argument("--trainFrac", type=float, default=0.9)
        parser.add_argument("--shuffle", action="store_true", default=True)
        parser.add_argument("--fast_dev_run", type=int, default=0)
        parser.add_argument("--reproducibility_seed", type=int, default=43)
        parser.add_argument("--dataset", default="SMLMDataset")
        parser.add_argument("--limit_val_batches", type=float, default=1.0)
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        # --- distributed training ---
        # Boolean switch: `use_fsdp: true` in a YAML config (or --use_fsdp on the
        # CLI) activates FSDP; the strategy itself is built by the trainer
        # subclass' _resolve_strategy from the model class' wrap policy.
        parser.add_argument("--use_fsdp", action="store_true", default=False)
        parser.add_argument("--devices", default="auto")
        parser.add_argument("--num_nodes", type=int, default=1)
        parser.add_argument("--precision", default=None)
        # Nested FSDP tuning block (YAML string or dict), e.g. sharding_strategy.
        parser.add_argument("--fsdpConfig", default=None)
        # --- callbacks ---
        parser.add_argument("--ModelCheckpoint_save_top_k", type=int, default=3)
        parser.add_argument("--ModelCheckpoint_monitor", default="Train LOSS")
        parser.add_argument("--ModelCheckpoint_mode", default="min")
        parser.add_argument("--EarlyStopping_monitor", default="Train LOSS")
        parser.add_argument("--EarlyStopping_mode", default="min")
        parser.add_argument("--EarlyStopping_patience", type=int, default=50)
        # Iteration info line every N train batches (IterationInfoCallback).
        parser.add_argument("--info_log_interval", type=int, default=10)
        # >0 enables GarbageCollectionCallback with that collect interval.
        parser.add_argument("--gc_collect_interval", type=int, default=0)
        # --- logging ---
        # parser.add_argument("--wandbProjectName", default="")
        return parser

    @classmethod
    def _model_config_defaults(cls) -> dict:
        """Subclasses override to supply default model-architecture config keys.

        These keys are popped from the flat args namespace and merged into
        ``modelConfig`` by :meth:`parse_args`.
        """
        return {}

    @classmethod
    def _dataset_config_defaults(cls) -> dict:
        """Subclasses override to supply default dataset config keys.

        These keys are popped from the flat args namespace and merged into
        ``datasetConfig`` by :meth:`parse_args`.
        """
        return {}

    @classmethod
    def parse_args(cls) -> dict:
        """Parse CLI / config-file args and return a kwargs dict for ``cls(**kwargs)``.

        Handles:
        * ``--num_workers`` auto-detection when still ``None`` after parsing
        * ``modelConfig`` / ``datasetConfig`` nested-YAML-string unpacking
        * Flat model/dataset keys merged over per-class defaults

        Subclasses should override :meth:`_model_config_defaults` and
        :meth:`_dataset_config_defaults` rather than overriding this method.
        """
        args = cls.get_args().parse_args()
        if getattr(args, "num_workers", None) is None:
            num_threads = os.cpu_count()
            base = int(num_threads * 0.8) if isinstance(num_threads, int) else 8
            # Windows spawns each worker (re-importing torch's full CUDA DLL stack);
            # too many concurrent loads exhaust the commit limit -> WinError 1114 (shm.dll).
            cap = 4 if sys.platform == "win32" else base
            args.num_workers = min(base, cap)
        print(f"CPU threads (num_workers): {args.num_workers}")

        args_dict = vars(args)

        def _as_dict(value) -> dict:
            if not value:
                return {}
            if isinstance(value, dict):
                return value
            return yaml.safe_load(value) or {}

        nested_model = _as_dict(args_dict.pop("modelConfig", None))
        nested_dataset = _as_dict(args_dict.pop("datasetConfig", None))

        # fsdpConfig arrives as a YAML string from configargparse; normalize to a
        # dict (or None when absent) so _resolve_strategy gets a plain mapping.
        args_dict["fsdpConfig"] = _as_dict(args_dict.pop("fsdpConfig", None)) or None

        model_defaults = cls._model_config_defaults()
        dataset_defaults = cls._dataset_config_defaults()

        # Pop any flat keys that belong to these configs so they don't leak into **kwargs
        flat_model = {k: args_dict.pop(k) for k in list(model_defaults.keys()) if k in args_dict}
        flat_dataset = {k: args_dict.pop(k) for k in list(dataset_defaults.keys()) if k in args_dict}

        # Priority: defaults < nested YAML block > flat args (nested wins when present)
        args_dict["modelConfig"] = {**model_defaults, **(nested_model if nested_model else flat_model)}
        args_dict["datasetConfig"] = {**dataset_defaults, **(nested_dataset if nested_dataset else flat_dataset)}

        return args_dict

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

   
        try:
            super().fit(segmentor, train, val, **kwargs)
            self.hparams.ckptPath = self.checkpoint_callback.best_model_path

        except:
            print(f"Exited prematurely with exception: {full_stack()}")
            # latest_checkpoint_callback (monitor=None) stores its rolling
            # final-epoch file in best_model_path; the top-k callback no longer
            # tracks last_model_path.
            self.hparams.ckptPath = self.latest_checkpoint_callback.best_model_path

        finally:
            self.saveConfig()

    @abstractmethod
    def getDataset(self, dataType, kwargs):
        pass

    @abstractmethod
    def getModel(self, kwargs):
        pass

    @abstractmethod
    def getTrainValDataloader(self, dataset, segmentor):
        pass