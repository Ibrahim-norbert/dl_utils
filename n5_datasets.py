"""Volume/mask datasets serialised into a single N5 file.

:class:`BaseVolumeDataset` and :class:`BaseMaskDataset` were moved here from
:mod:`dl_utils.datasets`; :class:`MultiDirectoryN5Dataset` extends them from one source
directory to arbitrarily many, all written into **one** N5 whose keys mirror each
sample's path diverging from the common root of those directories::

    common root = /data
    /data/plateA/well01/img001.tif  (dapi)   ->  plateA/well01/img001/dapi/{raw,mask}
    /data/plateB/100uM/img001.tif   (gfp)    ->  plateB/100uM/img001/gfp/{raw,mask}

With every sample in one directory the key collapses to ``<stem>/<channel>/<subkey>``,
which is what the single-directory base classes produced before the move.
"""

import logging
import os
from functools import cached_property
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import z5py

from dl_utils.datasets import BaseDataset

logger = logging.getLogger(__name__)


class BaseVolumeDataset(BaseDataset):
    """Sample volumes of one dataset, chunked into a single N5 file.

    One group per sample volume, holding one dataset per role (``raw`` here, ``mask``
    in :class:`BaseMaskDataset`).  The sample mapper DataFrame is the index: it is
    extended in place — and persisted to :attr:`sampleMapperOutPath` — with the N5 key
    of every dataset written, so downstream code addresses volumes by key and never
    touches the source TIFFs again.
    """

    # NOTE: Coloumn where all N5 keys are saved
    N5_VOLUME_KEY_COLUMN: str = "volume_key"
    DATASET_FILE_EXTENSION: str = ".n5"
    RAW_KEY: str = "raw"
    # Bicubic for intensities; masks override with nearest-neighbour (see BaseMaskDataset).
    RAW_INTERPOLATION_ORDER: int = 3
    CHUNKS: Tuple[int, int, int] = (32, 32, 32)
    DEFAULT_RESOLUTION: Tuple[int, int, int] = (3, 1, 1)

    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 samplePathRegex=None, resolution: Optional[List[int]] = None,
                 channelKey: str = "dapi", channelColumn: str = "channel",
                 datasetName: Optional[str] = None,
                 **kwargs) -> None:
        # Assigned before super(), whose __init__ validates the sample paths — the
        # multi-directory subclass validates against these very attributes.
        self.resolution: List[int] = list(resolution or self.DEFAULT_RESOLUTION)
        self.channelKey: str = channelKey
        self.channelColumn: str = channelColumn
        self.datasetName: Optional[str] = datasetName

        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         samplePathRegex=samplePathRegex,
                         **kwargs)

        # Idempotent: only keys missing from the N5 are written, so re-running is cheap
        # and an interrupted conversion resumes where it stopped.
        self.createDatasetN5File()
        self.write_index(self.sampleMapperDF, self.sampleMapperOutPath)
        logger.info("N5 dataset ready: %s (%d samples) — index: %s",
                    self.n5Path, len(self.sampleMapperDF), self.sampleMapperOutPath)

    # ---- Paths -------------------------------------------------------

    @cached_property
    def sampleRoot(self) -> str:
        """Common ancestor directory of every sample volume — the origin of all keys.

        Taking the *directories* before the common prefix keeps a single sample, and a
        single directory, from collapsing onto the file itself.
        """
        directories = [os.path.dirname(os.path.abspath(str(path)))
                       for path in self.sampleMapperDF[self.samplePathColumn]]
        try:
            return os.path.commonpath(directories)
        except ValueError as error:  # different drives, or absolute mixed with relative
            raise ValueError(
                f"Sample paths share no common root: {sorted(set(directories))}") from error

    @cached_property
    def datasetDirName(self) -> str:
        """Dataset name — the common root's basename unless one was given explicitly."""
        return self.datasetName or os.path.basename(self.sampleRoot)

    @cached_property
    def datasetDir(self) -> str:
        datasetDir = os.path.join(self.save_dir, self.datasetDirName)
        os.makedirs(datasetDir, exist_ok=True)
        return datasetDir

    @cached_property
    def n5Path(self) -> str:
        return os.path.join(self.datasetDir,
                            f"{self.datasetDirName}{self.DATASET_FILE_EXTENSION}")

    @cached_property
    def sampleMapperOutPath(self) -> str:
        """The key-annotated sample mapper, persisted beside the N5 it indexes."""
        return os.path.join(self.datasetDir, f"{self.datasetDirName}_sample_mapper.csv")

    @property
    def confirmDatasetN5File(self) -> bool:
        return os.path.exists(self.n5Path)

    # ---- Keys --------------------------------------------------------

    def sampleKey(self, sampleVolumePath: str) -> str:
        """Path of *sampleVolumePath* diverging from :attr:`sampleRoot`, extension-free.

        Always ``/``-separated: N5 keys are POSIX-style, so ``os.path.join`` (which
        yields backslashes on Windows) must never be used to build them.
        """
        stem = os.path.splitext(os.path.abspath(str(sampleVolumePath)))[0]
        return os.path.relpath(stem, self.sampleRoot).replace(os.sep, "/")

    def channelOf(self, row: pd.Series) -> str:
        """Channel of a sample mapper row, falling back to the dataset-wide channel."""
        channel = row.get(self.channelColumn) if self.channelColumn else None
        return self.channelKey if channel is None or pd.isna(channel) else str(channel)

    def getN5GroupKey(self, sampleVolumePath: str, channel: Optional[str] = None) -> str:
        """Group holding every role (raw, mask, ...) of a single FOV."""
        return f"{self.sampleKey(sampleVolumePath)}/{channel or self.channelKey}"

    def datasetKey(self, sampleVolumePath: str, subkey: str,
                   channel: Optional[str] = None) -> str:
        return f"{self.getN5GroupKey(sampleVolumePath, channel)}/{subkey}"

    # ---- Writing -----------------------------------------------------

    @staticmethod
    def toUint16(volume: np.ndarray, normalise: bool) -> np.ndarray:
        """Cast to the storage dtype, min-max stretching intensities when *normalise*.

        Label volumes are never stretched: their values are identities, not intensities.
        """
        ceiling = np.iinfo(np.uint16).max
        if normalise:
            low, high = float(volume.min()), float(volume.max())
            volume = ((volume - low) / (high - low) if high > low
                      else np.zeros_like(volume)) * ceiling
        assert volume.min() >= 0 and volume.max() <= ceiling, (
            f"Volume range [{volume.min()}, {volume.max()}] does not fit uint16")
        return volume.astype(np.uint16)

    def prepareSampleVolume(self, sampleVolumePath: str, interpolationOrder: int) -> np.ndarray:
        """Load a TIF and bring it into storage form: isotropic voxels, ``uint16``.

        *interpolationOrder* carries the role, following the convention this codebase
        uses everywhere: 3 (bicubic, anti-aliased) for intensity volumes, which are
        min-max stretched over the uint16 range, and 0 (nearest-neighbour, no
        smoothing) for label volumes, whose IDs must come through untouched.  Deciding
        on the order rather than on the resampled dtype matters: an already-isotropic
        volume skips the resize, and must still be treated as what it is.
        """
        isLabels: bool = interpolationOrder == 0
        volume: np.ndarray = self.loadtiffVolume(str(sampleVolumePath))
        assert volume.ndim == 3, (
            f"Expected a 3D (Z, Y, X) volume, got shape {volume.shape}: {sampleVolumePath}")

        resampled, _ = self.resampleVolume(volume, self.resolution, interpolationOrder)

        if isLabels and resampled.shape != volume.shape:
            # Nearest-neighbour keeps label values exact, but an object thinner than the
            # sampling stride can fall between samples and vanish entirely.
            lost = np.setdiff1d(np.unique(volume), np.unique(resampled))
            if lost.size:
                logger.warning("Resampling %s dropped %d label(s): %s",
                               sampleVolumePath, lost.size, lost[:10])

        return self.toUint16(resampled, normalise=not isLabels)

    @staticmethod
    def requireGroup(n5File, groupKey: str):
        """Group at *groupKey*, creating every missing level along the way."""
        group = n5File
        for name in filter(None, groupKey.split("/")):
            group = group[name] if name in group else group.create_group(name)
        return group

    @staticmethod
    def containsKey(n5File, key: str) -> bool:
        """Membership test for a nested key — ``in`` only speaks about direct children."""
        groupKey, _, subkey = key.rpartition("/")
        node = n5File
        for name in filter(None, groupKey.split("/")):
            if name not in node:
                return False
            node = node[name]
        return subkey in node

    def writeN5Dataset(self, n5File, key: str, volume: np.ndarray, channel: str) -> None:
        """Write *volume* at *key*, chunked for partial reads and tagged with its channel."""
        groupKey, _, subkey = key.rpartition("/")
        dataset = self.requireGroup(n5File, groupKey).require_dataset(
            subkey,
            shape=volume.shape,
            # A chunk may not exceed the axis it chunks — small volumes are one chunk.
            chunks=tuple(min(chunk, size) for chunk, size in zip(self.CHUNKS, volume.shape)),
            dtype=volume.dtype,
        )
        dataset[:] = volume
        dataset.attrs["channel"] = channel

    def writeKeyColumn(self, pathColumn: str, subkey: str, keyColumn: str,
                       interpolationOrder: int,
                       validate: Optional[Callable[[Any, str, np.ndarray], None]] = None) -> None:
        """Write one dataset per sample under *subkey*; record its key in *keyColumn*.

        Keys always derive from the sample *volume* path, so every role of a sample
        (raw, mask, ...) lands in that sample's one group whatever column the pixels
        came from.  Samples already present are skipped, which is what makes
        construction idempotent and an interrupted run resumable.  The N5 is opened
        once for the whole column.
        """
        assert pathColumn in self.sampleMapperDF.columns, (
            f"Sample mapper has no {pathColumn!r} column: {list(self.sampleMapperDF.columns)}")

        # Keys are pure path arithmetic, so they are derived — and checked for
        # collisions — before a single voxel is written and overwrites its neighbour.
        keys = pd.Index([self.datasetKey(row[self.samplePathColumn], subkey, self.channelOf(row))
                         for _, row in self.sampleMapperDF.iterrows()])
        duplicates = keys[keys.duplicated()].unique().tolist()
        assert not duplicates, (
            f"Sample mapper maps several rows onto the same N5 key: {duplicates[:5]}")

        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:
            for key, (_, row) in zip(keys, self.sampleMapperDF.iterrows()):
                if self.containsKey(n5File, key):
                    logger.debug("Already in N5, skipping: %s", key)
                    continue

                volume: np.ndarray = self.prepareSampleVolume(row[pathColumn], interpolationOrder)
                if validate is not None:
                    validate(n5File, key, volume)
                self.writeN5Dataset(n5File, key, volume, self.channelOf(row))
                logger.info("Wrote %s <- %s", key, row[pathColumn])

        self.sampleMapperDF[keyColumn] = keys.to_numpy()

    def createDatasetN5File(self) -> None:
        """Store the volumes as n5 files for RAM saving chunked based signal loading"""
        self.writeKeyColumn(self.samplePathColumn, self.RAW_KEY,
                            self.N5_VOLUME_KEY_COLUMN, self.RAW_INTERPOLATION_ORDER)

    # ---- Reading -----------------------------------------------------

    def keyOf(self, idx: int, keyColumn: Optional[str] = None) -> str:
        return str(self.row(idx)[keyColumn or self.N5_VOLUME_KEY_COLUMN])

    def openSample(self, idx: int, keyColumn: Optional[str] = None):
        """Lazily opened, cached handle on one sample's dataset — reads no voxels.

        Handles are built per process and dropped on pickling (see
        :meth:`BaseDataset.__getstate__`), so DataLoader workers open their own.
        """
        key: str = self.keyOf(idx, keyColumn)
        if not hasattr(self, "_n5Handles"):  # unpickled without __setstate__
            self._n5Handles = {}
        if key not in self._n5Handles:
            self._n5Handles[key] = z5py.File(self.n5Path, "r")[key]
        return self._n5Handles[key]

    def readSample(self, idx: int, keyColumn: Optional[str] = None, bbox=None) -> np.ndarray:
        """Whole-volume read, or only *bbox* (a tuple of slices) when one is given."""
        handle = self.openSample(idx, keyColumn)
        return np.asarray(handle[bbox] if bbox is not None else handle[:])


class BaseMaskDataset(BaseVolumeDataset):
    """Adds each sample's instance-segmentation mask to its volume's N5 group."""

    N5_MASK_KEY_COLUMN: str = "mask_key"
    SAMPLE_MASK_PATH_COLUMN: str = "mask_path"
    MASK_KEY: str = "mask"
    # Nearest-neighbour: interpolating between label IDs invents objects.
    MASK_INTERPOLATION_ORDER: int = 0

    def createDatasetN5File(self) -> None:
        """Raw volumes first (base class), then each paired mask into the same group.

        Extends rather than replaces the base implementation — overriding it outright
        left the raw channel out of the N5 entirely.
        """
        super().createDatasetN5File()
        self.writeKeyColumn(self.SAMPLE_MASK_PATH_COLUMN, self.MASK_KEY,
                            self.N5_MASK_KEY_COLUMN, self.MASK_INTERPOLATION_ORDER,
                            validate=self.assertPairsWithRaw)

    def assertPairsWithRaw(self, n5File, maskKey: str, mask: np.ndarray) -> None:
        """A mask must be voxel-aligned with the raw volume it shares a group with."""
        rawKey = f"{maskKey.rpartition('/')[0]}/{self.RAW_KEY}"
        assert self.containsKey(n5File, rawKey), (
            f"No raw volume at {rawKey} to pair {maskKey} with")
        rawShape = tuple(n5File[rawKey].shape)  # metadata only — no voxels are read
        assert tuple(mask.shape) == rawShape, (
            f"Mask {maskKey} has shape {mask.shape} but its raw volume has {rawShape}")


class MultiDirectoryN5Dataset(BaseMaskDataset):
    """Volume/mask samples spread over many directories, gathered into ONE N5 file.

    The sample mapper DataFrame drives everything: one row per sample, carrying the
    volume path, the mask path and the channel.  Each sample is stored under
    ``<path diverging from the common root>/<channel>/{raw,mask}``, so files sharing a
    name in different directories cannot collide — the limitation that confined the
    base classes to a single directory.  Construction leaves the mapper extended with
    the ``volume_key`` / ``mask_key`` columns addressing those datasets, persisted to
    :attr:`sampleMapperOutPath`.

    Parameters
    ----------
    sampleMapperDFPath : str
        Sample mapper table (CSV/XLSX/...), one row per sample.
    save_dir : str
        Artefact root; the N5 and the extended mapper land in
        ``<save_dir>/<datasetDirName>/``.
    volumePathColumn, maskPathColumn, channelColumn : str
        Mapper columns holding the raw volume path, the mask path and the channel.
    """

    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 volumePathColumn: str = "sample_path",
                 maskPathColumn: str = "mask_path",
                 channelColumn: str = "channel",
                 **kwargs) -> None:
        # Assigned before super(): its __init__ validates the paths and runs the whole
        # N5 conversion off these columns.
        self.samplePathColumn = volumePathColumn
        self.SAMPLE_MASK_PATH_COLUMN = maskPathColumn
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         channelColumn=channelColumn, **kwargs)

    def validateSamplePaths(self) -> None:
        """Replaces the single-directory contract of the base class with a wider one.

        Samples may live anywhere as long as the mapper is complete, every file exists
        and the volumes share a common root for keys to diverge from.
        """
        missingColumns = [column for column
                          in (self.samplePathColumn, self.SAMPLE_MASK_PATH_COLUMN,
                              self.channelColumn)
                          if column not in self.sampleMapperDF.columns]
        assert not missingColumns, (
            f"Sample mapper is missing {missingColumns}; "
            f"it has {list(self.sampleMapperDF.columns)}")

        for column in (self.samplePathColumn, self.SAMPLE_MASK_PATH_COLUMN):
            missing = [str(path) for path in self.sampleMapperDF[column]
                       if not os.path.isfile(str(path))]
            assert not missing, (
                f"{len(missing)} {column!r} entries are not files, first: {missing[0]}")

        logger.info("Indexing %d samples from %d directories under %s",
                    len(self.sampleMapperDF),
                    self.sampleMapperDF[self.samplePathColumn].map(os.path.dirname).nunique(),
                    self.sampleRoot)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(raw, mask)`` for one sample, read straight from the shared N5."""
        return (self.readSample(idx, self.N5_VOLUME_KEY_COLUMN),
                self.readSample(idx, self.N5_MASK_KEY_COLUMN))

