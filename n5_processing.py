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

Construction only indexes; :meth:`BaseVolumeDataset.run` performs the conversion.  Reads
follow the ``LMTextureNucleiDataset`` convention (:meth:`~BaseVolumeDataset.get_hr_vol`,
:meth:`~BaseMaskDataset.get_hr_mask`, :meth:`~BaseMaskDataset.get_masked_hr_vol`) and open
their handle per call, so no unpicklable state is carried across a DataLoader fork.
"""
import glob
import logging
import numbers
import os
from functools import cached_property
from typing import Any, Callable, List, Optional, Tuple, Union, Hashable

import numpy as np
import pandas as pd
import skimage.io
import z5py
from matplotlib import pyplot as plt
from numpy import ndarray, dtype
from skimage.transform import resize
from sklearn import preprocessing

from dl_utils import MOBIE_LABEL_KEY, img2patch as img2ps, util, MOBIE_LABEL_KEY
from dl_utils.SampleLoader import SampleLoaderBioImage
from dl_utils.datasets import BaseDataset
from dl_utils.util_base import save2DFcolumn, savedataframe, merge_with_nucl_table
from dl_utils.vizualizations import CustomMatplotlib

logger = logging.getLogger(__name__)

SIGMA_THRESHOLD = 3
CHUNK_MULTIPLE = 8
BBOX_COLUMNS = ["bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x"]

def getArrayFromDF(df, column):
    """Extracts and converts a column from a DataFrame to a NumPy array."""
    return np.array(df[column].tolist())

def getBBOXFromDF(df: pd.DataFrame, df_indx: Optional[Union[int, Hashable]]) -> np.ndarray:
    try:
        bbox_values = [df.loc[df_indx, dim] for dim in BBOX_COLUMNS]
        return np.array(bbox_values, dtype=int).T  # Convert to NumPy float array
    except Exception as e:
        print(f"Following error {e} was obtained from df with coloumns {df.columns.tolist()}")


def getAllBBOXArray(data_df) -> ndarray[Any, dtype]:
    """
    Converts the bounding boxes in the dataframe to an array compatible with
    the img2patch module.

    Concatenates the label and bounding box coordinates into a single array.

    Returns:
    --------
    np.ndarray
        An array containing the labels and bounding box coordinates.
    """
    arr: ndarray[Any, dtype] = np.array(
        [getBBOXFromDF(data_df, df_indx=indx) for indx in data_df.index]
    )
    data_df.reset_index(inplace=True)
    return np.concatenate(
        [data_df.index.to_numpy()[np.newaxis, ...].T,
         arr],
        axis=-1
    )


def filterDF(df, column="z") -> pd.DataFrame:
    # Calculate mean and standard deviation
    res = getArrayFromDF(df, column=column)
    mean = res.mean()
    std_dev = res.std()

    # Define the range for three standard deviations
    lower_bound = mean - std_dev * SIGMA_THRESHOLD
    upper_bound = mean + std_dev * SIGMA_THRESHOLD
    
    print(
        f"Filtering bbox sides for the column {column} according to the upper and lower bound: "
        f"{lower_bound} and {upper_bound} what is the mean {mean} and stddev {std_dev} "
        f"and min and max {res.min()} and {res.max()}"
    )
    # Print number of label_ids before filtering
    max_z = df["z"].max()
    max_y = df["y"].max()
    max_x = df["x"].max()
    print(
        f"Number of label ids before: {df['label_id'].count()} and {max_z}, {max_y}, {max_x}")

    # TODO: Save the mean and standard deviation values to columns "{column}_length_mean" and "{column}_length_stddev" in the DataFrame if needed
    df.loc[f"{column}_length_mean"] = mean
    df.loc[f"{column}_length_stddev"] = std_dev

    # Define the range for filtering based on the bounds
    return df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound), :]


def mask2BBOXDF(maskVol: np.ndarray) -> pd.DataFrame:
    all_boxes: img2ps.BBoxes = img2ps.BBoxes.from_mask(maskVol)

    assert isinstance(all_boxes.dataFrame, pd.DataFrame), (
        f"Dataframe does not exist. If it is None ({all_boxes.dataFrame}), then"
        f"{all_boxes.__class__} has not been init with classmethod .fromMask")

    return all_boxes.dataFrame


def BBOXDF(all_boxes: object) -> pd.DataFrame:
    """Convert a BBoxes object into a pandas DataFrame with nucleus metadata."""

    assert isinstance(all_boxes.dataFrame, pd.DataFrame), (
        f"Dataframe does not exist. If it is None ({all_boxes.dataFrame}), then"
        f"{all_boxes.__class__} has not been init with classmethod .fromMask")

    return all_boxes.dataFrame


def bbox2DFAndID(all_boxes: object, dataset_path, datasetMask_path, key):
    dataframe = BBOXDF(all_boxes)

    N = dataframe.shape[0]

    additionalCols = {"dataset path": [dataset_path] * N,
                      "mask volume path": [datasetMask_path] * N,
                      "dataset acronym": [key] * N}

    for k, v in additionalCols.items():
        dataframe[k] = v

    return dataframe


def clean_dataframe(df, show_hist=False):
    """
    Filter a nucleus DataFrame by bounding-box dimensions (±3 std).

    Adds z/y/x side-length columns, applies per-axis sigma filtering, drops
    NaN rows, and returns the cleaned DataFrame.
    """
    # TURN2BBOX implementation
    bbox = getAllBBOXArray(df)

    bboxes_obj = img2ps.BBoxes(bboxes=bbox)

    labels = bboxes_obj.get("sides")[:, 0]
    res = bboxes_obj.get("sides")[:, 1:]

    if show_hist:
        counts, bins = np.histogram(res)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylabel("Count")
        plt.title("Before")
        plt.show()
        plt.close()

    df = save2DFcolumn(res[:, 0].tolist(), sorted_nucl_labels=labels, dataframe=df,
                       column_name="z")
    df = save2DFcolumn(res[:, 1].tolist(), sorted_nucl_labels=labels, dataframe=df,
                       column_name="y")
    df = save2DFcolumn(res[:, 2].tolist(), sorted_nucl_labels=labels, dataframe=df,
                       column_name="x")

    filtered_df = filterDF(df, column="z")
    filtered_df = filterDF(filtered_df, column="y")
    filtered_df = filterDF(filtered_df, column="x")

    max_z = df["z"].max()
    max_y = df["y"].max()
    max_x = df["x"].max()

    print(
        f"After filtering: number of label ids before filtering: {df['label_id'].count()} and max dimensions: {max_z}, {max_y}, {max_x}")

    # Drop rows with missing values in specific columns
    filtered_df.dropna(subset=["z", "x", "y"], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    if show_hist:
        res = np.array([
            getArrayFromDF(filtered_df, column="z"),
            getArrayFromDF(filtered_df, column="y"),
            getArrayFromDF(filtered_df, column="x"),
        ]).flatten()

        counts, bins = np.histogram(res)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylabel("Count")
        plt.title("after")
        plt.show()
        plt.close()

    print(f"Number of label ids after filtering: {filtered_df['label_id'].count()}")

    return filtered_df



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
    # Only these are stripped when deriving a key, so `img.ome.tif` keeps its `.ome`.
    VOLUME_FILE_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff")
    RAW_KEY: str = "raw"
    # Bicubic for intensities; masks override with nearest-neighbour (see BaseMaskDataset).
    RAW_INTERPOLATION_ORDER: int = 3
    CHUNKS: Tuple[int, int, int] = (32, 32, 32)
    DEFAULT_RESOLUTION: Tuple[int, int, int] = (3, 1, 1)

    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 samplePathRegex=None, resolution: Optional[List[int]] = None,
                 channelKey: str = "dapi", channelColumn: str = "channel",
                 **kwargs) -> None:
        # Assigned before super(), whose __init__ validates the sample paths — the
        # multi-directory subclass validates against these very attributes.
        self.resolution: List[int] = list(resolution or self.DEFAULT_RESOLUTION)
        self.channelKey: str = channelKey
        self.channelColumn: str = channelColumn
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         samplePathRegex=samplePathRegex,
                         **kwargs)

        assert os.path.basename(self.save_dir) == self.datasetDirName, f"The saving directory must have the same basename as the dataset name: {self.save_dir} == {self.datasetDirName}"
        assert self.sampleMapperDF[self.SAMPLE_PATH_COLUMN].nunique() == self.sampleMapperDF.shape[0], (
            f"The sample mapper must have unique paths for column: {self.SAMPLE_PATH_COLUMN}"
        )

    def run(self) -> None:
        """Build the N5 and persist the key-annotated sample mapper.

        Deliberately not called from ``__init__``: constructing a dataset should index,
        not convert.  Keeping the conversion behind an explicit call lets an existing N5
        be opened without reprocessing, and is what a Lightning ``prepare_data()`` hook
        should invoke so the work happens once rather than once per rank.

        Idempotent: only keys missing from the N5 are written, so re-running is cheap and
        an interrupted conversion resumes where it stopped.
        """
        os.makedirs(self.datasetDir, exist_ok=True)
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
                       for path in self.sampleMapperDF[self.SAMPLE_PATH_COLUMN]]
        if not directories:
            raise ValueError(
                f"Sample mapper {self.sampleMapperDFPath} has no rows, so there is no "
                f"common root to derive N5 keys from")
        try:
            return os.path.commonpath(directories)
        except ValueError as error:  # different drives, or absolute mixed with relative
            raise ValueError(
                f"Sample paths share no common root: {sorted(set(directories))}") from error

    @cached_property
    def datasetDirName(self) -> str:
        """Dataset name — the common root's basename unless one was given explicitly."""
        return os.path.basename(self.sampleRoot)

    @cached_property
    def datasetDir(self) -> str:
        """Artefact root for this dataset — resolved only; :meth:`run` creates it."""
        return os.path.join(self.save_dir, "training_processed")

    @cached_property
    def n5Path(self) -> str:
        return os.path.join(self.datasetDir,
                            f"{self.datasetDirName}{self.DATASET_FILE_EXTENSION}")

    @cached_property
    def sampleMapperOutPath(self) -> str:
        """The key-annotated sample mapper, persisted beside the N5 it indexes."""
        return os.path.join(self.datasetDir, f"{self.datasetDirName}_sample_mapper.csv")

    def getSampleDatasetDir(self, sampleVolumePath: str, *subdirs: str) -> str:
        """Per-sample artefact directory: :attr:`save_dir` plus the sample's own path.

        Derived from the same diverging path as the sample's N5 key, so artefacts mirror
        the N5 hierarchy and two volumes sharing a basename in different directories
        cannot overwrite each other's outputs.  Created on demand.
        """
        path = os.path.join(self.save_dir,
                            *self.sampleKey(sampleVolumePath).split("/"), *subdirs)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def confirmDatasetN5File(self) -> bool:
        return os.path.exists(self.n5Path)

    # ---- Keys --------------------------------------------------------

    def sampleKey(self, sampleVolumePath: str) -> str:
        """Path of *sampleVolumePath* diverging from :attr:`sampleRoot`, extension-free.

        Always ``/``-separated: N5 keys are POSIX-style, so ``os.path.join`` (which
        yields backslashes on Windows) must never be used to build them.

        Only a known volume extension is stripped — blind ``splitext`` turned
        ``img.ome.tif`` into ``img.ome`` and made the key depend on how many dots the
        filename happened to carry.
        """
        path = os.path.abspath(str(sampleVolumePath))
        stem, extension = os.path.splitext(path)
        if extension.lower() not in self.VOLUME_FILE_EXTENSIONS:
            stem = path
        return os.path.relpath(stem, self.sampleRoot).replace(os.sep, "/")

    def channelOf(self, row: pd.Series) -> str:
        """Channel of a sample mapper row, falling back to the dataset-wide channel.

        The channel becomes one key segment, so a value carrying ``/`` would silently
        deepen the hierarchy and move the sample somewhere nobody addresses.
        """
        channel = row.get(self.channelColumn) if self.channelColumn else None
        channel = self.channelKey if channel is None or pd.isna(channel) else str(channel)
        if "/" in channel or not channel.strip():
            raise ValueError(
                f"Channel {channel!r} is not a usable N5 key segment: it must be "
                f"non-blank and free of '/'")
        return channel

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
            # TODO: Just use Sklearn minmax but only on raw image not any masks
            low, high = float(volume.min()), float(volume.max())
            volume = ((volume - low) / (high - low) if high > low
                      else np.zeros_like(volume)) * ceiling
        if volume.min() < 0 or volume.max() > ceiling:
            raise ValueError(
                f"Volume range [{volume.min()}, {volume.max()}] does not fit uint16")
        return volume.astype(np.uint16)

    def resampleVolume(self, volume: np.ndarray, resolutions: list[int],
                       interpolationOrder: int = 0,
                       antiAliasing: Optional[bool] = None) -> tuple[np.ndarray, int]:
        """Resample *volume* to isotropic resolution.

        Normalises all axis resolutions relative to the finest (minimum) one,
        then rescales the volume accordingly.  If the volume is already
        isotropic the resize step is skipped.

        Parameters
        ----------
        volume : np.ndarray
            3-D input volume ``(Z, Y, X)``.
        resolutions : list of int
            Physical voxel size (e.g. in nm) for each axis in ZYX order.
        interpolationOrder : int, optional
            Spline interpolation order passed to :func:`skimage.transform.resize`.
            Use ``0`` for label/mask volumes (nearest-neighbour) and ``3`` for
            raw intensity volumes (bicubic).  Default ``0``.
        antiAliasing : bool, optional
            Whether to pre-smooth before down-sampling.  Defaults to
            ``interpolationOrder > 0``: smoothing a label volume blends
            neighbouring IDs into values that name no object, so it must stay off
            for masks and on for intensities.

        Returns
        -------
        volume : np.ndarray
            Resampled volume.  Intensities keep their input range
            (``preserve_range``), so label IDs survive the round trip.
        min_res : int
            The finest (smallest) voxel size from *resolutions*, which becomes
            the isotropic resolution of the output.
        """
        min_res: int = min(resolutions)
        resolutions = np.array(resolutions) / min_res

        logger.debug("Pre-resampled dimension: %s", volume.shape)

        if all(res == 1 for res in resolutions):
            logger.debug("Isotropic resolution detected. Skipping resizing.")
            return volume, min_res

        newShape = tuple(int(round(size * res)) for size, res in zip(volume.shape, resolutions))
        volume = resize(volume, newShape, order=interpolationOrder, preserve_range=True,
                        anti_aliasing=interpolationOrder > 0 if antiAliasing is None else antiAliasing)

        logger.debug("Resampled dimension: %s", volume.shape)
        return volume, min_res

    @staticmethod
    def loadFromSamplePath(path: str) -> np.ndarray:
        """Load a 3-D TIFF volume from *path*."""
        volume : np.ndarray = SampleLoaderBioImage.loadData(path)
        assert isinstance(volume, np.ndarray), (f"BaseVolumeDataset currently only supports loading {np.ndarray.__class__}"
         + f" not {volume.__class__} from {path}")
        d = volume.ndim
        if not d == 3:
            raise ValueError(
                f"Expected a 3D (Z, Y, X) volume, got shape {volume.shape}: {path}"
            )
        return volume

    def prepareSampleVolume(self, sampleVolumePath: str, interpolationOrder: int) -> dict:
        """Load a sample and bring it into storage form: isotropic voxels, ``uint16``.

        *interpolationOrder* carries the role, following the convention this codebase
        uses everywhere: 3 (bicubic, anti-aliased) for intensity volumes, which are
        min-max stretched over the uint16 range.

        """

        volume: np.ndarray = self.loadFromSamplePath(str(sampleVolumePath))

        resampled, _ = self.resampleVolume(volume, self.resolution, interpolationOrder)

        return {"volume": self.toUint16(resampled, normalise=True)}

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
        # TODO: require_dataset writes metadata before this fill, so a crash here leaves
        # an empty dataset that containsKey skips on the next run. Accepted for now.
        dataset[:] = volume
        dataset.attrs["channel"] = channel

    def __writeKeyVolumes__(self, pathColumn: str, subkey: str, keyColumn: str,
                            interpolationOrder: int,
                            validate: Optional[Callable[[Any, str, Optional[np.ndarray]], None]] = None
                            ) -> None:
        """Write one dataset per sample under *subkey*; record its key in *keyColumn*.

        Keys always derive from the sample *volume* path, so every role of a sample
        (raw, mask, ...) lands in that sample's one group whatever column the pixels
        came from.  Samples already present are skipped, which is what makes
        construction idempotent and an interrupted run resumable.  The N5 is opened
        once for the whole column.
        """
        if pathColumn not in self.sampleMapperDF.columns:
            raise KeyError(
                f"Sample mapper has no {pathColumn!r} column: "
                f"{list(self.sampleMapperDF.columns)}")

        # Keys are pure path arithmetic, so they are derived — and checked for
        # collisions — before a single voxel is written and overwrites its neighbour.
        # NOTE: Constructed from sample path, channel and volume type (e.g. raw, mask etc.) indicator subkey
        keys = pd.Index([self.datasetKey(row[self.SAMPLE_PATH_COLUMN], channel=self.channelOf(row), subkey=subkey)
                         for _, row in self.sampleMapperDF.iterrows()])

        # Note: Test if duplicated keys exist
        duplicates = keys[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Sample mapper maps several rows onto the same N5 key: {duplicates[:5]}")

        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:
            for key, (i, row) in zip(keys, self.sampleMapperDF.iterrows()):
                if self.containsKey(n5File, key):
                    # Still validated: the check reads metadata only, and skipping it on
                    # resume meant an existing dataset was never checked against its pair.
                    if validate is not None:
                        validate(n5File, key, None)
                    logger.debug("Already in N5, skipping: %s", key)
                    continue
                print(f"Writing {key} of indx {i} keys to N5 file {self.n5Path}")
                volume: np.ndarray = self.prepareSampleVolume(row[pathColumn], interpolationOrder)["volume"]

                if validate is not None:
                    validate(n5File, key, volume)
                self.writeN5Dataset(n5File, key, volume, self.channelOf(row))
                logger.info("Wrote %s <- %s", key, row[pathColumn])

        self.sampleMapperDF[keyColumn] = keys.to_numpy()

    def createDatasetN5File(self) -> None:
        """Store the volumes as n5 files for RAM saving chunked based signal loading"""
        self.__writeKeyVolumes__(self.SAMPLE_PATH_COLUMN, self.RAW_KEY,
                                 self.N5_VOLUME_KEY_COLUMN, self.RAW_INTERPOLATION_ORDER)

    # ---- Reading -----------------------------------------------------

    def keyOf(self, idx: int, keyColumn: Optional[str] = None) -> str:
        return str(self.sampleDFRow(idx)[keyColumn or self.N5_VOLUME_KEY_COLUMN])

    def openKey(self, key: str):
        """Open the N5 dataset at *key* — the one place a handle is created.

        Nothing is stored on the instance: a cached z5py handle is unpicklable, which is
        what forced the ``__getstate__``/``__setstate__`` pair these classes no longer
        need.  Opening is metadata-only, so DataLoader workers each open their own.
        """
        return z5py.File(self.n5Path, "r")[key]

    def _ensure_open(self, idx: int, keyColumn: Optional[str] = None):
        """Open the dataset of sample *idx*, addressed through the mapper's key column."""
        print(f"Opening key for sample index {idx} from column {keyColumn}")
        return self.openKey(self.keyOf(idx, keyColumn))

    def get_hr_vol(self, idx: int) -> np.ndarray:
        """Raw volume of sample *idx*, or only *bbox* (a tuple of slices) when given."""
        Volume = self._ensure_open(idx, self.N5_VOLUME_KEY_COLUMN)
        return np.asarray(Volume[:])


class BaseMaskDataset(BaseVolumeDataset):
    """Adds each sample's instance-segmentation mask to its volume's N5 group."""

    N5_MASK_KEY_COLUMN: str = "mask_key"
    SAMPLE_MASK_PATH_COLUMN: str = "mask_path"
    MASK_KEY: str = "mask"
    # Nearest-neighbour: interpolating between label IDs invents objects.
    MASK_INTERPOLATION_ORDER: int = 0


    def prepareSampleVolume(self, sampleVolumePath: str, interpolationOrder: int) -> dict:
        """Load a TIF and bring it into storage form: isotropic voxels, ``uint16``.

        *interpolationOrder* carries the role, following the convention this codebase
        uses everywhere: 3 (bicubic, anti-aliased) for intensity volumes, which are
        min-max stretched over the uint16 range, and 0 (nearest-neighbour, no
        smoothing) for label volumes, whose IDs must come through untouched.  Deciding
        on the order rather than on the resampled dtype matters: an already-isotropic
        volume skips the resize, and must still be treated as what it is.
        """
        isLabels: bool = interpolationOrder == 0

        volume: np.ndarray = self.loadFromSamplePath(str(sampleVolumePath))

        resampled, _ = self.resampleVolume(volume, self.resolution, interpolationOrder)

        if isLabels and resampled.shape != volume.shape:
            # Nearest-neighbour keeps label values exact, but an object thinner than the
            # sampling stride can fall between samples and vanish entirely.
            lost = np.setdiff1d(np.unique(volume), np.unique(resampled))
            if lost.size:
                logger.warning("Resampling %s dropped %d label(s): %s",
                               sampleVolumePath, lost.size, lost[:10])

        return {"volume": self.toUint16(resampled, normalise=not isLabels)}

    def __writeKeyMask__(self) -> None:
        """Write one dataset per sample under *subkey*; record its key in *keyColumn*.

        Keys always derive from the sample *volume* path, so every role of a sample
        (raw, mask, ...) lands in that sample's one group whatever column the pixels
        came from.  Samples already present are skipped, which is what makes
        construction idempotent and an interrupted run resumable.  The N5 is opened
        once for the whole column.
        """

        pathColumn = self.SAMPLE_MASK_PATH_COLUMN
        subkey = self.MASK_KEY
        keyColumn = self.N5_MASK_KEY_COLUMN
        validate = self.assertPairsWithRaw
        interpolationOrder = self.MASK_INTERPOLATION_ORDER

        if pathColumn not in self.sampleMapperDF.columns:
            raise KeyError(
                f"Sample mapper has no {pathColumn!r} column: "
                f"{list(self.sampleMapperDF.columns)}")

        # Keys are pure path arithmetic, so they are derived — and checked for
        # collisions — before a single voxel is written and overwrites its neighbour.
        # NOTE: Constructed from sample path, channel and volume type (e.g. raw, mask etc.) indicator subkey
        keys = pd.Index([self.datasetKey(row[self.SAMPLE_PATH_COLUMN], channel=self.channelOf(row), subkey=subkey)
                         for _, row in self.sampleMapperDF.iterrows()])

        # Note: Test if duplicated keys exist
        duplicates = keys[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Sample mapper maps several rows onto the same N5 key: {duplicates[:5]}")

        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:
            for key, (_, row) in zip(keys, self.sampleMapperDF.iterrows()):
                if self.containsKey(n5File, key):
                    # Still validated: the check reads metadata only, and skipping it on
                    # resume meant an existing dataset was never checked against its pair.
                    if validate is not None:
                        validate(n5File, key, None)
                    logger.debug("Already in N5, skipping: %s", key)
                    continue

                prepared_dict: dict = self.prepareSampleVolume(row[pathColumn], interpolationOrder)

                volume: np.ndarray = prepared_dict.pop("volume")
                if validate is not None:
                    validate(n5File, key, volume)

                self.writeN5Dataset(n5File, key, volume, self.channelOf(row))

                logger.info("Wrote %s <- %s", key, row[pathColumn])


        self.sampleMapperDF[keyColumn] = keys.to_numpy()



    def createDatasetN5File(self) -> None:
        """Raw volumes first (base class), then each paired mask into the same group.

        Extends rather than replaces the base implementation — overriding it outright
        left the raw channel out of the N5 entirely.
        """
        # NOTE: First create volume N5 file
        super().createDatasetN5File()
        # NOTE: Then create mask N5 file
        self.__writeKeyMask__()

    def assertPairsWithRaw(self, n5File, maskKey: str, mask: Optional[np.ndarray]) -> None:
        """A mask must be voxel-aligned with the raw volume it shares a group with.

        *mask* is the array about to be written, or ``None`` when the mask is already in
        the N5 — the pairing then holds between two stored datasets, and both shapes come
        from metadata alone.
        """
        rawKey = f"{maskKey.rpartition('/')[0]}/{self.RAW_KEY}"
        if not self.containsKey(n5File, rawKey):
            raise KeyError(f"No raw volume at {rawKey} to pair {maskKey} with")
        rawShape = tuple(n5File[rawKey].shape)  # metadata only — no voxels are read
        maskShape = tuple(n5File[maskKey].shape if mask is None else mask.shape)
        if maskShape != rawShape:
            raise ValueError(
                f"Mask {maskKey} has shape {maskShape} but its raw volume has {rawShape}")

    # TODO: Is the above needed ????
    # def _ensure_open(self) -> None:
    #     """Open the N5 file handle lazily so the object stays picklable."""
    #     if self._n5_file is None:
    #         if self.chunkedImagePath is None:
    #             raise ValueError("chunkedImagePath must be set before reading volumes.")
    #         self._n5_file = z5py.File(self.chunkedImagePath, 'r')
    #         self.Volume = self._n5_file[self.dapiKey + "/" + self.rawKey]
    #         self.Mask = self._n5_file[self.dapiKey + "/" + self.maskKey]


    # ---- Reading -----------------------------------------------------
    # The mask-dependent half of the accessors; the raw-only ones sit on
    # BaseVolumeDataset, which has no mask to read.

    def get_hr_mask(self, idx: int) -> np.ndarray:
        """Instance mask of sample *idx*, isolated to *label* when one is given.

        With *label*, every voxel not belonging to it is zeroed, so the result names one
        object; without, the full label volume comes through unchanged.
        """
        Mask = self._ensure_open(idx, self.N5_MASK_KEY_COLUMN)
        return Mask[:]

    def get_masked_hr_vol(self, idx: int) -> np.ndarray:
        """Raw volume of sample *idx* with everything outside the mask zeroed out."""
        vol: np.ndarray = self.get_hr_vol(idx)
        mask: np.ndarray = self.get_hr_mask(idx)
        return vol * (mask != 0)

    def maskOfPath(self, sampleVolumePath: str, channel: Optional[str] = None) -> np.ndarray:
        """Mask of the sample at *sampleVolumePath*, addressed by key rather than row.

        Index-building runs before the mapper carries key columns, so it cannot address
        samples positionally; the key is pure path arithmetic and needs no lookup.
        """
        return np.asarray(
            self.openKey(self.datasetKey(sampleVolumePath, self.MASK_KEY, channel))[:])
    #
    # def prepareVolume(self, volumePath: str, maskPath: Optional[str] = None,
    #                   resolution: Optional[List[int]] = None,
    #                   ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Load a raw/mask TIF pair and pre-process it, ready for N5 saving.
    #
    #     1. Load the raw TIF.
    #     2. Resample to isotropic voxels using *resolution* (ZYX) — order 3 for the raw
    #        intensities, order 0 (nearest neighbour) for the label mask, so labels survive.
    #     3. Min-max normalise the raw volume to ``[0, 1]``.
    #
    #     *maskPath* may be ``None`` or missing, in which case an all-zero mask matching the
    #     resampled shape is returned — what raw-only consumers such as SimCLR need.
    #
    #     TODO: overlaps BaseVolumeDataset.prepareSampleVolume, which does the same work but
    #     yields uint16 rather than float32 [0, 1]. This one survives because it serves the
    #     per-volume LMTextureNucleiDataset.volume2DF pipeline, not the N5 dataset classes;
    #     collapse the two once that pipeline moves onto BaseVolumeDataset.
    #     """
    #     from sklearn import preprocessing
    #
    #     resolution = list(resolution or self.resolution)
    #     volume: np.ndarray = self.loadFromSamplePath(volumePath)
    #
    #     zoom = np.array(resolution, dtype=float) / min(resolution)
    #     isotropic = all(z == 1.0 for z in zoom)
    #     logger.debug("Pre-resampled dimension: %s", volume.shape)
    #
    #     if not isotropic:
    #         newShape = tuple(int(s * z) for s, z in zip(volume.shape, zoom))
    #         volume = resize(volume, newShape, anti_aliasing=True, order=3)
    #         logger.debug("Resampled dimension: %s", volume.shape)
    #
    #     volume = (preprocessing.MinMaxScaler()
    #               .fit_transform(volume.reshape(-1, 1))
    #               .reshape(volume.shape)
    #               .astype(np.float32))
    #
    #     if maskPath and os.path.exists(maskPath):
    #         mask: np.ndarray = self.loadFromSamplePath(maskPath)
    #         if not isotropic:
    #             nLabels = np.unique(mask).size
    #             mask = resize(mask, volume.shape, anti_aliasing=True, order=0)
    #             if np.unique(mask).size != nLabels:
    #                 raise ValueError(
    #                     "Label count changed during mask resampling: "
    #                     f"{nLabels} -> {np.unique(mask).size}"
    #                 )
    #     # TODO: Do not save an empty mask if no masks exist instead raise error
    #     else:
    #         mask = np.zeros(volume.shape, dtype=np.uint16)
    #
    #     return volume, mask.astype(np.uint16)

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
    volumePathColumn, maskPathColumn : str
        Mapper columns holding the raw volume path and the mask path.  Required: they
        are the minimum a sample needs, and defaulting them would silently overwrite the
        column names a subclass declares at class level.
    channelColumn : str, optional
        Mapper column holding the channel.  ``None`` puts every sample under
        :attr:`channelKey`.
    """

    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 channelColumn: str = "channel",
                 **kwargs) -> None:
        # Assigned before super(): its __init__ validates the paths and runs the whole
        # N5 conversion off these columns.
        self.samplePathColumn = BaseDataset.SAMPLE_PATH_COLUMN
        self.SAMPLE_MASK_PATH_COLUMN = BaseMaskDataset.SAMPLE_MASK_PATH_COLUMN
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         channelColumn=channelColumn, **kwargs)

    def validateSamplePaths(self) -> None:
        """Replaces the single-directory contract of the base class with a wider one.

        Samples may live anywhere as long as the mapper is complete, every file exists
        and the volumes share a common root for keys to diverge from.
        """
        # channelColumn is optional — channelOf falls back to channelKey without it — so
        # a None must not be looked up as if it were a column name.
        required = [column for column
                    in (self.samplePathColumn, self.SAMPLE_MASK_PATH_COLUMN,
                        self.channelColumn)
                    if column]
        missingColumns = [column for column in required
                          if column not in self.sampleMapperDF.columns]
        if missingColumns:
            raise KeyError(
                f"Sample mapper is missing {missingColumns}; "
                f"it has {list(self.sampleMapperDF.columns)}")

        for column in (self.samplePathColumn, self.SAMPLE_MASK_PATH_COLUMN):
            missing = [str(path) for path in self.sampleMapperDF[column]
                       if not os.path.isfile(str(path))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} {column!r} entries are not files, first: {missing[0]}")

        logger.info("Indexing %d samples from %d directories under %s",
                    len(self.sampleMapperDF),
                    self.sampleMapperDF[self.samplePathColumn].map(os.path.dirname).nunique(),
                    self.sampleRoot)


class BaseObjectDataset(MultiDirectoryN5Dataset):
    """The base class dealt with samples, which where volumes. Here,
    this holds but the class also works with objects which exist in samples.
    This class can process multi directories but therefore cannot use labels to retrieve objects, instead it is the dataframe index"""

    objectMapperDF : Optional[Union[pd.DataFrame, list]] = []
    BBOX_COLUMNS = BBOX_COLUMNS
    BBOX_SLICE_COLUMN = "bbox_slices"
    OBJECT_LABEL_COLUMN = MOBIE_LABEL_KEY
    OBJECT_DF_PATH_COLUMN = "objectDF_path"

    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 channelColumn: str = "channel",
                 **kwargs) -> None:
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         channelColumn=channelColumn, **kwargs)

        self.sampleMapperDF = SampleLoaderBioImage.loadData(self.sampleMapperOutPath)

    # TODO: Label logic we could transfer to a torch.Dataset class ??? Which would ensure this class
    # TODO: have had processed to N5 beforehand ???


    # TODO: Maybe add method property to dynamically find objectMapperDF using self.datasetDir ?



    def __concatAllObjectBBOX__(self):

        # TODO: Need to test

        bboxPaths : list[str] = glob.glob(os.path.join(
            self.objectDFDir, "*", "**", "bbox_*.json"), recursive=True)

        assert bboxPaths.__len__() == self.sampleMapperDF.shape[0], f"Instead we have {bboxPaths.__len__()}/{self.sampleMapperDF.shape[0]}"

        objectMapperDF : list[pd.DataFrame] = [SampleLoaderBioImage.loadData(path) for path in bboxPaths]

        self.objectMapperDF: pd.DataFrame = self.sampleDF2ObjectDFMerge(objectDF=pd.concat(objectMapperDF, axis=0),
                                                                        sampleDF=self.sampleMapperDF, on=[self.N5_MASK_KEY_COLUMN,
                                                                                                          BaseDataset.SAMPLE_LABEL_COLUMN])

        self.objectMapperDF.to_json(self.objectDFPath)

    @NotImplementedError
    def __obtainAllObjectBBOX__(self):
        # TODO: Need to test

        keys = pd.Index(
            [self.datasetKey(row[self.SAMPLE_PATH_COLUMN], channel=self.channelOf(row), subkey=self.MASK_KEY)
             for _, row in self.sampleMapperDF.iterrows()])

        for key, (idx, row) in zip(keys, self.sampleMapperDF.iterrows()):

                if key in self.sampleMapperDF[self.N5_MASK_KEY_COLUMN].tolist():
                    mask = super().get_hr_mask(idx)
                    # NOTE: Only can compute after resampled volume. If placed before, computation will be wrong
                    dataFrameObject: pd.DataFrame = mask2BBOXDF(mask)
                    # NOTE: Save each key to map objects to samples
                    # TODO: Maybe use the checkpoint method instead ?????
                    dataFrameObject = self.registerSample2SingleVolumeObjectDF(dataFrameObject,
                                                                               self.N5_MASK_KEY_COLUMN, key,
                                                             sample_row=row)
                    self.objectMapperDF.append(dataFrameObject)
                else:
                    print(f"{key} not in column {self.N5_MASK_KEY_COLUMN} instead we have {row[self.N5_MASK_KEY_COLUMN]}")

        self.objectMapperDF: pd.DataFrame = self.sampleDF2ObjectDFMerge(objectDF=pd.concat(self.objectMapperDF, axis=0),
                                                                        sampleDF=self.sampleMapperDF, on=[self.N5_MASK_KEY_COLUMN,
                                                                                                          BaseDataset.SAMPLE_LABEL_COLUMN])


        self.removeSegmentationErrors()

        self.showTestVolumes()

    def removeSegmentationErrors(self):

        temp: list[pd.DataFrame] = []

        for channel, group in self.objectMapperDF.groupby(by=self.channelColumn):
            group = clean_dataframe(group, show_hist=False)
            temp.append(group)
            

        self.objectMapperDF = pd.concat(temp, ignore_index=True)

    def showTestVolumes(self):
        fig, ax = plt.subplots(1, 2)
        rng = np.random.default_rng(42)
        object_index = rng.choice(
            self.objectMapperDF.index[
                self.objectMapperDF[self.channelColumn].str.contains("Nuclei", na=False)
            ].to_numpy()
        )

        hr_vol = self.get_hr_vol(object_index)
        masked_hr_vol = self.get_masked_hr_vol(object_index)
        mid_slice = hr_vol.shape[0] // 2
        CustomMatplotlib.subImshow(ax[0], hr_vol[mid_slice], label="Raw object volume", vmin=0, vmax=hr_vol[mid_slice].max())
        CustomMatplotlib.subImshow(ax[1], masked_hr_vol[mid_slice], label="Masked raw object volume", vmin=0, vmax=masked_hr_vol[mid_slice].max())


    @cached_property
    def objectDFDir(self):
        p = os.path.join(self.datasetDir, "tables") #training_processed/tables/
        os.makedirs(p, exist_ok=True)
        return p

    @cached_property
    def objectDFPath(self) -> str:
        """Return the path to the cleaned nucleus-table JSON file.
        """
        return util.get_savedf_path(self.objectDFDir, typie=self.datasetDirName)


    def experimentRelativeDir(self, samplePath)->str:
        sampleSubDir = os.path.dirname(samplePath).replace(self.sampleRoot, "")  # ......\condition
        return sampleSubDir

    def singVolumeObjectDFPath(self, sample_idx : int, samplePath :str):

        sampleFileName = os.path.splitext(os.path.basename(samplePath))[0]

        dirs = f"{self.objectDFDir}{self.experimentRelativeDir(samplePath)}"
        path = util.get_savedf_path(dirs,
                                    typie=sampleFileName) # #training_processed/tables/single_volume/bbox_sampleFileName.json

        self.sampleMapperDF.loc[sample_idx, self.OBJECT_DF_PATH_COLUMN] = path

        return path

    def registerSample2SingleVolumeObjectDF(self, dataFrameObject, keyColumn, n5VolumeKey, sample_row : pd.Series):
        dataFrameObject[keyColumn] = n5VolumeKey
        dataFrameObject[BaseDataset.SAMPLE_LABEL_COLUMN] = sample_row.name
        return dataFrameObject



    def checkpointSingleVolumeObjectDF(self, dataFrameObject: pd.DataFrame, keyColumn, n5VolumeKey, pathColumn, sample_row : pd.Series) -> pd.DataFrame:
        # TODO: Later, create a function gathering all the dataframes if os.path.exists(self.objectDFPath)
        dataFrameObject = self.registerSample2SingleVolumeObjectDF(dataFrameObject, keyColumn, n5VolumeKey, sample_row)
        dataFrameObject.to_json(self.singVolumeObjectDFPath(int(sample_row.name), sample_row[pathColumn]))

        return dataFrameObject

    def __writeKeyMask__(self) -> None:
        """Write one dataset per sample under *subkey*; record its key in *keyColumn*.

        Keys always derive from the sample *volume* path, so every role of a sample
        (raw, mask, ...) lands in that sample's one group whatever column the pixels
        came from.  Samples already present are skipped, which is what makes
        construction idempotent and an interrupted run resumable.  The N5 is opened
        once for the whole column.
        """


        pathColumn = self.SAMPLE_MASK_PATH_COLUMN
        subkey= self.MASK_KEY
        keyColumn= self.N5_MASK_KEY_COLUMN
        validate = self.assertPairsWithRaw
        interpolationOrder = self.MASK_INTERPOLATION_ORDER


        if pathColumn not in self.sampleMapperDF.columns:
            raise KeyError(
                f"Sample mapper has no {pathColumn!r} column: "
                f"{list(self.sampleMapperDF.columns)}")

        # Keys are pure path arithmetic, so they are derived — and checked for
        # collisions — before a single voxel is written and overwrites its neighbour.
        # NOTE: Constructed from sample path, channel and volume type (e.g. raw, mask etc.) indicator subkey
        keys = pd.Index([self.datasetKey(row[self.SAMPLE_PATH_COLUMN], channel=self.channelOf(row), subkey=subkey)
                         for _, row in self.sampleMapperDF.iterrows()])

        # Note: Test if duplicated keys exist
        duplicates = keys[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Sample mapper maps several rows onto the same N5 key: {duplicates[:5]}")

        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:

            for key, (i, row) in zip(keys, self.sampleMapperDF.iterrows()):
                if self.containsKey(n5File, key):
                    # Still validated: the check reads metadata only, and skipping it on
                    # resume meant an existing dataset was never checked against its pair.
                    if validate is not None:
                        validate(n5File, key, None)
                    logger.debug("Already in N5, skipping: %s", key)
                    continue
                print(f"Writing {key} of indx {i} keys to N5 file {self.n5Path}")
                prepared_dict: dict = self.prepareSampleVolume(row[pathColumn], interpolationOrder)

                volume: np.ndarray = prepared_dict.pop("volume")

                # NOTE: Only can compute after resampled volume. If placed before, computation will be wrong
                dataFrameObject: pd.DataFrame = mask2BBOXDF(volume)

                if validate is not None:
                    validate(n5File, key, volume)

                self.writeN5Dataset(n5File, key, volume, self.channelOf(row))

                logger.info("Wrote %s <- %s", key, row[pathColumn])

                # NOTE: Save each key to map objects to samples
                self.objectMapperDF.append(self.checkpointSingleVolumeObjectDF(dataFrameObject,
                                                                      keyColumn, key, pathColumn, row))

        self.sampleMapperDF[keyColumn] = keys.to_numpy()

        self.objectMapperDF: pd.DataFrame = self.sampleDF2ObjectDFMerge(objectDF=pd.concat(self.objectMapperDF, axis=0),
                                                                        sampleDF=self.sampleMapperDF, on=[keyColumn, BaseDataset.SAMPLE_LABEL_COLUMN])

        self.removeSegmentationErrors()

    @staticmethod
    def sampleDF2ObjectDFMerge(objectDF, sampleDF, on : list[str]) -> pd.DataFrame:
        if sampleDF.columns.isin(BaseObjectDataset.BBOX_COLUMNS).any():
            sampleDF.drop(columns=BaseObjectDataset.BBOX_COLUMNS, inplace=True)

        return objectDF.merge(sampleDF, on=on
                                  ,how="left",
                                  validate="many_to_one", suffixes=("","_sample"))

    def createDatasetN5File(self) -> None:
        super().createDatasetN5File()
        self.objectMapperDF.to_json(self.objectDFPath)


    def objectDFRow(self, idx: int) -> pd.Series:
        """Positional row lookup.

        DataLoader samplers yield positions, not index labels, so every ``__getitem__``
        path must use ``.iloc``. Mixing ``.loc`` in is only correct while the index
        happens to be a clean RangeIndex.
        """
        return self.objectMapperDF.iloc[int(idx)]

    def get_hr_vol(self, idx: int) -> np.ndarray:
        """Raw volume of sample *idx*, or only *bbox* (a tuple of slices) when given."""
        # TODO: objectDFRow and getBBOXSlice goes through objectMapperDF twice needlessly
        print(f"get_hr_vol: {idx}")
        sample_idx : int = self.objectDFRow(idx)[self.SAMPLE_LABEL_COLUMN]
        Volume = self._ensure_open(sample_idx, self.N5_VOLUME_KEY_COLUMN)
        bbox = self.getBBOXSlice(self.objectMapperDF, idx)
        return np.asarray(Volume[bbox])

    def get_hr_mask(self, idx: int) -> np.ndarray:
        """Instance mask of sample *idx*, isolated to *label* when one is given.

        With *label*, every voxel not belonging to it is zeroed, so the result names one
        object; without, the full label volume comes through unchanged.
        """
        sample_idx: int = self.objectDFRow(idx)[self.SAMPLE_LABEL_COLUMN]
        object_label: int = self.objectDFRow(idx)[MOBIE_LABEL_KEY]
        Mask = self._ensure_open(sample_idx, self.N5_MASK_KEY_COLUMN)
        bbox = self.getBBOXSlice(self.objectMapperDF, idx)
        return (Mask[bbox] == object_label).astype(np.int16)


    def run(self) -> None:
        super().run()
        self.showTestVolumes()


    @staticmethod
    def getAllLabels(df)-> np.ndarray[Any, np.int16]:
        return df[BaseObjectDataset.OBJECT_LABEL_COLUMN].dropna().to_numpy().astype(np.int16).flatten()

    @staticmethod
    def dfBBOXIndex2Label(df, df_indx) -> int:
        indices = df.index.to_numpy()
        if df_indx in indices:
            out = df.loc[df_indx, BaseObjectDataset.OBJECT_LABEL_COLUMN]
            if isinstance(out, pd.Series):
                return int(out.iloc[0])
            return out
        else:
            raise KeyError(f"Label {df_indx} does not exist in dataset")



    @staticmethod
    def label2DFBBOXIndex(df, label)-> int:
        # TODO: Need to adjust for MultiDirectory object dataframe
        labels = BaseObjectDataset.getAllLabels(df)
        if label in labels:
            assert (labels == label).sum() == 1, f"The label {label} is more than once ({(labels == label).sum()}) in the object dataframe"
            return df.loc[df[BaseObjectDataset.OBJECT_LABEL_COLUMN] == label, :].index.tolist()[0]
        else:
            raise KeyError(f"Label {label} does not exist in dataset")


    @staticmethod
    def transformBBOXDF2Slice(df: pd.DataFrame, df_indx):
        b = getBBOXFromDF(df, df_indx).flatten()
        return (slice(b[0], b[1]), slice(b[2], b[3]), slice(b[4], b[5]))

    @staticmethod
    def getArrayFromBBOXDF(df: pd.DataFrame, df_indx) -> np.ndarray:

        # Extract and ensure numerical values
        bbox_values = [df.loc[df_indx, dim] for dim in BaseObjectDataset.BBOX_COLUMNS]

        if not all(isinstance(val, numbers.Number) for val in bbox_values):
            raise TypeError("All bounding box values must be numeric, but got: "
                            f"{[type(val) for val in bbox_values]}")

        return np.array(bbox_values, dtype=int).T  # Convert to NumPy int array

    @staticmethod
    def arrayBBOX2Slice(bbox: np.ndarray) -> tuple[slice, slice, slice]:
        """Convert a flattened bounding-box array into a tuple of slices.

        Parameters
        ----------
        bbox : np.ndarray
            1-D (or single-row 2-D) array of six integers ordered as
            ``[start_z, stop_z, start_y, stop_y, start_x, stop_x]``.

        Returns
        -------
        tuple of slice
            ``(slice_z, slice_y, slice_x)`` ready to index a 3-D array.

        Raises
        ------
        AssertionError
            If *bbox* has more than one row.
        """
        assert bbox.ndim < 2 or bbox.shape[0] == 1, "Too many dims in bbox"
        bbox = bbox.flatten()
        return (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]),
                slice(bbox[4], bbox[5]))

    @staticmethod
    def getBBOXSlice(df: pd.DataFrame,
                     idx: Optional[int] = None) -> Tuple[slice, slice, slice]:
        """Return the bounding-box slice tuple for *nucl_label* from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Nucleus table with bounding-box columns.
        label : int, optional
            Instance label of the nucleus.

        Returns
        -------
        tuple of slice
            ``(slice_z, slice_y, slice_x)`` for the nucleus bounding box.
        """
        b = getBBOXFromDF(df, idx).flatten()
        return slice(b[0], b[1]), slice(b[2], b[3]), slice(b[4], b[5])


    # @staticmethod
    # def addBBOXSliceCol(table: pd.DataFrame) -> pd.DataFrame:
    #     """Append a ``"bbox slices"`` column of per-nucleus slice tuples.
    #
    #     If any of the expected bounding-box columns are absent from *table*,
    #     ``merge_with_nucl_table`` is called first to populate them.
    #
    #     Parameters
    #     ----------
    #     table : pd.DataFrame
    #         Nucleus table.  Expected bounding-box columns:
    #         ``bb_min_z``, ``bb_max_z``, ``bb_min_y``, ``bb_max_y``,
    #         ``bb_min_x``, ``bb_max_x``.
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #         *table* with an added ``"bbox slices"`` column whose values are
    #         ``(slice_z, slice_y, slice_x)`` tuples at s3 resolution.
    #     """
    #
    #     # Merge nucleus table if any bounding-box columns are absent
    #     missing_cols: List[str] = [col for col in BaseObjectDataset.BBOX_COLUMNS if col not in table.columns]
    #     assert not missing_cols, "The bounding box coordinates are incomplete"
    #
    #     bbs: List[Tuple[slice]] = [BaseObjectDataset.getBBOXSlice(table, BaseObjectDataset.dfBBOXIndex2Label(table, row.name)) for _, row in table.iterrows()]
    #
    #     return util.save2DFcolumn(bbs, BaseObjectDataset.getAllLabels(table), table)
    #
    #
























if __name__ == "__main__":

    save_dir = r"E:\Project 2\Neurospheres\Ihssane\data"
    sampleMapperDFPath = os.path.join(save_dir, "sample_mapper_crop_cleaned.csv")

    inst = BaseObjectDataset(sampleMapperDFPath, save_dir)
    inst.run()
