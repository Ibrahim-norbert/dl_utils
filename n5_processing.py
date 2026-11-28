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

import logging
import numbers
import os
from functools import cached_property
from typing import Any, Callable, List, Optional, Tuple, Union, Hashable

import numpy as np
import pandas as pd
import z5py
from joblib import Parallel, delayed
from IPython.core.oinspect import object_info
from matplotlib import pyplot as plt
from numpy import ndarray, dtype
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from xarray.util.generate_ops import inplace

from . import img2patch as img2ps, util, MOBIE_LABEL_KEY
from .SampleLoader import SampleLoaderBioImage
from .datasets import BaseDataset
from .util import getArrayFromDF
from .util_base import save2DFcolumn, get_savedf_path
from .vizualizations import CustomMatplotlib
import fsspec


logger = logging.getLogger(__name__)

SIGMA_THRESHOLD = 3
CHUNK_MULTIPLE = 8
BBOX_COLUMNS = ["bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x"]


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

    # NOTE: The per-axis mean/stddev are printed above, never stored. They used to be kept
    # as `df.loc[f"{column}_length_mean"] = mean`, which is row enlargement, not column
    # assignment: it appended a row whose *every* column — sample_label_id, label_id,
    # channel, the bbox columns — held that scalar. Those rows kept the stat in `column`
    # too, so they passed the filter below, survived clean_dataframe's reset_index and
    # reached objectMapperDF as ordinary-looking objects.

    # Define the range for filtering based on the bounds
    return df.loc[df[column].between(lower_bound, upper_bound), :]


def mask2BBOXDF(maskVol: np.ndarray, image: Optional[np.ndarray] = None, features: tuple[str] = ()) -> pd.DataFrame:
    all_boxes: img2ps.BBoxes = img2ps.BBoxes.from_mask(maskVol, image, features)

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
    """
    N5 dataset folder created for volumes living in self.sampleRoot.
    Each volume is preprocessed through resampling before saved with
    unique N5 key. The sampleMapperDF is updated with the keys and overridden in disk.

    The samples are not normalized after interpolation.

    """



    # NOTE: Coloumn where all N5 keys are saved
    CHANNEL_COLUMN : str = "channel" # Ensure this exists in samplemapper dataframe
    N5_KEY_COLUMN: str = "volume_key"
    DATASET_FILE_EXTENSION: str = ".n5"
    # Only these are stripped when deriving a key, so `img.ome.tif` keeps its `.ome`.
    VOLUME_FILE_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff")
    N5_RAW_KEY: str = "raw"
    # Bicubic for intensities; masks override with nearest-neighbour (see BaseMaskDataset).
    RAW_INTERPOLATION_ORDER: int = 3
    N5_CHUNKING: Tuple[int, int, int] = (32, 32, 32)
    DEFAULT_RESOLUTION: Tuple[int, int, int] = (3, 1, 1)

    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 resolution: Optional[List[int]] = None,
                 n_jobs: int = 6,
                 **kwargs) -> None:
        # Assigned before super(), whose __init__ validates the sample paths — the
        # multi-directory subclass validates against these very attributes.
        self.resolution: List[int] = list(resolution or self.DEFAULT_RESOLUTION)
        # joblib convention: 1 = serial, -1 = every allocated CPU. Each worker holds one
        # TIF plus its float64 resampled copy, so RAM bounds this, not cores.
        self.n_jobs: int = n_jobs
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         **kwargs)
        # Mappers written on Windows carry `\` separators, which POSIX treats as part of a
        # filename — collapsing the hierarchy into one key segment. Normalised before
        # anything (sampleRoot, keys, loads) reads the paths.
        for column in self.pathColumns():
            self.sampleMapperDF[column] = self.sampleMapperDF[column].map(
                lambda path: str(util.fix_path(path)))
        assert self.CHANNEL_COLUMN in self.sampleMapperDF.columns, f"The {self.CHANNEL_COLUMN} column is not in the sampleMapper dataframe."

        assert os.path.basename(self.save_dir) == self.sampleDatasetName, f"The saving directory must have the same basename as the dataset name: {os.path.basename(self.save_dir)} == {self.sampleDatasetName}"
        assert self.sampleMapperDF[self.SAMPLE_PATH_COLUMN].nunique() == self.sampleMapperDF.shape[0], (
            f"The sample mapper must have unique paths for column: {self.SAMPLE_PATH_COLUMN}"
        )
        # Keys are pure path arithmetic: every instance derives them, so none has to read
        # them back from a mapper that run() would otherwise overwrite.
        self.sampleMapperDF[self.N5_KEY_COLUMN] = self.generateN5keys(
            self.SAMPLE_PATH_COLUMN, self.N5_RAW_KEY).to_numpy()

    def pathColumns(self) -> Tuple[str, ...]:
        """Mapper columns holding file paths."""
        return (self.SAMPLE_PATH_COLUMN,)

    def generateN5keys(self, pathColumn : str, sampletypekey: str) -> pd.Index:
        keys = pd.Index([self.sampleN5Key(row[pathColumn], sampletypekey=sampletypekey, channel=self.channelOf(row))
                         for _, row in self.sampleMapperDF.iterrows()])

        # Note: Test if duplicated keys exist
        duplicates = keys[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Sample mapper maps several rows onto the same N5 key: {duplicates[:5]}")

        return keys

    def run(self) -> None:
        """
        1. Creates directory in self.save_dir.
        2. Creates N5 folder from volumes living in self.sampleRoot
        3. Writes the key-annotated mapper to :attr:`sampleMapperOutPath`. The input mapper
           is left untouched: a subclass may have filtered it (e.g. to one channel).
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

        assert directories, (f"Sample mapper {self.sampleMapperDFPath} has no rows, so there is no "
                f"common root to derive N5 keys from")

        try:
            return os.path.commonpath(directories)
        except ValueError as error:  # different drives, or absolute mixed with relative
            raise ValueError(
                f"Sample paths share no common root: {sorted(set(directories))}") from error

    @cached_property
    def sampleDatasetName(self) -> str:
        """Dataset name — the common root's basename unless one was given explicitly."""
        # TODO: In another update provide method to save corresponding N5 directories next to deepest volume directories
        return os.path.basename(self.sampleRoot)

    @cached_property
    def datasetDir(self) -> str:
        """Artefact root for this dataset — resolved only; :meth:`run` creates it."""
        return os.path.join(self.save_dir, "training_processed")

    @cached_property
    def n5Path(self) -> str:
        return os.path.join(self.datasetDir,
                            f"{self.sampleDatasetName}{self.DATASET_FILE_EXTENSION}")

    @cached_property
    def sampleMapperOutPath(self) -> str:
        """The key-annotated sample mapper, persisted beside the N5 it indexes."""
        return os.path.join(self.datasetDir, f"{self.sampleDatasetName}_sample_mapper.csv")

    def getSampleDatasetSubDir(self, sampleVolumePath: str, *subdirs: str) -> str:
        """Per-sample artefact directory: :attr:`save_dir` plus the sample's own path.

        Derived from the same diverging path as the sample's N5 key, so artefacts mirror
        the N5 hierarchy and two volumes sharing a basename in different directories
        cannot overwrite each other's outputs.  Created on demand.
        """
        path = os.path.join(self.save_dir,
                            *self.samplePathMappingN5Key(sampleVolumePath).split("/"), *subdirs)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def confirmDatasetN5File(self) -> bool:
        return os.path.exists(self.n5Path)

    # ---- Keys --------------------------------------------------------

    def samplePathMappingN5Key(self, sampleVolumePath: str) -> str:
        """
        The N5 sample key is simply the relative path from
        the self.sampleRoot.

        """
        path = os.path.abspath(str(sampleVolumePath))
        stem, extension = os.path.splitext(path)
        assert extension.lower() in self.VOLUME_FILE_EXTENSIONS, (f"Expected one of {self.VOLUME_FILE_EXTENSIONS} but sample at following path has instead {extension}:"
                                                                  f"{stem}")
        return os.path.relpath(stem, self.sampleRoot).replace(os.sep, "/")

    def channelOf(self, row: pd.Series) -> str:
        """Channel of a sample mapper row, falling back to the dataset-wide channel.

        The channel becomes one key segment, so a value carrying ``/`` would silently
        deepen the hierarchy and move the sample somewhere nobody addresses.
        """
        channel : str  = str(row.get(self.CHANNEL_COLUMN))
        if "/" in channel or not channel.strip():
            raise ValueError(
                f"Channel {channel!r} is not a usable N5 key segment: it must be "
                f"non-blank and free of '/'")
        return channel

    def getN5GroupKey(self, sampleVolumePath: str, channel: Optional[str] = None) -> str:
        """Group holding every role (raw, mask, ...) of a single FOV."""
        return f"{self.samplePathMappingN5Key(sampleVolumePath)}/{channel}"

    def sampleN5Key(self, sampleVolumePath: str, sampletypekey: str, channel: Optional[str] = None) -> str:
        return f"{self.getN5GroupKey(sampleVolumePath, channel)}/{sampletypekey}"

    # ---- Writing -----------------------------------------------------

    # @staticmethod
    # def toUint16(volume: np.ndarray, normalise: bool) -> np.ndarray:
    #     """Cast to the storage dtype, min-max stretching intensities when *normalise*.
    #
    #     Label volumes are never stretched: their values are identities, not intensities.
    #     """
    #     ceiling = np.iinfo(np.uint16).max
    #     if normalise:
    #         # TODO: Just use Sklearn minmax but only on raw image not any masks
    #         low, high = float(volume.min()), float(volume.max())
    #         volume = ((volume - low) / (high - low) if high > low
    #                   else np.zeros_like(volume)) * ceiling
    #     if volume.min() < 0 or volume.max() > ceiling:
    #         raise ValueError(
    #             f"Volume range [{volume.min()}, {volume.max()}] does not fit uint16")
    #     return volume.astype(np.uint16)

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

        print("Pre-resampled dimension: %s", volume.shape)

        if all(res == 1 for res in resolutions):
            print("Isotropic resolution detected. Skipping resizing.")
            return volume, min_res

        newShape = tuple(int(round(size * res)) for size, res in zip(volume.shape, resolutions))
        dtype = volume.dtype
        volume = resize(volume, newShape, order=interpolationOrder,
                        preserve_range=True, anti_aliasing=False)
        if np.issubdtype(dtype, np.integer):
            volume = np.rint(volume)  # clip=True already bounds it to the input range
        volume = volume.astype(dtype)
        print("Resampled dimension: %s", volume.shape)
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

    def preprocessVolume(self, sampleVolumePath: str, interpolationOrder: int) -> dict:
        """Load a sample and bring it into storage form: isotropic voxels, ``uint16``.

        *interpolationOrder* carries the role, following the convention this codebase
        uses everywhere: 3 (bicubic, anti-aliased) for intensity volumes, which are
        min-max stretched over the uint16 range.

        """

        volume: np.ndarray = self.loadFromSamplePath(str(sampleVolumePath))

        resampled, _ = self.resampleVolume(volume, self.resolution, interpolationOrder)

        return {"volume": resampled}

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
        groupKey, _, subkey = key.rpartition("/") # (subDir, seperator, type key)

        dataset = self.requireGroup(n5File, groupKey).require_dataset(
            subkey,
            shape=volume.shape,
            # A chunk may not exceed the axis it chunks — small volumes are one chunk.
            chunks=tuple(min(chunk, size) for chunk, size in zip(self.N5_CHUNKING, volume.shape)),
            dtype=volume.dtype,
        )
        # TODO: require_dataset writes metadata before this fill, so a crash here leaves
        # an empty dataset that containsKey skips on the next run. Accepted for now.
        dataset[:] = volume
        dataset.attrs["channel"] = channel

    def writeKeyVolume(self, key: str, path: str, order: int, channel: str,
                       validate: Optional[Callable] = None) -> None:
        """Preprocess and write one sample, unless *key* is already in the N5.

        *validate* ``(n5File, key, volume)`` still runs on a resumed key (with ``None``):
        it reads metadata only, and skipping it would leave an existing dataset unchecked.
        """
        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:
            exists = self.containsKey(n5File, key)
            volume = None if exists else self.preprocessVolume(path, order)["volume"]
            if validate is not None:
                validate(n5File, key, volume)
            if exists:
                logger.debug("Already in N5, skipping: %s", key)
                return
            self.writeN5Dataset(n5File, key, volume, channel)
            logger.info("Wrote %s <- %s", key, path)

    def __writeKeys__(self, pathColumn: str, order: int, keyColumn: str,
                      validate: Optional[Callable] = None) -> None:
        """Write every sample's *pathColumn* volume at its *keyColumn* key, ``n_jobs`` at a time.

        Keys always derive from the raw sample path, so every role (raw, mask, ...) of a
        sample lands in the same group whatever directory its own file lives in.
        Each sample writes only its own dataset directory; the groups above it are shared,
        so they are created here, up front, and the workers never race on them.
        """
        keys = self.sampleMapperDF[keyColumn]  # derived at construction
        with z5py.File(self.n5Path, "a", use_zarr_format=False) as n5File:
            for key in keys:
                self.requireGroup(n5File, key.rpartition("/")[0])
        # Processes (loky), not threads: TIF decoding does not reliably release the GIL.
        Parallel(n_jobs=self.n_jobs)(
            delayed(self.writeKeyVolume)(key, row[pathColumn], order, self.channelOf(row), validate)
            for key, (_, row) in zip(keys, self.sampleMapperDF.iterrows()))

    def __writeKeyVolumes__(self) -> None:
        self.__writeKeys__(self.SAMPLE_PATH_COLUMN, self.RAW_INTERPOLATION_ORDER,
                           self.N5_KEY_COLUMN)

    def createDatasetN5File(self) -> None:
        """Store the volumes as n5 files for RAM saving chunked based signal loading"""
        self.__writeKeyVolumes__()

    # ---- Reading -----------------------------------------------------

    def keyOf(self, idx: int, keyColumn: Optional[str] = None) -> str:
        return str(self.sampleDFRow(idx)[keyColumn or self.N5_KEY_COLUMN])

    def openKey(self, key: str):
        """...(docstring unchanged)..."""
        try:
            return z5py.File(self.n5Path, "r")[key]
        except KeyError as e:
            raise KeyError(f"'{key}' not found in {self.n5Path}") from e

    def _ensure_open(self, idx: int, keyColumn: Optional[str] = None):
        """Open the dataset of sample *idx*, addressed through the mapper's key column."""
        return self.openKey(self.keyOf(idx, keyColumn))

    def get_hr_vol(self, idx: int) -> np.ndarray:
        """Raw volume of sample *idx*, or only *bbox* (a tuple of slices) when given."""
        Volume = self._ensure_open(idx, self.N5_KEY_COLUMN)
        return np.asarray(Volume[:])


class BaseMaskDataset(BaseVolumeDataset):
    """Adds each sample's instance-segmentation mask to its volume's N5 group."""
    N5_MASK_KEY_COLUMN: str = "mask_key"
    SAMPLE_MASK_PATH_COLUMN: str = "mask_path"
    N5_MASK_KEY: str = "mask"
    # Nearest-neighbour: interpolating between label IDs invents objects.
    MASK_INTERPOLATION_ORDER: int = 0

    def pathColumns(self) -> Tuple[str, ...]:
        return super().pathColumns() + (self.SAMPLE_MASK_PATH_COLUMN,)
    # NOTE: No idea why we use this
    # Distinct storage levels of a raw volume. toUint16 stretches EVERY volume over this
    # full range, so it is the one bin grid every sample already shares. Kept apart from
    # toUint16's `ceiling`: that is a max (65535), this is a count.
    INTENSITY_LEVELS: int = np.iinfo(np.uint16).max + 1

    # TODO: Confirm after creating the N5 Files, we still require the Tiff File paths, if so make it agnostic
    def __init__(self, sampleMapperDFPath: str, save_dir: str,
                 features : tuple[str] = ("area",),
                 resolution: Optional[List[int]] = None,
                 **kwargs) -> None:
        super().__init__(sampleMapperDFPath=sampleMapperDFPath,
                         save_dir=save_dir, resolution=resolution,
                         **kwargs)
        self.features: tuple[str] = features
        self.sampleMapperDF[self.N5_MASK_KEY_COLUMN] = self.generateN5keys(
            self.SAMPLE_PATH_COLUMN, self.N5_MASK_KEY).to_numpy()



    def preprocessVolume(self, sampleVolumePath: str, interpolationOrder: int) -> dict:
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

        return {"volume": resampled}

    # def _statsFromMask(self, rawVol : np.ndarray, maskVol : np.ndarray, feaures: tuple[str]) -> pd.DataFrame:
    #     """Region statistics of **one** volume: a row per labelled object, or a single row for
    #     the hole-filled foreground as a whole when *globalMask*."""
    #     labels = binary_fill_holes(maskVol > 0)
    #     df : pd.DataFrame = BBOXDF(img2ps.BBoxes.from_mask(labels,
    #                                           rawVol,
    #                                           feaures))
    #     df.drop(columns=MOBIE_LABEL_KEY, inplace=True)
    #     return df

    def __writeKeyMask__(self) -> None:
        """Same as :meth:`__writeKeyVolumes__`, each mask checked against its raw volume."""
        self.__writeKeys__(self.SAMPLE_MASK_PATH_COLUMN, self.MASK_INTERPOLATION_ORDER,
                           self.N5_MASK_KEY_COLUMN,
                           validate=self.assertPairsWithRaw)




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
        rawKey = f"{maskKey.rpartition('/')[0]}/{self.N5_RAW_KEY}"
        if not self.containsKey(n5File, rawKey):
            raise KeyError(f"No raw volume at {rawKey} to pair {maskKey} with")
        rawShape = tuple(n5File[rawKey].shape)  # metadata only — no voxels are read
        maskShape = tuple(n5File[maskKey].shape if mask is None else mask.shape)
        if maskShape != rawShape:
            raise ValueError(
                f"Mask {maskKey} has shape {maskShape} but its raw volume has {rawShape}")

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
            self.openKey(self.sampleN5Key(sampleVolumePath, self.N5_MASK_KEY, channel))[:])

    def volumeOfPath(self, sampleVolumePath: str, channel: Optional[str] = None) -> np.ndarray:
        """Raw counterpart of :meth:`maskOfPath`, addressed by key rather than row."""
        return np.asarray(
            self.openKey(self.sampleN5Key(sampleVolumePath, self.N5_RAW_KEY, channel))[:])

    # ---- Intensity statistics ----------------------------------------

    def maskedIntensityDensity(self, idx: int, nBins: int = 256) -> Optional[np.ndarray]:
        """Density-normalised histogram of sample *idx*'s foreground (``mask != 0``) voxels.

        Goes through :meth:`volumeOfPath`/:meth:`maskOfPath`, never :meth:`get_hr_vol`:
        the object subclasses override those accessors to return one object's bbox crop
        off ``objectMapperDF``, so sample-level code calling them would histogram a crop
        out of a table with a different length.  Key arithmetic also needs no
        ``volume_key`` column, which only ``run()`` assigns.

        ``bincount`` over the full uint16 range makes the bin grid structural: every
        volume lands on the same levels, and a reshape-sum folds them into *nBins* wider
        bins with no edge arithmetic to drift.  ``None`` for an empty mask — a density
        over zero voxels is 0/0, not a curve.
        """
        row = self.sampleDFRow(idx)
        path, channel = row[self.SAMPLE_PATH_COLUMN], self.channelOf(row)

        # Mask first: ~60x smaller on disk, so an empty one costs no raw read.
        foreground = self.maskOfPath(path, channel) != 0
        if not foreground.any():
            logger.warning("Sample %d (%s) has an empty mask; excluded.", idx, path)
            return None

        # Not get_masked_hr_vol: it ZEROES the background rather than dropping it, which
        # would pile every background voxel into bin 0.
        counts = np.bincount(self.volumeOfPath(path, channel)[foreground],
                             minlength=self.INTENSITY_LEVELS).reshape(nBins, -1).sum(axis=1)
        return counts / (counts.sum() / nBins)   # integrates to 1 over a [0, 1] axis

    def showChannelIntensityHistograms(
        self, nBins: int = 256, maxSamplesPerChannel: Optional[int] = 50, seed: int = 42,
        logScale: bool = True, save_dir: Optional[str] = None,
        title: str = "Masked intensity distribution per channel",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Mean +/-1 std foreground-intensity density per channel, over sample volumes.

        Every volume is histogrammed on the shared uint16 grid and normalised to a
        density; the line is the **mean over that channel's volumes**, the band its
        standard deviation.  Not a pooled histogram: pooling weights each volume by its
        foreground voxel count, so one large sphere outvotes fifty small ones and the
        between-volume spread — the thing worth seeing — disappears.

        :meth:`~BaseVolumeDataset.toUint16` min-max stretches **each volume
        independently**, so the axis is each volume's own dynamic range: compare shape,
        never absolute value.  A spike at x~0 is a flat source volume, not background
        leakage, and on a log axis the symmetric band looks asymmetric.

        *save_dir* ``None`` returns a live figure, which is what renders inline in a
        notebook; passing a directory writes the SVG and closes it, since
        ``CustomMatplotlib.saveFig`` always closes.
        """
        if not self.confirmDatasetN5File:
            raise FileNotFoundError(f"No N5 at {self.n5Path}: run() it before histogramming")
        if self.INTENSITY_LEVELS % nBins:
            raise ValueError(f"nBins must divide {self.INTENSITY_LEVELS}, got {nBins}")

        rng = np.random.default_rng(seed)
        centres = (np.arange(nBins) + 0.5) / nBins
        # Positional throughout, like sampleDFRow: the mapper is not always a RangeIndex.
        channels = self.sampleMapperDF.apply(self.channelOf, axis=1).to_numpy()

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for channel in np.unique(channels):      # alphabetical -> stable legend order
            positions = np.flatnonzero(channels == channel)
            if maxSamplesPerChannel and positions.size > maxSamplesPerChannel:
                # Sorted so the N5 is walked in storage order rather than at random.
                positions = np.sort(rng.choice(positions, maxSamplesPerChannel, replace=False))

            stack = [d for d in (self.maskedIntensityDensity(int(p), nBins) for p in positions)
                     if d is not None]
            if not stack:
                logger.warning("Channel %r contributed no usable volume; skipped.", channel)
                continue
            stack = np.stack(stack)
            mean = stack.mean(axis=0)
            # ddof=1 — the volumes sample the acquisition population — but nan on one row.
            std = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros(nBins)
            logger.info("Channel %r: %d/%d volumes histogrammed.",
                        channel, len(stack), positions.size)

            # The band must reuse the line's colour: letting the cycler advance twice
            # pairs each band with the next channel's colour.  Clipped at zero, which a
            # log axis otherwise autoscales to nothing.
            line, = ax.plot(centres, mean, label=f"{channel} (n={len(stack)})")
            ax.fill_between(centres, np.maximum(mean - std, 0), mean + std,
                            color=line.get_color(), alpha=0.2, linewidth=0)

        if not ax.lines:
            raise ValueError(f"No channel of {self.sampleMapperDFPath} yielded a usable "
                             f"volume — every mask empty, or no rows at all")

        ax.set(xlabel="Intensity (per-volume min-max normalised)", ylabel="Density",
               title=title, xlim=(0, 1), yscale="log" if logScale else "linear")
        ax.legend(frameon=False)
        fig.tight_layout()
        if save_dir is not None:
            # saveFig closes unconditionally, so call it only when there is something to
            # save: a closed figure does not render inline.
            CustomMatplotlib.saveFig(fig, save_dir, title, self.showChannelIntensityHistograms)
        return fig, ax

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
                 features : tuple[str] = ("area",),
                 **kwargs) -> None:
        # Assigned before super(): its __init__ validates the paths and runs the whole
        # N5 conversion off these columns.
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         features=features,
                         **kwargs)

    # TODO: Below not ready yet as BaseVolume asserts all sample paths to exist in a root directory
    @NotImplementedError
    def validateSamplePaths(self) -> None:
        """Replaces the single-directory contract of the base class with a wider one.

        Samples may live anywhere as long as the mapper is complete, every file exists
        and the volumes share a common root for keys to diverge from.
        """
        # CHANNEL_COLUMN is optional — channelOf falls back to channelKey without it — so
        # a None must not be looked up as if it were a column name.
        required = [column for column
                    in (self.SAMPLE_PATH_COLUMN, self.SAMPLE_MASK_PATH_COLUMN,
                        self.CHANNEL_COLUMN)
                    if column]
        missingColumns = [column for column in required
                          if column not in self.sampleMapperDF.columns]
        if missingColumns:
            raise KeyError(
                f"Sample mapper is missing {missingColumns}; "
                f"it has {list(self.sampleMapperDF.columns)}")

        for column in (self.SAMPLE_PATH_COLUMN, self.SAMPLE_MASK_PATH_COLUMN):
            missing = [str(path) for path in self.sampleMapperDF[column]
                       if not os.path.isfile(str(path))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} {column!r} entries are not files, first: {missing[0]}")

        logger.info("Indexing %d samples from %d directories under %s",
                    len(self.sampleMapperDF),
                    self.sampleMapperDF[self.SAMPLE_PATH_COLUMN].map(os.path.dirname).nunique(),
                    self.sampleRoot)



class BaseObjectBBOXDataset(MultiDirectoryN5Dataset):
    """The base class dealt with samples, which where volumes. Here,
    this holds but the class also works with objects which exist in samples.
    This class can process multi directories but therefore cannot use labels to retrieve objects, instead it is the dataframe index.

    sample:param """

    BBOX_COLUMNS = BBOX_COLUMNS
    BBOX_SLICE_COLUMN = "bbox_slices"
    OBJECT_LABEL_COLUMN = MOBIE_LABEL_KEY # Instra sample object label
    OBJECT_DF_PATH_COLUMN = "objectDF_path"

    def __init__(self, sampleMapperDFPath: str, save_dir: str, overrideObjectDF: bool = False,
                 features : tuple[str] = ("area",),
                 **kwargs) -> None:
        super().__init__(sampleMapperDFPath=sampleMapperDFPath, save_dir=save_dir,
                         features=features,
                         **kwargs)
        
        self.overrideObjectDF : bool = overrideObjectDF

        if not self.confirmDatasetN5File:
            # The table is read off the N5: until run() writes it, index nothing, so
            # len() == 0 tells callers to run() (see representationlearning.builders).
            self.objectMapperDF = pd.DataFrame()
        elif not os.path.exists(self.objectDFPath) or self.overrideObjectDF:
            # Attempt to obtain the bboxs to construct objectMapperDF
            print(f"Did not find an objectMapperDF at the following path, thus generating anew: {self.objectDFPath}")
            self.__generateObjectMappingDF__()

        else:
            print(f"Utilizing objectMapperDF at the following path:\n{self.objectDFPath}")
            self.objectMapperDF = self.indexByMultiLabel(SampleLoaderBioImage.loadData(self.objectDFPath))



    # TODO: Label logic we could transfer to a torch.Dataset class ??? Which would ensure this class
    # TODO: have had processed to N5 beforehand ???

    # TODO: Maybe add method property to dynamically find objectMapperDF using self.datasetDir ?

    # TODO: Find a way to first process to obtain dataobjectDF before instantiating this class

    def removeSegmentationErrors(self):

        temp: list[pd.DataFrame] = []

        # TODO: Later, remove ansolute value here and replace by variable
        for channel, group in self.objectMapperDF.groupby(by="channel"):
            group = clean_dataframe(group, show_hist=False)
            temp.append(group)
            
        if temp:
            self.objectMapperDF = pd.concat(temp, ignore_index=True)
            self.objectMapperDF.to_json(self.objectDFPath)
            # If all n5 volumes were skipped then object data frame must exist already
        elif os.path.exists(self.objectDFPath):
            self.objectMapperDF = SampleLoaderBioImage.loadData(self.objectDFPath)
        else:
            raise Exception("No n5 volumes were processed and object dataframe does not exist. Something went wrong")

    def randomCandidate(self) -> int:
        isNuclei = self.objectMapperDF[self.CHANNEL_COLUMN].str.contains("Nuclei", na=False).to_numpy()
        candidates = np.flatnonzero(isNuclei) if isNuclei.any() else np.arange(len(self.objectMapperDF))
        return int(np.random.choice(candidates))


    def showTestVolumes(self):
        """Plot one random object as a sanity check.  Never fails the conversion.

        Preferring a Nuclei object is a display choice, not a requirement: a dataset
        with no such channel used to end a successful ``run()`` with
        ``ValueError: a cannot be empty`` out of ``rng.choice``.
        """

        object_index : int = self.randomCandidate()

        fig, ax = plt.subplots(1, 2)

        hr_vol = self.get_hr_vol(object_index)
        masked_hr_vol = self.get_masked_hr_vol(object_index)
        mid_slice = hr_vol.shape[0] // 2
        CustomMatplotlib.subImshow(ax[0], hr_vol[mid_slice], label="Raw object volume", vmin=0, vmax=hr_vol[mid_slice].max())
        CustomMatplotlib.subImshow(ax[1], masked_hr_vol[mid_slice], label="Masked raw object volume", vmin=0, vmax=masked_hr_vol[mid_slice].max())
        return hr_vol, masked_hr_vol, object_index

    @cached_property
    def objectDFDir(self):
        p = os.path.join(self.datasetDir, "tables") #training_processed/tables/
        os.makedirs(p, exist_ok=True)
        return p

    @cached_property
    def objectDFPath(self) -> str:
        """Return the path to the cleaned object-table JSON file.

        Named after the channels the mapper holds as well as the dataset: each channel
        segments its own objects, so a one-channel instance must not load another's table.
        """
        channels = sorted(self.sampleMapperDF[self.CHANNEL_COLUMN].astype(str).unique())
        return get_savedf_path(self.objectDFDir,
                                    typie="_".join([self.sampleDatasetName, *channels]))


    def experimentRelativeDir(self, samplePath)->str:
        sampleSubDir = os.path.dirname(samplePath).replace(self.sampleRoot, "")  # ......\condition
        return sampleSubDir

    def singleSampleObjectMapperDFPath(self, samplePath: str):

        sampleFileName = os.path.splitext(os.path.basename(samplePath))[0]

        dirs = f"{self.objectDFDir}{self.experimentRelativeDir(samplePath)}"

        path = get_savedf_path(dirs,
                                    typie=sampleFileName) # #training_processed/tables/single_volume/bbox_sampleFileName.json

        return path

    def sampleTransformObjectDF(self, sample_label: int):
        # Positional: the readers below go through sampleDFRow (.iloc), and a filtered
        # mapper's index labels are not positions.
        sample_idx : int = self.sample2Index(sample_label)
        image: np.ndarray = super().get_hr_vol(sample_idx)
        # TODO: Continue here to implement with image intensity
        dataFrameObject = mask2BBOXDF(super().get_hr_mask(sample_idx), image, self.features)
        dataFrameObject[self.SAMPLE_LABEL_COLUMN] = sample_label
        return dataFrameObject

    def generateSingleSampleObjectDF(self, sample_row: pd.Series, override : bool = False) -> str:

        volumePath : str = sample_row[self.SAMPLE_MASK_PATH_COLUMN]
        sample_label: int = sample_row[self.SAMPLE_LABEL_COLUMN]

        dfPath : str = self.singleSampleObjectMapperDFPath(volumePath)

        if not os.path.exists(dfPath) or SampleLoaderBioImage.loadData(dfPath).dropna(subset=BBOX_COLUMNS).shape[0] == 0\
                or override:
            print(f"Not existing dataframe or BBOX empty dataframe or overriding: {dfPath}")

            dataFrameObject : pd.DataFrame = self.sampleTransformObjectDF(sample_label)

            dataFrameObject.to_json(dfPath)

        return dfPath

    def __len__(self) -> int:  # global length
        return len(self.objectMapperDF)

    def index2MultiLabel(self, object_index: int) -> tuple[int, int]:  # global idx -> (sample, object)
        return self.objectMapperDF.index[object_index]

    def multiLabel2Index(self, sample_label: int, object_label: int) -> int:  # (sample, object) -> global idx
        return self.objectMapperDF.index.get_loc((sample_label, object_label))

    def getObjectRow(self, object_index: int) -> pd.Series:
        return self.objectMapperDF.iloc[object_index]

    def getObjectRowbyMultiLabel(self, sample_label: int, object_label: int) -> pd.Series:
        return self.objectMapperDF.index.get_loc((sample_label, object_label))



    def __generateObjectMappingDF__(self) -> None:
        """
        """

        # TODO: Not sur how changing samplePath to maskPath will change df object mapper saving

        dfpaths = []

        for i, row in self.sampleMapperDF.iterrows():

            dfPath : str = self.generateSingleSampleObjectDF(row, self.overrideObjectDF)

            dfpaths.append(dfPath)

        self.objectMapperDF: pd.DataFrame = pd.concat([SampleLoaderBioImage.loadData(p) for p in dfpaths], axis=0)

        self.sampleDF2ObjectDFMerge()

        self.removeSegmentationErrors()

        self.objectMapperDF = self.indexByMultiLabel(self.objectMapperDF)

        self.objectMapperDF.to_json(self.objectDFPath, orient="records")

    def indexByMultiLabel(self, table: pd.DataFrame) -> pd.DataFrame:
        """Index *table* by (sample label, object label).

        Needed on load too: ``orient="records"`` persists no index, so a reloaded table
        otherwise falls back to a RangeIndex and ``index2MultiLabel`` returns ints.
        """
        keys = [self.SAMPLE_LABEL_COLUMN, self.OBJECT_LABEL_COLUMN]
        # drop=False keeps the columns, so row[SAMPLE_LABEL_COLUMN] still works
        table = table.set_index(pd.MultiIndex.from_frame(table[keys]))
        assert table.index.is_unique, "duplicate (sample_label, object_label) pairs"
        return table

    def sampleDF2ObjectDFMerge(self):
        assert isinstance(self.objectMapperDF, pd.DataFrame), f"We need objectMapperDF to be an dataframe not : {type(self.objectMapperDF)}"
        self.objectMapperDF = self.objectMapperDF.merge(self.sampleMapperDF, on=[self.SAMPLE_LABEL_COLUMN],how="left",validate="many_to_one",suffixes=("","sample_"))

    def objectCrop(self, idx: int, sampletypekey: str) -> Tuple[pd.Series, np.ndarray]:
        """Object *idx*'s (positional) row and its bbox crop of the *sampletypekey* dataset."""
        row = self.getObjectRow(idx)
        key = self.sampleN5Key(row[self.SAMPLE_PATH_COLUMN], sampletypekey, self.channelOf(row))
        bbox = self.arrayBBOX2Slice(row[self.BBOX_COLUMNS].to_numpy(dtype=int))
        return row, np.asarray(self.openKey(key)[bbox])

    def get_hr_vol(self, idx: int) -> np.ndarray:
        return self.objectCrop(idx, self.N5_RAW_KEY)[1]

    def get_hr_mask(self, idx: int) -> np.ndarray:
        row, crop = self.objectCrop(idx, self.N5_MASK_KEY)
        return (crop == row[self.OBJECT_LABEL_COLUMN]).astype(np.int16)

    def run(self) -> None:
        super().run()
        self.__generateObjectMappingDF__()
        self.showTestVolumes()


    @staticmethod
    def getAllLabels(df)-> np.ndarray[Any, np.int16]:
        return df[BaseObjectBBOXDataset.OBJECT_LABEL_COLUMN].dropna().to_numpy().astype(np.int16).flatten()

    @staticmethod
    def dfBBOXIndex2Label(df, df_indx) -> int:
        indices = df.index.to_numpy()
        if df_indx in indices:
            out = df.loc[df_indx, BaseObjectBBOXDataset.OBJECT_LABEL_COLUMN]
            if isinstance(out, pd.Series):
                return int(out.iloc[0])
            return out
        else:
            raise KeyError(f"Label {df_indx} does not exist in dataset")



    @staticmethod
    def label2DFBBOXIndex(df, label)-> int:
        # TODO: Need to adjust for MultiDirectory object dataframe
        labels = BaseObjectBBOXDataset.getAllLabels(df)
        if label in labels:
            assert (labels == label).sum() == 1, f"The label {label} is more than once ({(labels == label).sum()}) in the object dataframe"
            return df.loc[df[BaseObjectBBOXDataset.OBJECT_LABEL_COLUMN] == label, :].index.tolist()[0]
        else:
            raise KeyError(f"Label {label} does not exist in dataset")


    @staticmethod
    def transformBBOXDF2Slice(df: pd.DataFrame, df_indx):
        b = getBBOXFromDF(df, df_indx).flatten()
        return (slice(b[0], b[1]), slice(b[2], b[3]), slice(b[4], b[5]))

    @staticmethod
    def getArrayFromBBOXDF(df: pd.DataFrame, df_indx) -> np.ndarray:

        # Extract and ensure numerical values
        bbox_values = [df.loc[df_indx, dim] for dim in BaseObjectBBOXDataset.BBOX_COLUMNS]

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


if __name__ == "__main__":

    save_dir = r"ssh CALI3:~/scratch/Project 2/Spheres/hepatospheres/Ihssane/data"

    sampleMapperDFPath = os.path.join(save_dir, "filtered_mapper_canaliculi_nuclei.csv")
    inst = BaseObjectBBOXDataset(sampleMapperDFPath, save_dir, overrideObjectDF=False,
                                 features=("intensity_mean", "intensity_std"))
    inst.showTestVolumes()
    plt.show()
