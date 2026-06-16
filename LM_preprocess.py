import logging
import numbers
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray, dtype
import skimage.io
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import z5py

from dl_utils import NUCLEUS_LABEL_KEY, OLD_LABEL_KEY
from dl_utils import img2patch as img2ps
from dl_utils import util
from dl_utils.util import savedataframe, merge_with_nucl_table, save2DFcolumn

logger = logging.getLogger(__name__)


def get_array_from_df(df, column):
    """Extracts and converts a column from a DataFrame to a NumPy array."""
    return np.array(df[column].tolist())


# ---------------------------------------------------------------------------
# Helper: extract a single bounding box from a DataFrame row
# ---------------------------------------------------------------------------

def getbboxfromdf(df: pd.DataFrame, df_indx) -> np.ndarray:

    cols = ["bb_min_z", "bb_max_z", "bb_min_y",
            "bb_max_y", "bb_min_x", "bb_max_x"]

    # Extract and ensure numerical values
    bbox_values = [df.loc[df_indx, dim] for dim in cols]
    # if not all(isinstance(val, numbers.Number) for val in bbox_values):
    #     raise Exception as e:
    #         print("All bounding box values must be numeric. but got: " + 
    #                              f"{[type(val) for val in bbox_values]} with error {e}")

    return np.array(bbox_values, dtype=int).T  # Convert to NumPy float array


# ---------------------------------------------------------------------------
# Helper: convert all bounding boxes in a DataFrame to an array compatible
# with the img2patch module
# ---------------------------------------------------------------------------

def get_all_bbox_array(data_df) -> ndarray[Any, dtype]:
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
        [getbboxfromdf(data_df, df_indx=indx) for indx in data_df.index]
    )

    return np.concatenate(
        [data_df.loc[data_df.index.tolist(), NUCLEUS_LABEL_KEY].to_numpy()[np.newaxis, ...].T,
         arr],
        axis=-1
    )


# ---------------------------------------------------------------------------
# Helper: filter a DataFrame column by ±3 standard deviations
# ---------------------------------------------------------------------------

def filterDF(df, column="z") -> pd.DataFrame:
    # Calculate mean and standard deviation
    res = get_array_from_df(df, column=column)
    mean = res.mean()
    std_dev = res.std()

    # Define the range for three standard deviations
    lower_bound = mean - std_dev * 3
    upper_bound = mean + std_dev * 3
    upper_bound = ((upper_bound + 8) // 8) * 8

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
        f"Number of label ids before: {df['label_id'].count()}, {max_z}, {max_y}, {max_x}")
    

    # TODO: Save the mean and standard deviation values to columns "{column}_length_mean" and "{column}_length_stddev" in the DataFrame if needed
    df[f"{column}_length_mean"] = mean
    df[f"{column}_length_stddev"] = std_dev
    

    # Define the range for filtering based on the bounds
    return df.loc[(df[column] >= lower_bound) & (df[column] < upper_bound), :]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def BBox2DF(all_boxes: object, dataset_path, datasetMask_path, key):
    """Convert a BBoxes object into a pandas DataFrame with nucleus metadata."""
    all_boxes.bboxes = all_boxes.bboxes.astype(np.uint32)

    values = all_boxes.get("center")

    labels, z_min, z_max, y_min, y_max, x_min, x_max = all_boxes.bboxes.T.tolist()

    rows = {
        NUCLEUS_LABEL_KEY: labels,
        "bb_min_x": x_min, "bb_max_x": x_max,
        "bb_min_y": y_min, "bb_max_y": y_max,
        "bb_min_z": z_min, "bb_max_z": z_max,
        "anchor_z": values[:, 1].tolist(),
        "anchor_x": values[:, 3].tolist(),
        "anchor_y": values[:, 2].tolist(),
        "dataset path": [dataset_path] * len(labels),
        "mask volume path": [datasetMask_path] * len(labels),
        "dataset acronym": [key] * len(labels),
    }

    return pd.DataFrame(rows)


def clean_dataframe(df, show_hist=False):
    """
    Filter a nucleus DataFrame by bounding-box dimensions (±3 std).

    Adds z/y/x side-length columns, applies per-axis sigma filtering, drops
    NaN rows, and returns the cleaned DataFrame.
    """
    # TURN2BBOX implementation
    bbox = get_all_bbox_array(df)

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
        f"After filtering: number of label ids before: {df['label_id'].count()}, {max_z}, {max_y}, {max_x}")

    # Drop rows with missing values in specific columns
    filtered_df.dropna(subset=["z", "x", "y", NUCLEUS_LABEL_KEY], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    if show_hist:
        res = np.array([
            get_array_from_df(filtered_df, column="z"),
            get_array_from_df(filtered_df, column="y"),
            get_array_from_df(filtered_df, column="x"),
        ]).flatten()

        counts, bins = np.histogram(res)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylabel("Count")
        plt.title("after")
        plt.show()
        plt.close()

    print(f"Number of label ids before: {filtered_df['label_id'].count()}")

    return filtered_df


class VolumeProcessing:
    """Resampling and N5 serialisation utilities for LM organoid volumes.

    Encapsulates all preprocessing steps required to convert a raw ``.tif``
    volume and its instance-segmentation mask into the N5 format expected by
    downstream dataset classes, together with an accompanying nucleus-table
    JSON file.

    Parameters
    ----------
    dapiKey : str, optional
        Top-level group name inside the N5 file.  Default ``"dapi"``.
    rawKey : str, optional
        Dataset name for the raw intensity channel.  Default ``"raw"``.
    maskKey : str, optional
        Dataset name for the instance-segmentation mask.  Default ``"mask"``.
    resolution : list of int, optional
        Physical voxel size ``[Z, Y, X]`` used for isotropic resampling.
        Default ``[3, 1, 1]`` (axial resolution 3× worse than lateral).
    """

    def __init__(
        self,
        dapiKey: str = "dapi",
        rawKey: str = "raw",
        maskKey: str = "mask",
        resolution: Optional[List[int]] = None,
        chunkedImagePath: Optional[str] = None,
    ) -> None:
        self.dapiKey = dapiKey
        self.rawKey = rawKey
        self.maskKey = maskKey
        self.resolution: List[int] = resolution if resolution is not None else [3, 1, 1]
        self.chunkedImagePath: Optional[str] = chunkedImagePath
        self._n5_file = None  # opened lazily; supports multiprocessing pickle

    def _ensure_open(self) -> None:
        """Open the N5 file handle lazily so the object stays picklable."""
        if self._n5_file is None:
            if self.chunkedImagePath is None:
                raise ValueError("chunkedImagePath must be set before reading volumes.")
            self._n5_file = z5py.File(self.chunkedImagePath, 'r')
            self.Volume = self._n5_file[self.dapiKey + "/" + self.rawKey]
            self.Mask = self._n5_file[self.dapiKey + "/" + self.maskKey]

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resampleVolume(self, volume: np.ndarray, resolutions: List[int],
                       interpolationOrder: int = 0) -> Tuple[np.ndarray, int]:
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

        Returns
        -------
        volume : np.ndarray
            Resampled volume.
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

        volume = resize(
            volume,
            tuple((np.array(volume.shape) * resolutions).flatten()),
            anti_aliasing=True,
            order=interpolationOrder,
        )

        logger.debug("Resampled dimension: %s", volume.shape)
        return volume, min_res

    def resampleRawVolume(self, volume: np.ndarray,
                          resolutions: List[int]) -> Tuple[np.ndarray, int]:
        """Resample and min-max normalise a raw intensity volume.

        Calls :meth:`resampleVolume` with cubic (order-3) interpolation and
        subsequently rescales intensities to ``[0, 1]`` using
        :class:`sklearn.preprocessing.MinMaxScaler`.

        Parameters
        ----------
        volume : np.ndarray
            Raw 3-D intensity volume ``(Z, Y, X)``.
        resolutions : list of int
            Physical voxel size for each axis (ZYX).

        Returns
        -------
        volume : np.ndarray of float32
            Resampled and normalised volume cast to ``float32`` to halve
            memory usage compared to ``float64``.
        min_res : int
            Isotropic output resolution (finest input resolution).
        """
        volume, min_res = self.resampleVolume(volume, resolutions, interpolationOrder=3)
        volume = preprocessing.MinMaxScaler().fit_transform(
            volume.reshape(-1, 1)
        ).reshape(*volume.shape)
        return volume.astype(np.float32), min_res

    def resampleMaskVolume(self, maskVolume: np.ndarray, resolutions: List[int],
                           targetShape: tuple) -> np.ndarray:
        """Resample an instance-segmentation mask to *targetShape*.

        Uses nearest-neighbour interpolation (order 0) to preserve integer
        labels.  A secondary resize step corrects any shape mismatch between
        the resampled mask and the raw volume.

        Parameters
        ----------
        maskVolume : np.ndarray
            3-D label volume ``(Z, Y, X)``.
        resolutions : list of int
            Physical voxel size for each axis (ZYX).
        targetShape : tuple of int
            Desired output shape, typically matching the resampled raw volume.

        Returns
        -------
        np.ndarray
            Resampled mask at *targetShape*.

        Raises
        ------
        AssertionError
            If the number of unique labels changes during resampling, which
            would indicate label merging or creation.
        """
        maskVolume, _ = self.resampleVolume(maskVolume, resolutions, interpolationOrder=0)

        nLabels: int = np.unique(maskVolume).size
        dtype_before = maskVolume.dtype
        if maskVolume.shape != targetShape:
            logger.debug(
                "Not equal dims between resampled and raw: %s, %s",
                maskVolume.shape, targetShape,
            )
            maskVolume = resize(maskVolume, targetShape, anti_aliasing=True, order=0)
            logger.debug("Matched mask dimensions with volume: %s", maskVolume.shape)

        assert np.unique(maskVolume).size == nLabels, (
            "Number of labels changed during resampling! "
            f"{np.unique(maskVolume).size} vs {nLabels} and datatype before "
            f"{dtype_before} and after {maskVolume.dtype}"
        )

        return maskVolume

    # ------------------------------------------------------------------
    # Nucleus-table helpers
    # ------------------------------------------------------------------

    @staticmethod
    def getDFPath(volumePath: str) -> str:
        """Return the path to the cleaned nucleus-table JSON file.

        The file is expected at
        ``<volume_dir>/tables/<volume_stem>_cleaned.<ext>``.

        Parameters
        ----------
        volumePath : str
            Path to the raw ``.tif`` volume (or the sibling ``.n5`` file).

        Returns
        -------
        str
            Full path to the cleaned nucleus-table file.
        """
        data_df_dir: str = os.path.join(os.path.dirname(volumePath), "tables")
        volFileName = os.path.splitext(os.path.basename(volumePath))[0]
        return util.get_savedf_path(data_df_dir, typie=f"{volFileName}_cleaned")

    def volume2DF(self, volPath: str, maskVolPath: str,
                  chunkedImagePath: str) -> pd.DataFrame:
        """Build a nucleus table from scratch using the raw volume and mask.

        Resamples both volumes to isotropic resolution, writes them to the
        sibling ``.n5`` file via :meth:`writeChunkedImage`, extracts
        bounding boxes with :func:`img2ps.BBoxes.from_mask`, and saves both a
        raw and a cleaned DataFrame to the ``tables/`` sub-directory.

        Parameters
        ----------
        volPath : str
            Absolute path to the raw ``.tif`` volume.
        maskVolPath : str
            Absolute path to the instance-segmentation ``.tif`` mask.
        chunkedImagePath : str
            Destination ``.n5`` file path for the resampled data.

        Returns
        -------
        pd.DataFrame
            Cleaned nucleus table with a ``"modality"`` column set to
            ``"LM"``.
        """
        volume: np.ndarray = VolumeProcessing.loadVolume(volPath)
        volume = self.resampleRawVolume(volume, resolutions=self.resolution)[0]

        maskVol: np.ndarray = self.resampleMaskVolume(
            VolumeProcessing.loadVolume(maskVolPath),
            resolutions=self.resolution,
            targetShape=volume.shape,
        )

        self.writeChunkedImage(chunkedImagePath, volume=volume, mask=maskVol)

        bbox = img2ps.BBoxes.from_mask(maskVol)
        df: pd.DataFrame = BBox2DF(bbox, volPath, datasetMask_path=maskVolPath,
                                   key=np.nan)

        save_dir: str = os.path.join(os.path.dirname(chunkedImagePath), "tables")
        os.makedirs(save_dir, exist_ok=True)

        volFileName = os.path.splitext(os.path.basename(volPath))[0]
        savedataframe(save_dir=save_dir, dataframe=df, typie=volFileName)

        df = clean_dataframe(df, show_hist=False)
        savedataframe(save_dir=save_dir, dataframe=df,
                      typie=f"{volFileName}_cleaned")

        df["modality"] = "LM"
        return df

    # ------------------------------------------------------------------
    # N5 I/O
    # ------------------------------------------------------------------

    def writeChunkedImage(self, n5_file: str, volume: np.ndarray,
                          mask: np.ndarray) -> None:
        """Write *volume* and *mask* into an N5 file at *n5_file*.

        Both arrays are stored under ``<dapiKey>/`` with datasets
        ``<rawKey>`` and ``<maskKey>`` respectively, using ``(32, 32, 32)``
        chunks.  A channel attribute is attached to each dataset.  When
        ``DEBUG`` logging is active a read-back verification is performed.

        Parameters
        ----------
        n5_file : str
            Path to the target N5 file (created or appended).
        volume : np.ndarray of float32
            Normalised raw intensity volume in ``[0, 1]``.  Scaled to
            ``uint16`` before writing.
        mask : np.ndarray
            Instance-segmentation label volume.  Cast to ``uint16`` before
            writing.
        """
        volume = (volume * 65535).astype(np.uint16)
        mask = mask.astype(np.uint16)
        logger.debug("Volume intensity range: [%s, %s]", volume.min(), volume.max())
        logger.debug("Mask intensity range: [%s, %s]", mask.min(), mask.max())

        with z5py.File(n5_file, 'a', use_zarr_format=False) as f:
            group = f[self.dapiKey] if self.dapiKey in f else f.create_group(self.dapiKey)

            volume_ds = group.require_dataset(
                self.rawKey,
                shape=volume.shape,
                chunks=(32, 32, 32),
                dtype=volume.dtype,
            )
            volume_ds[:] = volume
            volume_ds.attrs['channel'] = self.dapiKey

            mask_ds = group.require_dataset(
                self.maskKey,
                shape=mask.shape,
                chunks=(32, 32, 32),
                dtype=mask.dtype,
            )
            mask_ds[:] = mask
            mask_ds.attrs['channel'] = self.dapiKey

        if logger.isEnabledFor(logging.DEBUG):
            with z5py.File(n5_file, 'r', use_zarr_format=False) as f:
                volume_read = f[f"{self.dapiKey}/{self.rawKey}"][:]
                mask_read = f[f"{self.dapiKey}/{self.maskKey}"][:]
            logger.debug("Written volume — shape: %s, dtype: %s",
                         volume_read.shape, volume_read.dtype)
            logger.debug("Written mask   — shape: %s, dtype: %s",
                         mask_read.shape, mask_read.dtype)

    # ------------------------------------------------------------------
    # Bounding-box helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bbox2slice(bbox: np.ndarray) -> Tuple[slice, slice, slice]:
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
    def addbbox_slices_col(table: pd.DataFrame) -> pd.DataFrame:
        """Append a ``"bbox slices"`` column of per-nucleus slice tuples.

        If any of the expected bounding-box columns are absent from *table*,
        ``merge_with_nucl_table`` is called first to populate them.

        Parameters
        ----------
        table : pd.DataFrame
            Nucleus table.  Expected bounding-box columns:
            ``bb_min_z``, ``bb_max_z``, ``bb_min_y``, ``bb_max_y``,
            ``bb_min_x``, ``bb_max_x``.

        Returns
        -------
        pd.DataFrame
            *table* with an added ``"bbox slices"`` column whose values are
            ``(slice_z, slice_y, slice_x)`` tuples.
        """
        cols: List[str] = ["bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y",
                           "bb_min_x", "bb_max_x"]

        missing: List[str] = [col for col in cols if col not in table.columns]
        if missing:
            table = merge_with_nucl_table(table)

        bbs: List[Tuple[slice, slice, slice]] = []
        for _, row in table.iterrows():
            b = getbboxfromdf(table, row.name).flatten()
            bbs.append((slice(b[0], b[1]), slice(b[2], b[3]), slice(b[4], b[5])))

        return save2DFcolumn(bbs, table[OLD_LABEL_KEY].to_numpy(), table,
                             "bbox slices")

    # ------------------------------------------------------------------
    # Volume / mask loading
    # ------------------------------------------------------------------

    @staticmethod
    def loadVolume(path: str) -> np.ndarray:
        """Load a 3-D TIFF volume from *path*.

        Parameters
        ----------
        path : str
            Path to a ``.tif`` or ``.tiff`` file.

        Returns
        -------
        np.ndarray
            Image array as returned by :func:`skimage.io.imread`.

        Raises
        ------
        AssertionError
            If *path* does not have a ``.tif`` or ``.tiff`` extension.
        """
        assert os.path.splitext(path)[1].lower() in ('.tif', '.tiff'), (
            f"Unsupported file format: {os.path.splitext(path)[1]}. "
            "Supported formats are .tif, .tiff"
        )
        return skimage.io.imread(path)

    @staticmethod
    def get_bbox_slice(df: pd.DataFrame,
                       nucl_label: Optional[int] = None) -> Tuple[slice, slice, slice]:
        """Return the bounding-box slice tuple for *nucl_label* from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Nucleus table with bounding-box columns.
        nucl_label : int, optional
            Instance label of the nucleus.

        Returns
        -------
        tuple of slice
            ``(slice_z, slice_y, slice_x)`` for the nucleus bounding box.
        """
        row_idx = df.index[df[NUCLEUS_LABEL_KEY] == nucl_label][0]
        b = getbboxfromdf(df, row_idx).flatten()
        return slice(b[0], b[1]), slice(b[2], b[3]), slice(b[4], b[5])

    def get_hr_vol(self, df: pd.DataFrame,
                   nucl_label: Optional[int] = None) -> np.ndarray:
        """Extract the high-resolution raw crop for *nucl_label*.

        Parameters
        ----------
        df : pd.DataFrame
            Nucleus table used to look up the bounding box.
        nucl_label : int, optional
            Instance label of the target nucleus.

        Returns
        -------
        np.ndarray
            Sub-array of ``self.Volume`` cropped to the nucleus bounding box.
        """
        self._ensure_open()
        bb: Tuple[slice, slice, slice] = VolumeProcessing.get_bbox_slice(df, nucl_label)
        return np.array(self.Volume[bb])

    def get_hr_mask(self, df: pd.DataFrame,
                    nucl_label: Optional[int] = None) -> np.ndarray:
        """Extract and isolate the binary mask for *nucl_label*.

        Returns the bounding-box crop of ``self.Mask`` with all pixels that do
        not belong to *nucl_label* zeroed out.

        Parameters
        ----------
        df : pd.DataFrame
            Nucleus table used to look up the bounding box.
        nucl_label : int, optional
            Instance label of the target nucleus.

        Returns
        -------
        np.ndarray
            Binary (or label-valued) mask crop containing only *nucl_label*.
        """
        self._ensure_open()
        bb: Tuple[slice, slice, slice] = VolumeProcessing.get_bbox_slice(df, nucl_label)
        mask: np.ndarray = np.array(self.Mask[bb])
        return mask * (mask == nucl_label)

    def get_masked_hr_vol(self, df: pd.DataFrame,
                          nucl_label: Optional[int] = None) -> np.ndarray:
        """Return the raw crop for *nucl_label* with background zeroed out.

        Combines :meth:`get_hr_vol` and :meth:`get_hr_mask` to produce a
        clean single-nucleus volume with all surrounding signal suppressed.

        Parameters
        ----------
        df : pd.DataFrame
            Nucleus table used to look up the bounding box.
        nucl_label : int, optional
            Instance label of the target nucleus.

        Returns
        -------
        np.ndarray
            Raw intensity crop with out-of-nucleus voxels set to zero.
        """
        vol = self.get_hr_vol(df, nucl_label)
        mask = self.get_hr_mask(df, nucl_label)
        return vol * (mask != 0)
 