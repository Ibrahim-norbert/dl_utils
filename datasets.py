from dl_utils.SampleLoader import SampleLoaderBioImage
from dl_utils import util
from torch.utils.data import Subset, Dataset
import numpy as np
import pandas as pd
import os
from typing import Literal
import numpy as np
import os
from typing import Any, List, Literal, Tuple
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, List, Literal
import logging
from numpy import ndarray, dtype
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
import z5py
logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """Base class for nuclei data
    Calculates general properties, loads low resoltion (s3) nuclei"""

    def __init__(self, datasetPath: str, sampleColumn: str, **kwargs) -> None:
        
        self.datasetPath=datasetPath
        self.sampleColumn: str = sampleColumn
        self.datasetDF: pd.DataFrame = self.loadDataFrame(
            datasetPath=datasetPath)
        self.samples: np.ndarray[Literal["1"], np.dtype[np.int32]] = np.unique(
            self.get_all_labels(self.datasetDF, self.sampleColumn)
        )

    def writeChunkedImage(self, filePath: str, volume: np.ndarray, groupkey : str, key: str) -> None:
        """Write *volume* into an N5 file at *n5_file*.

        Array is stored under the group ``groupkey/`` with datasets
        ``key``, using ``(32, 32, 32)`` chunks. When ``DEBUG``
        logging is active a read-back verification is performed to confirm
        the write succeeded.

        Parameters
        ----------
        filePath : str
            Path to the target N5 file (created or appended).
        volume : If necessary, scaled to
            ``uint16`` before writing.
        """

        n5_file = util.replaceFileExt(filePath, ".n5")

        logger.debug("Volume intensity range: [%s, %s]", volume.min(), volume.max())
        # TODO: For now ensure image is in grayscale
        if np.issubdtype(volume.dtype, np.floating):
            assert volume.min() >= 0 and volume.max() <= 1, f"We do not have floating point grayscale range of 0 - 1 but instead {volume.min()} - {volume.max()}"
            # Should integer, else saving n5 file would take too long
            volume = (volume * 65535).astype(np.uint16)
            logger.debug("Volume intensity range after scaling: [%s, %s]", volume.min(), volume.max())

        assert volume.dtype in [np.int8, np.int16], f"The data type of the volume must be int16 or lower but we have {volume.dtype}"

        with z5py.File(n5_file, 'a', use_zarr_format=False) as f:
            if groupkey not in f:
                group = f.create_group(groupkey)
            else:
                group = f[groupkey]

            volume_ds = group.require_dataset(
                key,
                shape=volume.shape,
                chunks=(32, 32, 32),
                dtype=volume.dtype,
            )
            volume_ds[:] = volume

        # Verification read-back — only runs when DEBUG logging is enabled
        if logger.isEnabledFor(logging.DEBUG):
            with z5py.File(n5_file, 'r', use_zarr_format=False) as f:
                volume_read = f[f"{groupkey}/{key}"][:]
            logger.debug("Written volume — shape: %s, dtype: %s", volume_read.shape, volume_read.dtype)


    @staticmethod
    def getbboxfromdf(df: pd.DataFrame, df_indx) -> np.ndarray:

        cols = ["bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x"]

        print(df_indx)
        # Extract and ensure numerical values
        bbox_values = [df.loc[df_indx, dim] for dim in cols]
        # print(bbox_values, "before series check")
        # bbox_values = [val.iloc[0] if not isinstance(val, numbers.Number) else val for val in bbox_values]
        # print(bbox_values, "after series check")
        if not all(isinstance(val, numbers.Number) for val in bbox_values):
            raise DataclassTypeError("All bounding box values must be numeric. but got: "
                                     f"{[type(val) for val in bbox_values]}")

        return np.array(bbox_values, dtype=int).T  # Convert to NumPy float array
    
    @staticmethod
    def bbox2slice(bbox: np.ndarray) -> tuple[slice, slice, slice]:
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
            ``(slice_z, slice_y, slice_x)`` tuples at s3 resolution.
        """
        cols: List[str] = ["bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x"]

        # Merge nucleus table if any bounding-box columns are absent
        missing_cols: List[str] = [col for col in cols if col not in table.columns]
        assert not missing_cols, "The bounding box coordinates are incomplete"

        bbs: List[Tuple[slice[Any, Any, Any]]] = [BaseDataset.get_slice(table, row.name) for _, row in table.iterrows()]

        return util.save2DFcolumn(bbs, BaseDataset.get_all_labels(table, ), table, "bbox slices")

    def resampleVolume(self, volume: np.ndarray, resolutions: list[int],
                       interpolationOrder: int = 0) -> tuple[np.ndarray, int]:
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

        volume = resize(volume, tuple((np.array(volume.shape) * resolutions).flatten()),
                        anti_aliasing=True, order=interpolationOrder)

        logger.debug("Resampled dimension: %s", volume.shape)
        return volume, min_res

    @staticmethod
    def get_slice(df: pd.DataFrame, df_indx: int) -> tuple[slice, slice, slice]:
        """Return the bounding-box slice tuple for a nucleus at *df_indx*.

        Parameters
        ----------
        df : pd.DataFrame
            Nucleus table containing bounding-box columns.
        df_indx : int
            Row index (DataFrame index label) of the target nucleus.

        Returns
        -------
        tuple of slice
            ``(slice_z, slice_y, slice_x)`` covering the nucleus bounding box.
        """
        bbox: ndarray[Tuple[Any], dtype[Any]] = BaseDataset.getbboxfromdf(df, df_indx)
        slice_val: Tuple[slice[Any, Any, Any]] = BaseDataset.bbox2slice(bbox)
        return slice_val

    def loadDataFrame(self, datasetPath: str) -> pd.DataFrame:
        data_df = SampleLoaderBioImage.loadData(datasetPath)
        if data_df is not None:
            assert isinstance(data_df, pd.DataFrame)
            return data_df
        else:
            raise Exception(f"Loaded data is None")

    def index2DataPoint(self, idx: int) -> int:
        return self.samples[idx]

    def dataPoint2index(self, label: int) -> int:
        return np.where(self.samples == label)[0].flatten()[0]

    @staticmethod
    def get_all_labels(df, label_col):
        return df[label_col].dropna().to_numpy().flatten()

    @staticmethod
    def df_index2DataPoint(df, df_indx, dataPointCol):
        indices = df.index.to_numpy()
        if df_indx in indices:
            out = df.loc[df_indx, dataPointCol]
            if isinstance(out, pd.Series):
                return out.iloc[0]
            return out
        else:
            raise KeyError(f"Label {df_indx} does not exist in dataset")

    @staticmethod
    def label2df_index(df, dataPoint, dataPointCol):
        labels = BaseDataset.get_all_labels(df, dataPointCol)

        if dataPoint in labels:
            return df.loc[df[dataPointCol] == dataPoint, :].index.tolist()[0]
        else:
            raise KeyError(f"Label {dataPoint} does not exist in dataset")

    def __len__(self) -> int:
        return self.samples.size

    @staticmethod
    def getBatchedItem(dataset, n) -> List[Any]:
        assert n <= len(
            dataset
        ), f"Parameter n specified with value {n} is larger than dataset length"
        return [dataset[i] for i in range(n)]

    # @staticmethod
    # def get_center_of_mass(df, nucl_label: int) -> np.ndarray[Any, dtype]:
    #     """ Get center of mass of a nucleus for s3 resolution"""

    #     return com_pxl

    @staticmethod
    def save_array(save_dir, array: np.ndarray, filename: str) -> None:
        """Save a NumPy array to a specified file if the save directory is valid."""
        if save_dir and os.path.exists(save_dir):
            path: str = os.path.join(save_dir, f"{filename}.npy")
            np.save(path, array)

    @staticmethod
    def getDataloader(dataset, batch_size, num_workers, persistent_workers, **kwargs):
        # print(f"All entered parameters: {locals()}")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    @staticmethod
    def subset_dataset(dataset, indices) -> Subset:
        """
        Create a subset of the dataset by selecting specific labels.

        Parameters:
            labels (np.ndarray): An array of labels to include in the subset.
            nucl (bool, optional): Whether to use nuclear labels (`True`) or other labels (`False`). Default is `True`.

        Returns:
            Subset: A PyTorch `Subset` object containing the selected data.

        Behavior:
            - Converts the input `labels` into dataset indices using `self.label2index()`.
            - Creates a new column in `self.data_df` to categorize cell types.
            - Uses the subset indices to create a `Subset` object from the original dataset.
            - Subsets `self.data_df` to match the indices in the subset.
        """

        # Create the PyTorch Subset object
        mini_dataset: Subset = Subset(dataset, indices)

        return mini_dataset

