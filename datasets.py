import glob
import logging
import numbers
import os
import shutil

import numpy as np
import pandas as pd
import yaml
from functools import cached_property
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader

from dl_utils import LABEL_KEY
from dl_utils.SampleLoader import SampleLoaderBioImage
from dl_utils.SampleTypes import Data, Vertices
from dl_utils import util_base as util

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
class BaseDataset:
    """Base class for nuclei data
    Calculates general properties, loads low resoltion (s3) nuclei"""

    def __init__(self, datasetPath: str, sampleColumn: str, **kwargs) -> None:
        
        self.datasetPath=datasetPath
        self.sampleColumn: str = sampleColumn
        self.datasetDF: pd.DataFrame = self.loadDataFrame(
            datasetPath=datasetPath)
        
        if sampleColumn == "index":
            self.datasetDF[sampleColumn] = self.datasetDF.index
            
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

        import z5py

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
    def getbboxfromdf(df: pd.DataFrame, df_indx) -> np.ndarray:

        cols = ["bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x"]

        # Extract and ensure numerical values
        bbox_values = [df.loc[df_indx, dim] for dim in cols]
        if not all(isinstance(val, numbers.Number) for val in bbox_values):
            raise TypeError("All bounding box values must be numeric, but got: "
                            f"{[type(val) for val in bbox_values]}")

        return np.array(bbox_values, dtype=int).T  # Convert to NumPy int array
    
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

    def index2Sample(self, idx: int) -> int:
        return self.samples[idx]

    def Sample2index(self, label: int) -> int:
        return np.where(self.samples == label)[0].flatten()[0]

    @staticmethod
    def get_all_labels(df, label_col):
        return df[label_col].dropna().to_numpy().flatten()

    @staticmethod
    def df_index2Sample(df, df_indx, dataPointCol):
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
    def subset_dataset(dataset, indices) -> "SubsetDataset":
        """
        Create a subset of the dataset by selecting specific labels.

        Parameters:
            labels (np.ndarray): An array of labels to include in the subset.
            nucl (bool, optional): Whether to use nuclear labels (`True`) or other labels (`False`). Default is `True`.

        Returns:
            SubsetDataset: A subset object containing the selected data.

        Behavior:
            - Converts the input `labels` into dataset indices using `self.label2index()`.
            - Creates a new column in `self.data_df` to categorize cell types.
            - Uses the subset indices to create a subset from the original dataset.
            - Subsets `self.data_df` to match the indices in the subset.
        """

        mini_dataset = SubsetDataset(dataset, indices)

        return mini_dataset





class BaseVolumeCollectionDataset(BaseDataset):
    """Base class for collection datasets containing paired volume and mask paths.

    Handles loading or building a master index DataFrame from either a CSV
    listing volume/mask pairs or by scanning a directory.  Per-volume dataset
    instances are *not* created at this level — subclasses override
    :meth:`createDatasetFromDataFramePath` and
    :meth:`createDatasetDFfromDirectory` to populate any per-volume registry.

    Subclasses should set the class-level column attributes to match their
    specific data schema.

    Class Attributes
    ----------------
    sampleColumn : str
        Column used as the global sample index in the master CSV.
    keptIndicesColumn : str
        Column storing the per-volume DataFrame row index.
    volumePathColumn : str
        Column for the raw volume file path.
    maskPathColumn : str
        Column for the instance-segmentation mask file path.
    """

    sampleColumn: str = "sample index"
    volumePathColumn: str = "volume path"
    maskPathColumn: str = "mask path"

    def __init__(
        self,
        datasetDir: Optional[str] = None,
        fileExtension: str = "tif",
        datasetDataframePath: Optional[str] = None,
        datasetConfig: Optional[dict] = None,
    ) -> None:
        self.datasetConfig: dict = datasetConfig or {}
        self.fileExtension: str = fileExtension
        self.datasetDataframePath: Optional[str] = datasetDataframePath

        if datasetDataframePath and os.path.isfile(datasetDataframePath):
            csv_df = self.loadDataFrame(datasetDataframePath)
            required_cols = [self.volumePathColumn, self.maskPathColumn]
            missing = [c for c in required_cols if c not in csv_df.columns]
            assert not missing, (
                f"Dataset CSV is missing required columns: {missing}. "
                f"Expected: {required_cols}, found: {csv_df.columns.tolist()}"
            )
            try:
                common = os.path.commonpath(csv_df[self.volumePathColumn].tolist())
                self.datasetDir = common if os.path.isdir(common) else os.path.dirname(common)
            except ValueError:
                # Paths span multiple drives — fall back to the CSV's own directory
                self.datasetDir = os.path.dirname(os.path.abspath(datasetDataframePath))
            self.createDatasetFromDataFramePath(datasetDataframePath)
        else:
            assert datasetDir is not None, (
                "datasetDir is required when no datasetDataframePath is given."
            )
            self.datasetDir = datasetDir
            self.createDatasetDFfromDirectory(datasetDir, fileExtension=self.fileExtension)

        super().__init__(datasetPath=self.dfPath, sampleColumn=self.sampleColumn)

    @cached_property
    def dfPath(self) -> str:
        """Absolute path to the master index CSV file.

        Written one directory above *datasetDir* so it is not confused with
        per-volume tables.
        """
        parent = os.path.dirname(self.datasetDir)
        stem = os.path.basename(self.datasetDir)
        return os.path.join(parent, f"{stem}_dataset.csv")

    def createDatasetFromDataFramePath(self, manualDFPath: str) -> None:
        """Build the master index DataFrame from a hand-crafted CSV.

        Deduplicates to one row per unique volume path, assigns a sequential
        sample index, and writes the result to :attr:`dfPath`.  Subclasses
        override this to additionally populate any per-volume instance registry.

        Parameters
        ----------
        manualDFPath : str
            Path to the source CSV.
        """
        if os.path.exists(self.dfPath):
            return

        dataDF = self.loadDataFrame(manualDFPath)
        result = dataDF.drop_duplicates(subset=self.volumePathColumn).reset_index(drop=True)
        result[self.sampleColumn] = result.index
        result.to_csv(self.dfPath)

    def createDatasetDFfromDirectory(self, directory: str, fileExtension: str) -> None:
        """Discover volume/mask pairs in *directory* and build master index.

        Volumes are expected directly in *directory*; masks must reside in a
        ``mask/`` sub-directory with identical filenames.  Subclasses should
        override to additionally populate any per-volume instance registry.

        Parameters
        ----------
        directory : str
            Root directory to scan.
        fileExtension : str
            Glob extension, e.g. ``"tif"``.

        Raises
        ------
        AssertionError
            If no mask files are found, or if volume/mask counts differ.
        """
        if os.path.exists(self.dfPath):
            return
        
        volumePaths: List[str] = glob.glob(os.path.join(directory, f"*{fileExtension}"))
        maskPaths: List[str] = glob.glob(os.path.join(directory, "mask", f"*{fileExtension}"))

        assert maskPaths, (
            f"No mask files found. Volumes/masks found: "
            f"{len(volumePaths)}/{len(maskPaths)} in {directory}"
        )
        assert len(volumePaths) == len(maskPaths), (
            f"Volume and mask counts must match: "
            f"{len(volumePaths)} volumes, {len(maskPaths)} masks."
        )

        result = pd.DataFrame({
            self.volumePathColumn: volumePaths,
            self.maskPathColumn: maskPaths,
        })
        result[self.sampleColumn] = result.index
        result.to_csv(self.dfPath)

    def copy_samples(self, src_root: str, dest_root: str) -> None:
        """Copy volumes and masks to *dest_root*, preserving the sub-tree below *src_root*.

        For every file ``<src_root>/a/b/file.tif`` the destination is
        ``<dest_root>/a/b/file.tif``.  Intermediate directories are created as
        needed.  An updated index is saved as ``dataset.xlsx`` inside *dest_root*.

        Parameters
        ----------
        src_root : str
            Common ancestor whose sub-tree structure should be preserved.
            Every volume and mask path must be located under this directory.
        dest_root : str
            Root of the destination tree.

        Raises
        ------
        FileNotFoundError
            If a source file listed in the index does not exist.
        ValueError
            If a source file is not located under *src_root*.
        """
        src_root  = os.path.abspath(src_root)
        dest_root = os.path.abspath(dest_root)

        df = self.datasetDF.copy()

        for idx, row in df.iterrows():
            for col in (self.volumePathColumn, self.maskPathColumn):
                src = os.path.abspath(row[col])

                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Source file not found: {src}")
                if not src.startswith(src_root):
                    raise ValueError(f"{src!r} is not under src_root {src_root!r}")

                rel  = os.path.relpath(src, src_root)
                dest = os.path.join(dest_root, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                df.at[idx, col] = dest

        os.makedirs(dest_root, exist_ok=True)
        df.to_excel(os.path.join(dest_root, "dataset.xlsx"), index=False)
        logger.info("Copied %d samples from %s to %s", len(df), src_root, dest_root)

    @staticmethod
    def getConfig(path: str) -> dict:
        """Load a YAML configuration file and return it as a dictionary.

        Parameters
        ----------
        path : str
            Absolute path to a ``.yaml`` configuration file.

        Returns
        -------
        dict
            Parsed YAML contents.
        """
        with open(path, "r") as f:
            return yaml.safe_load(f)


class VerticesDataset(BaseDataset):
    """'
    Semantic label guide:
        * 0 --> ground
        * 1 --> focal plant
        * 2 --> surrounding plants

    Ground label guide:
        * 0 --> not ground
        * 1 --> ground

    """

    def __init__(
        self,
        sampleColumnCSV: str,
        datasetPath: str,
        sampleColumn: str,
        datasetDir: Union[str, None] = None,
        verticesCol=["x", "y"],
        labels="cluster_id",
        standardizedCoords=["x", "y"],
        standardizedLabels=LABEL_KEY,
        **kwargs,
    ) -> None:
        self.config: dict[str, Any] = locals()
        self.config.pop("self", None)
        self.config.pop("__class__", None)
        self.datasetDir = datasetDir

        if self.datasetDir is not None and os.path.exists(self.datasetDir):
            datasetPath = self.dir2CSV(
                [self.datasetDir], os.path.join(self.datasetDir, "..", "..", "..")
            )

        super().__init__(datasetPath=datasetPath, sampleColumn=sampleColumn, **kwargs)

        self.sampleColumnCSV: str = sampleColumnCSV

        self.verticesCol: Union[str, list[str]] = verticesCol
        self.standardizedLabels = standardizedLabels
        self.standardizedCoords = standardizedCoords

        if self.datasetDF.columns.isin(self.standardizedCoords).all():
            self.verticesCol: pd.DataFrame = self.datasetDF.loc[
                :, self.standardizedCoords
            ]

        if self.standardizedLabels in self.datasetDF.columns:
            self.standardizedLabels = self.datasetDF.loc[:, self.standardizedLabels]

        self.verticesInstanceLabelCol: str = labels

    @classmethod
    def from_dir(
        cls,
        datasetDir: str,
        saveDir: str,
        sampleColumn: str,
        **kwargs,
    ) -> "VerticesDataset":
        """Construct a VerticesDataset by scanning all CSVs under datasetDir.

        Args:
            datasetDir: Root directory containing CSV files (searched recursively).
            sampleCol: Column name identifying the sample/file path.
            dataPointCol: Column name used as the unique data-point identifier.
            **kwargs: Forwarded to VerticesDataset.__init__ (verticesCol, labels, etc.).
        """
        assert os.path.exists(datasetDir), f"Directory does not exist: {datasetDir}"
        datasetPath = cls.dir2CSV(
            [datasetDir], saveDir
        )
        return cls(
            sampleColumnCSV="filePath",
            datasetPath=datasetPath,
            sampleColumn=sampleColumn,
            datasetDir=None,  # already resolved above; prevent double call
            **kwargs,
        )
    
    @staticmethod
    def saveCSV(filePath: str, d_clustering_array: np.ndarray, columns: list) -> None:
        df = pd.DataFrame(d_clustering_array, columns=columns)
        df.to_csv(filePath, index=False)

    @staticmethod
    def dir2CSV(dataDirs: list[str], save_dir: str) -> str:

        import glob

        files = []
        for dataDir in dataDirs:
            assert os.path.exists(dataDir), (
                "The following path does not seem to exist: " + dataDir
            )
            files += glob.glob(os.path.join(dataDir, "**", "*.csv"), recursive=True)
        assert files, "Looks empty"
        os.makedirs(save_dir, exist_ok=True)
        dataCol = "filePath"
        df: pd.DataFrame = pd.DataFrame.from_dict({dataCol: files})
        path = os.path.join(save_dir, "smlm.csv")
        df.to_csv(path)
        logger.info("Saved here: %s", path)
        return path

    def Sample2df_index(self, sample) -> int:

        if sample in self.samples:
            return self.datasetDF.loc[
                self.datasetDF[self.sampleColumn] == sample, :
            ].index.tolist()[0]
        else:
            raise KeyError(f"Label {sample} does not exist in dataset")

    def getFilePathFromDF(self, index) -> str:
        label: str = self.samples[index]
        dfindex: int = self.Sample2df_index(label)
        dir: str = self.datasetDF.loc[dfindex, self.sampleColumnCSV]
        return dir #os.path.join(dir, label)

    @staticmethod
    def getVertices(
        item_df: pd.Series, verticesCol: list[str]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.asarray(np.array(item_df[verticesCol]).tolist()).squeeze()

    @staticmethod
    def file2Dataframe(file: Union[str, Path]):
        return SampleLoaderBioImage.loadData(file)

    @staticmethod
    def file2Vertices(
        file, verticesCol, verticesInstanceLabelCol
    ) -> Tuple[np.ndarray[Tuple[Any], np.dtype[Any]]]:
        dataframe : pd.DataFrame = VerticesDataset.file2Dataframe(file=file)

        if not all(isinstance(eval(col), str) for col in dataframe.columns):
            if all(isinstance(eval(col), (int, float)) for col in dataframe.columns):
                dataframe.loc[-1] = [eval(x) for x in dataframe.columns.tolist()]
                dataframe.index = dataframe.index + 1
                dataframe.sort_index(inplace=True)
            n = len(verticesCol)
            dataframe.rename(
                columns=dict(zip(dataframe.columns[:n], verticesCol)), inplace=True)

        vertices: np.ndarray = VerticesDataset.getVertices(dataframe, verticesCol)

        if isinstance(dataframe, pd.DataFrame):
            label: np.ndarray = (
                dataframe.get(
                    verticesInstanceLabelCol,
                    pd.Series(np.zeros(vertices.shape[0])),
                )
                .to_numpy()
                .squeeze()
            )
        else:
            # We use this else statemente incase we are dealing with a pyddata object
            if verticesInstanceLabelCol in dataframe:
                label = np.asarray(
                    dataframe[verticesInstanceLabelCol].tolist()
                ).squeeze()
            else:
                label = pd.Series(np.zeros(vertices.shape[0])).to_numpy().squeeze()

        return vertices, label

    @staticmethod
    def file2Localization(
        file: Union[str, Path],
        verticesCol: Union[str, list[str]],
        verticesInstanceLabelCol: str,
    ) -> Vertices:

        vertices, label = VerticesDataset.file2Vertices(
            file,
            verticesCol=verticesCol,
            verticesInstanceLabelCol=verticesInstanceLabelCol,
        )

        return Vertices(Data(vertices, label), file)
    
    @staticmethod
    def arrayDFPreprocessing(array: Union[list, np.ndarray]):
        if isinstance(array, np.ndarray):

            if array.ndim > 2:
                raise NotImplementedError(
                    "Handling for multi-column sorted_nucl_labels is not implemented."
                )
            else:
                array = array.tolist()
        return array

    @staticmethod
    def addNewColumn2DF(array: Union[list, np.ndarray], df, column_label_name):
        elementList = VerticesDataset.arrayDFPreprocessing(array)
        assert (
            elementList.__len__() == df.shape[0]
        ), f"The indexable object must be of the same length as the dataframe but we have: {elementList.__len__()} and {df.shape[0]}"
        df[column_label_name] = elementList
        return df

    @staticmethod
    def save2DFcolumn(
        sorted_results: list,
        sorted_nucl_labels: np.ndarray,
        dataframe: pd.DataFrame,
        column_name: str = "Embedding",
        column_label_name: str = "cluster_id",
    ) -> pd.DataFrame:
        """
        Adds a new column to the dataframe with values from sorted_results,
        mapped according to sorted_nucl_labels.

        Parameters:
        - sorted_results: List of values to be added as the new column.
        - sorted_nucl_labels: 1D or 2D numpy array of nucleus labels.
        - dataframe: The DataFrame to which the new column will be added.
        - column_name: The name of the new column (default is "Embedding").

        Returns:
        - Updated DataFrame with the new column.
        """

        sorted_nucl_labels = np.asarray(sorted_nucl_labels)
        # Accept a column/row vector (shape [n, 1] or [1, n]); reject genuine 2D label sets.
        if sorted_nucl_labels.ndim == 2:
            if 1 in sorted_nucl_labels.shape:
                sorted_nucl_labels = sorted_nucl_labels.ravel()
            else:
                raise NotImplementedError(
                    "Handling for multi-column sorted_nucl_labels is not implemented."
                )

        sorted_results = VerticesDataset.arrayDFPreprocessing(sorted_results)

        # Create a dictionary for fast lookup of results by label
        label_to_result = dict(zip(sorted_nucl_labels, sorted_results))

        # Map each NUCLEUS_LABEL_KEY in the dataframe to its corresponding result, or NaN if not found
        dataframe[column_name] = (
            dataframe[column_label_name].map(label_to_result).fillna(np.nan)
        )

        return dataframe
    

    def __getitem__(self, index: int) -> Union[None, Vertices]:

        try:
            file: str = self.getFilePathFromDF(index)

            verticesCol = self.verticesCol
            if isinstance(self.verticesCol, pd.DataFrame):
                verticesCol: list = self.verticesCol.iloc[index]

            verticesInstanceLabelCol = self.verticesInstanceLabelCol
            if isinstance(self.verticesInstanceLabelCol, pd.DataFrame):
                verticesInstanceLabelCol: list = self.verticesInstanceLabelCol.iloc[
                    index
                ]

            return VerticesDataset.file2Localization(
                file, verticesCol, verticesInstanceLabelCol
            )

        except Exception:
            # Returning None lets collate_fn drop the sample; log with traceback.
            logger.exception(
                "Failed to load sample at index %s (file: %s)",
                index, self.getFilePathFromDF(index),
            )
            return None


