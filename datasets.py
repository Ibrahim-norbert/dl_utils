import glob
import logging
import numbers
import os

from numpy import ndarray, dtype
from skimage.transform import resize
import shutil

import numpy as np
import pandas as pd
import yaml
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from torch.utils.data import DataLoader

from dl_utils import LABEL_KEY
from dl_utils.SampleLoader import SampleLoaderBioImage
from dl_utils.SampleTypes import Data, Vertices
from dl_utils import util_base as util

logger = logging.getLogger(__name__)
class BaseDataset:
    """Base class for nuclei data
    Calculates general properties, loads low resoltion (s3) nuclei"""

    # Class-scoped so subclasses and constructors can retarget them; assigning these in
    # __init__ silently overwrote the overrides subclasses declare (see
    # BaseImageCollectionDataset.sampleColumn).
    samplePathColumn: str = "sample_path"
    sampleColumn: str = "index"

    def __init__(self, sampleMapperDFPath: str, save_dir: str, samplePathRegex: str = None, **kwargs) -> None:

        self.sampleMapperDFPath = sampleMapperDFPath

        # NOTE: Dataset dataframe must be a sample mapper, where each sample a path is
        self.sampleMapperDF: pd.DataFrame = self.loadDataFrame(sampleMapperDFPath)

        self.samplePathRegex = samplePathRegex
        self.validateSamplePaths()

        if self.sampleColumn not in self.sampleMapperDF.columns:
            self.sampleMapperDF[self.sampleColumn] = np.arange(len(self.sampleMapperDF))
        assert not self.sampleMapperDF.duplicated(subset=self.sampleColumn).any(), (
            "Ensure sample mapper does not have duplicates"
        )
        self.samples: np.ndarray = self.sampleMapperDF[self.sampleColumn].to_numpy()
        assert save_dir is not None and isinstance(save_dir, str), f"The save directory must be a string, not {save_dir}"
        self.save_dir: str = os.path.abspath(save_dir)

    def validateSamplePaths(self) -> None:
        """Contract the sample paths must satisfy — one directory per dataset.

        A hook rather than an inline assertion so that datasets deliberately spanning
        several directories (:class:`dl_utils.n5_datasets.MultiDirectoryN5Dataset`) can
        replace the contract instead of working around it.
        """
        if self.sampleMapperDF[self.samplePathColumn].map(os.path.dirname).nunique() > 1:
            raise ValueError(
                "All sample paths must be located in the same directory"
            )


    # NOTE: writeChunkedImage, writeVolumePair and the openVolume/readVolume/readBBox/
    # readZSlice family were removed here. They were a second N5 implementation
    # duplicating dl_utils.n5_datasets.BaseVolumeDataset, and the read family depended on
    # createN5 / n5Paths / dapiKey / rawKey, none of which this class ever defined — so
    # every call into it raised AttributeError. Use BaseVolumeDataset / BaseMaskDataset.

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

    @staticmethod
    def loadDataFrame(datasetPath: str) -> pd.DataFrame:
        data_df = SampleLoaderBioImage.loadData(datasetPath)
        assert isinstance(data_df, pd.DataFrame)
        return data_df

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

    def row(self, idx: int) -> pd.Series:
        """Positional row lookup.

        DataLoader samplers yield positions, not index labels, so every ``__getitem__``
        path must use ``.iloc``. Mixing ``.loc`` in is only correct while the index
        happens to be a clean RangeIndex.
        """
        return self.sampleMapperDF.iloc[int(idx)]

    @staticmethod
    def write_index(df: pd.DataFrame, path: str) -> None:
        """Single writer for index CSVs.

        Never writes the DataFrame index, so a subsequent read cannot produce the
        ``Unnamed: 0`` column that previously leaked into metadata merges. Any such
        column already present is dropped, so repeated rebuilds cannot accrete them.
        """
        df.loc[:, ~df.columns.str.match(r"^Unnamed")].to_csv(path, index=False)

    @staticmethod
    def read_index(path: str) -> pd.DataFrame:
        """Single reader, matching :meth:`write_index`.

        Yields a positional RangeIndex. Tolerates legacy files that were written with
        the index by dropping the resulting ``Unnamed`` column.
        """
        df = pd.read_csv(path)
        return df.loc[:, ~df.columns.str.match(r"^Unnamed")]

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

    @staticmethod
    def loadtiffVolume(path: str) -> np.ndarray:
        """Load a 3-D TIFF volume from *path*."""
        import skimage.io
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".tif", ".tiff"):
            raise ValueError(
                f"Unsupported file format: {ext}. Supported formats are .tif, .tiff"
            )
        return skimage.io.imread(path)

    def prepareVolume(self, volumePath: str, maskPath: Optional[str] = None,
                      resolution: Optional[List[int]] = None,
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a raw/mask TIF pair and pre-process it, ready for N5 saving.

        1. Load the raw TIF.
        2. Resample to isotropic voxels using *resolution* (ZYX) — order 3 for the raw
           intensities, order 0 (nearest neighbour) for the label mask, so labels survive.
        3. Min-max normalise the raw volume to ``[0, 1]``.

        *maskPath* may be ``None`` or missing, in which case an all-zero mask matching the
        resampled shape is returned — what raw-only consumers such as SimCLR need.

        TODO: overlaps BaseVolumeDataset.prepareSampleVolume, which does the same work but
        yields uint16 rather than float32 [0, 1]. This one survives because it serves the
        per-volume LMTextureNucleiDataset.volume2DF pipeline, not the N5 dataset classes;
        collapse the two once that pipeline moves onto BaseVolumeDataset.
        """
        from sklearn import preprocessing

        resolution = list(resolution or self.resolution)
        volume: np.ndarray = self.loadtiffVolume(volumePath)

        zoom = np.array(resolution, dtype=float) / min(resolution)
        isotropic = all(z == 1.0 for z in zoom)
        logger.debug("Pre-resampled dimension: %s", volume.shape)

        if not isotropic:
            newShape = tuple(int(s * z) for s, z in zip(volume.shape, zoom))
            volume = resize(volume, newShape, anti_aliasing=True, order=3)
            logger.debug("Resampled dimension: %s", volume.shape)

        volume = (preprocessing.MinMaxScaler()
                  .fit_transform(volume.reshape(-1, 1))
                  .reshape(volume.shape)
                  .astype(np.float32))

        if maskPath and os.path.exists(maskPath):
            mask: np.ndarray = self.loadtiffVolume(maskPath)
            if not isotropic:
                nLabels = np.unique(mask).size
                mask = resize(mask, volume.shape, anti_aliasing=True, order=0)
                if np.unique(mask).size != nLabels:
                    raise ValueError(
                        "Label count changed during mask resampling: "
                        f"{nLabels} -> {np.unique(mask).size}"
                    )
        # TODO: Do not save an empty mask if no masks exist instead raise error
        else:
            mask = np.zeros(volume.shape, dtype=np.uint16)

        return volume, mask.astype(np.uint16)

    # NOTE: no __getstate__/__setstate__ here. They existed only to strip cached z5py
    # handles; the N5 datasets now open a handle per read and cache nothing, so there is
    # no unpicklable state to drop and subclasses pickle by the default protocol.


class BaseImageCollectionDataset(BaseDataset):
    """CSV-indexed image collection for CNN-style models.

    One row is one image. Deliberately carries **no** N5 conversion, patch geometry,
    positional embeddings or master-index machinery — those belong to the patch-based
    transformer datasets and are pure overhead for a convolutional encoder.

    Subclasses implement :meth:`load_image`; ``__getitem__`` returns a plain
    ``(image, label)`` pair unless the subclass overrides it.
    """

    volumePathColumn: str = "dataset path"
    sampleColumn: str = "label_id"

    def __init__(
        self,
        datasetDataframePath: str,
        DFBinaryFilters: Optional[Union[str, List[str]]] = None,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        # NOTE: BaseDataset.__init__ is intentionally not called — it builds a master
        # index this class does not need. The two attributes it would set are set here.
        self.datasetConfig: dict = kwargs
        self.datasetDataframePath: str = datasetDataframePath

        df = self.loadDataFrame(datasetDataframePath)
        columns = ([DFBinaryFilters] if isinstance(DFBinaryFilters, str)
                   else list(DFBinaryFilters or []))
        for col in columns:
            if col in df.columns:
                df = df[df[col] == 1]
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        self.datasetDF: pd.DataFrame = df.reset_index(drop=True)
        # Artefact root, resolved once by the common base class from save_dir plus the
        # directory basename of the volume paths. Must follow the load above, which is
        # what supplies those paths.
        self.initDatasetDir(save_dir or kwargs.get("save_dir"),
                            sampleMapperDFPath=(str(self.datasetDF[self.volumePathColumn].iloc[0])
                                                if self.volumePathColumn in self.datasetDF.columns
                                                   and len(self.datasetDF) else datasetDataframePath))
        # TIF -> N5 for every listed volume, in the main process, so reads in
        # __getitem__ are plain N5 slice fetches.
        self.initVolumeStore(self.datasetDF, self.volumePathColumn,
                             getattr(self, "maskPathColumn", None),
                             kwargs.get("resolution"))
        if self.sampleColumn not in self.datasetDF.columns:
            self.datasetDF[self.sampleColumn] = self.datasetDF.index
        self.samples: np.ndarray = self.datasetDF[self.sampleColumn].to_numpy()

    def load_image(self, path: str):
        raise NotImplementedError

    def row(self, idx: int) -> pd.Series:
        """Positional row lookup — samplers yield positions, never index labels."""
        return self.datasetDF.iloc[int(idx)]

    def __len__(self) -> int:
        return len(self.datasetDF)

    def __getitem__(self, idx: int):
        row = self.row(idx)
        return self.load_image(str(row[self.volumePathColumn])), int(row[self.sampleColumn])


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
        # NOTE: datasetDir here is an INPUT directory to scan for CSVs, not the artefact
        # root the collection datasets give the same name. This class never calls

        # TODO: rename this to inputDir to retire the collision outright.
        self.datasetDir = datasetDir

        if self.datasetDir is not None and os.path.exists(self.datasetDir):
            datasetPath = self.dir2CSV(
                [self.datasetDir], os.path.join(self.datasetDir, "..", "..", "..")
            )

        super().__init__(sampleColumn=sampleColumn, **kwargs)

        self.sampleColumnCSV: str = sampleColumnCSV

        self.verticesCol: Union[str, list[str]] = verticesCol
        self.standardizedLabels = standardizedLabels
        self.standardizedCoords = standardizedCoords

        if self.sampleMapperDF.columns.isin(self.standardizedCoords).all():
            self.verticesCol: pd.DataFrame = self.sampleMapperDF.loc[
                :, self.standardizedCoords
            ]

        if self.standardizedLabels in self.sampleMapperDF.columns:
            self.standardizedLabels = self.sampleMapperDF.loc[:, self.standardizedLabels]

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
            return self.sampleMapperDF.loc[
                   self.sampleMapperDF[self.sampleColumn] == sample, :
                   ].index.tolist()[0]
        else:
            raise KeyError(f"Label {sample} does not exist in dataset")

    def getFilePathFromDF(self, index) -> str:
        label: str = self.samples[index]
        dfindex: int = self.Sample2df_index(label)
        dir: str = self.sampleMapperDF.loc[dfindex, self.sampleColumnCSV]
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


