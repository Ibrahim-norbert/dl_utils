from dl_utils.SampleLoader import SampleLoaderBioImage
from torch.utils.data import Subset, Dataset
import numpy as np
import pandas as pd
import os
import glob
import yaml
from functools import cached_property
from typing import Any, List, Literal, Optional, Tuple, Union
from pathlib import Path
from torch.utils.data import DataLoader
from dl_utils.SampleTypes import Data, Vertices
from dl_utils import LABEL_KEY



class BaseDataset(Dataset):
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
            self.standardizedLabels = self.datasetDf.loc[:, self.standardizedLabels]

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
        print(f"Saved here: {path}")
        df.to_csv(path)
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

        if not isinstance(sorted_nucl_labels, np.ndarray):
            sorted_nucl_labels = np.array(sorted_nucl_labels)
        # Flatten sorted_nucl_labels if it has only one column (2D array with shape [n, 1])
        if sorted_nucl_labels.ndim == 2:
            if sorted_nucl_labels.shape[0] == 1 or sorted_nucl_labels.shape[1] == 1:
                if len(sorted_nucl_labels) == sorted_nucl_labels.size:
                    sorted_nucl_labels = sorted_nucl_labels.flatten()
                else:
                    raise ValueError(
                        "Sorted nucleus labels and sorted results do not match in length"
                    )
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

        except Exception as e:
            print(
                f"That is the following problem at index {index} and file : {self.getFilePathFromDF(index)}: {e}"
            )


