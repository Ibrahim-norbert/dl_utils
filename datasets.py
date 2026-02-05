from dl_utils.SampleLoader import SampleLoaderBioImage
from torch.utils.data import Subset, Dataset
import numpy as np
import pandas as pd
import os
from typing import Literal
import numpy as np
import os
from typing import Any, List, Literal
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, List, Literal



class BaseDataset(Dataset):
    """Base class for nuclei data
    Calculates general properties, loads low resoltion (s3) nuclei"""

    def __init__(self, datasetDFPath: str, sampleColumn: str, **kwargs) -> None:
        
        self.datasetDFPath=datasetDFPath
        self.sampleColumn: str = sampleColumn
        self.datasetDF: pd.DataFrame = self.loadDataFrame(
            datasetDFPath=datasetDFPath)
        self.samples: np.ndarray[Literal["1"], np.dtype[np.int32]] = np.unique(
            self.get_all_labels(self.datasetDF, self.sampleColumn)
        )

    def loadDataFrame(self, datasetDFPath: str) -> pd.DataFrame:
        data_df = SampleLoaderBioImage.loadData(datasetDFPath)
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

