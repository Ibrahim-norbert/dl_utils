import numbers
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray, dtype
from typing import Any

from dl_utils import NUCLEUS_LABEL_KEY
from dl_utils import img2patch as img2ps

# class DataclassTypeError(Exception):
#     pass  # Custom error handling if needed

# ---------------------------------------------------------------------------
# Helper: extract a named column from a DataFrame as a NumPy array
# ---------------------------------------------------------------------------

def save2DFcolumn(
    sorted_results: list,
    sorted_nucl_labels: np.ndarray,
    dataframe: pd.DataFrame,
    column_name: str = "Embedding"
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
                raise  ValueError("Sorted nucleus labels and sorted results do not match in length")
        else:
            raise NotImplementedError("Handling for multi-column sorted_nucl_labels is not implemented.")

    if isinstance(sorted_results, np.ndarray):

        if sorted_results.ndim > 2:
            raise NotImplementedError("Handling for multi-column sorted_nucl_labels is not implemented.")
        else:
            sorted_results = sorted_results.tolist()

    # Create a dictionary for fast lookup of results by label
    label_to_result = dict(zip(sorted_nucl_labels, sorted_results))

    # Map each NUCLEUS_LABEL_KEY in the dataframe to its corresponding result, or NaN if not found
    dataframe[column_name] = dataframe.label_id.map(label_to_result).fillna(np.nan)

    return dataframe


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
        "mask path": [datasetMask_path] * len(labels),
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
