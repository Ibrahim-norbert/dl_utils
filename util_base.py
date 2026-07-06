import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import psutil
import yaml

# Import the module (not the values) so that NUCL_TABLE / LM_DF are read at call
# time. These constants start as None and are reassigned elsewhere at runtime;
# `from .constants import NUCL_TABLE` would bind the None value permanently.
from . import constants
from .constants import NUCLEUS_LABEL_KEY


def replaceFileExt(filePath : str, newExt : str):
    fileExt = os.path.splitext(filePath)[-1]
    return filePath.replace(fileExt, newExt)

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

    sorted_nucl_labels = np.asarray(sorted_nucl_labels)
    # Accept a column/row vector (shape [n, 1] or [1, n]) but reject genuine 2D label sets.
    if sorted_nucl_labels.ndim == 2:
        if 1 in sorted_nucl_labels.shape:
            sorted_nucl_labels = sorted_nucl_labels.ravel()
        else:
            raise NotImplementedError("Handling for multi-column sorted_nucl_labels is not implemented.")

    if isinstance(sorted_results, np.ndarray):
        if sorted_results.ndim > 2:
            raise NotImplementedError("Handling for >2D sorted_results is not implemented.")
        sorted_results = sorted_results.tolist()

    # Create a dictionary for fast lookup of results by label
    label_to_result = dict(zip(sorted_nucl_labels, sorted_results))

    # Map each NUCLEUS_LABEL_KEY in the dataframe to its corresponding result, or NaN if not found
    dataframe[column_name] = dataframe.label_id.map(label_to_result).fillna(np.nan)

    return dataframe


def savedataframe(dataframe, save_dir,  **kwargs):
    # Remove unnamed columns
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    dataframe.drop(columns=dataframe.columns[dataframe.columns.duplicated()], inplace=True)
    dataframe.drop_duplicates(subset=NUCLEUS_LABEL_KEY, inplace = True)

    dataframe.reset_index(inplace=True, drop=True)

    if not "modality" in dataframe.columns:
        if "LM" in save_dir:
            dataframe["modality"] = "LM"
        elif "EM" in save_dir:
            dataframe["modality"] = "EM"

    dataframe.to_json(get_savedf_path(save_dir, **kwargs))

def get_savedf_path(save_dir: str, typie=''):

    if typie != "":
        return os.path.join(save_dir, f"dataframe_{typie}.json")
    else:
        return os.path.join(save_dir, f"dataframe.json")


def readdataframe(path: str, name='') -> pd.DataFrame:

    if "json" in path and "dataframe" in path and os.path.exists(path):
        data_df = pd.read_json(path)
    else:
        assert name is not None, ("If path is save_dir then please do not set parameter 'name' as None")

        if os.path.exists(path):
            path = get_savedf_path(path, typie=name)

        if not os.path.exists(path):
            print(f"Provided path does not exist: {path}")
            return None

        data_df = pd.read_json(path)

    if not "modality" in data_df.columns:

        if "LM" in path:
            data_df["modality"] = "LM"
        elif "EM" in path:
            data_df["modality"] = "EM"

    return data_df


def loadcheckpoints(train_run_path: str):
    return sorted(glob.glob(os.path.join(train_run_path, '*.pth')), key=os.path.getmtime)


class CustomArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        # Parse known arguments
        args = super().parse_args(*args, **kwargs)

        # Identify specified and default arguments
        specified = {}
        defaults = {}

        for action in self._actions:
            arg = action.dest
            if arg is not None and hasattr(args, arg):  # Skip actions without a dest attribute
                value = getattr(args, arg)
                if value != action.default:
                    specified[arg] = value
                else:
                    defaults[arg] = value

        return args, specified, defaults


def update_args_from_yaml(args, config_file):
    if config_file is not None:
        with open(config_file, "r") as file:
            yaml_args = yaml.safe_load(file)
            for key, value in yaml_args.items():
                setattr(args, key, value)
    return args


def clean_config(config: dict):
    """
    Clean the configuration dictionary by:
    - Removing keys with None values.
    - Ensuring values are of type float, int, str, or None.
    - (Optional) Sorting keys alphabetically.

    Args:
        config (dict): Configuration dictionary to clean.

    Returns:
        dict: Cleaned configuration dictionary.
    """
    # Allowed value types
    allowed_types = (float, int, str, type(None))

    # Remove keys with invalid or None values
    cleaned_config = {
        k: v for k, v in config.items()
        if isinstance(v, allowed_types)
    }

    return cleaned_config


def log_memory_usage():
    process = psutil.Process()  # Get current process
    memory_info = process.memory_info()  # Get memory details
    print(f"RSS (Resident Set Size): {memory_info.rss / 1e6:.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1e6:.2f} MB")


def save_img(label, masked=None, img=None, mask=None, savename="", save_dir=os.getcwd()):

    filename = f"{label}_"
    if masked is not None:
        path = os.path.join(save_dir, filename + "masked_" + savename + ".npy")
        np.save(path, masked)

    elif img is not None:
        path = os.path.join(save_dir, filename + "img_" + savename + ".npy")
        np.save(path, img)

    elif mask is not None:
        path = os.path.join(save_dir, filename + "mask_" + savename + ".npy")
        np.save(path, mask)


def mkdir(root_dir, dirname="patches"):

    dir = os.path.join(root_dir, dirname)

    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir


def mkresultsdir(training_dir):
    return mkdir(training_dir, dirname="results")


def mkepochdir(training_dir, epoch):

    save_dir = mkresultsdir(training_dir)

    epochdir = os.path.join(save_dir, f"epoch_{epoch}")

    if not os.path.exists(epochdir):
        os.mkdir(epochdir)

    return epochdir


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def patchify(vol, patch_size):
    """
    Extract patches from the input volume, agnostic to 2D or 3D inputs.

    Parameters:
    vol: numpy.ndarray of shape (Y, X) for 2D or (Z, Y, X) for 3D
        Each spatial dimension must be divisible by patch_size (they need not be equal).

    Returns:
    numpy.ndarray of shape (L, *([patch_size] * ndim))
        2D: (L, p, p)      with L = (Y/p) * (X/p)
        3D: (L, p, p, p)   with L = (Z/p) * (Y/p) * (X/p)
    """
    p = patch_size
    ndim = vol.ndim

    assert all(s % p == 0 for s in vol.shape), \
        f"Input dimensions {vol.shape} must be divisible by the patch size {p}."

    grid = tuple(s // p for s in vol.shape)  # patches per spatial axis

    # Reshape so each spatial axis splits into (grid_i, p):
    #   2D: (h, p, w, p)      3D: (d, p, h, p, w, p)
    split_shape = tuple(x for g in grid for x in (g, p))
    x = vol.reshape(split_shape)

    # Group all grid axes first, then all patch axes:
    #   2D: 'hpwq->hwpq'      3D: 'dfhpwq->dhwfpq'
    src = ''.join(chr(ord('a') + i) for i in range(2 * ndim))
    grid_axes = src[0::2]   # even positions = grid indices
    patch_axes = src[1::2]  # odd positions = within-patch indices
    x = np.einsum(f'{src}->{grid_axes + patch_axes}', x)

    return x.reshape(int(np.prod(grid)), *([p] * ndim))


def merge_with_nucl_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a nucleus DataFrame with the global nucleus table (NUCL_TABLE for EM,
    LM_DF for LM) to attach bounding-box and anchor columns.

    The DataFrame must contain a 'modality' column ('EM' or 'LM').
    """
    assert "modality" in df.columns, \
        f"Please specify modality in dataframe: {list(df.columns)}"

    left_df = constants.NUCL_TABLE if "EM" in df["modality"].unique() else constants.LM_DF
    if left_df is None:
        raise RuntimeError(
            "constants.NUCL_TABLE / constants.LM_DF is not populated; "
            "set the relevant nucleus table before calling merge_with_nucl_table."
        )

    left_df = left_df.loc[:, [
        "label_id",
        "anchor_z", "anchor_x", "anchor_y",
        "bb_min_z", "bb_max_z", "bb_min_y", "bb_max_y", "bb_min_x", "bb_max_x",
    ]]

    assert df.label_id.isin(left_df.label_id).sum() == df.label_id.size, (
        f"Dataframe does not have nuclei labels as label_id: "
        f"{df.label_id.isin(left_df.label_id).sum()}/{df.label_id.size}"
    )

    df = df.merge(left_df, on="label_id", how="left", suffixes=(None, "_nucl"))

    cols2drop = [col for col in df.columns if col.endswith("_nucl")]
    if cols2drop:
        df.drop(columns=cols2drop, inplace=True)

    return df
