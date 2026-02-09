from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import skimage
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,  confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import anndata as ad
import scanpy
from dl_utils import (NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY)

sns.set_context("poster")

# TODO: Do not add these packages as they are not useable for an utility package

# import anndata as ad
# import scanpy


def savedataframe(dataframe, save_dir,  **kwargs):
    # Remove unnamed columns
    # Remove unnamed columns
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    dataframe.drop(
        columns=dataframe.columns[dataframe.columns.duplicated()], inplace=True)
    dataframe.drop_duplicates(subset=NUCLEUS_LABEL_KEY, inplace=True)

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
                raise ValueError(
                    "Sorted nucleus labels and sorted results do not match in length")
        else:
            raise NotImplementedError(
                "Handling for multi-column sorted_nucl_labels is not implemented.")

    if isinstance(sorted_results, np.ndarray):

        if sorted_results.ndim > 2:
            raise NotImplementedError(
                "Handling for multi-column sorted_nucl_labels is not implemented.")
        else:
            sorted_results = sorted_results.tolist()

    # Create a dictionary for fast lookup of results by label
    label_to_result = dict(zip(sorted_nucl_labels, sorted_results))

    # Map each NUCLEUS_LABEL_KEY in the dataframe to its corresponding result, or NaN if not found
    dataframe[column_name] = dataframe.label_id.map(
        label_to_result).fillna(np.nan)

    return dataframe


def convert_array_to_data_url(path):
    """
    Convert a 2D NumPy array to a base64-encoded image.
    Args:
        array: A 2D NumPy array (grayscale image).
    Returns:
        A base64-encoded image as a data URL.
    """
    array = skimage.io.imread(path)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(array, cmap="gray")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded_image}"


def get_array_from_df(df, column):
    """Extracts and converts a column from a DataFrame to a NumPy array."""
    # Try to get a column, if does not exist give null array of dataframe length
    # if column not in df.columns:
    #     return np.array([None] * len(df))
    #return np.asarray(np.array(df[column]).tolist()).squeeze()
    return np.array(df[column].tolist())


class Classification:

    # Try random forest

    @staticmethod
    def train_classifier(embeddings, labels, method="KNN", metric="cosine"):
        """
        Train a classifier based on the specified metric.
        """

        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=5, metric=metric),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        if method not in classifiers:
            raise ValueError(
                f"Unsupported classifier method: {method}. Available methods are: {list(classifiers.keys())}")

        classifier = classifiers[method]
        classifier.fit(embeddings, labels)
        return classifier

    @staticmethod
    def createConfusionMatrixFigure(x, gt, classifier, save_dir=None):
        # Predict for missing cell types
        """
        Predict using the classifier and evaluate if ground truth labels are provided.
        """
        y_pred = classifier.predict(x)
        # If label mapping is provided, map the predicted and true labels
        ks: list[int] = classifier.classes_
        cm = confusion_matrix(gt, y_pred, normalize="true")
        ticks = ks
        # Plot confusion matrix using Seaborn
        fig, ax = plt.subplots(1, 1)
        ax = sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            cmap="Blues",
            xticklabels=ticks,
            yticklabels=ticks,
            cbar=False,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("Ground Truth Labels")
        plt.xticks(rotation=45)
        plt.yticks()
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, "confusion_matrix.png")
            )  # Save the confusion matrix image
        plt.close()
        return fig

    @staticmethod
    def getAccuracy(x, gt, classifier):
        y_pred = classifier.predict(x)
        accuracy = accuracy_score(gt, y_pred)
        # print(f"Accuracy: {accuracy:.2f}")
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))
        # probabilities = classifier.predict_proba(X_test)
        return accuracy


class EmbeddingAnalysis:
    def __init__(self, df_path=r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_resnet_masking_075_patches4096\results\epoch_99\dataframe_analyzed.json", labelColumn=NUCLEUS_LABEL_KEY,
                 type=MASKED_FEATURES_KEY, save_dir=None):
        # json or tab ?
        self.df_path = df_path
        self.type = type  # TOKEN_KEY #MASKED_AVG_TOKEN_FEATURES_KEY #FEATURES_KEY #TOKEN_KEY #MASKED_FEATURES_KEY #AVG_TOKEN_FEATURES_KEY #MASKED_AVG_TOKEN_FEATURES_KEY #AVG_TOKEN_FEATURES_KEY #MASKED_AVG_TOKEN_FEATURES_KEY  #TOKEN_KEY #FEATURES_KEY
        # assert dfpath is json
        assert df_path.endswith('.json'), "Dataframe path must be a JSON file."
        self.data_df = pd.read_json(df_path)
        self.embeddings = StandardScaler().fit_transform(
            get_array_from_df(self.data_df, self.type))

        assert isinstance(
            self.embeddings, np.ndarray), f"The embeddings are instead: {type(self.embeddings)}"
        print(f"The embeddings are of shape: {self.embeddings.shape}")
        self.labelColumn = labelColumn
        self.labels = get_array_from_df(self.data_df, labelColumn)
        self.save_dir = save_dir
        self.classColumn = "cluster"
        self.classification = Classification

    def tailorClassColour(self, class_ids):
        # TODO: Confirm that predLabels is generated
        temp = self.subplots_kwargs
        temp["c"] = costumMatplotlib.labels2colors(class_ids)
        return temp

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[self.labelColumn] == label]

    def get_array_from_df(self, column):
        """Extracts and converts a column from a DataFrame to a NumPy array."""
        # Try to get a column, if does not exist give null array of dataframe length
        # if column not in df.columns:
        #     return np.array([None] * len(df))

        return np.array(self.data_df[column].tolist())

    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color",
                       legend_title: str = "Nuclei labels", save_dir: str = "./"):
        import plotly.express as px
        import os
        from . import MoBie_coloring

        color_space = MoBie_coloring.GlasbeyARGBLut()
        if classColoumn not in self.data_df.columns:
            self.data_df[classColoumn] = 0

        classLabels = self.data_df[classColoumn].unique().astype(int).tolist()
        self.data_df[classColoumn] = self.data_df[classColoumn].astype(
            np.int16)
        map_cluster_2_color = {
            k: f"rgba{color_space.rgba_tuple_by_index(k)}"
            for k in classLabels
        }
        map_cluster_2_color[0] = "rgba(128, 128, 128, 0.5)"

        # Define the patches directory
        patches_dir = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\organoidTestData\patches"

        # Add image HTML tags to the DataFrame
        # self.data_df['image_html'] = self.data_df.index.map(
        #     # +1 because your files are named nucleus_hr_1.npy, etc.
        #     lambda idx: self.load_png_for_nucleus(idx + 1, patches_dir)
        # )

        fig = px.scatter(
            self.data_df,
            x=xColumn,
            y=yColumn,
            color=classColoumn,
            color_discrete_map=map_cluster_2_color
        )

        # Custom hover template to display images
        fig.update_traces(
            hovertemplate="<b>Cluster: %{color}</b><br>" +
            "%{customdata[0]}" +
            "<extra></extra>"
        )

        # Customize layout
        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                font=dict(size=16),
                yanchor="bottom",
                xanchor="left",
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="black",
                borderwidth=1,
                x=1,
                y=0
            ),
            width=1000,
            height=800,
            legend_title_text=legend_title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
        )

        # Save the plot as an HTML file
        output_path = os.path.join(save_dir, f"{classColoumn}_UMAP.svg")
        fig.write_image(output_path)
        # fig.write_html(output_path)

        # Show plot
        fig.show()

    def UMAP(self, n_neighbors=15, min_dist=0.1,
             n_components=2, random_state=42, metric="euclidean", **kwargs):

        import umap

        # Fit UMAP to reduce 1D embeddings to 2D for visualization
        umap_reducer = umap.UMAP(n_neighbors=n_neighbors,
                                 min_dist=min_dist,
                                 n_components=n_components,
                                 random_state=random_state,
                                 metric=metric, **kwargs).fit(self.embeddings)

        umap_array = umap_reducer.transform(self.embeddings)

        self.data_df = save2DFcolumn(umap_array[:, 0],
                                     sorted_nucl_labels=self.labels,
                                     dataframe=self.data_df,
                                     column_name="UMAP x")

        self.data_df = save2DFcolumn(umap_array[:, 1],
                                     sorted_nucl_labels=self.labels,
                                     dataframe=self.data_df,
                                     column_name="UMAP y")

        savedataframe(self.data_df, os.path.dirname(
            self.df_path), typie="analyzed")

    def pca(self):

        # Perform PCA
        pca_model = PCA()
        print(f"Detected following type for emebddings: {type(self.embeddings)}")
        # if not isinstance(self.embeddings, np.ndarray):
        #     print(f"Detected following type for emebddings: {type(self.embeddings)}")
        #     self.embeddings = np.array(self.embeddings)
        return pca_model.fit_transform(self.embeddings)

    def clustering(self, resolution: float, n_iterations: int, n_neighbors: int, distance_metric: str = "euclidean"):

        labels = np.zeros(self.embeddings.shape[0])

        embedding = ad.AnnData(X=self.embeddings)

        scanpy.pp.neighbors(embedding, n_neighbors=n_neighbors,
                            n_pcs=None,
                            metric=distance_metric,
                            random_state=111)

        adata = scanpy.tl.leiden(embedding, resolution=resolution,
                                 random_state=111, n_iterations=n_iterations, copy=True)

        # Map the subcluster labels back to the main dataframe
        for indx, sub_label in enumerate(adata.obs["leiden"].unique()):
            indices = adata.obs[adata.obs["leiden"]
                                == sub_label].index.astype(int)
            labels[indices] = indx

        self.predLabels = labels.astype(int) + 1
        self.data_df[self.classColumn] = self.predLabels

        return self.predLabels


if __name__ == '__main__':
    df_path = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_masking_075_patch4 SAM masks\results\epoch_10\dataframe_analyzed.json"
    embeddingAnalysis = EmbeddingAnalysis(df_path=df_path)

    embeddingAnalysis.UMAP()

    embeddingAnalysis.clustering(
        resolution=0.05, n_iterations=10, n_neighbors=5)

    args = {
        "xColumn": "UMAP x",
        "yColumn": "UMAP y",
        "classColoumn": embeddingAnalysis.classColumn,
        "legend_title": "Nuclei labels",
        "save_dir": os.path.dirname(embeddingAnalysis.df_path)
    }

    embeddingAnalysis.specialScatter(**args)

    embeddingAnalysis.generateMask()
