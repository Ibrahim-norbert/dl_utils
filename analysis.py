import base64
import os
from io import BytesIO



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,  confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from . import (NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY)
from .LM_preprocess import get_array_from_df
from .util import savedataframe

sns.set_context("poster")

# TODO: Do not add these packages as they are not useable for an utility package
# import anndata as ad
# import scanpy


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


class Classification:

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
        """
        Predict using the classifier and evaluate if ground truth labels are provided.
        """
        y_pred = classifier.predict(x)
        ks: list[int] = classifier.classes_
        cm = confusion_matrix(gt, y_pred, normalize="true")
        ticks = ks
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
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()
        return fig

    @staticmethod
    def getAccuracy(x, gt, classifier):
        y_pred = classifier.predict(x)
        return accuracy_score(gt, y_pred)


class EmbeddingAnalysis:
    def __init__(self, df_path=r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_resnet_masking_075_patches4096\results\epoch_99\dataframe_analyzed.json", type = MASKED_FEATURES_KEY, labelColumn=NUCLEUS_LABEL_KEY, save_dir=None):
        self.df_path = df_path
        self.type = type
        assert df_path.endswith('.json'), "Dataframe path must be a JSON file."
        self.data_df = pd.read_json(df_path)
        self.embeddings = StandardScaler().fit_transform(
            get_array_from_df(self.data_df, self.type))

        assert isinstance(
            self.embeddings, np.ndarray), f"The embeddings are instead: {type(self.embeddings)}"
        print(f"The embeddings are of shape: {self.embeddings.shape}")
<<<<<<< HEAD
        self.labelColumn = NUCLEUS_LABEL_KEY
        self.labels = get_array_from_df(self.data_df, NUCLEUS_LABEL_KEY)
        self.save_dir = os.path.dirname(df_path)
=======
        self.labelColumn = labelColumn
        self.labels = get_array_from_df(self.data_df, labelColumn)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
>>>>>>> origin/Project1Changes
        self.classColumn = "cluster"
        self.classification = Classification

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[NUCLEUS_LABEL_KEY] == label]

    def get_array_from_df(self, column):
        """Extracts and converts a column from self.data_df to a NumPy array."""
        return np.array(self.data_df[column].tolist())

    def load_png_for_nucleus(self, nucleus_id, patches_dir):
        """
        Load a PNG file for a nucleus and convert it to a base64-encoded image tag.

        Parameters:
        - nucleus_id: The ID of the nucleus.
        - patches_dir: Directory where the nucleus patches are stored.

        Returns:
        - HTML image tag with base64-encoded image or a message if the file doesn't exist.
        """
        png_filename = os.path.join(patches_dir, f"nucleus_hr_{nucleus_id}.png")

        if os.path.exists(png_filename):
            try:
                with Image.open(png_filename) as img:
                    img = img.resize((200, 200))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                return f'<img src="data:image/png;base64,{img_str}" width="200" height="200">'
            except Exception as e:
                print(f"Error loading image {png_filename}: {e}")
                return "Error loading image"
        else:
            npy_filename = os.path.join(patches_dir, f"nucleus_hr_{nucleus_id}.npy")
            if os.path.exists(npy_filename):
                try:
                    input_vol = np.load(npy_filename)
                    z_slice = input_vol.shape[0] // 2
                    img_array = input_vol[z_slice]
                    img_array = ((img_array - img_array.min()) /
                                 (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img = img.resize((200, 200))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return f'<img src="data:image/png;base64,{img_str}" width="200" height="200">'
                except Exception as e:
                    print(f"Error creating image from {npy_filename}: {e}")
                    return "Error creating image"
            else:
                return "No image available"

    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color",
<<<<<<< HEAD
                       legend_title: str = "Nuclei labels", save_dir: str = ""):
=======
                       legend_title: str = "Nuclei labels", save_dir: str = "./"):
>>>>>>> origin/Project1Changes
        
        import plotly.express as px
        from . import MoBie_coloring

        color_space = MoBie_coloring.GlasbeyARGBLut()
        if classColoumn not in self.data_df.columns:
            self.data_df[classColoumn] = 0

        classLabels = sorted(self.data_df[classColoumn].unique().astype(int).tolist())
        self.data_df[classColoumn] = self.data_df[classColoumn].astype(int).astype(str)
        map_cluster_2_color = {
            str(k): f"rgba{color_space.rgba_tuple_by_index(k)}"
            for k in classLabels
        }
        map_cluster_2_color["0"] = "rgba(128, 128, 128, 0.5)"

        patches_dir = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\organoidTestData\patches"

        self.data_df['image_html'] = self.data_df.index.map(
            lambda idx: self.load_png_for_nucleus(idx + 1, patches_dir)
        )

        fig = px.scatter(
            self.data_df,
            x=xColumn,
            y=yColumn,
            color=classColoumn,
            color_discrete_map=map_cluster_2_color,
            category_orders={classColoumn: [str(k) for k in classLabels]},
        )

        fig.update_traces(
            hovertemplate="<b>Cluster: %{color}</b><br>" +
                          "%{customdata[0]}" +
                          "<extra></extra>"
        )

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
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

        output_path = os.path.join(save_dir, f"{classColoumn}_UMAP.svg")
        fig.write_image(output_path)
        fig.show()


    def UMAP(self, n_neighbors=15, min_dist=0.1,
             n_components=2, random_state=42, metric="euclidean", **kwargs):


        import umap

        umap_reducer = umap.UMAP(n_neighbors=n_neighbors,
                                 min_dist=min_dist,
                                 n_components=n_components,
                                 random_state=random_state,
                                 metric=metric, **kwargs).fit(self.embeddings)

        umap_array = umap_reducer.transform(self.embeddings)

<<<<<<< HEAD
        self.data_df = save2DFcolumn(umap_array[:, 0],
                                     sorted_nucl_labels=self.labels,
                                     dataframe=self.data_df,
                                     column_name="UMAP x")
=======
        umap_df = pd.DataFrame({"UMAP x": umap_array[:, 0],
                                "UMAP y": umap_array[:, 1]},
                               index=self.data_df.index)
        
        return umap_df
    
    @staticmethod
    def concatColumnDF(df1, df2):
        return pd.concat([df1, df2], axis=1)
    
    def concatDF(self, df):
        self.data_df = self.concatColumnDF(self.data_df, df)
        return self.data_df
    
    def pca(self):
>>>>>>> origin/Project1Changes

        # Perform PCA
        pca_model = PCA()
        print(f"Detected following type for emebddings: {type(self.embeddings)}")
        # if not isinstance(self.embeddings, np.ndarray):
        #     print(f"Detected following type for emebddings: {type(self.embeddings)}")
        #     self.embeddings = np.array(self.embeddings)

        fg_pcs = pca_model.fit_transform(self.embeddings)

        print(f"Explained variance   : {pca_model.explained_variance_ratio_[:5].round(3)}")
        print(f"Cumulative (first 3) : {pca_model.explained_variance_ratio_[:3].sum():.3f}")
        return fg_pcs

    def clustering(self, resolution: float, n_iterations: int, n_neighbors: int, distance_metric: str = "euclidean"):
<<<<<<< HEAD

        import anndata as ad
        import scanpy

=======
        import anndata as ad
        import scanpy
>>>>>>> origin/Project1Changes
        labels = np.zeros(self.embeddings.shape[0])

        embedding = ad.AnnData(X=self.embeddings)

        scanpy.pp.neighbors(embedding, n_neighbors=n_neighbors,
                            n_pcs=None,
<<<<<<< HEAD
                            metric=distance_metric,  # type: ignore[arg-type]
=======
                            metric=distance_metric,
>>>>>>> origin/Project1Changes
                            random_state=111)

        scanpy.tl.leiden(embedding, resolution=resolution,
                         random_state=111, n_iterations=n_iterations)

<<<<<<< HEAD
        for indx, sub_label in enumerate(embedding.obs["leiden"].unique()):
            indices = embedding.obs[embedding.obs["leiden"]
                                    == sub_label].index.astype(int)
=======
        # Map the subcluster labels back to the main dataframe
        for indx, sub_label in enumerate(adata.obs["leiden"].unique()):
            indices = adata.obs[adata.obs["leiden"]
                                == sub_label].index.astype(int)
>>>>>>> origin/Project1Changes
            labels[indices] = indx

        self.predLabels = labels.astype(int) + 1
        self.data_df[self.classColumn] = self.predLabels

        return self.predLabels

    def generateMask(self, resolution=0.5,
                     n_iterations=10, n_neighbors=15,
                     distance_metric: str = "euclidean"):

        maskVolumePath = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\organoidTestData\dataset\mask\NS6_OE_06_w4SPI-405.tif"

        if not hasattr(self, "predLabels"):
            self.clustering(resolution=resolution, n_iterations=n_iterations,
                            n_neighbors=n_neighbors, distance_metric=distance_metric)

        maskVol = skimage.io.imread(maskVolumePath)

        max_id = maskVol.max()
        mapping_array = np.zeros(max_id + 1, dtype=np.uint16)
        mapping_array[self.labels] = self.predLabels

        maskVol = mapping_array[maskVol]

        fileName = os.path.basename(maskVolumePath).replace(".tif", "_clustered.tiff")
        output_dir = os.path.join(os.path.dirname(maskVolumePath), "..")
        path = os.path.join(output_dir, fileName)

        skimage.io.imsave(path, maskVol.astype(np.int16))
        print(f"Result saved to: {path}")


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

