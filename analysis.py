import base64
import os
import pathlib
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
from matplotlib.figure import Figure
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from dl_utils import NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY, EMBED_KEY
from dl_utils.vizualizations import costumMatplotlib
from dl_utils.LM_preprocess import get_array_from_df

sns.set_context("poster")

# TODO: Do not add these packages as they are not useable for an utility package
# import anndata as ad
# import scanpy


def convert_array_to_data_url(path) -> str:
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
    encoded_image: str = base64.b64encode(buf.getvalue()).decode()
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
    def createConfusionMatrixFigure(x, gt, classifier, save_dir=None, mapping: dict=None) -> Figure:
        """
        Predict using the classifier and evaluate if ground truth labels are provided.
        """
        y_pred = classifier.predict(x)
        ks: list[int] = classifier.classes_
        cm = confusion_matrix(gt, y_pred, normalize="true")
        ticks = [mapping[k] for k in ks] if mapping is not None else ks
        fig, ax = plt.subplots(1, 1)
        ax: plt.Axes = sns.heatmap(
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
    def __init__(self, df_path=r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_resnet_masking_075_patches4096\results\epoch_99\dataframe_analyzed.json", type=EMBED_KEY, instancelabelColumn=NUCLEUS_LABEL_KEY, gtColumn=NUCLEUS_LABEL_KEY, 
        classMapping: dict = {}, save_dir=None, binary: bool = False, dbscan: bool = False, leiden: bool = True, leiden_resolution: float = 1.0, leiden_n_iterations: int = 2, leiden_n_neighbors: int = 15, leiden_distance_metric: str = "euclidean") -> None:
        self.df_path: str = df_path
        self.type: str = type
        assert df_path.endswith('.json'), "Dataframe path must be a JSON file."
        self.data_df: pd.DataFrame = pd.read_json(df_path)
        self.embeddings = StandardScaler().fit_transform(
            get_array_from_df(self.data_df, self.type))
        self.subplots_kwargs: dict[str, int] = {"s": 6}

        self.dbscan: bool = dbscan
        self.leiden: bool = leiden
        self.leiden_resolution: float = leiden_resolution
        self.leiden_n_iterations: int = leiden_n_iterations
        self.leiden_n_neighbors: int = leiden_n_neighbors
        self.leiden_distance_metric: str = leiden_distance_metric
        self.classifier_method = "LogisticRegression"
        self.classMappedColumn = "Mapped"

        assert isinstance(
            self.embeddings, np.ndarray), f"The embeddings are instead: {type(self.embeddings)}"
        print(f"The embeddings are of shape: {self.embeddings.shape}")
        self.gtColumn: str = gtColumn
        self.getLabels()
        self.classMapping = classMapping
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.classColumn = "Class"
        self.instancelabelColumn = instancelabelColumn
        self.classification = Classification
        self.results_df: pd.DataFrame = pd.DataFrame(index=self.data_df.index)
        self.predLabels = self.classify(
            binary=binary, mapping=classMapping
        )  # self.getClusters(self.embeddings)  
    
    def getLabels(self):
        self.gtlabels : np.ndarray = get_array_from_df(self.data_df, self.gtColumn)
        self.instancelabels : np.ndarray = get_array_from_df(self.data_df, self.instancelabelColumn)
        nan_mask = ~pd.isna(self.gtlabels)
        if nan_mask.sum() < len(self.gtlabels):
            warnings.warn(
                f"Labels contain {(~nan_mask).sum()} NaN value(s). "
                f"self.labels ({nan_mask.sum()}) is a subset of self.embeddings ({len(self.embeddings)}). "
                "Filtering both to non-NaN entries."
            )
            self.traingt = self.gtlabels[nan_mask]
            self.trainEmbeddings = self.embeddings[nan_mask]
            self.gtlabels[~nan_mask] = 0
        else:
            self.traingt = self.gtlabels
            self.trainEmbeddings = self.embeddings
    @classmethod
    def from_dataframe(
        cls,
        data_df: pd.DataFrame,
        type=MASKED_FEATURES_KEY,
        instancelabelColumn=NUCLEUS_LABEL_KEY,
        gtColumn=NUCLEUS_LABEL_KEY,
        save_dir=None,
        classMapping: dict = {},
        binary: bool = False,
        dbscan: bool = False,
        leiden: bool = True,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
    ) -> "EmbeddingAnalysis":
        instance: Self = cls.__new__(cls)
        instance.df_path = None
        instance.type = type
        instance.data_df = data_df.copy()
        instance.embeddings = StandardScaler().fit_transform(
            get_array_from_df(instance.data_df, type)
        )
        instance.subplots_kwargs = {"s": 1}
        instance.dbscan = dbscan
        instance.leiden = leiden
        instance.leiden_resolution = leiden_resolution
        instance.leiden_n_iterations = leiden_n_iterations
        instance.leiden_n_neighbors = leiden_n_neighbors
        instance.leiden_distance_metric = leiden_distance_metric
        instance.classifier_method = "LogisticRegression"
        instance.classification = Classification
        instance.gtColumn = gtColumn
        instance.instancelabelColumn = instancelabelColumn

        instance.classMapping = classMapping
        instance.getLabels()
        instance.save_dir = save_dir
        instance.classMappedColumn = "Mapped"
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        instance.classColumn = "cluster"
        instance.results_df = pd.DataFrame(index=instance.data_df.index)
        assert isinstance(instance.embeddings, np.ndarray), \
            f"The embeddings are instead: {type(instance.embeddings)}"
        print(f"The embeddings are of shape: {instance.embeddings.shape}")
        instance.predLabels = instance.classify(binary=binary, mapping=classMapping)
        return instance

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[NUCLEUS_LABEL_KEY] == label]

    def get_array_from_df(self, column):
        """Extracts and converts a column from self.data_df to a NumPy array."""
        return np.array(self.data_df[column].tolist())

    def load_png_for_nucleus(self, nucleus_id, patches_dir) -> str:
        """
        Load a PNG file for a nucleus and convert it to a base64-encoded image tag.

        Parameters:
        - nucleus_id: The ID of the nucleus.
        - patches_dir: Directory where the nucleus patches are stored.

        Returns:
        - HTML image tag with base64-encoded image or a message if the file doesn't exist.
        """
        png_filename: str = os.path.join(patches_dir, f"nucleus_hr_{nucleus_id}.png")


        npy_filename: str = os.path.join(patches_dir, f"nucleus_hr_{nucleus_id}.npy")
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
                img_str: str = base64.b64encode(buffered.getvalue()).decode()
                return f'<img src="data:image/png;base64,{img_str}" width="200" height="200">'
            except Exception as e:
                print(f"Error creating image from {npy_filename}: {e}")
                return "Error creating image"
   

    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color", mapping : dict ={},
                       legend_title: str = "Classes", save_dir: str = "./") -> None:
        
        import plotly.express as px
        from . import MoBie_coloring

        color_space = MoBie_coloring.GlasbeyARGBLut()

        # Work on a local copy so results_df is never mutated for display purposes
        plot_df = self.results_df.copy()

        if classColoumn not in plot_df.columns:
            print(f"{self.classColumn} is not a column in DataFrame")
            plot_df[classColoumn] = 0

        plot_df[classColoumn] = plot_df[classColoumn].fillna(0)
        classLabels = sorted(plot_df[classColoumn].unique().astype(int).tolist())

        def _label(k: int) -> str:
            return mapping.get(k, str(k)) if mapping else str(k)

        map_cluster_2_color: dict[str, str] = {
            _label(k): f"rgba{color_space.rgba_tuple_by_index(k)}"
            for k in classLabels
        }
        map_cluster_2_color[_label(0)] = "rgba(128, 128, 128, 0.5)"

        # Replace column values with display names so Plotly legend matches color map keys
        plot_df[classColoumn] = plot_df[classColoumn].astype(int).map(_label)
        display_labels = [_label(k) for k in classLabels]

        fig = px.scatter(
            plot_df,
            x=xColumn,
            y=yColumn,
            color=classColoumn,
            color_discrete_map=map_cluster_2_color,
            category_orders={classColoumn: display_labels},
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

        output_path: str = os.path.join(save_dir, f"{classColoumn}_UMAP.svg")
        fig.write_image(output_path)
        fig.show()


    def UMAP(self, n_neighbors=15, min_dist=0.1,
             n_components=3, random_state=42, metric="euclidean", **kwargs) -> pd.DataFrame:


        import umap

        umap_reducer = umap.UMAP(n_neighbors=n_neighbors,
                                 min_dist=min_dist,
                                 n_components=n_components,
                                 random_state=random_state,
                                 metric=metric, **kwargs).fit(self.embeddings)

        umap_array = umap_reducer.transform(self.embeddings)

        umap_df = pd.DataFrame({"UMAP x": umap_array[:, 0],
                                "UMAP y": umap_array[:, 1],
                                "UMAP z": umap_array[:, 2]},
                               index=self.data_df.index)

        return umap_df
    def classify(self, binary: bool = False, train_size=0.6, mapping: dict[str, int] = None) -> np.ndarray:


        X_train, X_val, y_train, y_val = train_test_split(
            self.trainEmbeddings,
            self.traingt,
            train_size=train_size,
        )

        print(f"\n{np.unique(y_train)}, \n{np.unique(y_val)}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # self.trainEmbeddings = X_train
        self.classifier = self.classification.train_classifier(
            X_train, y_train, method=self.classifier_method
        )

        # Compute accuracy on validation set

        # Compute accuracy on training set
        self.train_accuracy = self.classification.getAccuracy(
            X_val, y_val, self.classifier
        )
        print(f"Validation Accuracy: {self.train_accuracy}")

        self.predLabels = self.classifier.predict(self.embeddings)
        self.results_df[self.classifier_method] = self.predLabels
        self.classColumn = self.classifier_method

        if mapping is not None:
            self.results_df[self.classMappedColumn] = np.array([mapping.get(label, label) for label in self.predLabels])
        else:
            self.classMappedColumn = self.classifier_method
            

        
        self.classification.createConfusionMatrixFigure(x=X_val, gt=y_val, classifier=self.classifier,
                                                        save_dir=self.save_dir, mapping=mapping)

        # Predict class for remainder embeddings
        return self.predLabels
    
    def performDBSCAN(
        self, preds, shape, DBSCAN_eps=0.5, DBSCAN_min_samples=10
    ) -> np.ndarray:
        clustering: DBSCAN = DBSCAN(
            eps=DBSCAN_eps, min_samples=int(DBSCAN_min_samples)
        ).fit(preds)
        final_clusters: np.ndarray = clustering.labels_

        return final_clusters.reshape(*shape)
    
    @staticmethod
    def concatColumnDF(df1, df2) -> pd.DataFrame:
        result = df1.copy()
        for col in df2.columns:
            result[col] = df2[col].values
        return result
    
    def performLeiden(
        self,
        preds: np.ndarray,
        shape: tuple,
        resolution: float = 1.0,
        n_iterations: int = 2,
        n_neighbors: int = 15,
        distance_metric: str = "euclidean",
    ) -> np.ndarray:
        import anndata as ad
        import scanpy

        labels = np.zeros(preds.shape[0])

        embedding = ad.AnnData(X=preds)

        scanpy.pp.neighbors(
            embedding,
            n_neighbors=n_neighbors,
            n_pcs=None,
            metric=distance_metric,  # type: ignore[arg-type]
            random_state=111,
        )
        scanpy.tl.leiden(
            embedding,
            resolution=resolution,
            random_state=111,
            n_iterations=n_iterations,
        )

        for indx, sub_label in enumerate(embedding.obs["leiden"].unique()):
            indices = embedding.obs[embedding.obs["leiden"] == sub_label].index.astype(
                int
            )
            labels[indices] = indx

        return labels.astype(int).reshape(*shape)
    
    def getClusters(self, preds, DBSCAN_eps=0.5, DBSCAN_min_samples=10):

        shape: tuple = preds.shape[:-1]

        preds = StandardScaler().fit_transform(preds)

        if self.leiden:
            labels: np.ndarray = self.performLeiden(
                preds,
                shape,
                resolution=self.leiden_resolution,
                n_iterations=self.leiden_n_iterations,
                n_neighbors=self.leiden_n_neighbors,
                distance_metric=self.leiden_distance_metric,
            )
        else:
            labels = self.performDBSCAN(
                preds=preds,
                shape=shape,
                DBSCAN_eps=DBSCAN_eps,
                DBSCAN_min_samples=DBSCAN_min_samples,
            )

        return labels
    def saveDataFrame(self, suffix: str = "_analyzed", extension=".csv") -> str:
        if self.df_path is not None:
            stem = pathlib.Path(self.df_path).stem
        else:
            stem = "dataframe"
        filename = f"{stem}{suffix}{extension}"
        save_dir = self.save_dir if self.save_dir is not None else "."
        out_path = os.path.join(save_dir, filename)
        self.results_df.to_csv(out_path)
        print(f"Dataframe saved to: {out_path}")
        return out_path

    def concatDF(self, df) -> pd.DataFrame:
        self.results_df = self.concatColumnDF(self.results_df, df)
        return self.results_df
    
    def UMAPResults(self) -> None:
        umap: pd.DataFrame = self.UMAP()

        self.concatDF(umap)

        args: dict[str, str] = {
            "xColumn": "UMAP x",
            "yColumn": "UMAP y",
            "classColoumn": self.classColumn,
            "legend_title": "Class",
            "save_dir": self.save_dir,
            "mapping": self.classMapping
        }

        self.specialScatter(**args)
    @staticmethod
    def vizualiseCoord(
        points: dict, title="", labels=None, save_dir=None, **subplots_kwargs
    ):

        return costumMatplotlib.simpleScatter(
            points,
            labels,
            title=title,
            save_dir=save_dir,
            **subplots_kwargs,
        )[0]
    def vizualisePCA(self, pcas, title="") -> Figure:
        points = costumMatplotlib.points2Dict(pcas[:, :2])

        return costumMatplotlib.simpleScatter(
            points,
            self.gtlabels,
            title=title,
            save_dir=self.save_dir,
            **self.subplots_kwargs,
        )[0]
    
    def pca(self):

        # Perform PCA
        pca_model = PCA()
        print(f"Detected following type for emebddings: {type(self.embeddings)}")
        # if not isinstance(self.embeddings, np.ndarray):
        #     print(f"Detected following type for emebddings: {type(self.embeddings)}")
        #     self.embeddings = np.array(self.embeddings)

        fg_pcs = pca_model.fit_transform(self.embeddings)

        print(f"Explained variance   : {pca_model.explained_variance_ratio_[:5].round(3)}")
        print(f"Cumulative (first 3) : {pca_model.explained_variance_ratio_[:3].sum():.3f}")

        umap_df = pd.DataFrame({"PCA x": fg_pcs[:, 0],
                        "PCA y": fg_pcs[:, 1],
                        "PCA z": fg_pcs[:, 2]},
                        index=self.data_df.index)
        
        self.concatDF(umap_df)
        
        return fg_pcs

    
    def clustering(self, resolution: float, n_iterations: int, n_neighbors: int, distance_metric: str = "euclidean"):
        import anndata as ad
        import scanpy
        labels = np.zeros(self.embeddings.shape[0])

        embedding = ad.AnnData(X=self.embeddings)

        scanpy.pp.neighbors(embedding, n_neighbors=n_neighbors,
                            n_pcs=None,
                            metric=distance_metric,
                            random_state=111)

        scanpy.tl.leiden(embedding, resolution=resolution,
                         random_state=111, n_iterations=n_iterations)

        # Map the subcluster labels back to the main dataframe
        for indx, sub_label in enumerate(adata.obs["leiden"].unique()):
            indices = adata.obs[adata.obs["leiden"]
                                == sub_label].index.astype(int)
            labels[indices] = indx

        self.predLabels = labels.astype(int) + 1
        self.results_df[self.classColumn] = self.predLabels

        return self.predLabels

    def generateMask(self, maskVolumePath, resolution=0.5,
                     n_iterations=10, n_neighbors=15,
                     distance_metric: str = "euclidean") -> None:

        if not hasattr(self, "predLabels"):
            self.clustering(resolution=resolution, n_iterations=n_iterations,
                            n_neighbors=n_neighbors, distance_metric=distance_metric)

        maskVol = skimage.io.imread(maskVolumePath)

        max_id = maskVol.max()
        mapping_array = np.zeros(max_id + 1, dtype=np.uint16)
        mapping_array[self.instancelabels] = self.predLabels

        maskVol = mapping_array[maskVol]

        fileName: str = os.path.basename(maskVolumePath).replace(".tif", "_predicted.tif")

        path: str = os.path.join(self.save_dir, fileName)

        skimage.io.imsave(path, maskVol.astype(np.int16))
        print(f"Result saved to: {path}")


if __name__ == '__main__':
    import pandas as pd

    df1path = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_masking_075_patch4 SAM masks 2\results\epoch_30\dataframe_analyzed.json"
    df2path= r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\06_cellpose_sam\tables\test.xlsx"
    df1 = pd.read_json(df1path)
    df2 = pd.read_excel(df2path)
    # df2.columns = df2.iloc[0]       # use first row as column names
    # df2 = df2.iloc[1:].reset_index(drop=True)  # drop first row, reset index

    df1['Class'] = df1["label_id"].map(df2.set_index('label_id')['Class']) + 1

    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    from dl_utils.analysis import EmbeddingAnalysis
    from dl_utils import EMBED_KEY, NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY, MASKED_AVG_TOKEN_FEATURES_KEY

    save_dir = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\06_cellpose_sam\predictedMask"

    mapping = {1: "Neuron", 2: "Glial"}
    # --- Instantiate from the already-loaded dataframe ---
    ea = EmbeddingAnalysis.from_dataframe(
        data_df=df1,
        type=MASKED_AVG_TOKEN_FEATURES_KEY,
        gtColumn='Class',
        classMapping=mapping,
        save_dir=save_dir,
        binary=False
    )

    # --- Classification (already done in __init__, just display accuracy) ---
    print(f"Classifier : {ea.classifier_method}")
    print(f"Validation accuracy: {ea.train_accuracy:.4f}")

    # --- PCA ---
    pcas = ea.pca()
    # pca_fig = ea.vizualisePCA(pcas, title="PCA")
    #pca_fig.show()

    # --- UMAP ---
    ea.UMAPResults()

    ea.saveDataFrame()

    maskpath=r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\06_cellpose_sam\predictedMask\NS6_OE_06_w4SPI-405.tif"
    ea.generateMask(maskpath)

