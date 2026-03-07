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


def convert_array_to_data_url(path) -> str:
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
        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=5, metric=metric),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        if method not in classifiers:
            raise ValueError(
                f"Unsupported classifier method: {method}. "
                f"Available: {list(classifiers.keys())}"
            )
        classifier = classifiers[method]
        classifier.fit(embeddings, labels)
        return classifier

    @staticmethod
    def createConfusionMatrixFigure(x, gt, classifier, save_dir=None, mapping: dict = None) -> Figure:
        y_pred = classifier.predict(x)
        ks: list[int] = classifier.classes_
        cm = confusion_matrix(gt, y_pred, normalize="true")
        ticks = [mapping[k] for k in ks] if mapping is not None else ks
        fig, ax = plt.subplots(1, 1)
        ax: plt.Axes = sns.heatmap(
            cm, ax=ax, annot=True, cmap="Blues",
            xticklabels=ticks, yticklabels=ticks, cbar=False,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("Ground Truth Labels")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()
        return fig

    @staticmethod
    def getAccuracy(x, gt, classifier):
        return accuracy_score(gt, classifier.predict(x))


class EmbeddingAnalysis:

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        df_path: str = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_resnet_masking_075_patches4096\results\epoch_99\dataframe_analyzed.json",
        type: str = EMBED_KEY,
        instancelabelColumn: str = NUCLEUS_LABEL_KEY,
        gtColumn: str = None,
        classMapping: dict = {},
        save_dir: str = None,
        binary: bool = False,
        dbscan: bool = False,
        leiden: bool = True,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
    ) -> None:
        assert df_path.endswith(".json"), "Dataframe path must be a JSON file."
        self.df_path: str = df_path
        self.type: str = type
        self.data_df: pd.DataFrame = pd.read_json(df_path)
        self.embeddings: np.ndarray = StandardScaler().fit_transform(
            get_array_from_df(self.data_df, type)
        )
        self._apply_common_setup(
            self,
            gtColumn=gtColumn,
            instancelabelColumn=instancelabelColumn,
            classMapping=classMapping,
            save_dir=save_dir,
            dbscan=dbscan,
            leiden=leiden,
            leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
            default_class_column="Class",
        )
        self.predLabels = (
            self.classify(binary=binary, mapping=classMapping)
            if self._has_gt
            else np.zeros(len(self.data_df), dtype=int)
        )

    @classmethod
    def from_dataframe(
        cls,
        data_df: pd.DataFrame,
        type: str = MASKED_FEATURES_KEY,
        instancelabelColumn: str = NUCLEUS_LABEL_KEY,
        gtColumn: str = None,
        save_dir: str = None,
        classMapping: dict = {},
        binary: bool = False,
        dbscan: bool = False,
        leiden: bool = True,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
    ) -> "EmbeddingAnalysis":
        instance = cls.__new__(cls)
        instance.df_path = None
        instance.type = type
        instance.data_df = data_df.copy()
        instance.embeddings = StandardScaler().fit_transform(
            get_array_from_df(instance.data_df, type)
        )
        cls._apply_common_setup(
            instance,
            gtColumn=gtColumn,
            instancelabelColumn=instancelabelColumn,
            classMapping=classMapping,
            save_dir=save_dir,
            dbscan=dbscan,
            leiden=leiden,
            leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
            default_class_column="cluster",
            subplots_kwargs={"s": 1},
        )
        instance.predLabels = (
            instance.classify(binary=binary, mapping=classMapping)
            if instance._has_gt
            else np.zeros(len(instance.data_df), dtype=int)
        )
        return instance

    @classmethod
    def fromEmbeddingArrayAndMetaDataFrame(
        cls,
        embeddings: np.ndarray,
        meta_df: "pd.DataFrame",
        save_dir: str = None,
        gtColumn: str = None,
        instancelabelColumn: str = NUCLEUS_LABEL_KEY,
        leiden: bool = False,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
    ) -> "EmbeddingAnalysis":
        """Construct directly from a raw embedding array and a metadata DataFrame.

        Unlike ``__init__`` and ``from_dataframe``, the embeddings are supplied
        as a pre-computed ``np.ndarray``.  No classifier is trained; scatter
        plots are coloured by *gtColumn* (or uniformly when absent).
        """
        instance = cls.__new__(cls)
        instance.df_path = None
        instance.type = None
        instance.data_df = meta_df.reset_index(drop=True).copy()
        instance.embeddings = StandardScaler().fit_transform(embeddings)
        gt_present = gtColumn is not None and gtColumn in meta_df.columns
        cls._apply_common_setup(
            instance,
            gtColumn=gtColumn,
            instancelabelColumn=instancelabelColumn,
            classMapping={},
            save_dir=save_dir,
            dbscan=False,
            leiden=leiden,
            leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
            default_class_column=gtColumn if gt_present else "Class",
        )
        # Colour by ground-truth label; skip classifier training
        instance.predLabels = instance.gtlabels
        return instance

    @staticmethod
    def _apply_common_setup(
        instance: "EmbeddingAnalysis",
        *,
        gtColumn,
        instancelabelColumn,
        classMapping,
        save_dir,
        dbscan,
        leiden,
        leiden_resolution,
        leiden_n_iterations,
        leiden_n_neighbors,
        leiden_distance_metric,
        default_class_column: str = "Class",
        subplots_kwargs: dict = None,
    ) -> None:
        instance.gtColumn = gtColumn
        instance.instancelabelColumn = instancelabelColumn
        instance.classMapping = classMapping
        instance.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        instance.dbscan = dbscan
        instance.leiden = leiden
        instance.leiden_resolution = leiden_resolution
        instance.leiden_n_iterations = leiden_n_iterations
        instance.leiden_n_neighbors = leiden_n_neighbors
        instance.leiden_distance_metric = leiden_distance_metric
        instance.classifier_method = "LogisticRegression"
        instance.classification = Classification
        instance.classMappedColumn = "Mapped"
        instance.classColumn = default_class_column
        instance.subplots_kwargs = subplots_kwargs if subplots_kwargs is not None else {"s": 6}
        instance.results_df = pd.DataFrame(index=instance.data_df.index)
        instance.getLabels()
        print(f"Embeddings shape: {instance.embeddings.shape}")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def _has_gt(self) -> bool:
        return self.gtColumn is not None and self.gtColumn in self.data_df.columns

    # ------------------------------------------------------------------ #
    # Label initialisation                                                 #
    # ------------------------------------------------------------------ #

    def getLabels(self) -> None:
        n = len(self.data_df)
        self.gtlabels: np.ndarray = (
            get_array_from_df(self.data_df, self.gtColumn)
            if self._has_gt
            else np.zeros(n, dtype=int)
        )
        self.instancelabels: np.ndarray = get_array_from_df(self.data_df, self.instancelabelColumn)
        nan_mask = ~pd.isna(self.gtlabels)
        if nan_mask.sum() < n:
            warnings.warn(
                f"Labels contain {(~nan_mask).sum()} NaN value(s). "
                f"Training on {nan_mask.sum()} of {len(self.embeddings)} samples."
            )
            self.gtlabels[~nan_mask] = 0
            self.traingt = self.gtlabels[nan_mask]
            self.trainEmbeddings = self.embeddings[nan_mask]
        else:
            self.traingt = self.gtlabels
            self.trainEmbeddings = self.embeddings

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def classify(self, binary: bool = False, train_size: float = 0.6, mapping: dict = None) -> np.ndarray:
        X_train, X_val, y_train, y_val = train_test_split(
            self.trainEmbeddings, self.traingt, train_size=train_size,
        )
        print(f"Classes — train: {np.unique(y_train)}, val: {np.unique(y_val)}")
        print(f"Samples — train: {len(X_train)}, val: {len(X_val)}")

        self.classifier = self.classification.train_classifier(
            X_train, y_train, method=self.classifier_method
        )
        self.train_accuracy = self.classification.getAccuracy(X_val, y_val, self.classifier)
        print(f"Validation accuracy: {self.train_accuracy:.4f}")

        self.predLabels = self.classifier.predict(self.embeddings)
        self.results_df[self.classifier_method] = self.predLabels
        self.classColumn = self.classifier_method

        if mapping is not None:
            self.results_df[self.classMappedColumn] = np.array(
                [mapping.get(label, label) for label in self.predLabels]
            )
        else:
            self.classMappedColumn = self.classifier_method

        self.classification.createConfusionMatrixFigure(
            x=X_val, gt=y_val, classifier=self.classifier,
            save_dir=self.save_dir, mapping=mapping,
        )
        return self.predLabels

    # ------------------------------------------------------------------ #
    # Clustering                                                           #
    # ------------------------------------------------------------------ #

    def performDBSCAN(self, preds, shape, DBSCAN_eps=0.5, DBSCAN_min_samples=10) -> np.ndarray:
        clustering: DBSCAN = DBSCAN(
            eps=DBSCAN_eps, min_samples=int(DBSCAN_min_samples)
        ).fit(preds)
        return clustering.labels_.reshape(*shape)

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
            embedding, n_neighbors=n_neighbors, n_pcs=None,
            metric=distance_metric, random_state=111,  # type: ignore[arg-type]
        )
        scanpy.tl.leiden(
            embedding, resolution=resolution, random_state=111, n_iterations=n_iterations,
        )
        for indx, sub_label in enumerate(embedding.obs["leiden"].unique()):
            indices = embedding.obs[embedding.obs["leiden"] == sub_label].index.astype(int)
            labels[indices] = indx
        return labels.astype(int).reshape(*shape)

    def getClusters(self, preds, DBSCAN_eps=0.5, DBSCAN_min_samples=10):
        shape: tuple = preds.shape[:-1]
        preds = StandardScaler().fit_transform(preds)
        if self.leiden:
            return self.performLeiden(
                preds, shape,
                resolution=self.leiden_resolution,
                n_iterations=self.leiden_n_iterations,
                n_neighbors=self.leiden_n_neighbors,
                distance_metric=self.leiden_distance_metric,
            )
        return self.performDBSCAN(preds=preds, shape=shape,
                                   DBSCAN_eps=DBSCAN_eps, DBSCAN_min_samples=DBSCAN_min_samples)

    def clustering(self, resolution: float, n_iterations: int, n_neighbors: int,
                   distance_metric: str = "euclidean"):
        import anndata as ad
        import scanpy

        labels = np.zeros(self.embeddings.shape[0])
        embedding = ad.AnnData(X=self.embeddings)
        scanpy.pp.neighbors(embedding, n_neighbors=n_neighbors, n_pcs=None,
                            metric=distance_metric, random_state=111)
        scanpy.tl.leiden(embedding, resolution=resolution,
                         random_state=111, n_iterations=n_iterations)
        for indx, sub_label in enumerate(embedding.obs["leiden"].unique()):
            indices = embedding.obs[embedding.obs["leiden"] == sub_label].index.astype(int)
            labels[indices] = indx
        self.predLabels = labels.astype(int) + 1
        self.results_df[self.classColumn] = self.predLabels
        return self.predLabels

    # ------------------------------------------------------------------ #
    # Dimensionality reduction                                             #
    # ------------------------------------------------------------------ #

    def pca(self) -> np.ndarray:
        pca_model = PCA()
        pcs = pca_model.fit_transform(self.embeddings)
        print(f"Explained variance      : {pca_model.explained_variance_ratio_[:5].round(3)}")
        print(f"Cumulative (first 3)    : {pca_model.explained_variance_ratio_[:3].sum():.3f}")
        pca_df = pd.DataFrame(
            {"PCA x": pcs[:, 0], "PCA y": pcs[:, 1], "PCA z": pcs[:, 2]},
            index=self.data_df.index,
        )
        self.concatDF(pca_df)
        return pcs

    def UMAP(self, n_neighbors=15, min_dist=0.1, n_components=3,
             random_state=42, metric="euclidean", **kwargs) -> pd.DataFrame:
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
            random_state=random_state, metric=metric, **kwargs,
        ).fit(self.embeddings)
        umap_array = reducer.transform(self.embeddings)
        return pd.DataFrame(
            {"UMAP x": umap_array[:, 0], "UMAP y": umap_array[:, 1], "UMAP z": umap_array[:, 2]},
            index=self.data_df.index,
        )

    def UMAPResults(self) -> Figure:
        self.concatDF(self.UMAP())
        points = {"x": np.asarray(self.results_df["UMAP x"]), "y": np.asarray(self.results_df["UMAP y"])}
        return costumMatplotlib.simpleScatter(
            points, self.gtlabels, title="UMAP",
            save_dir=self.save_dir, **getattr(self, "subplots_kwargs", {"s": 6}),
        )[0]

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def vizualisePCA(self, pcas, title="") -> Figure:
        points = costumMatplotlib.points2Dict(pcas[:, :2])
        return costumMatplotlib.simpleScatter(
            points, self.gtlabels, title=title,
            save_dir=self.save_dir, **self.subplots_kwargs,
        )[0]

    @staticmethod
    def vizualiseCoord(points: dict, title="", labels=None, save_dir=None, **subplots_kwargs):
        return costumMatplotlib.simpleScatter(
            points, labels, title=title, save_dir=save_dir, **subplots_kwargs,
        )[0]

    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color",
                       mapping: dict = {}, legend_title: str = "Classes",
                       save_dir: str = "./") -> None:
        import plotly.express as px
        from . import MoBie_coloring

        color_space = MoBie_coloring.GlasbeyARGBLut()
        plot_df = self.results_df.copy()

        if classColoumn not in plot_df.columns:
            print(f"{classColoumn} is not a column in DataFrame")
            plot_df[classColoumn] = 0

        plot_df[classColoumn] = plot_df[classColoumn].fillna(0)
        classLabels = sorted(plot_df[classColoumn].unique().astype(int).tolist())

        def _label(k: int) -> str:
            return mapping.get(k, str(k)) if mapping else str(k)

        color_map = {_label(k): f"rgba{color_space.rgba_tuple_by_index(k)}" for k in classLabels}
        color_map[_label(0)] = "rgba(128, 128, 128, 0.5)"
        plot_df[classColoumn] = plot_df[classColoumn].astype(int).map(_label)

        fig = px.scatter(
            plot_df, x=xColumn, y=yColumn, color=classColoumn,
            color_discrete_map=color_map,
            category_orders={classColoumn: [_label(k) for k in classLabels]},
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(size=16), yanchor="bottom", xanchor="left",
                        bgcolor="rgba(255,255,255,0.7)", bordercolor="black",
                        borderwidth=1, x=1, y=0),
            width=1000, height=800,
            legend_title_text=legend_title,
            xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        )
        if save_dir is not None:
            fig.write_image(os.path.join(save_dir, f"{classColoumn}_UMAP.svg"))
            fig.show()

    # ------------------------------------------------------------------ #
    # Results & mask                                                       #
    # ------------------------------------------------------------------ #

    def saveDataFrame(self, suffix: str = "_analyzed", extension: str = ".csv") -> str:
        stem = pathlib.Path(self.df_path).stem if self.df_path is not None else "dataframe"
        save_dir = self.save_dir if self.save_dir is not None else "."
        out_path = os.path.join(save_dir, f"{stem}{suffix}{extension}")
        self.results_df.to_csv(out_path)
        print(f"Dataframe saved to: {out_path}")
        return out_path

    def generateMask(self, maskVolumePath, resolution=0.5, n_iterations=10,
                     n_neighbors=15, distance_metric: str = "euclidean") -> None:
        if not hasattr(self, "predLabels"):
            self.clustering(resolution=resolution, n_iterations=n_iterations,
                            n_neighbors=n_neighbors, distance_metric=distance_metric)
        maskVol = skimage.io.imread(maskVolumePath)
        mapping_array = np.zeros(maskVol.max() + 1, dtype=np.uint16)
        mapping_array[self.instancelabels] = self.predLabels
        maskVol = mapping_array[maskVol]
        fileName = os.path.basename(maskVolumePath).replace(".tif", "_predicted.tif")
        save_dir = self.save_dir if self.save_dir is not None else os.path.dirname(maskVolumePath)
        path = os.path.join(save_dir, fileName)
        skimage.io.imsave(path, maskVol.astype(np.int16))
        print(f"Result saved to: {path}")

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def concatColumnDF(df1, df2) -> pd.DataFrame:
        result = df1.copy()
        for col in df2.columns:
            result[col] = df2[col].values
        return result

    def concatDF(self, df) -> pd.DataFrame:
        self.results_df = self.concatColumnDF(self.results_df, df)
        return self.results_df

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[NUCLEUS_LABEL_KEY] == label]

    def get_array_from_df(self, column):
        return np.array(self.data_df[column].tolist())

    def load_png_for_nucleus(self, nucleus_id, patches_dir) -> str:
        npy_filename = os.path.join(patches_dir, f"nucleus_hr_{nucleus_id}.npy")
        if os.path.exists(npy_filename):
            try:
                input_vol = np.load(npy_filename)
                mid_z = input_vol[input_vol.shape[0] // 2]
                mid_z = ((mid_z - mid_z.min()) / (mid_z.max() - mid_z.min()) * 255).astype(np.uint8)
                img = Image.fromarray(mid_z).resize((200, 200))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f'<img src="data:image/png;base64,{img_str}" width="200" height="200">'
            except Exception as e:
                return f"Error creating image: {e}"
        return "No image available"


if __name__ == '__main__':
    import pandas as pd

    df1path = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\checkpoints\LM_batch-16_20-epochs_masking_075_patch4 SAM masks 2\results\epoch_30\dataframe_analyzed.json"
    df2path = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\06_cellpose_sam\tables\test.xlsx"
    df1 = pd.read_json(df1path)
    df2 = pd.read_excel(df2path)

    df1['Class'] = df1["label_id"].map(df2.set_index('label_id')['Class']) + 1

    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    from dl_utils.analysis import EmbeddingAnalysis
    from dl_utils import EMBED_KEY, NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY, MASKED_AVG_TOKEN_FEATURES_KEY

    save_dir = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\06_cellpose_sam\predictedMask"
    mapping = {1: "Neuron", 2: "Glial"}

    ea = EmbeddingAnalysis.from_dataframe(
        data_df=df1,
        type=MASKED_AVG_TOKEN_FEATURES_KEY,
        gtColumn='Class',
        classMapping=mapping,
        save_dir=save_dir,
        binary=False,
    )

    print(f"Classifier : {ea.classifier_method}")
    print(f"Validation accuracy: {ea.train_accuracy:.4f}")

    pcas = ea.pca()
    ea.UMAPResults()
    ea.saveDataFrame()

    maskpath = r"C:\Users\imansaray\repos\PhD_subprojects\representationlearning\data\06_cellpose_sam\predictedMask\NS6_OE_06_w4SPI-405.tif"
    ea.generateMask(maskpath)
