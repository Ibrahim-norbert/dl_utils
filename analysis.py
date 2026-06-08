import base64
from typing import Optional
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from dl_utils import LABEL_KEY, NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY, EMBED_DICT_EMBED
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
            "LinearRegression": LinearRegression()
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
    def createConfusionMatrixFigure(x, gt, classifier, save_dir=None, mapping: dict = {}, gtColumn="") -> Figure:
        y_pred = classifier.predict(x)
        ks: list[int] = classifier.classes_
        cm = confusion_matrix(gt, y_pred, normalize="true")
        ticks = [mapping.get(k, k) for k in ks] if mapping is not None else ks

        n = len(ks)
        cell_size = max(0.6, min(1.2, 12 / n))
        fig_size = max(4, n * cell_size)
        annot = n <= 20
        fontsize = max(6, min(12, int(120 / n)))

        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        ax: plt.Axes = sns.heatmap(
            cm, ax=ax, annot=annot, cmap="Blues",
            xticklabels=ticks, yticklabels=ticks, cbar=True,
            fmt=".2f" if annot else "",
            annot_kws={"size": fontsize} if annot else {},
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("Ground Truth Labels")
        plt.xticks(rotation=45, ha="right", fontsize=fontsize)
        plt.yticks(rotation=0, fontsize=fontsize)
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(
                save_dir, "{}confusion_matrix.png").format(gtColumn), dpi=300)
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
        type: str = EMBED_DICT_EMBED,
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
        metadDataFramePath: Optional[str] = None,
        classifier_method: str = "LogisticRegression",
    ) -> None:
        assert df_path.endswith(".json"), "Dataframe path must be a JSON file."
        if save_dir is None:
            save_dir = os.path.dirname(df_path)
        self.df_path: str = df_path
        self.type: str = type
        self.data_df: pd.DataFrame = pd.read_json(df_path)
        print(f"Available coloumns: {self.data_df.columns.tolist()}")
        embeddings = get_array_from_df(self.data_df, type)
        nan_mask = np.isnan(embeddings)
        assert not nan_mask.any(
        ), f"Yes we have {nan_mask.sum()} nans in the array"

        assert np.unique(embeddings.flatten()).__len__(
        ) > 1, f"The embeddings are uninformative with the constant value of {np.unique(embeddings.flatten())}"

        self.embeddings: np.ndarray = StandardScaler().fit_transform(embeddings)

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
            metadDataFramePath=metadDataFramePath,
            classifier_method=classifier_method,
        )

        if self.classMapping and self._has_gt:
            self.plot_gromov_wasserstein_heatmap()
            self.plot_kl_divergence_heatmap()

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
        metadDataFramePath: str = None,
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
            metadDataFramePath=metadDataFramePath,
            default_class_column="cluster",
            subplots_kwargs={"s": 1},
        )

        return instance

    @classmethod
    def fromEmbeddingArrayAndMetaDataFrame(
        cls,
        embeddings: np.ndarray,
        meta_df: "pd.DataFrame",
        save_dir: str = None,
        gtColumn: str = None,
        instancelabelColumn: str = LABEL_KEY,
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
        metadDataFramePath: Optional[str] = None,
        classifier_method: str = "LogisticRegression",
    ) -> None:
        instance.gtColumn = gtColumn
        instance.instancelabelColumn = instancelabelColumn
        instance.classMapping = classMapping
        instance.save_dir = os.path.join(
            save_dir, "EmbeddingAnalysis") if save_dir is not None else None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        instance.metadDataFramePath = metadDataFramePath
        if metadDataFramePath is not None and os.path.exists(metadDataFramePath) and metadDataFramePath.endswith((".csv", ".xlsx", ".json")):
            data_df = pd.read_csv(metadDataFramePath) if metadDataFramePath.endswith(
                ".csv") else pd.read_excel(metadDataFramePath)
            assert instancelabelColumn in data_df.columns, f"Instance label column '{instancelabelColumn}' not found in metadata DataFrame."
            instance.data_df = instance.data_df.merge(
                data_df, on=instancelabelColumn, how="left")
            print(f"Metadata DataFrame loaded from: {metadDataFramePath}")

        instance.dbscan = dbscan
        instance.leiden = leiden
        instance.leiden_resolution = leiden_resolution
        instance.leiden_n_iterations = leiden_n_iterations
        instance.leiden_n_neighbors = leiden_n_neighbors
        instance.leiden_distance_metric = leiden_distance_metric
        instance.classifier_method = classifier_method
        instance.classification = Classification
        instance.classMappedColumn = "Mapped"
        instance.classColumn = default_class_column
        instance.subplots_kwargs = subplots_kwargs if subplots_kwargs is not None else {
            "s": 6}
        instance.results_df = pd.DataFrame(index=instance.data_df.index)
        instance.results_df[instancelabelColumn] = instance.data_df[instancelabelColumn]
        instance.getLabels()
        print(f"Embeddings shape: {instance.embeddings.shape}")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def _has_gt(self) -> bool:
        if isinstance(self.gtColumn, list):
            return all(c in self.data_df.columns for c in self.gtColumn)
        return self.gtColumn is not None and self.gtColumn in self.data_df.columns

    # ------------------------------------------------------------------ #
    # Label initialisation                                                 #
    # ------------------------------------------------------------------ #

    def getLabels(self) -> None:
        n = len(self.data_df)
        _gtColumn_was_list = isinstance(self.gtColumn, list)

        if isinstance(self.gtColumn, list):
            combined_col = "_".join(self.gtColumn)
            subset = self.data_df[self.gtColumn]

            # Detect one-hot encoded columns: each column only contains 0/1
            # (handles int, float, bool, and string variants after JSON round-trip)
            def _is_binary(col):
                try:
                    return pd.to_numeric(col.dropna()).isin([0, 1]).all()
                except (ValueError, TypeError):
                    return False

            is_onehot = all(_is_binary(subset[c]) for c in self.gtColumn)

            if is_onehot:
                # Use the active column name as label; NaN where none or multiple active
                def _onehot_label(row):
                    active = [c for c in self.gtColumn if float(row[c]) == 1.0]
                    if len(active) == 1:
                        return active[0]
                    return np.nan  # all-zero or multi-active → treated as unlabelled

                self.data_df[combined_col] = subset.apply(
                    _onehot_label, axis=1)
            else:
                def _fmt(v):
                    if pd.isna(v):
                        return "NA"
                    if isinstance(v, float) and v.is_integer():
                        return str(int(v))
                    return str(v)

                self.data_df[combined_col] = (
                    subset.apply(lambda col: col.map(_fmt))
                    .agg("_".join, axis=1)
                )

            self.gtColumn = combined_col

        self.gtlabels: np.ndarray = (
            get_array_from_df(self.data_df, self.gtColumn)
            if self._has_gt
            else np.zeros(n, dtype=int)
        )

        if self._has_gt and self.gtlabels.dtype.kind in ('U', 'S', 'O'):
            if not self.classMapping:
                unique_vals = [v for v in pd.unique(
                    self.gtlabels) if not pd.isna(v)]
                str_to_int = {v: i + 1 for i,
                              v in enumerate(sorted(unique_vals, key=str))}
                self.classMapping = {i: v for v, i in str_to_int.items()}
            else:
                str_to_int = {v: k for k, v in self.classMapping.items()}
            self.gtlabels = np.array(
                [str_to_int.get(v, np.nan) for v in self.gtlabels], dtype=float
            )
            self.data_df[self.gtColumn] = self.gtlabels

        valid_mask = ~pd.isna(self.gtlabels)
        if 0 in self.gtlabels[valid_mask]:
            self.gtlabels = np.where(
                valid_mask, self.gtlabels + 1, self.gtlabels)
            self.data_df[self.gtColumn] = self.gtlabels

        self.instancelabels: np.ndarray = get_array_from_df(
            self.data_df, self.instancelabelColumn)
        nan_mask = ~pd.isna(self.gtlabels)
        # When gtColumn was a list, additionally exclude zero/inactive rows so that
        # only explicitly labelled (active) samples participate in training.
        # Mutual exclusivity is already guaranteed upstream: _onehot_label returns
        # NaN for multi-active rows, so they are caught by nan_mask as well.
        # if _gtColumn_was_list:
        #     train_mask = nan_mask & (self.gtlabels > 0)
        # else:
        #     train_mask = nan_mask
        if nan_mask.sum() < n:
            warnings.warn(
                f"Labels contain {n - nan_mask.sum()} inactive/NaN value(s). "
                f"Training on {nan_mask.sum()} of {len(self.embeddings)} samples."
            )
            # Turn nan values to 0 but filter for training classifier
            self.gtlabels[~nan_mask] = 0
            self.traingt = self.gtlabels[nan_mask].astype(int)
            self.trainEmbeddings = self.embeddings[nan_mask]
        else:
            self.traingt = self.gtlabels.astype(int) if np.issubdtype(
                self.gtlabels.dtype, np.floating) else self.gtlabels
            self.trainEmbeddings = self.embeddings

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def classify(self, binary: bool = False, train_size: float = 0.6, mapping: dict = None, classifier_weights=None) -> np.ndarray:

        if classifier_weights is None:
            X_train, X_val, y_train, y_val = train_test_split(
                self.trainEmbeddings, self.traingt, train_size=train_size,
                stratify=self.traingt, random_state=42
            )
            print(
                f"Classes — train: {np.unique(y_train)}, val: {np.unique(y_val)}")
            print(f"Samples — train: {len(X_train)}, val: {len(X_val)}")

            self.classifier = self.classification.train_classifier(
                X_train, y_train, method=self.classifier_method
            )

        else:
            self.classifier = classifier_weights
            X_val = self.trainEmbeddings
            y_val = self.traingt

        self.train_accuracy = self.classification.getAccuracy(
            X_val, y_val, self.classifier)
        print(f"Validation accuracy: {self.train_accuracy:.4f}")
        self.predLabels = self.classifier.predict(self.embeddings)

        self.results_df[self.classifier_method] = self.predLabels
        self.classColumn = self.classifier_method

        proba = self.classifier.predict_proba(self.embeddings)
        for i, cls in enumerate(self.classifier.classes_):
            self.results_df[f"proba_{cls}"] = proba[:, i]

        if self.classMapping is not None:
            self.results_df[self.classMappedColumn] = np.array(
                [self.classMapping.get(label, label)
                 for label in self.predLabels]
            )
        else:
            self.classMappedColumn = self.classifier_method

        self.classification.createConfusionMatrixFigure(
            x=X_val, gt=y_val, classifier=self.classifier,
            save_dir=self.save_dir, mapping=mapping or self.classMapping, gtColumn=self.gtColumn
        )
        return self.predLabels

    # ------------------------------------------------------------------ #
    # Clustering                                                           #
    # ------------------------------------------------------------------ #

    def performDBSCAN(self, embeddings, shape, DBSCAN_eps=0.5, DBSCAN_min_samples=10) -> np.ndarray:
        embeddings = StandardScaler().fit_transform(embeddings)
        clustering: DBSCAN = DBSCAN(
            eps=DBSCAN_eps, min_samples=int(DBSCAN_min_samples)
        ).fit(embeddings)
        self.classColumn = "DBSCAN_cluster"
        labels = clustering.labels_.reshape(*shape).astype(int) + 1
        self.predLabels = labels
        return labels

    def performLeiden(
        self,
        embeddings: np.ndarray,
        shape: tuple,
        resolution: float = 1.0,
        n_iterations: int = 2,
        n_neighbors: int = 15,
        distance_metric: str = "euclidean",
    ) -> np.ndarray:
        import anndata as ad
        import scanpy
        embeddings = StandardScaler().fit_transform(embeddings)
        labels = np.zeros(embeddings.shape[0])
        embedding = ad.AnnData(X=embeddings)
        scanpy.pp.neighbors(
            embedding, n_neighbors=n_neighbors, n_pcs=None,
            # type: ignore[arg-type]
            metric=distance_metric, random_state=111, use_rep="X"
        )
        scanpy.tl.leiden(
            embedding, resolution=resolution, random_state=111, n_iterations=n_iterations,
        )
        for indx, sub_label in enumerate(embedding.obs["leiden"].unique()):
            indices = embedding.obs[embedding.obs["leiden"]
                                    == sub_label].index.astype(int)
            labels[indices] = indx
        labels = labels.astype(int).reshape(*shape) + 1
        self.predLabels = labels.flatten()
        return labels

    def cluster(self, embeddings=None, **kwargs):
        shape: tuple = self.embeddings.shape[:-1]

        if embeddings is None:
            embeddings = self.embeddings
        if self.leiden:
            labels = self.performLeiden(
                embeddings=embeddings, shape=shape, **kwargs)

        else:
            labels = self.performDBSCAN(
                embeddings=embeddings, shape=shape, **kwargs)

        self.results_df[self.classColumn] = labels

        return labels

    # ------------------------------------------------------------------ #
    # Dimensionality reduction                                             #
    # ------------------------------------------------------------------ #

    def pca(self) -> np.ndarray:
        pca_model = PCA()
        nan_mask = np.isnan(self.embeddings)
        assert not nan_mask.any(
        ), f"Yes we have {nan_mask.sum()} nans in the array"
        pcs = pca_model.fit_transform(self.embeddings)
        nan_mask = np.isnan(pcs)
        assert not nan_mask.any(
        ), f"Yes we have {nan_mask.sum()} nans in the array"
        print(
            f"Explained variance      : {pca_model.explained_variance_ratio_[:5].round(3)}")
        print(
            f"Cumulative (first 3)    : {pca_model.explained_variance_ratio_[:3].sum():.3f}")
        pca_df = pd.DataFrame(
            {"PCA x": pcs[:, 0], "PCA y": pcs[:, 1], "PCA z": pcs[:, 2]},
            index=self.data_df.index,
        )
        self.explained_variance_ratio = pca_model.explained_variance_ratio_[
            :3].sum()
        self.concatDF(pca_df)
        return pcs

    def UMAP(self, n_neighbors=15, min_dist=0.1, n_components=3,
             random_state=42, metric="euclidean", **kwargs) -> pd.DataFrame:
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
            random_state=random_state, metric=metric, n_jobs=10, **kwargs,
        ).fit(self.embeddings)
        umap_array = reducer.transform(self.embeddings)
        return pd.DataFrame(
            {"UMAP x": umap_array[:, 0], "UMAP y": umap_array[:,
                                                              1], "UMAP z": umap_array[:, 2]},
            index=self.data_df.index,
        )

    def UMAPResults(self, **kwargs):
        self.concatDF(self.UMAP())
        return self.specialScatter(
            xColumn="UMAP x", yColumn="UMAP y",
            xaxis_title="UMAP Dimension 1", yaxis_title="UMAP Dimension 2",
            classColoumn=self.classColumn,
            legend_title="UMAP - {}".format(
                self.gtColumn if self._has_gt else self.classColumn),
            save_dir=self.save_dir, **kwargs
        )

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def vizualisePCA(self, pcas=None, title="", **kwargs):
        return self.specialScatter(
            xColumn="PCA x", yColumn="PCA y",
            xaxis_title="PC 1", yaxis_title="PC 2",
            classColoumn=self.classColumn,
            legend_title="PC - {}".format(
                self.gtColumn if self._has_gt else self.classColumn),
            save_dir=self.save_dir, **kwargs
        )

    @staticmethod
    def vizualiseCoord(points: dict, title="", labels=None, save_dir=None, **subplots_kwargs):
        return costumMatplotlib.simpleScatter(
            points, labels, title=title, save_dir=save_dir, **subplots_kwargs,
        )[0]

    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color",
                       mapping: dict = {}, legend_title: str = "Classes",
                       save_dir: str = "./", precomputed_colors: bool = False,
                       show_axes: bool = False, color_background: bool = False, withLegendTitle=True, **kwargs):
        import plotly.express as px
        from . import MoBie_coloring

        plot_df = self.results_df.copy()

        if precomputed_colors:
            # Color column already contains plotly-compatible rgba strings — use directly
            fig = px.scatter(plot_df, x=xColumn, y=yColumn)
            fig.update_traces(marker=dict(
                color=plot_df[classColoumn].tolist(), size=4))
        else:
            color_space = MoBie_coloring.GlasbeyARGBLut()

            if classColoumn not in plot_df.columns:
                print(f"{classColoumn} is not a column in DataFrame")
                plot_df[classColoumn] = 0

            plot_df[classColoumn] = plot_df[classColoumn].fillna(0)
            classLabels = sorted(
                plot_df[classColoumn].unique().astype(int).tolist())

            def _label(k: int) -> str:
                if k == 0 and color_background:
                    return "background"
                return mapping.get(k, self.classMapping.get(k, str(k)))

            color_map = {
                _label(k): f"rgba{color_space.rgba_tuple_by_index(k)}" for k in classLabels}
            if not color_background:
                color_map[_label(0)] = "rgba(128, 128, 128, 0.5)"
            plot_df[classColoumn] = plot_df[classColoumn].astype(
                int).map(_label)

            fig = px.scatter(
                plot_df, x=xColumn, y=yColumn, color=classColoumn,
                color_discrete_map=color_map,
                category_orders={classColoumn: [
                    _label(k) for k in classLabels]},
            )

        if not withLegendTitle:
            legend_title = None

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=show_axes,
                       visible=True, showline=show_axes, linecolor="black",
                       ticks="outside" if show_axes else "", tickfont=dict(color="white")),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=show_axes,
                       visible=True, showline=show_axes, linecolor="black",
                       ticks="outside" if show_axes else "", tickfont=dict(color="white")),
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

        fig.update_traces(marker=dict(size=kwargs.get(
            "markerSize", 3), opacity=kwargs.get("markerOpacity", 0.8)))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_image(os.path.join(
                save_dir, f"{classColoumn}_{legend_title}.svg"))
            fig.show()
        return fig

    # ------------------------------------------------------------------ #
    # Results & mask                                                       #
    # ------------------------------------------------------------------ #

    def saveDataFrame(self, suffix: str = "_analyzed", extension: str = ".csv") -> str:
        stem = pathlib.Path(
            self.df_path).stem if self.df_path is not None else "dataframe"
        save_dir = self.save_dir if self.save_dir is not None else "."
        out_path = os.path.join(save_dir, f"{stem}{suffix}{extension}")
        self.results_df.to_csv(out_path)
        print(f"Dataframe saved to: {out_path}")
        return out_path

    def generateMask(self, maskVolumePath, resolution=0.5, n_iterations=10,
                     n_neighbors=15, distance_metric: str = "euclidean", classColoumn=None) -> str:
        if not hasattr(self, "predLabels"):
            self.clustering(resolution=resolution, n_iterations=n_iterations,
                            n_neighbors=n_neighbors, distance_metric=distance_metric)

        maskVol = skimage.io.imread(maskVolumePath)
        mapping_array = np.zeros(maskVol.max() + 1, dtype=np.uint16)

        if self.classColumn == classColoumn:
            mapping_array[self.instancelabels] = self.predLabels.astype(int)
        elif self.gtColumn == classColoumn:
            mapping_array[self.instancelabels] = self.gtlabels.astype(int)
        elif classColoumn in self.results_df.columns:
            mapping_array[self.instancelabels] = self.results_df[classColoumn].values.astype(
                int)
        else:
            raise ValueError(
                f"Column '{classColoumn}' not found in results DataFrame.")

        maskVol = mapping_array[maskVol]
        fileName = os.path.basename(maskVolumePath).replace(
            ".tif", f"_{classColoumn}.tif")
        save_dir = self.save_dir if self.save_dir is not None else os.path.dirname(
            maskVolumePath)
        path = os.path.join(save_dir, fileName)
        skimage.io.imsave(path, maskVol.astype(np.int16))
        print(f"Result saved to: {path}")
        return path

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

    def get_top_instances_per_class(self, n: int = 5) -> list[tuple]:
        """Return (class_name, instance_label) pairs for the top-n most confident instances per class.

        Yields n entries per class ordered highest-confidence first.
        class_name is resolved via classMapping when available.
        """
        mapping = self.classMapping or {}
        return [
            (mapping.get(cls, cls), lbl)
            for cls in self.classifier.classes_
            for lbl in self.instancelabels[self.results_df[f"proba_{cls}"].argsort()[-1:-n - 1:-1]]
        ]

    def plot_gromov_wasserstein_heatmap(self, max_samples: int = 500) -> Figure:
        """Compute and plot pairwise Gromov-Wasserstein distances between classes."""
        import ot

        classes = sorted(self.classMapping.keys())
        class_names = [self.classMapping[c] for c in classes]
        n_classes = len(classes)

        rng = np.random.default_rng(42)
        class_embeddings = {}
        for c in classes:
            mask = self.gtlabels == c
            emb = self.embeddings[mask]
            if len(emb) > max_samples:
                emb = emb[rng.choice(len(emb), max_samples, replace=False)]
            class_embeddings[c] = emb

        gw_matrix = np.zeros((n_classes, n_classes))
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i >= j:
                    continue
                Xi, Xj = class_embeddings[ci], class_embeddings[cj]
                Ci = ot.dist(Xi, Xi)
                Cj = ot.dist(Xj, Xj)
                Ci /= Ci.max() + 1e-12
                Cj /= Cj.max() + 1e-12
                pi = np.ones(len(Xi)) / len(Xi)
                pj = np.ones(len(Xj)) / len(Xj)
                gw = ot.gromov.gromov_wasserstein2(
                    Ci, Cj, pi, pj, "square_loss", verbose=False)
                gw_matrix[i, j] = gw_matrix[j, i] = gw

        fig, _ = costumMatplotlib.simpleHeatmap(
            gw_matrix,
            xticklabels=class_names,
            yticklabels=class_names,
            title="Gromov-Wasserstein Distance Between Classes",
            save_dir=self.save_dir,
        )
        return fig

    def plot_kl_divergence_heatmap(self, max_samples: int = 500) -> Figure:
        """Compute and plot pairwise symmetrised KL divergence between classes (Gaussian approximation via LedoitWolf)."""
        classes = sorted(self.classMapping.keys())
        class_names = [self.classMapping[c] for c in classes]
        n_classes = len(classes)

        rng = np.random.default_rng(42)
        means, covs = {}, {}
        for c in classes:
            mask = self.gtlabels == c
            emb = self.embeddings[mask]
            if len(emb) > max_samples:
                emb = emb[rng.choice(len(emb), max_samples, replace=False)]
            means[c] = emb.mean(axis=0)
            covs[c] = LedoitWolf().fit(emb).covariance_

        def _kl(mu_p, sigma_p, mu_q, sigma_q):
            sigma_q_inv = np.linalg.inv(sigma_q)
            diff = mu_q - mu_p
            _, logdet_p = np.linalg.slogdet(sigma_p)
            _, logdet_q = np.linalg.slogdet(sigma_q)
            return 0.5 * (
                np.trace(sigma_q_inv @ sigma_p)
                + diff @ sigma_q_inv @ diff
                - len(mu_p)
                + logdet_q - logdet_p
            )

        kl_matrix = np.zeros((n_classes, n_classes))
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i >= j:
                    continue
                sym = (_kl(means[ci], covs[ci], means[cj], covs[cj])
                       + _kl(means[cj], covs[cj], means[ci], covs[ci])) / 2
                kl_matrix[i, j] = kl_matrix[j, i] = sym

        fig, _ = costumMatplotlib.simpleHeatmap(
            kl_matrix,
            xticklabels=class_names,
            yticklabels=class_names,
            title="Symmetrised KL Divergence Between Classes",
            save_dir=self.save_dir,
        )
        return fig

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[NUCLEUS_LABEL_KEY] == label]

    def get_array_from_df(self, column):
        return np.array(self.data_df[column].tolist())

    def load_png_for_nucleus(self, nucleus_id, patches_dir) -> str:
        npy_filename = os.path.join(
            patches_dir, f"nucleus_hr_{nucleus_id}.npy")
        if os.path.exists(npy_filename):
            try:
                input_vol = np.load(npy_filename)
                mid_z = input_vol[input_vol.shape[0] // 2]
                mid_z = ((mid_z - mid_z.min()) / (mid_z.max() -
                         mid_z.min()) * 255).astype(np.uint8)
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
    from dl_utils import EMBED_DICT_EMBED, NUCLEUS_LABEL_KEY, MASKED_FEATURES_KEY, MASKED_AVG_TOKEN_FEATURES_KEY

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
