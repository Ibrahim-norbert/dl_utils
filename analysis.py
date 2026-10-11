import base64
import itertools
import logging
import os
import pathlib
import warnings
from dataclasses import dataclass
from functools import reduce
from io import BytesIO
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import seaborn as sns
import skimage
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_hex, Normalize, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from plotly.graph_objs import Scatter
from sklearn.cluster import DBSCAN
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from constants import LABEL_KEY, MOBIE_LABEL_KEY, MASKED_FEATURES_KEY, EMBED_DICT_EMBED
import MoBie_coloring as MoBie_coloring
from vizualizations import CustomMatplotlib
from util import getArrayFromDF

sns.set_context("poster")

# kaleido (Plotly static-image export) drives a headless Chromium via the
# choreographer package; both loggers have no explicit level of their own, so
# they inherit whatever the root logger is set to (e.g. logging.basicConfig
# level=INFO in a training entrypoint) and emit browser-lifecycle chatter on
# every fig.write_image(...) call. Pin them explicitly — an explicit per-logger
# level always wins over inherited root level, regardless of import order.
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)

# Metadata table readers, keyed by file suffix.
_TABLE_READERS = {".csv": pd.read_csv, ".xlsx": pd.read_excel, ".json": pd.read_json}


def _png_buffer_to_data_url(buf: BytesIO) -> str:
    """Encode the PNG bytes held in *buf* as a ``data:image/png;base64`` URL."""
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded_image}"


def _plotly_rgb_palette(name: str, keys) -> dict:
    """Map each key to a plotly ``rgb(r, g, b)`` string from seaborn palette *name*."""
    palette = sns.color_palette(name, len(keys))
    return {k: f"rgb{tuple(int(c * 255) for c in rgb)}" for k, rgb in zip(keys, palette)}


def convert_array_to_data_url(path) -> str:
    array = skimage.io.imread(path)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(array, cmap="gray")
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return _png_buffer_to_data_url(buf)


class _LabelColumn:
    """Ground-truth spec: one label column, or several condition columns merged into one.

    Standardises it to 1-based float class ids (NaN = missing/unknown) using pandas
    primitives: ``idxmax`` for one-hot columns, ``pd.factorize`` for first-appearance
    order and ``pd.Categorical`` for the encoding.
    """

    def __init__(self, gtColumn):
        self.sources: list = list(gtColumn) if isinstance(gtColumn, list) else []
        self.name: str = "_".join(self.sources) or gtColumn

    def standardise(self, data_df: pd.DataFrame, classMapping: dict) -> tuple[pd.DataFrame, str, dict]:
        df = data_df.assign(**{self.name: self._merge(data_df[self.sources])}) if self.sources else data_df.copy()
        # Only string-valued label columns are mapped to integer ids.
        if self.name not in df.columns or (values := getArrayFromDF(df, self.name)).dtype.kind not in "USO":
            return df, self.name, classMapping
        classMapping = classMapping or self._class_mapping(values)
        df[self.name] = self._encode(values, classMapping)
        return df, self.name, classMapping

    @classmethod
    def _merge(cls, subset: pd.DataFrame) -> pd.Series:
        """One-hot (0/1/NaN) -> name of the single active column, NaN if none or several; else "_"-joined values."""
        numeric = subset.apply(pd.to_numeric, errors="coerce")
        if numeric.isin([0, 1]).where(subset.notna(), True).all(axis=None):
            active = numeric.eq(1)
            return active.idxmax(axis=1).where(active.sum(axis=1) == 1)
        return reduce(lambda a, b: a + "_" + b, (subset[c].map(cls._format) for c in subset.columns))

    @staticmethod
    def _format(v) -> str:
        """NaN -> "NA", 2.0 -> "2", anything else -> str(v)."""
        if pd.isna(v):
            return "NA"
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

    def _class_mapping(self, values: np.ndarray) -> dict:
        """{1: class, ...}: classes named in ``sources`` first, then the rest by first appearance."""
        present = list(pd.factorize(values)[1])   # NaN excluded
        seen = set(present)
        return dict(enumerate(dict.fromkeys([v for v in self.sources if v in seen] + present), start=1))

    @staticmethod
    def _encode(values: np.ndarray, classMapping: dict) -> np.ndarray:
        """Class id per value; NaN for missing values or values absent from the mapping."""
        codes = pd.Categorical(values, categories=list(classMapping.values())).codes
        # code -1 (missing/unknown) indexes the trailing NaN
        return np.append(np.fromiter(classMapping, dtype=float), np.nan)[codes]


class Classification:

    @staticmethod
    def train_classifier(embeddings, labels, method="KNN", metric="cosine"):
        # Factories, so only the requested model is ever constructed.
        factories = {
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric=metric),
            "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
            "LinearRegression": LinearRegression,
        }
        if method not in factories:
            raise ValueError(
                f"Unsupported classifier method: {method}. "
                f"Available: {list(factories)}"
            )
        return factories[method]().fit(embeddings, labels)

    @staticmethod
    def createConfusionMatrixFigure(x, gt, classifier, save_dir=None, mapping: dict = {}, gtColumn="") -> Figure:
        ks = classifier.classes_
        # labels=ks keeps matrix rows/cols aligned with the tick labels below.
        cm = confusion_matrix(gt, classifier.predict(x), labels=ks, normalize="true")
        ticks = [mapping.get(k, k) for k in ks] if mapping is not None else ks

        n = len(ks)
        cell_size = max(0.6, min(1.2, 12 / n))
        fig_size = max(4, n * cell_size)
        annot = n <= 20
        fontsize = max(6, min(12, int(120 / n)))

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            cm, ax=ax, annot=annot, cmap="Blues",
            xticklabels=ticks, yticklabels=ticks, cbar=True,
            fmt=".2f" if annot else "",
            annot_kws={"size": fontsize} if annot else {},
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("Ground Truth Labels")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
        fig.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{gtColumn}_confusion_matrix.png"), dpi=300)
        plt.close(fig)
        return fig

    @staticmethod
    def getAccuracy(x, gt, classifier):
        return accuracy_score(gt, classifier.predict(x))


@dataclass
class ClusteringConfig:
    """DBSCAN/Leiden hyper-parameters for :meth:`EmbeddingAnalysis.cluster`.

    Groups the six clustering knobs that were previously threaded as loose
    keyword arguments through every constructor and the shared setup routine, so
    defaults live in one place and each call site passes a single object.
    """
    dbscan: bool = False
    leiden: bool = True
    resolution: float = 1.0
    n_iterations: int = 2
    n_neighbors: int = 15
    distance_metric: str = "euclidean"


PlotBackend = Literal["plotly", "matplotlib"]


class EmbeddingAnalysis:
    # matplotlib for raster sinks (TensorBoard): plotly's static export needs a
    # headless browser.
    plot_backend: PlotBackend = "plotly"

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
            self,
            df_path: Optional[str] = None,
            type: str = EMBED_DICT_EMBED,
            instancelabelColumn: str = MOBIE_LABEL_KEY,
            gtColumn: str = None,
            classMapping: dict = None,
            save_dir: str = None,
            dbscan: bool = False,
            leiden: bool = True,
            leiden_resolution: float = 1.0,
            leiden_n_iterations: int = 2,
            leiden_n_neighbors: int = 15,
            leiden_distance_metric: str = "euclidean",
            metadDataFramePath: Optional[str] = None,
            classifier_method: str = "LogisticRegression",
            greyBackground:bool = False,
            plot_backend: PlotBackend = "plotly",
            data_df: Optional[pd.DataFrame] = None,
            embeddings: Optional[np.ndarray] = None,
            subplots_kwargs : Optional[dict] = None,
            **kwargs,
    ) -> None:
        """Load, validate and prepare the data. Runs no analysis.

        Every constructor ends up here; call :meth:`analyse` afterwards to train
        the classifier (or run DBSCAN without ground truth) and draw the
        default figures.
        """
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {sorted(kwargs)}")

        data_df = self._read_dataframe(df_path, data_df)
        print(f"Available coloumns: {data_df.columns.tolist()}")
        embeddings = self._validated_embeddings(data_df, type, embeddings)
        if save_dir is None and df_path is not None:
            save_dir = os.path.dirname(df_path)

        data_df = self._merge_metadata(data_df, metadDataFramePath, instancelabelColumn)
        classMapping = classMapping or {}
        if gtColumn is not None:
            # Standardise label column(s) into a single integer column before getLabels runs.
            data_df, gtColumn, classMapping = self._standardise_label_column(
                data_df, gtColumn, classMapping
            )

        # Data
        self.df_path = df_path
        self.type = type
        self.data_df = data_df
        self.embeddings = StandardScaler().fit_transform(embeddings)
        self.metadDataFramePath = metadDataFramePath
        self.save_dir = os.path.join(save_dir, "EmbeddingAnalysis") if save_dir is not None else None
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        # Labels
        self.gtColumn = gtColumn
        self.instancelabelColumn = instancelabelColumn
        self.classMapping = classMapping or {}
        self.greyBackground = greyBackground

        # Grouped config is the source of truth; the flat attributes below are
        # kept for backwards compatibility (e.g. ``cluster()`` reads ``self.leiden``).
        self.clustering = ClusteringConfig(
            dbscan=dbscan, leiden=leiden, resolution=leiden_resolution,
            n_iterations=leiden_n_iterations, n_neighbors=leiden_n_neighbors,
            distance_metric=leiden_distance_metric,
        )
        self.dbscan = self.clustering.dbscan
        self.leiden = self.clustering.leiden
        self.leiden_resolution = self.clustering.resolution
        self.leiden_n_iterations = self.clustering.n_iterations
        self.leiden_n_neighbors = self.clustering.n_neighbors
        self.leiden_distance_metric = self.clustering.distance_metric

        # Classification & plotting
        self.classifier_method = classifier_method
        self.classification = Classification
        self.classMappedColumn = "Mapped"
        self.classColumn = "Class"
        self.plot_backend = plot_backend
        self.subplots_kwargs = subplots_kwargs if subplots_kwargs is not None else {"s": 6}

        self.results_df = self._scalar_columns(data_df)
        self.getLabels()
        print(f"Embeddings shape: {self.embeddings.shape}")
        self.color_space = MoBie_coloring.GlasbeyARGBLut()

    def analyse(self, **kwargs) -> np.ndarray:
        """Default analysis, formerly run automatically at the end of ``__init__``.

        Trains the classifier when ground truth is available (DBSCAN otherwise),
        adds the ground-truth columns to ``results_df`` and, when a class mapping
        exists, plots the Gromov-Wasserstein and KL class-distance heatmaps.
        """
        # TODO: DBSCAN
        self.predLabels = self.classify(**kwargs) if self._has_gt else self._dbscan(self.embeddings, **kwargs)

        # Add ground truth column as mapping
        if self._has_gt:
            self.results_df["Mapping_gtColumns"] = self._mapped_labels(self.gtlabels)
            self.results_df["gtColumns"] = self.gtlabels

        print("We have the following columns in the results DataFrame:", self.results_df.columns.tolist())

        if self.classMapping and self._has_gt:
            self.plot_gromov_wasserstein_heatmap()
            self.plot_kl_divergence_heatmap()
        return self.predLabels


    @classmethod
    def from_dataframe(
            cls,
            data_df: pd.DataFrame,
            type: str = MASKED_FEATURES_KEY,
            instancelabelColumn: str = MOBIE_LABEL_KEY,
            gtColumn: str = None,
            save_dir: str = None,
            classMapping: dict = None,
            dbscan: bool = False,
            leiden: bool = True,
            leiden_resolution: float = 1.0,
            leiden_n_iterations: int = 2,
            leiden_n_neighbors: int = 15,
            leiden_distance_metric: str = "euclidean",
            metadDataFramePath: str = None,
            *args,
            **kwargs
    ) -> "EmbeddingAnalysis":
        instance = cls(
            data_df=data_df,
            type=type,
            instancelabelColumn=instancelabelColumn,
            gtColumn=gtColumn,
            save_dir=save_dir,
            classMapping=classMapping,
            dbscan=dbscan,
            leiden=leiden,
            leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
            metadDataFramePath=metadDataFramePath,
            *args,
            **kwargs
        )
        instance.classColumn = "cluster"
        instance.subplots_kwargs = {"s": 1}
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
        instance = cls(
            data_df=meta_df.reset_index(drop=True),
            embeddings=embeddings,
            type=None,
            instancelabelColumn=instancelabelColumn,
            gtColumn=gtColumn,
            save_dir=save_dir,
            dbscan=False,
            leiden=leiden,
            leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
        )
        # Colour by ground-truth label (the combined column for a list of GT columns)
        if instance._has_gt:
            instance.classColumn = instance.gtColumn
        # Skip classifier training
        instance.predLabels = instance.gtlabels
        return instance

    # ------------------------------------------------------------------ #
    # Data preparation helpers (pure: inputs in, value out)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_dataframe(df_path: Optional[str], data_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """A private copy of *data_df*, or the JSON file at *df_path* when no frame is given."""
        if data_df is not None:
            return data_df.copy()
        if not (isinstance(df_path, str) and df_path.endswith(".json")):
            raise ValueError("Either data_df or df_path (a path to a JSON file) must be given.")
        return pd.read_json(df_path)

    @classmethod
    def _validated_embeddings(cls, data_df: pd.DataFrame, type, embeddings) -> np.ndarray:
        """Embeddings as an array (read from column *type* when not given), checked for use."""
        embeddings = np.asarray(getArrayFromDF(data_df, type) if embeddings is None else embeddings)
        if len(embeddings) != len(data_df):
            raise ValueError(
                f"embeddings has {len(embeddings)} rows but data_df has {len(data_df)}")
        cls._check_no_nans(embeddings)
        if embeddings.size == 0 or np.all(embeddings == embeddings.flat[0]):
            raise ValueError(
                f"The embeddings are uninformative with the constant value of {np.unique(embeddings)}")
        return embeddings

    @staticmethod
    def _merge_metadata(data_df: pd.DataFrame, path: Optional[str], instancelabelColumn: str) -> pd.DataFrame:
        """Left-join the metadata table at *path* on *instancelabelColumn*; no-op without one."""
        reader = _TABLE_READERS.get(pathlib.Path(path).suffix) if path else None
        if reader is None or not os.path.exists(path):
            return data_df
        meta_df = reader(path)
        if instancelabelColumn not in meta_df.columns:
            raise ValueError(
                f"Instance label column '{instancelabelColumn}' not found in metadata DataFrame.")
        print(f"Metadata DataFrame loaded from: {path}")
        return data_df.merge(meta_df, on=instancelabelColumn, how="left")

    @staticmethod
    def _scalar_columns(data_df: pd.DataFrame) -> pd.DataFrame:
        """Copy of the columns holding scalars (drops list/array/dict-valued columns)."""
        def _is_scalar_column(series: pd.Series) -> bool:
            kind = series.dtype.kind
            if kind in "fiu":  # numeric dtypes cannot hold containers
                return True
            return kind in "USO" and not series.map(
                lambda x: isinstance(x, (list, dict, np.ndarray))).any()

        return data_df[[c for c, s in data_df.items() if _is_scalar_column(s)]].copy()

    @staticmethod
    def _check_no_nans(array: np.ndarray) -> None:
        n_nans = int(np.isnan(array).sum())
        if n_nans:
            raise ValueError(f"Found {n_nans} NaN(s) in the array")

    @staticmethod
    def _standardise_label_column(data_df, gtColumn, classMapping) -> tuple["pd.DataFrame", str, dict]:
        """(docstring unchanged)"""
        return _LabelColumn(gtColumn).standardise(data_df, classMapping)

    @property
    def _has_gt(self) -> bool:
        return self.gtColumn is not None and self.gtColumn in self.data_df.columns

    def _class_name(self, label) -> str:
        return str(self.classMapping.get(int(label), int(label)))

    def _mapped_labels(self, labels) -> list:
        return [self.classMapping.get(label, label) for label in labels]

    # ------------------------------------------------------------------ #
    # Label initialisation                                                 #
    # ------------------------------------------------------------------ #

    def getLabels(self) -> None:
        """Derive training labels from the (already-standardised) gtColumn.

        Label *derivation* — list→combined-column joining, one-hot detection,
        and string→int mapping — is performed once upstream in
        :meth:`_standardise_label_column` (called from ``_init_common``).
        By the time this runs ``self.gtColumn`` is always a single column name
        whose values are numeric, so this method only handles the remaining
        bookkeeping: the zero-offset shift, instance labels, and the
        NaN/train-mask split used to fit the classifier.
        """

        n = len(self.data_df)
        self.gtlabels: np.ndarray = (
            getArrayFromDF(self.data_df, self.gtColumn)
            if self._has_gt
            else np.zeros(n, dtype=int)
        )


        self.instancelabels: np.ndarray = getArrayFromDF(self.data_df, self.instancelabelColumn) if (self.instancelabelColumn is not None and self.instancelabelColumn in self.data_df.columns) else np.zeros(n, dtype=int)

        valid = ~pd.isna(self.gtlabels)
        n_valid = int(valid.sum())
        if n_valid < n:
            warnings.warn(
                f"Labels contain {n - n_valid} inactive/NaN value(s). "
                f"Training on {n_valid} of {len(self.embeddings)} samples."
            )
            self.traingt = self.gtlabels[valid].astype(int)
            self.trainEmbeddings = self.embeddings[valid]
        else:
            self.traingt = (self.gtlabels.astype(int)
                            if np.issubdtype(self.gtlabels.dtype, np.floating)
                            else self.gtlabels)
            self.trainEmbeddings = self.embeddings
    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def classify(self, train_size: float = 0.6, mapping: dict = None, classifier_weights=None) -> np.ndarray:
        if classifier_weights is None:
            X_train, X_val, y_train, y_val = train_test_split(
                self.trainEmbeddings, self.traingt, train_size=train_size,
                stratify=self.traingt, random_state=42
            )
            print(f"Classes — train: {np.unique(y_train)}, val: {np.unique(y_val)}")
            print(f"Samples — train: {len(X_train)}, val: {len(X_val)}")
            self.classifier = self.classification.train_classifier(
                X_train, y_train, method=self.classifier_method
            )
        else:
            self.classifier = classifier_weights
            X_val, y_val = self.trainEmbeddings, self.traingt

        self.train_accuracy = self.classification.getAccuracy(X_val, y_val, self.classifier)
        print(f"Validation accuracy: {self.train_accuracy:.4f}")
        self.predLabels = self.classifier.predict(self.embeddings)

        self.results_df[self.classifier_method] = self.predLabels
        self.classColumn = self.classifier_method

        proba = self.classifier.predict_proba(self.embeddings)
        self.results_df = self.results_df.assign(
            **{f"proba_{cls}": p for cls, p in zip(self.classifier.classes_, proba.T)}
        )

        if self.classMapping is not None:
            self.results_df[self.classMappedColumn] = np.array(self._mapped_labels(self.predLabels))
        else:
            self.classMappedColumn = self.classifier_method

        self.classification.createConfusionMatrixFigure(
            x=X_val, gt=y_val, classifier=self.classifier,
            save_dir=self.save_dir, mapping=mapping or self.classMapping, gtColumn=self.gtColumn
        )
        return self.predLabels

    def _dbscan(self, embeddings: np.ndarray, DBSCAN_eps=0.5, DBSCAN_min_samples=10) -> np.ndarray:
        embeddings = StandardScaler().fit_transform(embeddings)
        clustering = DBSCAN(eps=DBSCAN_eps, min_samples=int(DBSCAN_min_samples)).fit(embeddings)
        self.classColumn = "DBSCAN_cluster"
        self.predLabels = clustering.labels_.astype(int) + 1
        self.results_df[self.classColumn] = self.predLabels
        return self.predLabels

    def tune_clustering(
            self,
            space: dict,
            clusterer_fn,
            max_evals: int = 100,
    ) -> dict:
        """Tune any clustering algorithm on ``self.embeddings`` using
        Tree-structured Parzen Estimator (hyperopt).

        Parameters
        ----------
        space:
            Hyperopt search space dict mapping parameter names to hyperopt
            distributions, e.g.::

                from hyperopt import hp
                from hyperopt.pyll import scope
                space = {
                    "eps":       hp.uniform("eps", 0.05, 10.0),
                    "minpoints": scope.int(hp.quniform("minpoints", 3, 125, q=1)),
                }

        clusterer_fn:
            Callable ``(params: dict, embeddings: np.ndarray) -> np.ndarray``
            that receives the sampled parameter dict and the embedding matrix
            and returns integer cluster labels of shape ``(N,)``, or a string
            naming a method on this instance (e.g. ``"performDBSCAN"`` or
            ``"performLeiden"``).  The named method must have the same
            ``(params, embeddings)`` signature.

        max_evals:
            Number of hyperopt iterations.

        Returns
        -------
        dict
            Best parameter dict found (same keys as *space*).
        """
        from hyperopt import fmin, tpe
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics.cluster import contingency_matrix

        def _mean_iou(gt, pred):
            cm = contingency_matrix(gt, pred).astype(float)
            union = cm.sum(axis=1, keepdims=True) + cm.sum(axis=0, keepdims=True) - cm
            iou_matrix = np.where(union > 0, cm / union, 0.0)
            row_idx, col_idx = linear_sum_assignment(-iou_matrix)
            return iou_matrix[row_idx, col_idx].mean()

        embeddings = self.embeddings
        if isinstance(clusterer_fn, str):
            method = getattr(self, clusterer_fn)
            shape = (len(embeddings),)
            run = lambda params: method(embeddings, shape, **params)  # noqa: E731
        else:
            run = lambda params: clusterer_fn(params, embeddings)  # noqa: E731

        return fmin(
            fn=lambda params: -_mean_iou(self.gtlabels, run(params)),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
        )

    def _leiden(
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

        adata = ad.AnnData(X=StandardScaler().fit_transform(embeddings))
        scanpy.pp.neighbors(
            adata, n_neighbors=n_neighbors, n_pcs=None,
            metric=distance_metric, random_state=111, use_rep="X",  # type: ignore[arg-type]
        )
        scanpy.tl.leiden(
            adata, resolution=resolution, random_state=111, n_iterations=n_iterations,
        )
        # Relabel clusters 1..k in order of first appearance.
        codes, _ = pd.factorize(adata.obs["leiden"])
        labels = codes.reshape(shape) + 1
        self.predLabels = labels.flatten()
        return labels

    def cluster(self, embeddings=None, **kwargs):
        if embeddings is None:
            embeddings = self.embeddings
        if self.leiden:
            labels = self._leiden(embeddings=embeddings, shape=embeddings.shape[:-1], **kwargs)
        else:
            labels = self._dbscan(embeddings=embeddings, **kwargs)

        self.results_df[self.classColumn] = labels
        return labels

    # ------------------------------------------------------------------ #
    # Dimensionality reduction                                             #
    # ------------------------------------------------------------------ #

    def pca(self) -> np.ndarray:
        self._check_no_nans(self.embeddings)
        pca_model = PCA()
        pcs = pca_model.fit_transform(self.embeddings)
        self._check_no_nans(pcs)

        ratio = pca_model.explained_variance_ratio_
        print(f"Explained variance      : {ratio[:5].round(3)}")
        print(f"Cumulative (first 3)    : {ratio[:3].sum():.3f}")
        self.explained_variance_ratio = ratio[:3].sum()
        self.concatDF(pd.DataFrame(
            pcs[:, :3], columns=["PCA x", "PCA y", "PCA z"], index=self.data_df.index,
        ))
        return pcs

    def _umap(self, n_neighbors=15, min_dist=0.1, n_components=3, random_state=42, metric="euclidean", **kwargs) -> pd.DataFrame:
        import umap

        umap_array = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
            random_state=random_state, metric=metric, n_jobs=10, **kwargs,
        ).fit_transform(self.embeddings)
        return pd.DataFrame(
            umap_array[:, :3], columns=["UMAP x", "UMAP y", "UMAP z"], index=self.data_df.index,
        )

    def visualiseUMAP(self, classColoumn: str | None = None, umap_kwargs: dict | None = None, **kwargs):
        col = classColoumn if classColoumn is not None else self.classColumn
        # umap_kwargs flow to UMAP() (e.g. n_neighbors); remaining kwargs to the
        # scatter plot. UMAP requires n_neighbors < n_samples, so small feature
        # spaces (e.g. one point per pooled sample) must pass a capped value.
        self.concatDF(self._umap(**(umap_kwargs or {})))
        return EmbeddingAnalysis.specialScatter(
            self,
            xColumn="UMAP x", yColumn="UMAP y",
            xaxis_title="UMAP Dimension 1", yaxis_title="UMAP Dimension 2",
            classColoumn=col,
            legend_title=f"UMAP - {self.gtColumn if self._has_gt else self.classColumn}",
            save_dir=self.save_dir, **kwargs
        )

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def plot_distance_matrix(self, metric: str = "cosine") -> Figure:
        """Plot a pairwise distance matrix sorted by GT class label.

        Uses the labelled training embeddings (``trainEmbeddings`` /
        ``traingt``).  Each row is normalised to sum to 1 so relative
        distances are comparable across rows with different scales.
        """
        from scipy.spatial.distance import cdist

        sort = np.argsort(self.traingt)
        feats = self.trainEmbeddings[sort]
        labels = self.traingt[sort]
        # labels are sorted, so each class is one contiguous block
        classes, starts, counts = np.unique(labels, return_index=True, return_counts=True)
        tick_lbl = [self._class_name(c) for c in classes]
        print(f"[plot_distance_matrix] {len(classes)} classes found in traingt: {tick_lbl}")

        D = cdist(feats, feats, metric=metric)  # type: ignore[call-overload]
        D /= D.sum(axis=1, keepdims=True) + 1e-12

        tick_pos = starts + (counts - 1) / 2
        boundaries = (starts + counts)[:-1] - 0.5

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(False)
        im = ax.imshow(D, aspect="auto", cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, label=f"{metric} distance (row-normalised)")

        for b in boundaries:
            ax.axhline(b, color="white", lw=0.8, alpha=0.6)
            ax.axvline(b, color="white", lw=0.8, alpha=0.6)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl, fontsize=8)
        ax.set_xlabel("Class")
        ax.set_ylabel("Class")
        ax.set_title(
            f"Pairwise {metric} distance matrix (row-normalised)\n"
            f"({len(feats)} samples, sorted by GT label)"
        )
        fig.tight_layout()

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, f"distance_matrix_{metric}.png"), dpi=300)

        return fig

    def visualisePCA(self, classColoumn: str | None = None, precomputed_colors: bool = True):
        col = classColoumn if classColoumn is not None else self.classColumn
        # Call EmbeddingAnalysis.specialScatter directly so that predicted-label
        # columns are never temporarily zeroed by subclass overrides, allowing
        # every sample to be coloured by its predicted class.
        return EmbeddingAnalysis.specialScatter(
            self,
            xColumn="PCA x", yColumn="PCA y",
            xaxis_title="PC 1", yaxis_title="PC 2",
            classColoumn=col,
            legend_title=f"PC - {self.gtColumn if self._has_gt else col}",
            save_dir=self.save_dir,
            precomputed_colors=precomputed_colors
        )
    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color",
                       mapping: Optional[dict] = None, legend_title: str = "Classes",
                       save_dir: Optional[str] = "./", precomputed_colors: bool = False,
                       heatmap: bool = False,
                       show_axes: bool = False, color_background: bool = False, withLegendTitle=True,
                       show: bool = False, **kwargs):
        """Scatter coloured by *classColoumn*, drawn with ``self.plot_backend``.

        - default: integer class labels -> named classes with a legend
        - ``precomputed_colors``: the column already holds per-point colours
        - ``heatmap``: float values are normalised over vmin..vmax (``vmin``/``vmax``
          via kwargs, default: min..max of the non-background values) and shown
          with a normalised 0..1 colour bar titled *classColoumn*.
        Background (0) and NaN are always grey and never part of a colour scale.
        """
        df, col, lut = self.results_df.copy(), classColoumn, self.color_space
        title = legend_title if withLegendTitle else None
        is_mpl = self.plot_backend == "matplotlib"
        draw, save = ((self._scatter_matplotlib, "savefig") if is_mpl
                      else (self._scatter_plotly, "write_image"))
        to_hex = "#{:02x}{:02x}{:02x}{:02x}".format
        grey = "#808080" if color_background else "#80808080"
        palette = None

        if heatmap:
            # (n, 4) uint8 RGBA, unpacked once from the packed ARGB LUT
            argb = np.asarray(lut.indices, dtype=np.uint32)[:, None]
            rgba = (argb >> np.array([16, 8, 0, 24], dtype=np.uint32)) & 0xFF
            n = len(rgba)

            values = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan).to_numpy(dtype=float)
            valid = values[~np.isnan(values)]
            lo, hi = (valid.min(), valid.max()) if valid.size else (0.0, 1.0)  # adaptive data range
            vmin, vmax = kwargs.get("vmin"), kwargs.get("vmax")
            vmin = lo if vmin is None else float(vmin)
            vmax = hi if vmax is None else float(vmax)
            vmax = vmin + ((vmax - vmin) or 1.0)  # equal limits -> range 1

            norm = np.clip((values - vmin) / (vmax - vmin), 0, 1)
            slot = np.nan_to_num(norm * (n - 1), nan=n).astype(int)  # NaN/background -> grey slot n

            df[col] = np.array([to_hex(*c) for c in rgba.tolist()] + [grey])[slot]

        elif not precomputed_colors:
            labels = pd.Series(df.get(col, 0), index=df.index).fillna(0).astype(int)
            names = {**self.classMapping, **(mapping or {})}
            names.update({0: "background"} if color_background else {})
            label_of = {k: names.get(k, str(k)) for k in sorted(labels.unique())}

            palette = {name: to_hex(*lut.rgba_tuple_by_index(k)) if k else grey
                       for k, name in label_of.items()}
            df[col] = labels.map(label_of)

        fig = draw(df, xColumn, yColumn, col, palette,
                   xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                   legend_title=None if heatmap else title,
                   show_axes=show_axes,
                   marker_size=kwargs.get("markerSize", 3),
                   opacity=kwargs.get("markerOpacity", 0.8))

        if heatmap:  # normalised 0..1 colour bar instead of legend, built from the same LUT
            ticks = np.linspace(0, 1, 5)
            ticktext = [f"{t:.2g}" for t in ticks]
            cbar_title = col if withLegendTitle else None
            if is_mpl:
                ax = fig.axes[0]
                for legend in filter(None, [ax.get_legend(), *fig.legends]):
                    legend.remove()
                sm = ScalarMappable(Normalize(0, 1), ListedColormap(rgba / 255.0))
                fig.colorbar(sm, ax=ax, ticks=ticks, label=cbar_title).set_ticklabels(ticktext)
            else:
                scale = [[i / (n - 1), f"rgba({r},{g},{b},{a / 255:.3f})"]
                         for i, (r, g, b, a) in enumerate(rgba.tolist())]
                fig.update_layout(showlegend=False)
                fig.add_trace(Scatter(  # invisible trace that only carries the colour bar
                    x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
                    marker=dict(colorscale=scale, cmin=0, cmax=1, color=[0], showscale=True,
                                colorbar=dict(title=cbar_title, tickvals=list(ticks), ticktext=ticktext))))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            getattr(fig, save)(os.path.join(save_dir, f"{col}_{legend_title}.svg"))
        # Displaying is independent of saving. Off by default - training/validation
        # logging routes figures to TensorBoard/WandB and must not render anything.
        if show:
            fig.show()
        return fig



    @staticmethod
    def _scatter_plotly(plot_df, xColumn, yColumn, classColoumn, palette, *,
                        xaxis_title, yaxis_title, legend_title, show_axes,
                        marker_size, opacity) -> plotly.graph_objs.Figure:
        import plotly.express as px

        if palette is None:
            fig = px.scatter(plot_df, x=xColumn, y=yColumn)
            fig.update_traces(marker=dict(color=plot_df[classColoumn].tolist()))
        else:
            fig = px.scatter(
                plot_df, x=xColumn, y=yColumn, color=classColoumn,
                color_discrete_map=palette,
                category_orders={classColoumn: list(palette)},
            )

        axis = dict(showgrid=False, zeroline=False, showticklabels=show_axes,
                    visible=True, showline=show_axes, linecolor="black",
                    ticks="outside" if show_axes else "", tickfont=dict(color="white"))
        fig.update_layout(
            xaxis=axis, yaxis=axis,
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
        fig.update_traces(marker=dict(size=marker_size, opacity=opacity))
        return fig

    @staticmethod
    def _scatter_matplotlib(plot_df, xColumn, yColumn, classColoumn, palette, *,
                            xaxis_title, yaxis_title, legend_title, show_axes,
                            marker_size, opacity) -> Figure:
        colors = plot_df[classColoumn]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(plot_df[xColumn], plot_df[yColumn],
                   c=colors if palette is None else colors.map(palette),
                   s=marker_size ** 2, alpha=opacity, linewidths=0)  # s is an area
        if palette is not None:
            ax.legend(handles=[Line2D([], [], ls="", marker="o", color=c, label=label)
                               for label, c in palette.items()],
                      title=legend_title, loc="lower left", bbox_to_anchor=(1, 0))
        ax.set(xlabel=xaxis_title, ylabel=yaxis_title)
        ax.grid(False)
        if not show_axes:
            ax.set(xticks=[], yticks=[])
            ax.spines[:].set_visible(False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Results & mask                                                       #
    # ------------------------------------------------------------------ #

    def saveDataFrame(self, suffix: str = "_analyzed", extension: str = ".csv") -> str:
        stem = pathlib.Path(self.df_path).stem if self.df_path is not None else "dataframe"
        out_path = os.path.join(self.save_dir or ".", f"{stem}{suffix}{extension}")
        self.results_df.to_csv(out_path)
        print(f"Dataframe saved to: {out_path}")
        return out_path

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
            for lbl in self.instancelabels[
                np.argsort(self.results_df[f"proba_{cls}"].to_numpy())[::-1][:n]]
        ]

    def _class_subsamples(self, max_samples: int) -> list[np.ndarray]:
        """Embeddings of each mapped class (sorted by class key), capped at *max_samples*."""
        rng = np.random.default_rng(42)
        subsamples = []
        for c in sorted(self.classMapping):
            emb = self.embeddings[self.gtlabels == c]
            if len(emb) > max_samples:
                emb = emb[rng.choice(len(emb), max_samples, replace=False)]
            subsamples.append(emb)
        return subsamples

    def _plot_pairwise_class_heatmap(self, per_class: list, distance, title: str) -> Figure:
        """Fill a symmetric class×class matrix with ``distance(a, b)`` and plot it."""
        n = len(per_class)
        matrix = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            matrix[i, j] = matrix[j, i] = distance(per_class[i], per_class[j])

        class_names = [self.classMapping[c] for c in sorted(self.classMapping)]
        fig, _ = CustomMatplotlib.simpleHeatmap(
            matrix,
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            save_dir=self.save_dir,
        )
        return fig

    def plot_gromov_wasserstein_heatmap(self, max_samples: int = 500) -> Figure:
        """Compute and plot pairwise Gromov-Wasserstein distances between classes."""
        import ot

        def _metric_space(X):
            # Normalised intra-class cost matrix + uniform weights, computed once per class.
            C = ot.dist(X, X)
            C /= C.max() + 1e-12
            return C, ot.unif(len(X))

        spaces = [_metric_space(X) for X in self._class_subsamples(max_samples)]
        return self._plot_pairwise_class_heatmap(
            spaces,
            lambda a, b: ot.gromov.gromov_wasserstein2(
                a[0], b[0], a[1], b[1], "square_loss", verbose=False),
            "Gromov-Wasserstein Distance Between Classes",
        )

    def plot_kl_divergence_heatmap(self, max_samples: int = 500) -> Figure:
        """Compute and plot pairwise symmetrised KL divergence between classes (Gaussian approximation via LedoitWolf)."""
        def _gaussian(X):
            # (mean, cov, inv(cov), logdet(cov)), computed once per class.
            cov = LedoitWolf().fit(X).covariance_
            return X.mean(axis=0), cov, np.linalg.inv(cov), np.linalg.slogdet(cov)[1]

        def _kl(p, q):
            mu_p, cov_p, _, logdet_p = p
            mu_q, _, inv_q, logdet_q = q
            diff = mu_q - mu_p
            return 0.5 * (
                    np.einsum("ij,ji->", inv_q, cov_p)  # trace(inv_q @ cov_p)
                    + diff @ inv_q @ diff
                    - len(mu_p)
                    + logdet_q - logdet_p
            )

        gaussians = [_gaussian(X) for X in self._class_subsamples(max_samples)]
        return self._plot_pairwise_class_heatmap(
            gaussians,
            lambda p, q: (_kl(p, q) + _kl(q, p)) / 2,
            "Symmetrised KL Divergence Between Classes",
        )

    def plot_dendrogram(
            self,
            metric: str = "cosine",
            linkage: str = "average",
            save_dir: str = None,
            show: bool = False,
    ):
        """Hierarchical clustering dendrogram of class centroids in embedding space.

        Computes one centroid per class (mean of ``trainEmbeddings``), then
        runs agglomerative clustering on the pairwise distance matrix and
        renders an interactive Plotly dendrogram.

        Parameters
        ----------
        metric:
            Pairwise distance metric passed to ``scipy.spatial.distance.pdist``
            (e.g. ``"cosine"``, ``"euclidean"``).
        linkage:
            Linkage method passed to ``scipy.cluster.hierarchy.linkage``
            (e.g. ``"average"``, ``"ward"``, ``"complete"``).
        save_dir:
            Directory in which to save ``dendrogram.html``.  Defaults to
            ``self.save_dir``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.figure_factory as ff
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import pdist

        classes = np.unique(self.traingt)
        centroids = np.stack([self.trainEmbeddings[self.traingt == c].mean(axis=0) for c in classes])

        fig = ff.create_dendrogram(
            centroids,
            orientation="left",
            labels=[self._class_name(c) for c in classes],
            distfun=lambda X: pdist(X, metric=metric),
            linkagefun=lambda d: sch.linkage(d, method=linkage),
        )
        fig.update_layout(
            title_text=f"Class dendrogram — {metric} distance, {linkage} linkage",
            template="plotly_dark",
            width=800,
            height=max(400, len(classes) * 40),
            xaxis=dict(title=f"{metric} distance"),
            yaxis=dict(title=""),
        )

        save_dir = save_dir or self.save_dir
        if save_dir:
            out = os.path.join(save_dir, "dendrogram.svg")
            fig.write_image(out, format="svg")
            print(f"[plot_dendrogram] Saved to {out}")

        if show:  # see specialScatter: never render from a logging path
            fig.show()
        return fig

    # ------------------------------------------------------------------ #
    # Trajectory analysis (PAGA + diffusion pseudotime)                   #
    # ------------------------------------------------------------------ #

    def trajectory(
            self,
            n_neighbors: int = 15,
            n_dcs: int = 10,
            resolution: float = 0.5,
            root_group: str = None,
            distance_metric: str = "euclidean",
            color_by: str = None,
            save_dir: str = None,
            use_gt: bool = False,
    ) -> pd.DataFrame:
        """Trajectory analysis via PAGA + diffusion pseudotime (scanpy).

        Workflow mirrors SeuratExtend / Slingshot in Python:
        1. Build a k-NN graph on ``self.embeddings``.
        2. Leiden clustering to define cell groups.
        3. PAGA to infer the coarse trajectory graph between groups.
        4. Diffusion-map embedding + diffusion pseudotime (DPT) to order
           cells along each branch.

        Results are stored in ``self.data_df`` (columns ``"leiden_trajectory"``,
        ``"dpt_pseudotime"``) and in ``self.results_df``, and a PAGA +
        pseudotime figure is written to *save_dir* (or ``self.save_dir``).

        Parameters
        ----------
        n_neighbors:
            Number of neighbours for the k-NN graph.
        n_dcs:
            Number of diffusion components to compute.
        resolution:
            Leiden resolution controlling the granularity of cell groups.
        root_group:
            Leiden cluster label (string) to treat as the trajectory root.
            When ``None`` the group with the lowest median pseudotime on the
            first diffusion component is chosen automatically.
        distance_metric:
            Distance metric for the k-NN graph.
        color_by:
            Column in ``self.data_df`` used to colour the UMAP panels.
            Defaults to ``self.gtColumn`` when available, else ``"dpt_pseudotime"``.
        save_dir:
            Directory in which to save the figure.  Defaults to
            ``self.save_dir``.

        Returns
        -------
        pd.DataFrame
            ``self.data_df`` updated with ``"leiden_trajectory"`` and
            ``"dpt_pseudotime"`` columns.
        """
        import anndata as ad
        import scanpy as sc

        save_dir = save_dir or self.save_dir

        # ── 1. Build AnnData from scaled embeddings ───────────────────────
        adata = ad.AnnData(X=self.embeddings.astype(np.float32))
        if color_by is None:
            color_by = self.gtColumn if self._has_gt else "dpt_pseudotime"

        # Carry group labels into obs so scanpy can colour by them
        if self._has_gt:
            adata.obs[self.gtColumn] = self.data_df[self.gtColumn].astype(str).values

        # ── 2. k-NN graph ─────────────────────────────────────────────────
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=None,
                        metric=distance_metric, random_state=111)

        # ── 3a. Group definition: GT labels or Leiden clustering ───────────
        if use_gt and self._has_gt:
            gt_labels = (
                self.data_df[self.gtColumn]
                .map(lambda v: self.classMapping.get(int(v), str(v))
                if pd.notna(v) and v != 0 else "unlabelled")
                .astype(str)
            )
            adata.obs["leiden_trajectory"] = pd.Categorical(gt_labels)
        else:
            sc.tl.leiden(adata, resolution=resolution, random_state=111,
                         key_added="leiden_trajectory")

        # ── 3. PAGA (trajectory graph between Leiden groups) ──────────────
        sc.tl.paga(adata, groups="leiden_trajectory")
        # sc.pl.paga populates adata.uns['paga']['pos'], required by umap(init_pos='paga')
        sc.pl.paga(adata, show=False, plot=False)

        # ── 4. UMAP for visualisation ─────────────────────────────────────
        # Reuse the UMAP stored by visualiseUMAP (in results_df) when available;
        # fall back to a PAGA-initialised UMAP otherwise.
        if {"UMAP x", "UMAP y"} <= set(self.results_df.columns):
            adata.obsm["X_umap"] = self.results_df[["UMAP x", "UMAP y"]].to_numpy()
        else:
            sc.tl.umap(adata, init_pos="paga", random_state=42)

        # ── 5. Diffusion map + pseudotime ─────────────────────────────────
        sc.tl.diffmap(adata, n_comps=n_dcs)

        groups = adata.obs["leiden_trajectory"].astype(str).to_numpy()
        dc1 = adata.obsm["X_diffmap"][:, 1]
        unique_groups = np.unique(groups)

        def _lowest_median_group() -> str:
            medians = {g: np.median(dc1[groups == g]) for g in unique_groups}
            return str(min(medians, key=medians.get))

        if root_group is None:
            root_group = _lowest_median_group()
            print(f"[trajectory] Auto-selected root group: {root_group}")
        elif root_group not in unique_groups:
            # root_group may be a gtColumn value rather than a Leiden label —
            # find the Leiden cluster most enriched for that value.
            if self._has_gt:
                gt_vals = self.data_df[self.gtColumn].astype(str).to_numpy()
                enrichment = {
                    g: (gt_vals[groups == g] == str(root_group)).mean()
                    for g in unique_groups
                }
                resolved = max(enrichment, key=enrichment.get)
                print(
                    f"[trajectory] '{root_group}' not a Leiden label; "
                    f"resolved to group '{resolved}' "
                    f"(enrichment {enrichment[resolved]:.2f})."
                )
            else:
                print(
                    f"[trajectory] Warning: root_group='{root_group}' not found "
                    f"in Leiden labels {list(unique_groups)}. Falling back to auto."
                )
                resolved = _lowest_median_group()
            root_group = resolved

        # Root cell: the cell in root_group with the lowest DC1 value
        root_mask = groups == root_group
        root_idx = int(np.flatnonzero(root_mask)[np.argmin(dc1[root_mask])])
        adata.uns["iroot"] = root_idx
        sc.tl.dpt(adata, n_dcs=n_dcs)

        # ── 6. Write results back to data_df / results_df ─────────────────
        umap_coords = adata.obsm["X_umap"]
        pseudotime = adata.obs["dpt_pseudotime"].to_numpy()
        traj_columns = {
            "leiden_trajectory": groups,
            "dpt_pseudotime": pseudotime,
            "traj_umap_x": umap_coords[:, 0],
            "traj_umap_y": umap_coords[:, 1],
        }
        for df in (self.data_df, self.results_df):
            for col, values in traj_columns.items():
                df[col] = values

        # Store adata so plot_trajectory can reuse it without recomputing
        self._traj_adata = adata
        self._traj_color_by = color_by
        self._traj_root_group = root_group

        print(f"[trajectory] Root cell index: {root_idx}  (group '{root_group}')")
        print(f"[trajectory] Pseudotime range: {pseudotime.min():.3f} – {pseudotime.max():.3f}")

        return self.data_df

    def plot_trajectory(
            self,
            save_dir: str = None,
            paga_threshold: float = 0.05,
            show: bool = False,
            **trajectory_kwargs,
    ):
        """Build and return an interactive Plotly figure of the trajectory results.

        Layout (2 × 2 grid):

        * **Top-left** – UMAP coloured by Leiden group with the PAGA
          connectivity graph superimposed (edges scaled by weight, nodes at
          cluster centroids).
        * **Top-right** – UMAP coloured by ``self.gtColumn`` (only when a
          ground-truth column is available and differs from pseudotime).
        * **Bottom-left** – UMAP coloured by diffusion pseudotime.
        * **Bottom-right** – Violin plot of pseudotime per Leiden group.

        Parameters
        ----------
        save_dir:
            If given, the figure is saved as ``trajectory_plotly.html`` there.
        paga_threshold:
            Minimum PAGA connectivity weight for an edge to be drawn.
        **trajectory_kwargs:
            Forwarded to :meth:`trajectory` when it needs to be computed.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # ── Compute trajectory if needed ──────────────────────────────────
        if not hasattr(self, "_traj_adata"):
            self.trajectory(**trajectory_kwargs)

        adata = self._traj_adata
        color_by = self._traj_color_by
        df = self.data_df

        ux = df["traj_umap_x"].to_numpy()
        uy = df["traj_umap_y"].to_numpy()
        groups = df["leiden_trajectory"].astype(str).to_numpy()
        pseudotime = df["dpt_pseudotime"].to_numpy()
        # PAGA connectivities are indexed in the categorical order of the groups,
        # so node i of the graph must be unique_groups[i].
        unique_groups = [str(g) for g in adata.obs["leiden_trajectory"].cat.categories]
        group_masks = {g: groups == g for g in unique_groups}

        # Shared palette — tab20 avoids the grey-for-0 override in specialScatter
        group_color = _plotly_rgb_palette("tab20", unique_groups)

        has_extra = self._has_gt and color_by != "dpt_pseudotime"

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "UMAP + PAGA – Leiden groups",
                f"UMAP – {color_by}" if has_extra else "",
                "UMAP – pseudotime",
                "Pseudotime per group",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "violin"}],
            ],
        )

        # ── Panel 1: UMAP (Leiden) with PAGA overlay ──────────────────────
        for g, mask in group_masks.items():
            fig.add_trace(go.Scatter(
                x=ux[mask], y=uy[mask],
                mode="markers",
                name=f"Group {g}",
                marker=dict(size=4, color=group_color[g], opacity=0.6),
                legendgroup=f"group_{g}",
                hovertemplate=f"Group {g}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

        # PAGA edges superimposed
        conn = adata.uns["paga"]["connectivities"].toarray()
        node_x = [ux[mask].mean() for mask in group_masks.values()]
        node_y = [uy[mask].mean() for mask in group_masks.values()]

        for i, j in itertools.combinations(range(len(unique_groups)), 2):
            w = float(conn[i, j])
            if w < paga_threshold:
                continue
            fig.add_trace(go.Scatter(
                x=[node_x[i], node_x[j], None],
                y=[node_y[i], node_y[j], None],
                mode="lines",
                line=dict(width=w * 10, color="rgba(255,255,255,0.55)"),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)

        # PAGA nodes (same colors as scatter, larger markers + labels)
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=unique_groups,
            textposition="top center",
            marker=dict(
                size=20,
                color=[group_color[g] for g in unique_groups],
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
            hovertemplate="Group %{text}<extra></extra>",
        ), row=1, col=1)

        # ── Panel 2 (optional): UMAP coloured by gtColumn ─────────────────
        if has_extra:
            gt_vals = df[color_by].astype(str).to_numpy()
            unique_gt = sorted(np.unique(gt_vals))
            gt_color = _plotly_rgb_palette("Set2", unique_gt)
            for v in unique_gt:
                mask = gt_vals == v
                fig.add_trace(go.Scatter(
                    x=ux[mask], y=uy[mask],
                    mode="markers",
                    name=str(v),
                    marker=dict(size=4, color=gt_color[v], opacity=0.7),
                    hovertemplate=f"{color_by}={v}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
                    legendgroup=f"gt_{v}",
                ), row=1, col=2)

        # ── Panel 3: UMAP coloured by pseudotime ─────────────────────────
        fig.add_trace(go.Scatter(
            x=ux, y=uy,
            mode="markers",
            marker=dict(
                size=4,
                color=pseudotime,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Pseudotime", x=1.02),
                opacity=0.8,
            ),
            showlegend=False,
            hovertemplate="pt=%{marker.color:.3f}<extra></extra>",
        ), row=2, col=1)

        # ── Panel 4: Violin – pseudotime per Leiden group ─────────────────
        for g, mask in group_masks.items():
            fig.add_trace(go.Violin(
                y=pseudotime[mask],
                name=f"Group {g}",
                box_visible=True,
                meanline_visible=True,
                fillcolor=group_color[g],
                line_color="white",
                opacity=0.8,
                showlegend=False,
                legendgroup=f"group_{g}",
            ), row=2, col=2)

        # ── Layout ────────────────────────────────────────────────────────
        fig.update_layout(
            height=900,
            title_text="Trajectory analysis — PAGA + diffusion pseudotime",
            template="plotly_dark",
            legend=dict(itemsizing="constant", font=dict(size=10)),
        )
        fig.update_xaxes(showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)

        save_dir = save_dir or self.save_dir
        if save_dir:
            out = os.path.join(save_dir, "trajectory_plotly.html")
            fig.write_html(out)
            print(f"[plot_trajectory] Saved to {out}")

        if show:  # see specialScatter: never render from a logging path
            fig.show()
        return fig

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[MOBIE_LABEL_KEY] == label]

    def get_array_from_df(self, column):
        return np.array(self.data_df[column].tolist())


