from __future__ import annotations

import base64
import itertools
import logging
import os
import pathlib
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dl_utils import (EMBED_DICT_EMBED, LABEL_KEY, MASKED_FEATURES_KEY,
                      MOBIE_LABEL_KEY, MoBie_coloring)
from dl_utils.util import getArrayFromDF
from dl_utils.vizualizations import CustomMatplotlib

# NOTE: no sns.set_context() at import.  Setting a context here mutated global
# matplotlib state for every importer of dl_utils.analysis, which EmbeddingAnalysis
# then had to undo per figure via `_plottingContext`.  Figures now take their
# context from that manager alone; the one figure that wants "poster" asks for it.
#
# skimage.io is likewise imported lazily at its two call sites: the top-level
# `import skimage` never bound the `.io` submodule, so `skimage.io.imread` only
# worked when some other module had imported it first -- and that submodule import
# is what trips the MKL/Intel-OpenMP crash in this environment.

# kaleido (Plotly static-image export) drives a headless Chromium via the
# choreographer package; both loggers have no explicit level of their own, so
# they inherit whatever the root logger is set to (e.g. logging.basicConfig
# level=INFO in a training entrypoint) and emit browser-lifecycle chatter on
# every fig.write_image(...) call. Pin them explicitly — an explicit per-logger
# level always wins over inherited root level, regardless of import order.
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)


def _scalarColumns(df: pd.DataFrame) -> list[str]:
    """Columns holding plain scalars -- the ones that belong in ``results_df``.

    Tests the first valid value per column rather than every cell.  These frames
    are homogeneous per column and carry embedding vectors as lists, so the old
    ``.apply(isinstance, ...)`` over every cell of every column dominated
    construction time on large tables.
    """
    def isScalar(col: pd.Series) -> bool:
        idx = col.first_valid_index()
        return idx is None or not isinstance(col[idx], (list, dict, np.ndarray))

    return [c for c in df.columns
            if df[c].dtype.kind in "fiuUSO" and isScalar(df[c])]


def _isBinaryColumn(col: pd.Series) -> bool:
    """True when every non-null value in *col* is 0 or 1."""
    try:
        return pd.to_numeric(col.dropna()).isin([0, 1]).all()
    except (ValueError, TypeError):
        return False


def _oneHotLabel(row, columns: list):
    """The name of the single active column in a one-hot *row*, else NaN."""
    active = [c for c in columns if float(row[c]) == 1.0]
    return active[0] if len(active) == 1 else np.nan


def _formatLabelValue(v) -> str:
    """Render one cell of a composite label, collapsing ``1.0`` to ``"1"``."""
    if pd.isna(v):
        return "NA"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _klDivergence(mu_p, sigma_p, mu_q, sigma_q) -> float:
    """KL(P || Q) between two multivariate Gaussians."""
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


def _png_buffer_to_data_url(buf: BytesIO) -> str:
    """Encode the PNG bytes held in *buf* as a ``data:image/png;base64`` URL."""
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded_image}"


def convert_array_to_data_url(path) -> str:
    import skimage.io

    array = skimage.io.imread(path)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(array, cmap="gray")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return _png_buffer_to_data_url(buf)


class Classification:

    # Factories rather than instances: the old dict constructed all four estimators
    # on every call just to select one.  LinearRegression is a *regressor* kept for
    # backwards compatibility -- it has no `classes_`/`predict_proba`, and
    # `EmbeddingAnalysis.classify` guards the paths that need them.
    _FACTORIES: dict[str, Callable] = {
        "KNN": lambda metric: KNeighborsClassifier(n_neighbors=5, metric=metric),
        "LogisticRegression": lambda metric: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": lambda metric: RandomForestClassifier(n_estimators=100, random_state=42),
        "LinearRegression": lambda metric: LinearRegression(),
    }

    @staticmethod
    def train_classifier(embeddings, labels, method="KNN", metric="cosine"):
        if method not in Classification._FACTORIES:
            raise ValueError(
                f"Unsupported classifier method: {method}. "
                f"Available: {list(Classification._FACTORIES)}"
            )
        return Classification._FACTORIES[method](metric).fit(embeddings, labels)

    @staticmethod
    def createConfusionMatrixFigure(x, gt, classifier, save_dir=None,
                                    mapping: dict = None, gtColumn="") -> Figure:
        mapping = mapping or {}
        ks: list[int] = classifier.classes_
        cm = confusion_matrix(gt, classifier.predict(x), normalize="true")
        ticks = [mapping.get(k, k) for k in ks]

        n = len(ks)
        fig_size = max(4, n * max(0.6, min(1.2, 12 / n)))
        annot = n <= 20
        fontsize = max(6, min(12, int(120 / n)))

        # This is the one figure that wanted the module-level "poster" context, so it
        # now asks for it explicitly instead of every importer paying for it.
        with sns.plotting_context("poster"):
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
            sns.heatmap(
                cm, ax=ax, annot=annot, cmap="Blues",
                xticklabels=ticks, yticklabels=ticks, cbar=True,
                fmt=".2f" if annot else "",
                annot_kws={"size": fontsize} if annot else {},
            )
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("Ground Truth Labels")
            # OO API: the old plt.xticks/plt.tight_layout/plt.savefig acted on the
            # *current* figure and worked here only because it happened to be this one.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
            fig.tight_layout()

        CustomMatplotlib.saveFig(fig, save_dir=save_dir,
                                 filename=f"{gtColumn}_confusion_matrix.png")
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


class EmbeddingAnalysis:

    # Figures this class draws get their own seaborn context.  "notebook" is right
    # for anything carrying one label per row plus a legend; the single-panel
    # confusion matrix asks for "poster" itself.  Override per subclass
    # (`plotContext = "talk"`) or per instance (`a.plotContext = "poster"`).
    plotContext: str = "notebook"
    fontScale: float = 1.0

    def _plottingContext(self):
        """seaborn context manager for figures this class draws."""
        return sns.plotting_context(self.plotContext, font_scale=self.fontScale)

    # ------------------------------------------------------------------ #
    # ClusteringConfig aliases                                             #
    # ------------------------------------------------------------------ #

    def _configProperty(field: str) -> property:  # noqa: N805 - class-body helper
        """Read/write alias onto ``self.clustering``.

        These six names used to be *copies* of the ClusteringConfig fields, assigned
        once at construction, so ``ea.leiden`` and ``ea.clustering.leiden`` could
        silently drift apart.  As properties there is exactly one value.
        """
        return property(lambda self: getattr(self.clustering, field),
                        lambda self, value: setattr(self.clustering, field, value))

    dbscan                 = _configProperty("dbscan")
    leiden                 = _configProperty("leiden")
    leiden_resolution      = _configProperty("resolution")
    leiden_n_iterations    = _configProperty("n_iterations")
    leiden_n_neighbors     = _configProperty("n_neighbors")
    leiden_distance_metric = _configProperty("distance_metric")
    del _configProperty

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        df_path: str | None = None,
        type: str = EMBED_DICT_EMBED,
        instancelabelColumn: str = MOBIE_LABEL_KEY,
        gtColumn: str | list[str] | None = None,
        classMapping: dict | None = None,
        label_order: list | None = None,
        save_dir: str | None = None,
        binary: bool = False,
        dbscan: bool = False,
        leiden: bool = True,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
        metadDataFramePath: Optional[str] = None,
        classifier_method: str | None = "LogisticRegression",
        greyBackground: bool = False,
        data_df: pd.DataFrame | None = None,
        embeddings: np.ndarray | None = None,
        default_class_column: str = "Class",
    ) -> None:
        """Analyse a set of embeddings.

        Give exactly one source: *df_path* (a JSON table), *data_df* (an in-memory
        table whose *type* column holds the vectors), or *embeddings* together with
        *data_df* as metadata -- see :meth:`_resolveSource`.  The two classmethod
        constructors are named shortcuts for the latter two forms; they are ordinary
        calls to this constructor, not instances forged with ``__new__``.

        *classifier_method* of ``None`` trains no classifier and colours plots by the
        ground-truth column instead.

        There is deliberately no ``**kwargs``: it only ever fed an
        ``assert not kwargs``, which replaced Python's own precise TypeError for an
        unexpected keyword with a weaker assertion that vanishes under ``python -O``.
        """
        # -- inputs --------------------------------------------------------
        self.df_path: str | None = df_path
        self.type: str | None = type
        self.data_df, vectors = self._resolveSource(df_path, data_df, embeddings, type)
        self.embeddings: np.ndarray = self._standardiseEmbeddings(vectors)

        # -- configuration -------------------------------------------------
        self.instancelabelColumn = instancelabelColumn
        self.classifier_method = classifier_method
        self.greyBackground = greyBackground
        self.classColumn = default_class_column
        self.classMappedColumn = "Mapped"
        self.subplots_kwargs = {"s": 6}
        self.classification = Classification   # kept: part of the public surface
        self.metadDataFramePath = metadDataFramePath
        self.clustering = ClusteringConfig(
            dbscan=dbscan, leiden=leiden, resolution=leiden_resolution,
            n_iterations=leiden_n_iterations, n_neighbors=leiden_n_neighbors,
            distance_metric=leiden_distance_metric,
        )
        self.save_dir = self._prepareSaveDir(save_dir, df_path)

        # -- table preparation ---------------------------------------------
        if metadDataFramePath is not None:
            self._mergeMetadata(metadDataFramePath)
        self.data_df, self.gtColumn, self.classMapping = self._standardise_label_column(
            self.data_df, gtColumn, classMapping or {}, label_order)
        self.results_df = self.data_df[_scalarColumns(self.data_df)].copy()
        self.getLabels()
        print(f"Embeddings shape: {self.embeddings.shape}")

        # -- labels ---------------------------------------------------------
        # classify() assigns predLabels itself.  The old code assigned its
        # (labels, figure) tuple here, leaving predLabels a *tuple* for the whole
        # lifetime of the object.
        if self._has_gt and classifier_method is not None:
            self.classify(binary=binary, mapping=classMapping)
        else:
            self.classifier = None
            self.predLabels = (self.gtlabels if self._has_gt
                               else np.zeros(len(self.data_df), dtype=int))

        # Add ground truth column as mapping
        self.results_df["Mapping_gtColumns"] = [self.classMapping.get(label, label) for label in self.gtlabels]
        self.results_df["gtColumns"] = self.gtlabels

        print("We have the following columns in the results DataFrame:", self.results_df.columns.tolist())

        if self.classMapping and self._has_gt:
            self._plotClassDiagnostics()

    # ------------------------------------------------------------------ #
    # Construction steps                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolveSource(df_path, data_df, embeddings, type) -> tuple[pd.DataFrame, np.ndarray]:
        """Resolve the three accepted input forms to ``(data_df, embeddings)``.

        Exactly one form applies, checked in order of specificity:

        * *embeddings* + *data_df* -- a pre-computed array with a metadata table;
        * *data_df*                -- an in-memory table, vectors in its *type* column;
        * *df_path*                -- the same table on disk as JSON.

        Making the input form a parameter is what lets every entry point reach the
        real ``__init__``; the alternate constructors previously had no way in and
        forged instances with ``cls.__new__(cls)``.
        """
        if embeddings is not None:
            if data_df is None:
                raise ValueError("`embeddings` must be accompanied by `data_df` as metadata.")
            return data_df.reset_index(drop=True).copy(), np.asarray(embeddings)
        if data_df is not None:
            return data_df.copy(), getArrayFromDF(data_df, type)
        if not (isinstance(df_path, str) and df_path.endswith(".json")):
            raise ValueError(
                f"Dataframe path must be a path to a JSON file, got {df_path!r}. "
                "Pass `data_df=` or `embeddings=` for the in-memory forms."
            )
        frame = pd.read_json(df_path)
        print(f"Available coloumns: {frame.columns.tolist()}")
        return frame, getArrayFromDF(frame, type)

    @staticmethod
    def _standardiseEmbeddings(vectors: np.ndarray) -> np.ndarray:
        """Validate and z-score the embedding matrix.

        These checks were ``assert``s, which are stripped under ``python -O`` -- the
        wrong tool for validating what a caller handed in.
        """
        if np.isnan(vectors).any():
            raise ValueError(f"Embeddings contain {int(np.isnan(vectors).sum())} NaN value(s).")
        if len(np.unique(vectors)) <= 1:
            raise ValueError(
                f"Embeddings are uninformative: constant value {np.unique(vectors)}."
            )
        return StandardScaler().fit_transform(vectors)

    @staticmethod
    def _prepareSaveDir(save_dir: str | None, df_path: str | None) -> str | None:
        """Resolve and create the output directory, defaulting beside *df_path*.

        The old code created the *parent* and left ``self.save_dir`` itself
        uncreated, which is why most writers re-ran makedirs defensively -- and why
        ``plot_dendrogram``, which did not, failed on its first call.
        """
        if save_dir is None and df_path is None:
            return None
        base = save_dir if save_dir is not None else os.path.dirname(df_path)
        resolved = os.path.join(base, "EmbeddingAnalysis")
        os.makedirs(resolved, exist_ok=True)
        return resolved

    def _mergeMetadata(self, path: str) -> None:
        """Left-join a metadata table onto ``data_df`` on the instance-label column."""
        readers = {".csv": pd.read_csv, ".xlsx": pd.read_excel, ".json": pd.read_json}
        suffix = pathlib.Path(path).suffix.lower()
        if suffix not in readers or not os.path.exists(path):
            raise ValueError(f"Metadata path must be an existing "
                             f"{'/'.join(readers)} file, got {path!r}")
        # .json was accepted by the old guard and then handed to read_excel.
        meta = readers[suffix](path)
        if self.instancelabelColumn not in meta.columns:
            raise ValueError(f"Instance label column {self.instancelabelColumn!r} "
                             f"not found in metadata DataFrame {path!r}.")
        self.data_df = self.data_df.merge(meta, on=self.instancelabelColumn, how="left")
        print(f"Metadata DataFrame loaded from: {path}")

    @property
    def color_space(self):
        return MoBie_coloring.GlasbeyARGBLut()

    @property
    def classOrder(self) -> list:
        """Class names in label order — ``classMapping`` key order, background (0) excluded.

        ``_standardise_label_column`` assigns codes 1..N in label order, so the
        sorted keys *are* that order.  This is the single definition of "class
        order" for plots that group by name rather than by code.
        """
        return [self.classMapping[k] for k in sorted(self.classMapping) if k != 0]

    @classmethod
    def from_dataframe(
        cls,
        data_df: pd.DataFrame,
        type: str = MASKED_FEATURES_KEY,
        instancelabelColumn: str = MOBIE_LABEL_KEY,
        gtColumn: str | list[str] | None = None,
        save_dir: str | None = None,
        classMapping: dict | None = None,
        label_order: list | None = None,
        binary: bool = False,
        dbscan: bool = False,
        leiden: bool = True,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
        metadDataFramePath: str | None = None,
        classifier_method: str | None = "LogisticRegression",
        greyBackground: bool = False,
    ) -> "EmbeddingAnalysis":
        """Construct from an in-memory table; vectors come from its *type* column.

        An ordinary call to ``__init__`` -- this used to forge an instance with
        ``cls.__new__(cls)`` and hand-seed its attributes, because ``__init__``
        could only read from a JSON path.  It also trains a classifier when a
        *gtColumn* is given, as ``__init__`` always has; the *binary* parameter was
        accepted and silently unused before.
        """
        instance = cls(
            data_df=data_df, type=type, instancelabelColumn=instancelabelColumn,
            gtColumn=gtColumn, save_dir=save_dir, classMapping=classMapping,
            label_order=label_order, binary=binary, dbscan=dbscan, leiden=leiden,
            leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
            metadDataFramePath=metadDataFramePath,
            classifier_method=classifier_method, greyBackground=greyBackground,
        )
        instance.subplots_kwargs = {"s": 1}   # read at plot time; not an init concern
        return instance

    @classmethod
    def fromEmbeddingArrayAndMetaDataFrame(
        cls,
        embeddings: np.ndarray,
        meta_df: "pd.DataFrame",
        save_dir: str | None = None,
        gtColumn: str | list[str] | None = None,
        instancelabelColumn: str = LABEL_KEY,
        label_order: list | None = None,
        leiden: bool = False,
        leiden_resolution: float = 1.0,
        leiden_n_iterations: int = 2,
        leiden_n_neighbors: int = 15,
        leiden_distance_metric: str = "euclidean",
    ) -> "EmbeddingAnalysis":
        """Construct directly from a raw embedding array and a metadata DataFrame.

        Unlike the other two forms the embeddings are supplied as a pre-computed
        ``np.ndarray``.  No classifier is trained (``classifier_method=None``);
        scatter plots are coloured by *gtColumn*, or uniformly when absent.
        """
        columns = [gtColumn] if isinstance(gtColumn, str) else (gtColumn or [])
        present = bool(columns) and all(c in meta_df.columns for c in columns)
        return cls(
            embeddings=embeddings, data_df=meta_df, gtColumn=gtColumn,
            instancelabelColumn=instancelabelColumn, label_order=label_order,
            save_dir=save_dir, leiden=leiden, leiden_resolution=leiden_resolution,
            leiden_n_iterations=leiden_n_iterations,
            leiden_n_neighbors=leiden_n_neighbors,
            leiden_distance_metric=leiden_distance_metric,
            classifier_method=None,
            default_class_column="_".join(columns) if present else "Class",
        )

    @staticmethod
    def _standardise_label_column(
        data_df: pd.DataFrame,
        gtColumn: str | list[str] | None,
        classMapping: dict,
        label_order: list | None = None,
    ) -> tuple["pd.DataFrame", str | None, dict]:
        """Convert *gtColumn* into a single integer label column in *data_df*.

        *gtColumn* may be a list of condition columns -- collapsed into one combined
        column, one-hot when every column is binary and joined value-wise otherwise
        -- or a single column name, in which case only the string->integer mapping
        applies.  ``None`` passes straight through, so callers need no guard.
        Returns ``(data_df, new_gtColumn_name, updated_classMapping)``.

        *label_order* pins the order in which label values are assigned integer
        codes — and therefore legend order, dendrogram leaves and colours, which
        are all functions of the code.  Values absent from the data are skipped
        and values absent from *label_order* are appended after it, so a superset
        or a partial list are both safe.  It overrides the column order implied by
        a list *gtColumn*; an explicit *classMapping* overrides both.
        """
        if gtColumn is None:
            return data_df, None, classMapping

        frame = data_df
        columnOrder: list | None = None   # the order implied by a list gtColumn

        if isinstance(gtColumn, list):
            columnOrder = list(gtColumn)  # preserve caller-specified order
            combined = "_".join(gtColumn)
            subset = frame[gtColumn]
            frame = frame.copy()
            if all(_isBinaryColumn(subset[c]) for c in gtColumn):
                frame[combined] = subset.apply(_oneHotLabel, axis=1, columns=gtColumn)
            else:
                frame[combined] = (subset.apply(lambda col: col.map(_formatLabelValue))
                                         .agg("_".join, axis=1))
            gtColumn = combined

        if gtColumn in frame.columns:
            values = getArrayFromDF(frame, gtColumn)
            if values.dtype.kind in ("U", "S", "O"):
                seen = [v for v in pd.unique(values) if not pd.isna(v)]
                if classMapping:
                    toInt = {v: k for k, v in classMapping.items()}
                else:
                    order = label_order if label_order is not None else columnOrder
                    if order is None:
                        ordered = seen   # no order given: first-appearance in the data
                    else:
                        # caller-supplied order; append any unseen values after it
                        ordered = [v for v in order if v in seen]
                        ordered += [v for v in seen if v not in ordered]
                    toInt = {v: i + 1 for i, v in enumerate(ordered)}
                    classMapping = {i: v for v, i in toInt.items()}
                if frame is data_df:
                    frame = frame.copy()
                frame[gtColumn] = np.array(
                    [toInt.get(v, np.nan) for v in values], dtype=float)

        return frame, gtColumn, classMapping

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
        """Derive training labels from the (already-standardised) gtColumn.

        Label *derivation* — list→combined-column joining, one-hot detection,
        and string→int mapping — is performed once upstream in
        :meth:`_standardise_label_column`, called from ``__init__``.
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

        if not self.greyBackground:
            valid_mask = ~pd.isna(self.gtlabels)
            if 0 in self.gtlabels[valid_mask]:
                self.gtlabels = np.where(
                    valid_mask, self.gtlabels + 1, self.gtlabels)

        self.instancelabels: np.ndarray = getArrayFromDF(
            self.data_df, self.instancelabelColumn)
        nan_mask = ~pd.isna(self.gtlabels)
        if nan_mask.sum() < n:
            warnings.warn(
                f"Labels contain {n - nan_mask.sum()} inactive/NaN value(s). "
                f"Training on {nan_mask.sum()} of {n} samples.",
                UserWarning, stacklevel=2,
            )
            # Turn nan values to 0 but filter for training classifier
            self.gtlabels[~nan_mask] = 0
            self.traingt = self.gtlabels[nan_mask].astype(int)
            self.trainEmbeddings = self.embeddings[nan_mask]
        else:
            self.traingt = self.gtlabels.astype(int) if np.issubdtype(
                self.gtlabels.dtype, np.floating) else self.gtlabels
            self.trainEmbeddings = self.embeddings

        # Write the labels back unconditionally.  The old code did this only inside
        # the zero-shift branch above, so data_df and gtlabels could disagree about
        # which integer codes the NaN rows carry.
        if self._has_gt:
            self.data_df[self.gtColumn] = self.gtlabels

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def classify(self, binary: bool = False, train_size: float = 0.6,
                 mapping: dict = None, classifier_weights=None) -> tuple[np.ndarray, Figure | None]:
        """Fit (or reuse) a classifier and record its predictions on ``results_df``.

        Returns ``(predLabels, confusionFigure)``.  Both are also stored on the
        instance, as ``self.predLabels`` and ``self.confusionFigure`` -- the return
        annotation used to claim a bare ``np.ndarray`` while returning the pair,
        which is how ``__init__`` came to store a tuple in ``self.predLabels``.
        """
        if classifier_weights is None:
            X_train, X_val, y_train, y_val = train_test_split(
                self.trainEmbeddings, self.traingt, train_size=train_size,
                stratify=self.traingt, random_state=42
            )
            print(
                f"Classes — train: {np.unique(y_train)}, val: {np.unique(y_val)}")
            print(f"Samples — train: {len(X_train)}, val: {len(X_val)}")

            self.classifier = Classification.train_classifier(
                X_train, y_train, method=self.classifier_method
            )
        else:
            self.classifier = classifier_weights
            X_val, y_val = self.trainEmbeddings, self.traingt

        # "LinearRegression" is in the registry for backwards compatibility but is a
        # regressor: no classes_, no predict_proba, and accuracy_score rejects its
        # continuous output.  One check gates every classifier-only step below.
        isClassifier = hasattr(self.classifier, "classes_")
        self.train_accuracy = (
            Classification.getAccuracy(X_val, y_val, self.classifier) if isClassifier
            else self.classifier.score(X_val, y_val)      # R^2 for a regressor
        )
        print(f"Validation {'accuracy' if isClassifier else 'R^2'}: {self.train_accuracy:.4f}")

        self.predLabels = self.classifier.predict(self.embeddings)
        self.classColumn = self.classifier_method
        self.results_df[self.classColumn] = self.predLabels
        # classMapping is normalised to {} in __init__, so the old
        # `if self.classMapping is not None: ... else: ...` had a dead else branch.
        self.results_df[self.classMappedColumn] = np.array(
            [self.classMapping.get(label, label) for label in self.predLabels]
        )

        self.confusionFigure = None
        if isClassifier:
            proba = self.classifier.predict_proba(self.embeddings)
            for i, cls in enumerate(self.classifier.classes_):
                self.results_df[f"proba_{cls}"] = proba[:, i]
            self.confusionFigure = Classification.createConfusionMatrixFigure(
                x=X_val, gt=y_val, classifier=self.classifier,
                save_dir=self.save_dir, mapping=mapping or self.classMapping,
                gtColumn=self.gtColumn
            )
        return self.predLabels, self.confusionFigure

    # ------------------------------------------------------------------ #
    # Clustering                                                           #
    # ------------------------------------------------------------------ #

    def _rescaled(self, embeddings: np.ndarray) -> np.ndarray:
        """Z-score *embeddings*, unless they are the already-standardised ones.

        ``self.embeddings`` is standardised once in ``__init__``; re-fitting a
        StandardScaler on it is a no-op that costs a full pass over the matrix.
        """
        return embeddings if embeddings is self.embeddings else \
            StandardScaler().fit_transform(embeddings)

    def performDBSCAN(self, embeddings, shape, DBSCAN_eps=0.5, DBSCAN_min_samples=10) -> np.ndarray:
        clustering: DBSCAN = DBSCAN(
            eps=DBSCAN_eps, min_samples=int(DBSCAN_min_samples)
        ).fit(self._rescaled(embeddings))
        self.classColumn = "DBSCAN_cluster"
        labels = clustering.labels_.reshape(*shape).astype(int) + 1
        self.predLabels = labels.reshape(-1)
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

        embedding = ad.AnnData(X=self._rescaled(embeddings))
        scanpy.pp.neighbors(
            embedding, n_neighbors=n_neighbors, n_pcs=None,
            # type: ignore[arg-type]
            metric=distance_metric, random_state=111, use_rep="X"
        )
        scanpy.tl.leiden(
            embedding, resolution=resolution, random_state=111, n_iterations=n_iterations,
        )
        # factorize reproduces the old per-cluster masking loop exactly -- same
        # first-appearance ordering -- in one vectorised pass.
        codes, _ = pd.factorize(embedding.obs["leiden"])
        labels = codes.astype(int).reshape(*shape) + 1

        # Was missing, while performDBSCAN set its own: Leiden labels therefore
        # landed in whatever classColumn happened to be, overwriting the
        # classifier's prediction column whenever cluster() followed classify().
        self.classColumn = "Leiden_cluster"
        self.predLabels = labels.reshape(-1)
        return labels

    def cluster(self, embeddings=None, **kwargs):
        """Cluster the embeddings; labels land in a column named for the backend."""
        backend = self.performLeiden if self.leiden else self.performDBSCAN
        labels = backend(
            embeddings=self.embeddings if embeddings is None else embeddings,
            shape=self.embeddings.shape[:-1], **kwargs,
        )
        self.results_df[self.classColumn] = labels.reshape(-1)
        return labels

    # ------------------------------------------------------------------ #
    # Dimensionality reduction                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _componentFrame(array: np.ndarray, prefix: str, index) -> pd.DataFrame:
        """Frame of ``"<prefix> x/y/z"`` columns for however many components exist.

        The old code indexed columns 0..2 unconditionally, so a narrow embedding
        (fewer than three components) raised IndexError.
        """
        axes = ["x", "y", "z"][:array.shape[1]]
        return pd.DataFrame(
            {f"{prefix} {axis}": array[:, i] for i, axis in enumerate(axes)},
            index=index,
        )

    def pca(self) -> np.ndarray:
        pca_model = PCA()
        if np.isnan(self.embeddings).any():
            raise ValueError(
                f"Embeddings contain {int(np.isnan(self.embeddings).sum())} NaN value(s).")
        pcs = pca_model.fit_transform(self.embeddings)
        if np.isnan(pcs).any():
            raise ValueError(
                f"PCA produced {int(np.isnan(pcs).sum())} NaN value(s).")
        print(
            f"Explained variance      : {pca_model.explained_variance_ratio_[:5].round(3)}")
        print(
            f"Cumulative (first 3)    : {pca_model.explained_variance_ratio_[:3].sum():.3f}")
        pca_df = self._componentFrame(pcs, "PCA", self.data_df.index)

        # Keep the fitted model and the loadings; plotPCALoadings reads them.
        self.pca_model = pca_model
        self.loadings = pd.DataFrame(
            pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
            columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
            index=np.arange(self.embeddings.shape[1]),
        )
        print("PCA loadings:")
        print(self.loadings)

        self.explained_variance_ratio = pca_model.explained_variance_ratio_[:3].sum()

        self.concatDF(pca_df)
        return pcs

    def UMAP(self, n_neighbors=15, min_dist=0.1, n_components=3,
             random_state=42, metric="euclidean", **kwargs) -> pd.DataFrame:
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
            random_state=random_state, metric=metric, n_jobs=10, **kwargs,
        ).fit(self.embeddings)
        # Honours n_components: the old code always built a "UMAP z" column and
        # raised IndexError when fewer than three were requested.
        return self._componentFrame(
            reducer.transform(self.embeddings), "UMAP", self.data_df.index)

    def UMAPResults(self, classColoumn: str | None = None,
                    marker_size: int = 4, background_color: str = "rgba(0,0,0,0)",
                    umap_kwargs: dict | None = None,
                    **kwargs):
        col = classColoumn if classColoumn is not None else self.classColumn
        # umap_kwargs flow to UMAP() (e.g. n_neighbors); remaining kwargs to the
        # scatter plot. UMAP requires n_neighbors < n_samples, so small feature
        # spaces (e.g. one point per pooled sample) must pass a capped value.
        self.concatDF(self.UMAP(**(umap_kwargs or {})))
        return EmbeddingAnalysis.specialScatter(
            self,
            xColumn="UMAP x", yColumn="UMAP y",
            xaxis_title="UMAP Dimension 1", yaxis_title="UMAP Dimension 2",
            classColoumn=col,
            legend_title="UMAP - {}".format(
                self.gtColumn if self._has_gt else self.classColumn),
            save_dir=self.save_dir, **kwargs
        )

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _returnFigure(fig: Figure, show: bool = False) -> Figure:
        """Hand back a matplotlib figure without Jupyter drawing it twice.

        pyplot keeps every figure built with plt.subplots in its registry, and
        the inline backend flushes that registry at the end of a cell -- so a
        figure that is returned *and* displayed renders twice.  Closing drops it
        from the registry; the object still renders wherever it is displayed
        explicitly, and savefig keeps working on it.  Matplotlib counterpart of
        the `show` flag on specialScatter/plot_dendrogram.
        """
        if show:
            from IPython.display import display
            display(fig)
        plt.close(fig)
        return fig

    def _saveFigure(self, fig: Figure, filename: str, save_dir: str = None) -> None:
        """Write *fig* into *save_dir* (default ``self.save_dir``), creating it.

        Delegates to ``CustomMatplotlib.saveFig`` so the whole package has one
        makedirs-and-savefig; this replaced six near-identical blocks whose dpi and
        bbox_inches had drifted apart.  ``close=False`` because ``_returnFigure``
        owns closing.
        """
        CustomMatplotlib.saveFig(
            fig, save_dir=save_dir if save_dir is not None else self.save_dir,
            filename=filename, close=False,
        )

    @staticmethod
    def _returnPlotlyFigure(fig, save_dir: str, filename: str, show: bool = False):
        """Plotly counterpart of :meth:`_returnFigure`.

        Saving and showing stay independent: training/validation logging routes
        figures to TensorBoard/WandB and must never pop one open, so *show*
        defaults to False everywhere.
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, filename)
            if filename.endswith(".html"):
                fig.write_html(out)
            else:
                fig.write_image(out)
            print(f"Saved to {out}")
        if show:
            fig.show()
        return fig

    def plot_distance_matrix(self, metric: str = "cosine", show: bool = False,
                             max_samples: int = 2000) -> Figure:
        """Plot a pairwise distance matrix sorted by GT class label.

        Uses the labelled training embeddings (``trainEmbeddings`` /
        ``traingt``).  Each row is normalised to sum to 1 so relative
        distances are comparable across rows with different scales.

        *max_samples* caps the N x N distance matrix, which was previously
        unbounded -- at this project's ~70k objects that is tens of gigabytes.
        The subsample is seeded, like the one in :meth:`_classSamples`.
        """
        from scipy.spatial.distance import cdist

        rows = np.arange(len(self.traingt))
        if len(rows) > max_samples:
            rows = np.random.default_rng(42).choice(rows, max_samples, replace=False)
        sort = rows[np.argsort(self.traingt[rows])]
        feats = self.trainEmbeddings[sort]
        labels = self.traingt[sort]
        classes = np.unique(labels)
        print(f"[plot_distance_matrix] {len(classes)} classes found in traingt: "
              f"{[self.classMapping.get(int(c), int(c)) for c in classes]}")

        D = cdist(feats, feats, metric=metric)  # type: ignore[call-overload]
        D = D / (D.sum(axis=1, keepdims=True) + 1e-12)

        tick_pos, tick_lbl, boundaries = [], [], []
        for i, cls in enumerate(classes):
            positions = np.where(labels == cls)[0]
            tick_pos.append(positions.mean())
            tick_lbl.append(str(self.classMapping.get(int(cls), int(cls))))
            if i < len(classes) - 1:
                boundaries.append(positions[-1] + 0.5)

        # Built inside the context manager so the artists pick up its rc values.
        with self._plottingContext():
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.grid(False)
            im = ax.imshow(D, aspect="auto", cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax, label=f"{metric} distance (row-normalised)")

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
            plt.tight_layout()

        self._saveFigure(fig, f"distance_matrix_{metric}.png")
        return self._returnFigure(fig, show)

    def plotPCALoadings(
        self,
        n_components: int = 3,
        by: str = "feature",
        columnMapping: dict | list | None = None,
        features: slice | list | None = None,
        sort_by: str | None = "PC1",
        palette: str | list = "Set2",
        residual_label: str | None = None,
        residual_color: str = "lightgrey",
        figsize: tuple | None = None,
        save_dir: str | None = None,
        show: bool = False,
    ) -> Figure:
        """Stacked fractional bar plot of the PCA loadings, in either direction.

        The variancePartition ``plotPercentBars`` idiom: every bar spans 100 %.

        ``by="feature"`` draws one bar per input feature, split into the fraction
        of *that feature's* variance carried by PC1, PC2, ... with the remaining
        components pooled into a ``residual_label`` segment.  Because
        ``self.embeddings`` is standardised, the squared loadings of a feature
        sum to its total variance, so each bar normalises exactly.

        ``by="component"`` answers the opposite question -- how much each feature
        contributes to the strongest directions of variance.  One bar per
        component, split by feature (squared loadings normalised down the column,
        the standard PCA contribution), with each bar labelled by the component's
        explained-variance ratio.

        Parameters
        ----------
        n_components
            Number of components: segments per bar (``by="feature"``) or bars
            (``by="component"``).
        columnMapping
            Feature labels, keyed by position in the embedding vector
            (``{0: "area (Nuclei)", ...}``) or given as a plain list in feature
            order.  Unmapped features fall back to ``"feature {i}"``.
        features
            Optional subset/slice of features, applied by position
            (``slice(0, 5)`` mirrors ``varPart[1:5, ]``).  Under
            ``by="component"`` the features left out are pooled into the residual
            segment -- worth using for wide embeddings, which would otherwise
            produce one legend entry per dimension.
        sort_by
            Component whose values order the features, descending; ``None`` keeps
            feature order.  Under ``by="feature"`` that orders the bars, under
            ``by="component"`` the segments within every bar (the bars themselves
            stay in PC order).

        Type size comes from the class (``plotContext`` / ``fontScale``), not from
        the module-level "poster" context, which crowds one-label-per-feature bars.
        """
        if by not in ("feature", "component"):
            raise ValueError(f"by must be 'feature' or 'component', got {by!r}")
        if not hasattr(self, "loadings"):
            self.pca()

        sq = self.loadings.pow(2)
        mapping = dict(enumerate(columnMapping)) if isinstance(columnMapping, list) \
            else (columnMapping or {})

        def name(i):
            return str(mapping.get(i, mapping.get(str(i), f"feature {i}")))

        if by == "feature":
            residual_label = residual_label or "Other PCs"
            frac = sq.div(sq.sum(axis=1), axis=0)           # feature variance -> PCs
            keep = list(frac.columns[:n_components])
            frac = frac[keep].assign(**{residual_label: 1.0 - frac[keep].sum(axis=1)})
            if features is not None:
                frac = frac.iloc[features]
            if sort_by:
                frac = frac.sort_values(sort_by, ascending=False)
            bars = [name(i) for i in frac.index]
            xlabel = "Variance explained (%)"
        else:
            residual_label = residual_label or "Other features"
            contrib = sq.div(sq.sum(axis=0), axis=1).iloc[:, :n_components]  # PC variance -> features
            shown = contrib if features is None else contrib.iloc[features]
            if sort_by:
                shown = shown.sort_values(sort_by, ascending=False)
            frac = shown.T                                  # index = PCs, columns = features
            frac.columns = self._uniqueLabels([name(i) for i in shown.index], shown.index)
            if len(shown) < len(contrib):
                frac[residual_label] = 1.0 - frac.sum(axis=1)
            bars = [f"{pc} ({r:.1%})" for pc, r in
                    zip(frac.index, self.pca_model.explained_variance_ratio_)]
            xlabel = "Contribution to component (%)"

        bars = self._uniqueLabels(bars, frac.index)

        segments = [c for c in frac.columns if c != residual_label]
        colors = dict(zip(segments, sns.color_palette(palette, len(segments))))
        colors[residual_label] = residual_color

        # seaborn has no stacked barplot: draw reverse-cumulative widths so each
        # narrower bar overpaints the left edge of the previous one.  Everything
        # is built inside the context manager so the artists pick up its rc.
        cum = frac.cumsum(axis=1) * 100
        with self._plottingContext():
            fig, ax = plt.subplots(figsize=figsize or (10, 0.6 * len(frac) + 2))
            for col in reversed(frac.columns):
                sns.barplot(x=cum[col].to_numpy(), y=bars, order=bars, orient="y",
                            color=colors[col], label=col, errorbar=None, width=0.7, ax=ax)

            ax.set(xlim=(0, 100), xlabel=xlabel, ylabel="")
            handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], legend_labels[::-1], loc="center left",
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
            sns.despine(left=True)
            fig.tight_layout()

        self._saveFigure(fig, f"pca_loadings_bars_by_{by}.png", save_dir)
        return self._returnFigure(fig, show)

    def plotFeatureMeans(self, columnMapping: dict | list, groupBy: str,
                         features: slice | list | None = None, sort_by: str | None = None,
                         palette: str | list = "Set2", figsize: tuple | None = None,
                         save_dir: str | None = None, show: bool = False) -> Figure:
        """Grouped bars of mean feature values, each feature scaled to its own maximum.

        Means come from the *unscaled* vectors in ``data_df[self.type]``:
        ``self.embeddings`` is standardised, so its per-feature means are 0.
        Dividing each feature by its largest group mean puts the strongest group at
        1.0 and reads the rest as a fraction of it, which is what lets features in
        different units share one axis.

        ``columnMapping`` labels features by position (``{0: "area (Nuclei)", ...}``)
        or as a list in feature order.  ``groupBy`` resolves against ``results_df``
        first, so cluster/prediction columns work, then ``data_df``; code-valued
        label columns map back through ``self.classMapping`` (``_standardise_label_column``
        stores them as floats).  ``sort_by="spread"``
        puts the most discriminative features first, a group name sorts by that group.
        """
        frame = pd.DataFrame(getArrayFromDF(self.data_df, self.type), index=self.data_df.index)
        frame = frame.iloc[:, features] if features is not None else frame

        groups = self.results_df[groupBy] if groupBy in self.results_df else self.data_df[groupBy]
        if self.classMapping and groups.dtype.kind in "iuf":
            # Ordered categorical so groupby (and seaborn's hue) follow the class
            # order rather than sorting the names alphabetically.  Unlabelled rows
            # (code 0, absent from classOrder) become NaN and drop out.
            groups = pd.Categorical(
                groups.map(lambda v: self.classMapping.get(int(v), v) if pd.notna(v) else v),
                categories=self.classOrder, ordered=True,
            )

        means = frame.groupby(groups, observed=True).mean().T   # features x groups
        means = means.div(means.abs().max(axis=1), axis=0)      # each feature -> its own max

        labels = dict(enumerate(columnMapping)) if isinstance(columnMapping, list) else columnMapping
        means.index = [str(labels.get(k, labels.get(str(k), f"feature {k}"))) for k in means.index]

        with self._plottingContext():
            fig, ax = plt.subplots(figsize=figsize or (10, .3 * means.size + 1.5))
            sns.barplot(data=means.rename_axis("feature").reset_index().melt("feature", var_name=groupBy),
                        x="value", y="feature", hue=groupBy, orient="y",
                        palette=palette, errorbar=None, ax=ax)
            ax.set(xlabel="Mean value (fraction of per-feature maximum)", ylabel="")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, title=groupBy)
            sns.despine(left=True)
            fig.tight_layout()

        self._saveFigure(fig, "feature_means.png", save_dir)
        return self._returnFigure(fig, show)

    @staticmethod
    def _uniqueLabels(labels: list, keys) -> list:
        """Suffix duplicate labels with their key so bars/segments stay distinct."""
        if len(set(labels)) == len(labels):
            return list(labels)
        return [f"{l} [{k}]" for l, k in zip(labels, keys)]

    def vizualisePCA(self, pcas=None, title="", classColoumn: str | None = None,
                     precomputed_colors: bool = False, axis_range: tuple | None = (-1, 1),
                     show_axes: bool = True):
        col = classColoumn if classColoumn is not None else self.classColumn
        # Call EmbeddingAnalysis.specialScatter directly so that predicted-label
        # columns are never temporarily zeroed by subclass overrides, allowing
        # every sample to be coloured by its predicted class.
        return EmbeddingAnalysis.specialScatter(
            self,
            xColumn="PCA x", yColumn="PCA y",
            xaxis_title="PC 1", yaxis_title="PC 2",
            classColoumn=col,
            legend_title="PC - {}".format(self.gtColumn if self._has_gt else col),
            save_dir=self.save_dir,
            precomputed_colors=precomputed_colors,
            axis_range=axis_range, show_axes=show_axes,
            **self.subplots_kwargs
        )

    @staticmethod
    def vizualiseCoord(points: dict, title="", labels=None, save_dir=None, **subplots_kwargs):
        return CustomMatplotlib.simpleScatter(
            points, labels, title=title, save_dir=save_dir, **subplots_kwargs,
        )[0]

    def specialScatter(self, xColumn, yColumn, xaxis_title="UMAP Dimension 1",
                       yaxis_title="UMAP Dimension 2", classColoumn: str = "color",
                       mapping: dict = None, legend_title: str = "Classes",
                       save_dir: Optional[str] = "./", precomputed_colors: bool = False,
                       show_axes: bool = False, color_background: bool = False, withLegendTitle=True,
                       show: bool = False, axis_range: tuple | None = None,
                       **kwargs) -> plotly.graph_objs.Figure:
        import plotly.express as px

        mapping = mapping or {}
        plot_df = self.results_df.copy()

        if axis_range is not None:
            # Rescale the cloud into `axis_range` and pin both axes to it, so the
            # plot is centred on the origin with a fixed, readable scale. One
            # shared factor for x and y: scaling them apart would stretch the
            # component geometry the scatter exists to show.
            lo, hi = axis_range
            span = float(np.nanmax(np.abs(plot_df[[xColumn, yColumn]].to_numpy())))
            if span > 0:
                plot_df[[xColumn, yColumn]] = (
                    plot_df[[xColumn, yColumn]] * ((hi - lo) / 2 / span) + (hi + lo) / 2
                )

        if precomputed_colors:
            # Color column already contains plotly-compatible rgba strings — use directly
            fig = px.scatter(plot_df, x=xColumn, y=yColumn)
            fig.update_traces(marker=dict(
                color=plot_df[classColoumn].tolist(), size=4))
        else:

            if classColoumn not in plot_df.columns:
                print(f"{classColoumn} is not a column in DataFrame")
                plot_df[classColoumn] = 0

            if pd.api.types.is_numeric_dtype(plot_df[classColoumn]):
                # Integer class codes: colour through the Glasbey LUT, with 0
                # reserved for unlabelled/background.
                plot_df[classColoumn] = plot_df[classColoumn].fillna(0)
                classLabels = sorted(
                    plot_df[classColoumn].unique().astype(int).tolist())

                def _label(k: int) -> str:
                    if k == 0 and color_background:
                        return "background"
                    return mapping.get(k, self.classMapping.get(k, str(k)))

                color_map = {
                    _label(k): f"rgba{self.color_space.rgba_tuple_by_index(k)}"
                    for k in classLabels}
                if not color_background:
                    color_map[_label(0)] = "rgba(128, 128, 128, 0.5)"
                plot_df[classColoumn] = plot_df[classColoumn].astype(int).map(_label)
                order = [_label(k) for k in classLabels]
            else:
                # Already-named categories (e.g. "leiden_trajectory", the mapped
                # class-name column).  The old code called .astype(int) on these
                # unconditionally and raised.
                plot_df[classColoumn] = plot_df[classColoumn].fillna("unlabelled").astype(str)
                order = sorted(plot_df[classColoumn].unique().tolist())
                palette = sns.color_palette("tab20", len(order))
                color_map = {name: f"rgb{tuple(int(c * 255) for c in rgb)}"
                             for name, rgb in zip(order, palette)}

            fig = px.scatter(
                plot_df, x=xColumn, y=yColumn, color=classColoumn,
                color_discrete_map=color_map,
                category_orders={classColoumn: order},
            )

        if not withLegendTitle:
            legend_title = None

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=show_axes,
                       visible=True, showline=show_axes, linecolor="black",
                       ticks="outside" if show_axes else "", tickfont=dict(color="black"),
                       range=list(axis_range) if axis_range is not None else None),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=show_axes,
                       visible=True, showline=show_axes, linecolor="black",
                       ticks="outside" if show_axes else "", tickfont=dict(color="black"),
                       range=list(axis_range) if axis_range is not None else None),
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
        return self._returnPlotlyFigure(
            fig, save_dir, f"{classColoumn}_{legend_title}.svg", show)

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
        """Paint *maskVolumePath*'s instance labels with a column of class values."""
        import skimage.io

        if not hasattr(self, "predLabels"):
            # Was `self.clustering(...)`, which is the ClusteringConfig dataclass --
            # not callable, so this path raised TypeError whenever it was reached.
            self.cluster(resolution=resolution, n_iterations=n_iterations,
                         n_neighbors=n_neighbors, distance_metric=distance_metric)

        # Defaulting to classColumn means the no-argument call works; it used to
        # fall through every branch and raise.
        column = classColoumn if classColoumn is not None else self.classColumn
        if column in self.results_df.columns:
            # results_df carries predictions for *all* instances, including those
            # without a ground-truth label.  (The first two branches of the old
            # chain had identical bodies behind a redundant guard.)
            values = self.results_df[column].to_numpy()
        elif column == self.gtColumn:
            values = self.gtlabels
        else:
            raise ValueError(
                f"Column {column!r} not found in results DataFrame. "
                f"Available: {self.results_df.columns.tolist()}")

        maskVol = skimage.io.imread(maskVolumePath)
        mapping_array = np.zeros(maskVol.max() + 1, dtype=np.uint16)
        mapping_array[self.instancelabels] = values.astype(int)
        maskVol = mapping_array[maskVol]
        fileName = os.path.basename(maskVolumePath).replace(
            ".tif", f"_{column}.tif")
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
        """Add *df2*'s columns to *df1*, aligned on *df1*'s index.

        Columns already in *df1* are replaced in place, so calling ``UMAPResults``
        (or ``pca``) twice updates the existing coordinate columns rather than
        appending a second set.  The old loop assigned ``.values`` column by
        column, which silently misaligns rows whenever the two frames carry
        different indices -- it worked only because every caller happened to build
        *df2* with ``df1``'s index.
        """
        aligned = df2.set_axis(df1.index)
        return df1.assign(**{c: aligned[c] for c in aligned.columns})

    def concatDF(self, df) -> pd.DataFrame:
        self.results_df = self.concatColumnDF(self.results_df, df)
        return self.results_df

    def get_top_instances_per_class(self, n: int = 5) -> list[tuple]:
        """Return (class_name, instance_label) pairs for the top-n most confident instances per class.

        Yields n entries per class ordered highest-confidence first.
        class_name is resolved via classMapping when available.
        """
        if getattr(self, "classifier", None) is None:
            raise RuntimeError(
                "No classifier was trained, so there are no per-class confidences. "
                "Construct with a classifier_method (not None) and a gtColumn, or "
                "call classify() first."
            )
        mapping = self.classMapping or {}
        return [
            (mapping.get(cls, cls), lbl)
            for cls in self.classifier.classes_
            for lbl in self.instancelabels[self.results_df[f"proba_{cls}"].argsort()[-1:-n - 1:-1]]
        ]

    def _classSamples(self, max_samples: int, seed: int = 42) -> dict:
        """Per-class embedding blocks, subsampled to at most *max_samples* rows."""
        rng = np.random.default_rng(seed)
        blocks = {}
        for c in sorted(self.classMapping):
            emb = self.embeddings[self.gtlabels == c]
            if len(emb) > max_samples:
                emb = emb[rng.choice(len(emb), max_samples, replace=False)]
            blocks[c] = emb
        return blocks

    def _pairwiseClassHeatmap(self, pairFn, title: str, max_samples: int = 500,
                              show: bool = False, prepareFn=None) -> Figure:
        """Symmetric class x class heatmap of *pairFn* over prepared class blocks.

        *prepareFn* runs once per class rather than once per pair.  Besides sharing
        the skeleton of the two public heatmaps, that removes a hidden O(k^2) cost:
        the Gromov-Wasserstein version used to recompute each class's cost matrix
        once for every partner class.
        """
        blocks = self._classSamples(max_samples)
        if prepareFn is not None:
            blocks = {c: prepareFn(X) for c, X in blocks.items()}
        classes = list(blocks)

        matrix = np.zeros((len(classes), len(classes)))
        for i, j in itertools.combinations(range(len(classes)), 2):
            matrix[i, j] = matrix[j, i] = pairFn(blocks[classes[i]], blocks[classes[j]])

        names = [self.classMapping[c] for c in classes]
        fig, _ = CustomMatplotlib.simpleHeatmap(
            matrix, xticklabels=names, yticklabels=names,
            title=title, save_dir=self.save_dir,
        )
        return self._returnFigure(fig, show)

    def _plotClassDiagnostics(self) -> None:
        """Draw both class-distance heatmaps, skipping any whose deps are missing.

        ``__init__`` used to call the two methods directly, so a missing optional
        package (POT, for Gromov-Wasserstein) broke construction outright.
        """
        try:
            self.plot_gromov_wasserstein_heatmap()
        except ImportError as exc:
            warnings.warn(f"Skipping Gromov-Wasserstein heatmap: {exc}",
                          UserWarning, stacklevel=2)
        self.plot_kl_divergence_heatmap()

    def plot_gromov_wasserstein_heatmap(self, max_samples: int = 500, show: bool = False) -> Figure:
        """Compute and plot pairwise Gromov-Wasserstein distances between classes."""
        import ot

        def prepare(X):
            C = ot.dist(X, X)
            return C / (C.max() + 1e-12), np.full(len(X), 1 / len(X))

        def gromovWasserstein(p, q):
            (Ci, pi), (Cj, pj) = p, q
            return ot.gromov.gromov_wasserstein2(
                Ci, Cj, pi, pj, "square_loss", verbose=False)

        return self._pairwiseClassHeatmap(
            gromovWasserstein, "Gromov-Wasserstein Distance Between Classes",
            max_samples, show, prepareFn=prepare)

    def plot_kl_divergence_heatmap(self, max_samples: int = 500, show: bool = False) -> Figure:
        """Compute and plot pairwise symmetrised KL divergence between classes (Gaussian approximation via LedoitWolf)."""
        def symmetrisedKL(p, q):
            return (_klDivergence(*p, *q) + _klDivergence(*q, *p)) / 2

        return self._pairwiseClassHeatmap(
            symmetrisedKL, "Symmetrised KL Divergence Between Classes",
            max_samples, show,
            prepareFn=lambda X: (X.mean(axis=0), LedoitWolf().fit(X).covariance_))

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
        from scipy.spatial.distance import pdist
        import scipy.cluster.hierarchy as sch

        classes = sorted(np.unique(self.traingt))
        centroids = np.stack([
            self.trainEmbeddings[self.traingt == c].mean(axis=0)
            for c in classes
        ])
        labels = [str(self.classMapping.get(int(c), int(c))) for c in classes]

        fig = ff.create_dendrogram(
            centroids,
            orientation="left",
            labels=labels,
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

        return self._returnPlotlyFigure(
            fig, save_dir or self.save_dir, "dendrogram.svg", show)

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
        if self._has_gt and self.gtColumn in self.data_df.columns:
            adata.obs[self.gtColumn] = (
                self.data_df[self.gtColumn]
                .astype(str)
                .values
            )

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
        # Reuse an already-computed UMAP when available (avoids recomputation).
        # Fall back to a PAGA-initialised UMAP otherwise.
        if "UMAP x" in self.data_df.columns and "UMAP y" in self.data_df.columns:
            adata.obsm["X_umap"] = self.data_df[["UMAP x", "UMAP y"]].to_numpy()
        else:
            sc.tl.umap(adata, init_pos="paga", random_state=42)

        # ── 5. Diffusion map + pseudotime ─────────────────────────────────
        sc.tl.diffmap(adata, n_comps=n_dcs)

        # Determine root cell: cell in root_group with lowest DC1 value
        groups = adata.obs["leiden_trajectory"].values
        dc1 = adata.obsm["X_diffmap"][:, 1]
        unique_groups = np.unique(groups)

        if root_group is None:
            group_medians = {g: np.median(dc1[groups == g]) for g in unique_groups}
            root_group = str(min(group_medians, key=group_medians.get))
            print(f"[trajectory] Auto-selected root group: {root_group}")
        elif root_group not in unique_groups:
            # root_group may be a gtColumn value rather than a Leiden label —
            # find the Leiden cluster most enriched for that value.
            gt_col = self.gtColumn if self._has_gt else None
            leiden_resolved = None
            if gt_col is not None and gt_col in self.data_df.columns:
                gt_vals = self.data_df[gt_col].astype(str).values
                enrichment = {
                    g: (gt_vals[groups == g] == str(root_group)).mean()
                    for g in unique_groups
                }
                leiden_resolved = max(enrichment, key=enrichment.get)
                print(
                    f"[trajectory] '{root_group}' not a Leiden label; "
                    f"resolved to group '{leiden_resolved}' "
                    f"(enrichment {enrichment[leiden_resolved]:.2f})."
                )
            if leiden_resolved is None:
                print(
                    f"[trajectory] Warning: root_group='{root_group}' not found "
                    f"in Leiden labels {list(unique_groups)}. Falling back to auto."
                )
                group_medians = {g: np.median(dc1[groups == g]) for g in unique_groups}
                leiden_resolved = str(min(group_medians, key=group_medians.get))
            root_group = leiden_resolved

        root_mask = groups == root_group
        dc1_vals = adata.obsm["X_diffmap"][root_mask, 1]
        root_idx = int(np.where(root_mask)[0][np.argmin(dc1_vals)])
        adata.uns["iroot"] = root_idx
        sc.tl.dpt(adata, n_dcs=n_dcs)

        # ── 6. Write results back to data_df / results_df ─────────────────
        umap_coords = adata.obsm["X_umap"]
        self.data_df["leiden_trajectory"] = adata.obs["leiden_trajectory"].astype(str).to_numpy()
        self.data_df["dpt_pseudotime"] = adata.obs["dpt_pseudotime"].to_numpy()
        self.data_df["traj_umap_x"] = umap_coords[:, 0]
        self.data_df["traj_umap_y"] = umap_coords[:, 1]
        self.results_df["leiden_trajectory"] = adata.obs["leiden_trajectory"].astype(str).to_numpy()
        self.results_df["dpt_pseudotime"] = adata.obs["dpt_pseudotime"].to_numpy()
        self.results_df["traj_umap_x"] = umap_coords[:, 0]
        self.results_df["traj_umap_y"] = umap_coords[:, 1]

        # Store adata so plot_trajectory can reuse it without recomputing
        self._traj_adata = adata
        self._traj_color_by = color_by
        self._traj_root_group = root_group

        print(f"[trajectory] Root cell index: {root_idx}  (group '{root_group}')")
        print(f"[trajectory] Pseudotime range: "
              f"{adata.obs['dpt_pseudotime'].min():.3f} – "
              f"{adata.obs['dpt_pseudotime'].max():.3f}")

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

        ux = df["traj_umap_x"].values
        uy = df["traj_umap_y"].values
        groups = df["leiden_trajectory"].astype(str).values
        pseudotime = df["dpt_pseudotime"].values
        unique_groups = sorted(np.unique(groups))

        # Shared palette — tab20 avoids the grey-for-0 override in specialScatter
        palette = sns.color_palette("tab20", len(unique_groups))
        group_color = {g: f"rgb{tuple(int(c * 255) for c in palette[i])}"
                       for i, g in enumerate(unique_groups)}

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
        # Scatter points
        for g in unique_groups:
            mask = groups == g
            fig.add_trace(go.Scatter(
                x=ux[mask], y=uy[mask],
                mode="markers",
                name=f"Group {g}",
                marker=dict(size=4, color=group_color[g], opacity=0.6),
                legendgroup=f"group_{g}",
                hovertemplate=f"Group {g}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

        # PAGA edges superimposed
        conn = np.array(adata.uns["paga"]["connectivities"].todense())
        node_x = [ux[groups == g].mean() for g in unique_groups]
        node_y = [uy[groups == g].mean() for g in unique_groups]

        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
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
            gt_vals = df[color_by].astype(str).values
            unique_gt = sorted(np.unique(gt_vals))
            gt_palette = sns.color_palette("Set2", len(unique_gt))
            gt_color = {v: f"rgb{tuple(int(c * 255) for c in gt_palette[i])}"
                        for i, v in enumerate(unique_gt)}
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
        for g in unique_groups:
            mask = groups == g
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
        for row in (1, 2):
            for col in (1, 2):
                fig.update_xaxes(showgrid=False, showticklabels=False, row=row, col=col)
                fig.update_yaxes(showgrid=False, showticklabels=False, row=row, col=col)

        return self._returnPlotlyFigure(
            fig, save_dir or self.save_dir, "trajectory_plotly.html", show)

    def getRowsByLabel(self, label):
        return self.data_df[self.data_df[MOBIE_LABEL_KEY] == label]

    def get_array_from_df(self, column):
        """Deprecated alias for :func:`dl_utils.util.getArrayFromDF`."""
        return getArrayFromDF(self.data_df, column)

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
                data_url = _png_buffer_to_data_url(buffered)
                return f'<img src="{data_url}" width="200" height="200">'
            except Exception as e:
                return f"Error creating image: {e}"
        return "No image available"
