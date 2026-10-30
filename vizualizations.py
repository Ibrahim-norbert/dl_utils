import os
from typing import List, Literal, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from sklearn.preprocessing import MinMaxScaler

from MoBie_coloring import GlasbeyARGBLut


class CustomMatplotlib:
    def __init__(self, style_set_context="paper") -> None:
        self.style_set_context = style_set_context

    @staticmethod
    def array2colors(x: np.ndarray):
        """Map an (N, 3) array to a list of opaque RGBA tuples for seaborn."""
        x = MinMaxScaler(feature_range=(0, 1), clip=True).fit_transform(x)
        assert x.shape[-1] == 3, "Input must have 3 columns (RGB)"
        return [(*row, 1) for row in x]

    @staticmethod
    def labels2colors(
        labels: list[int], alpha=1, colormap="tab20"
    ) -> Union[list[tuple[float]], None]:

        labels = np.asarray(labels)
        assert (
            labels.ndim == 1
        ), f"Labels must be a 1D array but it is {labels.ndim} with shape {labels.shape}"

        unique = np.unique(labels)
        mapper = cm.ScalarMappable(cmap=colormap)
        d_colors = mapper.to_rgba(unique)
        # Precompute label -> color once, then look up per label (O(n) instead of O(n^2)).
        label_to_color = {lbl: (r, g, b, alpha) for lbl, (r, g, b, _) in zip(unique, d_colors)}
        return [label_to_color[label] for label in labels]

    @staticmethod
    def points2Dict(points):
        ndim = points.shape[-1]
        axes = ["x", "y", "z"][:ndim]
        assert ndim <= 3, f"points2Dict supports up to 3 dimensions, got {ndim}"
        return {axes[i]: x.flatten() for i, x in enumerate(np.vsplit(points.T, ndim))}

    # Extensions that mark save_dir as a file path. A plain splitext test is not enough:
    # folder names like "RealData-Florian-30.10.2025" would read as having a ".2025" suffix.
    FIGURE_EXTENSIONS = (".svg", ".png", ".pdf", ".jpg", ".jpeg", ".webp", ".eps")

    @classmethod
    def resolve_save_path(cls, save_dir, default_name) -> str:
        """*save_dir* ending in a figure extension is the full output path; otherwise
        it is a folder and *default_name* is written inside it. Creates the folder."""
        path = (save_dir if os.path.splitext(save_dir)[1].lower() in cls.FIGURE_EXTENSIONS
                else os.path.join(save_dir, default_name))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return path

    @classmethod
    def saveFig(cls, fig, save_dir, title="", func=None, fileExtension="svg",
                filename: str = None, close: bool = True) -> None:
        """Write *fig* into *save_dir*, creating the directory if needed. A *save_dir*
        ending in a figure extension (e.g. ``.png``) is used as the full file path.

        *filename* overrides the ``{title}-{func.__name__}.{ext}`` convention for
        callers that already know the name they want; *close* can be turned off by
        callers that close the figure themselves (see
        ``EmbeddingAnalysis._returnFigure``), so a figure is never closed twice.
        Both default to the previous behaviour.
        """
        if save_dir is not None:
            name = filename if filename is not None else f"{title}-{func.__name__}.{fileExtension}"
            fig.savefig(cls.resolve_save_path(save_dir, name), dpi=500, bbox_inches="tight")  # keep outside legends
        if close:
            plt.close(fig)

        # TODO: Add functionality to save a subplot figure as seperate figures
        # fig.savefig(
        #     "/tmp/bottom.png",
        #     # we need a bounding box in inches
        #     bbox_inches=mtransforms.Bbox(
        #         # This is in "figure fraction" for the bottom half
        #         # input in [[xmin, ymin], [xmax, ymax]]
        #         [[0, 0], [1, 0.5]]
        #     ).transformed(
        #         # this take data from figure fraction -> inches
        #         #    transFigrue goes from figure fraction -> pixels
        #         #    dpi_scale_trans goes from inches -> pixels
        #         (fig.transFigure - fig.dpi_scale_trans)
        #     ),
        # )

    # initializing function

    @staticmethod
    def labels2glasbey(labels, color_space=GlasbeyARGBLut) -> list[tuple[float, float, float]]:
        """Integer class codes -> opaque RGB from the MoBIE LUT (NaN / codes <= 0
        are grey), so plots match the class colours shown in MoBIE."""
        codes = np.nan_to_num(np.asarray(labels, dtype=float)).astype(int)
        lut = color_space()
        code_to_color = {c: lut.rgba_plt_by_index(c)[:3] for c in np.unique(codes)}  # drop alpha: background's is ~0
        return [code_to_color[c] for c in codes]


    @classmethod
    def simpleScatter(
            cls,
            points: dict[str, np.ndarray],
            labels: Union[List[int],
            np.ndarray[Literal["1"], np.dtype[np.int32]]] = None,
            title: str = "",
            style_set_context="paper",
            save_dir=None,
            **kwargs,
    ) -> tuple[Figure, Axes]:
        import MoBie_coloring

        sns.set_context(style_set_context, font_scale=0.5)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Consumed here, never forwarded: seaborn/matplotlib don't know these.
        show_legend = kwargs.pop("showlegend", kwargs.pop("legend", False))
        palette = kwargs.pop("palette", None)
        color_space = kwargs.pop("color_space", None) or MoBie_coloring.GlasbeyARGBLut
        class_mapping = kwargs.pop("classMapping", None)
        if class_mapping:
            kwargs.setdefault("labelNames", class_mapping)  # subScatter renames the legend
        legend = show_legend if isinstance(show_legend, str) else ("auto" if show_legend else False)

        if labels is None:
            ax = CustomMatplotlib.subScatter(ax=ax, points=points, **kwargs)
        else:
            labels = np.asarray(labels)
            if isinstance(palette, dict):
                # labels are palette KEYS (e.g. class names from Analysis.specialScatter)
                ax = CustomMatplotlib.subScatter(
                    ax=ax, points=points, hue=labels, palette=palette,
                    hue_order=list(palette), legend=legend, **kwargs)
            elif labels.dtype.kind in "OUS":
                # no palette + strings -> already colours (e.g. '#e5e5ffff' from a heatmap)
                ax = CustomMatplotlib.subScatter(ax=ax, points=points, c=labels.tolist(), **kwargs)
            else:
                # integer labels -> the color_space LUT (Glasbey by default); 0 (background) is grey
                codes = np.nan_to_num(labels.astype(float)).astype(int)
                if palette is None:
                    lut = color_space() if isinstance(color_space, type) else color_space
                    palette = {k: lut.rgba_plt_by_index(int(k)) if k else (0.5, 0.5, 0.5, 1.0)
                               for k in np.unique(codes)}
                if show_legend or not isinstance(palette, dict):
                    ax = CustomMatplotlib.subScatter(
                        ax=ax, points=points, hue=codes, palette=palette, legend=legend, **kwargs)
                else:
                    ax = CustomMatplotlib.subScatter(
                        ax=ax, points=points, c=[palette[k] for k in codes], **kwargs)

        if ax.get_legend() is not None:
            sns.move_legend(ax, "center left", bbox_to_anchor=(1.02, 0.5))

        ax.set_title(title)

        fig.tight_layout()

        cls.saveFig(fig, save_dir=save_dir,
                    title=title, func=cls.simpleScatter)

        return fig, ax


    @classmethod
    def simpleHeatmap(
            cls,
            matrix: np.ndarray,
            xticklabels: list = None,
            yticklabels: list = None,
            title: str = "",
            save_dir: str = None,
            cmap: str = "viridis",
            fmt: str = ".4f",
            xtick_codes: list[int] = None,
            ytick_codes: list[int] = None,
            color_space=GlasbeyARGBLut,
    ) -> tuple[Figure, Axes]:
        """*xtick_codes* / *ytick_codes* are the integer class codes behind each
        tick (same order as the tick labels); each tick is coloured from the
        MoBIE *color_space* LUT so it matches the class colours in MoBIE
        (codes <= 0 are grey)."""
        n = matrix.shape[0]
        cell_size = max(1.2, min(2.0, 12 / n))
        fig_size = max(5, n * cell_size)
        fontsize = max(7, min(14, int(100 / n)))

        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        sns.heatmap(
            matrix, ax=ax, annot=True, fmt=fmt, cmap=cmap, square=True,
            xticklabels=xticklabels if xticklabels is not None else "auto",
            yticklabels=yticklabels if yticklabels is not None else "auto",
            annot_kws={"size": fontsize},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
        for ticks, codes in ((ax.get_xticklabels(), xtick_codes), (ax.get_yticklabels(), ytick_codes)):
            if codes is not None:
                for tick, colour in zip(ticks, cls.labels2glasbey(codes, color_space)):
                    tick.set_color(colour)
        fig.tight_layout()
        cls.saveFig(fig, save_dir=save_dir, title=title, func=cls.simpleHeatmap)
        return fig, ax

    @classmethod
    def simpleImshow(cls, matrix: np.ndarray, title: str = "", save_dir=None, **kwargs) -> tuple[Figure, Axes]:
        # plot = cls(style_set_context)
        # fig, ax = plt.subplots(1,1, figsize=plot.figsize)
        fig, ax = plt.subplots(1, 1)

        CustomMatplotlib.subImshow(ax, matrix, title, **kwargs)

        fig.tight_layout()

        cls.saveFig(fig, save_dir=save_dir,
                    title=title, func=cls.simpleImshow, fileExtension="png")

        return fig, ax

    @staticmethod
    def subImshow(ax, img : np.ndarray, label : str, **kwargs):
        ax.imshow(img, **kwargs)
        ax.grid(False)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)  # Ensure spine is visible

        if label is not None:
            ax.set_title(label)

        return ax

    @staticmethod
    def subScatter(ax, points: dict[str, np.ndarray], **kwargs) -> Axes:
        # For KWARGS, : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes
        s = kwargs.pop("markerSize", kwargs.pop("s", 4))
        kwargs["s"] = s

        legend = kwargs.pop("showlegend", kwargs.pop("legend", "auto"))
        kwargs["legend"] = legend

        # {label: name}; seaborn has no such argument, so relabel the legend afterwards
        labelNames = kwargs.pop("labelNames", None)

        ax = sns.scatterplot(ax=ax, **points, **kwargs)

        if labelNames and ax.get_legend() is not None:
            names = {str(k): str(v) for k, v in labelNames.items()}
            for text in ax.get_legend().get_texts():
                text.set_text(names.get(text.get_text(), text.get_text()))

        return ax

    @staticmethod
    def square_multi_subplot(
        data: list[np.ndarray],
        title: str = "",
        labels: Union[List[int], List[np.ndarray], None] = None,
        subplot_titles: Union[List[str], None] = None,
        save_dir=None,
        subplots_kwargs: Union[dict, list[dict], None] = None,
    ) -> plt.Figure:
        """# Parameters for plotting
        images_per_row = 6  # Number of images per row
        rows_per_plot = 2  # Number of rows in each plot

        Note:
        If you add the follwoing to the class then you cannot call a afunction with initializing using __getattrribue__: def __init__(self, style_set_context="notebook").
        Fix the call method by subplot:str functionality
        """
        # TODO: Fix the call method by subplot:str functionality

        if len(data) > 2:
            square_root = np.sqrt(len(data))
            # Confirm if integer
            if square_root * square_root == len(data):
                images_per_row = int(square_root)
                rows_per_plot = int(square_root)
            else:
                raise NotImplementedError(
                    "Currently only square number of images supported."
                )
        else:

            if len(data) == 2:
                images_per_row = 2
                rows_per_plot = 1
            else:
                raise NotImplementedError("Must plot at least 2 images.")

        fig_size: tuple[int, int] = (
            5 * images_per_row, 5 * rows_per_plot)  # x,y
        # Iterate through groups of images and save each as a separate plot

        fig, axes = plt.subplots(
            rows_per_plot, images_per_row, figsize=fig_size)
        # fig.patch.set_facecolor('black')  # Set figure background

        # Plot images in the current group
        for i in range(rows_per_plot):
            for j in range(images_per_row):
                if rows_per_plot == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                img_index: int = i * images_per_row + j

                if isinstance(subplot_titles, (list, tuple)):
                    subplot_title = (
                        subplot_titles[img_index]
                        if subplot_titles is not None
                        else None
                    )
                elif isinstance(subplot_titles, str):
                    subplot_title: str = f"{subplot_titles}-{img_index}"
                else:
                    subplot_title: str = ""

                ax.set_title(subplot_title)
                CustomMatplotlib.subScatter(
                    ax,
                    data[img_index],
                    label=labels[img_index] if labels is not None else None,
                    **subplots_kwargs[img_index]
                    if subplots_kwargs is not None and isinstance(subplots_kwargs, list)
                    else subplots_kwargs,
                )

        fig.tight_layout()

        # if save_path is None:

        # assert save_dir is not None, f"Either save path or save directory must be given. Both are currently: {save_path}, {save_dir}"

        # save_path = os.path.join(save_dir, f"NN_E_PCA_plot_{file_name}.png")

        # mkdir(os.path.dirname(save_dir), os.path.basename(save_dir))
        CustomMatplotlib.saveFig(
            fig,
            save_dir=save_dir,
            title=title,
            func=CustomMatplotlib.square_multi_subplot,
        )

        return fig

