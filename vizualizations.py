
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import seaborn as sns
from matplotlib.pyplot import cm
from sklearn.preprocessing import MinMaxScaler

from matplotlib.figure import Figure
import os
from ast import Tuple
from typing import List, Literal, Any, Union


class FeatureVizualizer:
    def __init__(self, features) -> None:
        self.features: Any = features
# TODO: Continue here

    def PCA(self) -> None:
        pass




class costumMatplotlib:
    def __init__(self, style_set_context="paper") -> None:

        # self.figsize = (12,6)
        self.style_set_context = style_set_context
        

    @property
    def colorDataFormat() -> Tuple:
        return Tuple

    @staticmethod
    def array2colors(x: np.ndarray):
        """Argument for color in sns scatter is color."""
        x = MinMaxScaler(feature_range=(0,1), clip=True).fit_transform(x)
        assert x.shape[-1] == 3, f"Axis y must be 3"
        colors = []
        for i in x:
            colors.append(tuple(i) + (1,))
        return colors

    @staticmethod
    def labels2colors(
        labels: list[int], alpha=1, colormap="tab20"
    ) -> Union[list[tuple[float]], None]:

        labels = np.array(labels)

        # sns.color_palette(palette="tab20",n_colors=np.unique(labels).size)
        assert (
            labels.ndim == 1
        ), f"Labels must be a 1D array but it is {labels.ndim} with shape {labels.shape}"
        mapper = cm.ScalarMappable(cmap=colormap)
        d_colors = mapper.to_rgba(np.unique(labels))  # Initialize the mapper
        colors = []
        for label in labels:
            i = np.where(np.unique(labels) == label)[0][0]
            r, g, b, a = d_colors[i]
            colors.append((r, g, b, alpha))
        return colors

    @staticmethod
    def points2Dict(points):
        return {
            ["x", "y", "z"][i]: x.flatten()
            for i, x in enumerate(np.vsplit(points.T, points.shape[-1]))
        }

    @classmethod
    def saveFig(cls, fig, save_dir, title, func, fileExtension="svg") -> None:
        plt.close()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path: str = os.path.join(save_dir, f"{title}-{func.__name__}")
            fig.savefig(f"{save_path}.{fileExtension}", dpi=500)

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

    @classmethod
    def simpleScatter(
        cls,
        points: dict[str, np.ndarray],
        labels: Union[List[int],
                      np.ndarray[Literal["1"], np.dtype[np.int32]]] = None,
        title: str = "",
        style_set_context="notebook",
        save_dir=None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        # plot = cls(style_set_context)
        # fig, ax = plt.subplots(1,1, figsize=plot.figsize)\
        sns.set_context("paper", font_scale=0.5)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if labels is not None:
            ax: Axes = costumMatplotlib.subScatter(
                ax=ax,
                color=costumMatplotlib.labels2colors(labels),
                points=points,
                **kwargs,
            )
        else:
            ax: Axes = costumMatplotlib.subScatter(
                ax=ax, points=points, **kwargs)

        ax.set_title(title)

        fig.tight_layout()

        cls.saveFig(fig, save_dir=save_dir,
                    title=title, func=cls.simpleScatter)
        
        plt.close(fig)
        
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
    ) -> tuple[Figure, Axes]:
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
        fig.tight_layout()
        cls.saveFig(fig, save_dir=save_dir, title=title, func=cls.simpleHeatmap)
        return fig, ax

    @classmethod
    def simpleImshow(
        cls,
        matrix: np.ndarray,
        title: str = "",
        style_set_context="notebook",
        save_dir=None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        # plot = cls(style_set_context)
        # fig, ax = plt.subplots(1,1, figsize=plot.figsize)
        fig, ax = plt.subplots(1, 1)

        costumMatplotlib.subImshow(ax, matrix, title)

        fig.tight_layout()

        cls.saveFig(fig, save_dir=save_dir,
                    title=title, func=cls.simpleImshow, fileExtension="png")
        
        

        return fig, ax

    @staticmethod
    def subImshow(ax, img, label, **kwargs):
        ax.imshow(img, cmap="magma", **kwargs)
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
    def  subScatter(ax, points: dict[str, np.ndarray], **kwargs) -> Axes:
        # For KWARGS, : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes
        
        # Ensure ticks are small
        #ax.tick_params(axis='both', which='both', labelsize=3, )
        # Set axis edges and ticks to verz small
#

        return sns.scatterplot(ax=ax, **points, **kwargs)

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
                costumMatplotlib.subScatter(
                    ax,
                    data[img_index],
                    label=labels[img_index] if labels is not None else None,
                    **subplots_kwargs[img_index]
                    if subplots_kwargs is not None and isinstance(subplots_kwargs, list)
                    else subplots_kwargs,
                )

        # file_name = f"label_{labels[0]}"
        # if kwargs["extra_info"] is not None:
        #     grouping = kwargs["extra_info"]["grouping"]
        #     if kwargs["extra_info"]["color_group"] is not None:
        #         color_group : tuple[float, float, float, float] = kwargs["extra_info"]["color_group"]
        #         # Highlight the figure border
        #         fig.patch.set_edgecolor(color_group)  # Blue figure border
        #         fig.patch.set_linewidth(50)  # Thicker border
        #         file_name = f"grouping_{grouping}_{file_name}"
        # Adjust layout and save the plot

        fig.tight_layout()

        # if save_path is None:

        # assert save_dir is not None, f"Either save path or save directory must be given. Both are currently: {save_path}, {save_dir}"

        # save_path = os.path.join(save_dir, f"NN_E_PCA_plot_{file_name}.png")

        # mkdir(os.path.dirname(save_dir), os.path.basename(save_dir))
        costumMatplotlib.saveFig(
            fig,
            save_dir=save_dir,
            title=title,
            func=costumMatplotlib.square_multi_subplot,
        )

        return fig
