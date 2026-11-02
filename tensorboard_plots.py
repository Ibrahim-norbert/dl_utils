import os
import re
from collections import Counter
from functools import cached_property
from typing import Iterable, Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorboard.backend.event_processing.event_accumulator import (
    SCALARS,
    EventAccumulator,
)

# importing analysis also registers the "plotly+scatter_axes" template used below
from analysis import _glasbey_rgb_palette
from vizualizations import CustomMatplotlib


class TensorBoardPlotter:
    """Plotly line plots of the scalars logged to TensorBoard event files.

    ``runs`` maps a display name to a run directory (the folder holding the
    ``events.out.tfevents.*`` files, e.g. a Lightning ``version_<n>`` folder). A
    list of directories uses each folder name as the display name.

    All scalars are read once into :attr:`df` with the columns
    ``run``, ``tag``, ``step``, ``value`` and ``wall_time``, sorted by tag, run and
    step. Each run keeps one colour across every figure (MoBIE Glasbey LUT, same as
    the other dl_utils plots). Every method works on one tag: :meth:`figure` returns
    its ``go.Figure`` and :meth:`save` writes one; :meth:`match_tags` finds tags.

    ``plotly_kwargs`` are instance-wide ``update_layout`` arguments applied to every
    figure (e.g. ``font_size=18``, ``width=800``); a per-call ``plotly_kwargs`` of
    :meth:`figure` is applied after them, so it wins on shared keys.
    """

    COLUMNS = ["run", "tag", "step", "value", "wall_time"]

    def __init__(
            self,
            runs: Union[Mapping[str, str], Iterable[str]],
            plotly_kwargs: Optional[dict] = None,
    ) -> None:
        if not isinstance(runs, Mapping):
            paths = list(runs)
            names = [os.path.basename(os.path.normpath(path)) for path in paths]
            duplicates = sorted(n for n, count in Counter(names).items() if count > 1)
            if duplicates:
                raise ValueError(
                    f"Run folders share the names {duplicates}; pass a {{name: path}} "
                    f"dict to tell them apart"
                )
            runs = dict(zip(names, paths))
        self.runs: dict[str, str] = dict(runs)
        self.plotly_kwargs: dict = plotly_kwargs if plotly_kwargs is not None else {}
        self.accumulators: dict[str, EventAccumulator] = {
            name: self.load(path) for name, path in self.runs.items()
        }
        self.df: pd.DataFrame = (
            pd.concat(
                [self.read_scalars(acc, name) for name, acc in self.accumulators.items()],
                ignore_index=True,
            )
            .astype({"run": "category", "tag": "category"})
            .sort_values(["tag", "run", "step"], ignore_index=True)
        )
        # one index of the rows per tag, so figure() never scans the whole frame
        self._by_tag = self.df.groupby("tag", observed=True, sort=False)
        # {run: "rgb(r, g, b)"}; a run name that parses as an integer uses that LUT code
        self.colors: dict[str, str] = _glasbey_rgb_palette(self.runs)

    @staticmethod
    def load(path: str) -> EventAccumulator:
        """The loaded event files directly inside the run directory ``path``."""
        # size_guidance 0 keeps every scalar instead of TensorBoard's reservoir sample
        accumulator = EventAccumulator(path, size_guidance={SCALARS: 0})
        accumulator.Reload()
        if not accumulator.Tags()[SCALARS]:
            # EventAccumulator reads only the files directly inside ``path``
            raise FileNotFoundError(
                f"No scalars in {path!r}: pass the folder that holds the "
                f"events.out.tfevents.* files (e.g. its version_<n> subfolder)"
            )
        return accumulator

    @classmethod
    def read_scalars(cls, accumulator: EventAccumulator, run: str) -> pd.DataFrame:
        """Every scalar event of ``accumulator`` as rows of :attr:`COLUMNS`, ``run``
        filling the run column."""
        # ScalarEvent is a (wall_time, step, value) namedtuple: its fields name the columns
        return (
            pd.concat(
                {
                    tag: pd.DataFrame(accumulator.Scalars(tag))
                    for tag in accumulator.Tags()[SCALARS]
                },
                names=["tag", None],
            )
            .reset_index("tag")
            .assign(run=run)[cls.COLUMNS]
        )

    def available_tags(self, run: Optional[str] = None) -> dict[str, dict[str, list[str]]]:
        """``{run: {tab: sorted tags}}`` of everything logged, grouped as TensorBoard's
        tabs (``"scalars"``, ``"images"``, ``"histograms"``, ``"distributions"``,
        ``"audio"``, ``"tensors"``); tabs without tags are left out. ``run`` restricts
        the result to that run.

        Logged figures (``add_figure``) appear under ``"images"`` and hyperparameters
        under ``"tensors"``. The graph entries of ``EventAccumulator.Tags()`` are
        flags, not tags, so they are not listed.
        """
        names = self.runs if run is None else [run]
        return {
            name: {
                tab: sorted(tags)
                for tab, tags in self.accumulators[name].Tags().items()
                if isinstance(tags, list) and tags
            }
            for name in names
        }

    @cached_property
    def tags(self) -> list[str]:
        """Scalar tags logged by at least one run, sorted."""
        return sorted(self.df["tag"].unique())

    def match_tags(self, pattern: str) -> list[str]:
        """Tags matching the regular expression ``pattern`` anywhere in the tag."""
        regex = re.compile(pattern)
        return [tag for tag in self.tags if regex.search(tag)]

    @staticmethod
    def smooth(values: np.ndarray, weight: float) -> np.ndarray:
        """TensorBoard's debiased exponential moving average.

        ``weight`` in [0, 1): 0 returns ``values`` unchanged. Non-finite values are
        passed through and do not enter the average, as in TensorBoard.

        The debiased average ``sum(w**i * x[t-i]) / sum(w**i)`` is exactly pandas'
        ``ewm(adjust=True)`` with ``alpha = 1 - w``; ``ignore_na=True`` keeps skipped
        values from advancing the weights.
        """
        if not 0 <= weight < 1:
            raise ValueError(f"weight must be in [0, 1), got {weight}")
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        smoothed = (
            pd.Series(np.where(finite, values, np.nan))
            .ewm(alpha=1 - weight, adjust=True, ignore_na=True)
            .mean()
            .to_numpy()
        )
        return np.where(finite, smoothed, values)

    def figure(
            self,
            tag: str,
            smoothing: float = 0.6,
            x: Literal["step", "wall_time"] = "step",
            log_y: bool = False,
            show_raw: bool = True,
            title: Optional[str] = None,
            yaxis_title: Optional[str] = None,
            plotly_kwargs: Optional[dict] = None,
            line_kwargs: Optional[dict] = None,
    ) -> go.Figure:
        """Line plot of ``tag`` with one line per run.

        With ``smoothing > 0`` the smoothed curve is drawn solid and, with
        ``show_raw``, the raw values as a faint line of the same colour.
        ``x="wall_time"`` plots hours since each run's first event of this tag.

        - ``plotly_kwargs``: per-call ``update_layout`` overrides, applied after the
          instance-wide :attr:`plotly_kwargs` (e.g. ``xaxis_range=[0, 5000]``,
          ``legend_orientation="h"``).
        - ``line_kwargs``: ``go.scatter.Line`` arguments for the smoothed lines
          (e.g. ``width=3``, ``dash="dot"``); the run colour is kept unless given.
        """
        try:
            data = self._by_tag.get_group(tag)
        except KeyError:
            raise KeyError(f"No run logged the tag {tag!r}; available: {self.tags}") from None

        traces = []
        # rows are already sorted by run and step (see __init__)
        for run, rows in data.groupby("run", observed=True, sort=False):
            xs = (
                rows["step"].to_numpy()
                if x == "step"
                else (rows["wall_time"] - rows["wall_time"].min()).to_numpy() / 3600
            )
            values = rows["value"].to_numpy()
            color = self.colors[run]
            if smoothing > 0 and show_raw:
                traces.append(
                    go.Scatter(
                        x=xs,
                        y=values,
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.25,
                        legendgroup=run,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            traces.append(
                go.Scatter(
                    x=xs,
                    y=self.smooth(values, smoothing),
                    mode="lines",
                    name=run,
                    line={"color": color, "width": 2, **(line_kwargs or {})},
                    legendgroup=run,
                )
            )

        # Separate calls on purpose: plotly merges nested dicts (e.g. xaxis=dict(...))
        # across update_layout calls, where one merged dict would replace them.
        return (
            go.Figure(data=traces)
            .update_layout(
                title_text=tag if title is None else title,
                xaxis_title="Step" if x == "step" else "Wall time [h]",
                yaxis_title=tag.split("/")[-1] if yaxis_title is None else yaxis_title,
                yaxis_type="log" if log_y else "linear",
                # axis lines on all four sides; the template sets their colour
                xaxis_showline=True,
                xaxis_mirror=True,
                yaxis_showline=True,
                yaxis_mirror=True,
                legend_title_text="Run",
                template="plotly+scatter_axes",
            )
            .update_layout(**self.plotly_kwargs)
            .update_layout(**(plotly_kwargs or {}))
        )

    @staticmethod
    def save(fig: go.Figure, tag: str, save_dir: str, extension: str = "png") -> str:
        """Write ``fig`` as ``<tag with "/" replaced by "_">.<extension>`` into the
        folder ``save_dir`` and return the path (static image formats need ``kaleido``)."""
        path = CustomMatplotlib.resolve_save_path(
            save_dir, f"{tag.replace('/', '_')}.{extension}"
        )
        write = fig.write_html if extension == "html" else fig.write_image
        write(path)
        return path
