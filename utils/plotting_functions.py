import os
import pickle
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 12
HUGE_SIZE = 14
MEGA_SIZE = 16

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("axes", titlesize=BIG_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIG_SIZE)  # fontsize of the x and y labels
plt.rc("figure", titlesize=MEGA_SIZE)  # fontsize of the figure title

plt.style.use('seaborn-paper')


def _set_fig_size(width: float, height: float, ncols: int, nrows: int, scale: str):
    # original figure width & height before scaling
    fig_width = width * ncols
    fig_height = height * nrows

    if scale == "auto":
        scale = min(1.0, 10.0 / fig_width)  # width of figure cannot exceede 10 inches
        scale = min(
            scale, 10.0 / fig_height
        )  # height of figure cannot exceede 10.0 inches

    figsize = (scale * fig_width, scale * fig_height)

    return figsize

def init_plot(
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    figsize=(6, 4),
    xscale="linear",
    yscale="linear",
    grid_on: bool = False,
    ax=None,
):
    if not ax:
        # If no axis provided, generate one (default)
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)
    else:
        fig = None
    ax.set(xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel)
    ax.grid(grid_on)

    return fig, ax


def init_subplot(
    nrows: int,
    ncols: int,
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    width: float = 6,
    height: float = 4,
    scale: str = "auto",
    sharex: bool = True,
    sharey: bool = True,
    grid_on: bool = False,
):
    figsize = _set_fig_size(width, height, ncols, nrows, scale)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)

    for ax in axs.flat:
        ax.grid(grid_on)  # maybe add grid lines

    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()  # adjust the padding between and around subplots to neaten things up

    return fig, axs


def init_multiplot(
        nrows: int,
        ncols: int,
        panels: list[str],
        title: Optional[str] = None,
        width: float = 6,
        height: float = 4,
        scale: str = "auto",
        grid_on: bool = False
):
    figsize = _set_fig_size(width, height, ncols, nrows, scale)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(nrows, ncols)

    panel_axs = []
    for panel in panels:
        slice_ = eval(f'np.s_[{panel}]')
        panel_axs.append(fig.add_subplot(grid[slice_]))

    for ax in panel_axs:
        ax.grid(grid_on)  # maybe add grid lines

    if title:
        fig.suptitle(title)
    fig.tight_layout()  # adjust the padding between and around subplots to neaten things up

    return fig, panel_axs


def save_fig(
    fig,
    save_dir: str,
    name: str,
    both_formats: bool = True,
    close: bool = True,
    save_data: bool = True,
    show_fig: bool = True,
):
    if save_data:
        data = locals()
        with open(os.path.join(save_dir, f"{name}.pickle"), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(save_dir, f"{name}.pdf")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    if both_formats:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches="tight", dpi=300)

    if show_fig:
        plt.show()

    if close:
        plt.close(fig)  # otherwise figure may hang around in memory
