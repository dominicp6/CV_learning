import os
import pickle

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


def init_plot(
    title: str,
    xlabel: str,
    ylabel: str,
    figsize=(6, 4),
    xscale="linear",
    yscale="linear",
    use_grid: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel)
    ax.grid(use_grid)
    fig.suptitle(title)

    return fig, ax


def init_subplot(
    nrows: int,
    ncols: int,
    title: str,
    xlabel: str,
    ylabel: str,
    width: float = 6,
    height: float = 4,
    scale: str = "auto",
    sharex: bool = True,
    sharey: bool = True,
    grid: bool = False,
):
    # original figure width & height before scaling
    fig_width = width * ncols
    fig_height = height * nrows

    if scale == "auto":
        scale = min(1.0, 10.0 / fig_width)  # width of figure cannot exceede 10 inches
        scale = min(
            scale, 10.0 / fig_height
        )  # height of figure cannot exceede 10.0 inches

    figsize = (scale * fig_width, scale * fig_height)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)

    for ax in axs.flat:
        ax.grid(grid)  # maybe add grid lines

    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()  # adjust the padding between and around subplots to neaten things up

    return fig, axs


def save_fig(
    fig,
    save_dir: str,
    name: str,
    both_formats: bool = True,
    close: bool = True,
    save_data: bool = True,
    show_fig: bool = True,
):
    if show_fig:
        plt.show()
    if save_data:
        data = locals()
        with open(os.path.join(save_dir, f"{name}.pickle"), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(save_dir, f"{name}.pdf")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    if both_formats:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches="tight", dpi=300)
    if close:
        plt.close(fig)  # otherwise figure may hang around in memory