#!/usr/bin/env python

"""General utilities to help with plotting.

   Author: Dominic Phillips (dominicp6)
"""

import warnings
from math import ceil, floor
from typing import Callable, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.signal import find_peaks
import numpy.typing as npt
import networkx as nx
from scipy.interpolate import SmoothBivariateSpline
from sklearn.cluster import DBSCAN

from utils.diffusion_utils import (
    free_energy_estimate,
    project_points_to_line,
    free_energy_estimate_2D,
)
# from MarkovStateModel import MSM

type_kramers_rates = list[tuple[tuple[int, int], float]]


def find_turning_points_interpolation(x, y, z, gradient_threshold=0.15, clustering_eps=0.05, show_plots=False, ax = None, interpolate_cluster_values=False):
    """
    Find the turning points of a 2D surface using interpolation.
    """
    # Interpolate the z surface with a 2D interpolation function
    interp_func = SmoothBivariateSpline(x, y, z, kx=3, ky=3)

    # Create a grid of points to evaluate the interpolation function
    X,Y = np.meshgrid(np.linspace(x.min(),x.max(),2500),np.linspace(y.min(),y.max(),2500))
    X = X.flatten()
    Y = Y.flatten()

    # Calculate the gradients of the z surface using the interpolation function
    fx = interp_func(X, Y, dx=1, dy=0, grid=False)
    fy = interp_func(X, Y, dx=0, dy=1, grid=False)

    if show_plots:
        plt.plot(fx**2 + fy**2)
        plt.yscale('log')
        plt.show()

    # Find candidate turning points by finding the points where the gradient is below a threshold
    if show_plots:
        num_candidates = []
        for threshold in np.linspace(0.01, 0.25, 100):
            candidates = np.where(np.sqrt(fx**2 + fy**2) < threshold)[0]
            num_candidates.append(len(candidates))
        plt.plot(np.linspace(0.01, 0.25, 100), num_candidates)
        plt.show()

    candidates = np.where(np.sqrt(fx**2 + fy**2) < gradient_threshold)[0]

    # Cluster the candidate turning points to remove any that are too close to each other
    candidate_turning_points = np.array([[X[i], Y[i]] for i in candidates])
    clustering = DBSCAN(eps=clustering_eps, min_samples=1).fit(candidate_turning_points)
    labels = clustering.labels_

    turning_points = []
    for label in np.unique(labels):
        # Skip any points that are not part of a cluster
        if label == -1:
            print("[Notice] Isolated point found. Skipping...")
        # Find the points in the cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_points = candidate_turning_points[cluster_indices]

        if interpolate_cluster_values:
            # Interpolate the z value at each point in the cluster
            cluster_values = interp_func(cluster_points[:, 0], cluster_points[:, 1], grid=False)
        else:
            cluster_values = []
            for point in cluster_points:
                # Find the closest (x,y) coordinate for each point in the cluster
                closest_point_idx = np.argmin(np.linalg.norm(np.array([x, y]).T - point, axis=1))
                # Find the z value at the closest (x,y) coordinate
                z_value = z[closest_point_idx]
                cluster_values.append(z_value)
            cluster_values = np.array(cluster_values)

        mean_point = np.mean(cluster_points, axis=0)
        # Compute the Hessian matrix of the z surface at the mean point
        fxx = interp_func(mean_point[0], mean_point[1], dx=2, dy=0)
        fyy = interp_func(mean_point[0], mean_point[1], dx=0, dy=2)
        fxy = interp_func(mean_point[0], mean_point[1], dx=1, dy=1)
        hessian = np.array([[fxx, fxy], [fxy, fyy]])
        eigenvalues = np.linalg.eigvals(hessian)
        # If the Hessian matrix has all negative eigenvalues, the point is a maximum
        if np.all(eigenvalues < 0):
            # Choose the point with the highest z value
            turning_points.append((cluster_points[np.argmax(cluster_values)], max(cluster_values), 'maximum'))
        # If the Hessian matrix has all positive eigenvalues, the point is a minimum
        elif np.all(eigenvalues > 0):
            # Choose the point with the lowest z value
            turning_points.append((cluster_points[np.argmin(cluster_values)], min(cluster_values), 'minimum'))
        else:
            turning_points.append((mean_point, interp_func(mean_point[0], mean_point[1], grid=False), 'saddle'))

    # Sort the turning points by their corresponding values
    turning_points.sort(key=lambda x: x[1])

    # Plot the turning points
    max_coords = np.array([x for x, _, _ in turning_points if _ == 'maximum'])
    min_coords = np.array([x for x, _, _ in turning_points if _ == 'minimum'])
    saddle_coords = np.array([x for x, _, _ in turning_points if _ == 'saddle'])
    if ax is not None:
        ax.scatter(max_coords[:, 0], max_coords[:, 1], c='r', label='maximum', zorder=2)
        ax.scatter(min_coords[:, 0], min_coords[:, 1], c='b', label='minimum', zorder=2)
        ax.scatter(saddle_coords[:, 0], saddle_coords[:, 1], c='g', label='saddle', zorder=2)
        ax.legend()
    if show_plots:
            plt.scatter(max_coords[:, 0], max_coords[:, 1], c='r', label='maximum')
            plt.scatter(min_coords[:, 0], min_coords[:, 1], c='b', label='minimum')
            plt.scatter(saddle_coords[:, 0], saddle_coords[:, 1], c='g', label='saddle')
            plt.legend()
            plt.show()

    return turning_points, ax


# Source: SciPy
def voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    Examples
    --------
    Set of point:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> points = rng.random((10,2))

    Voronoi diagram of the points:

    >>> from scipy.spatial import Voronoi, voronoi_plot_2d
    >>> vor = Voronoi(points)

    using `voronoi_plot_2d` for visualisation:

    >>> fig = voronoi_plot_2d(vor)

    using `voronoi_plot_2d` for visualisation with enhancements:

    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                 line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()

    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:,0], vor.points[:,1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))

    return ax.figure


def get_digit_text_width(fig, axis) -> float:
    r = fig.canvas.get_renderer()
    t = axis.text(0.5, 0.5, "1")

    bb = t.get_window_extent(renderer=r).transformed(axis.transData.inverted())

    t.remove()

    return bb.width / 2


def display_kramers_rates(kramers_rates: type_kramers_rates) -> None:
    print("Kramer's Rates")
    print("-" * 25)
    for rate in kramers_rates:
        initial_state = rate[0][0]
        final_state = rate[0][1]
        transition_rate = rate[1]
        print(
            rf"S{initial_state} --> S{final_state} : {'{:e}'.format(transition_rate)}"
        )
    print("-" * 25)


def get_minima(
    data_array: np.array, prominence: float, include_endpoint_minima: bool, ignore_high_minima: bool
) -> npt.NDArray[np.int]:
    number_of_minima = 0
    current_prominence = prominence
    minima = None

    while number_of_minima < 2:
        minima = find_peaks(-data_array, prominence=current_prominence)[0]
        energy = [data_array[index] for index in minima]

        if ignore_high_minima:
            minima = [
                point
                for idx, point in enumerate(minima)
                if not np.abs(energy[idx])
                > min(data_array) + 0.8 * (max(data_array) - min(data_array))
            ]

        number_of_minima = len(minima)
        current_prominence *= 0.975

    if current_prominence != prominence * 0.975:
        warnings.warn(
            f"Automatically reduced prominence from {prominence} "
            f"to {round(current_prominence / 0.975, 3)} so as to find at least two minima."
        )

    if include_endpoint_minima:
        if data_array[0] < data_array[1]:
            minima = np.insert(minima, 0, 0)
        if data_array[-1] < data_array[-2]:
            minima = np.append(minima, len(data_array) - 1)

    return minima


def plot_minima(minima_list: npt.NDArray[np.int], y_variable: np.array) -> None:
    for idx, minima in enumerate(minima_list):
        print("Minima ", (round(minima[0], 3), round(minima[1], 3)))
        plt.text(
            minima[0],
            minima[1] - 0.075 * (max(y_variable) - min(y_variable)),
            f"S{idx}",
            fontsize=16,
            color="b",
        )

# TODO: np, move into MSM file
# def display_state_boundaries(msm: MSM, y_coordinate: list[float]) -> list[float]:
#     voronoi_cell_boundaries = [
#         (msm.sorted_state_centers[i + 1] + msm.sorted_state_centers[i]) / 2
#         for i in range(len(msm.sorted_state_centers) - 1)
#     ]
#     for boundary in voronoi_cell_boundaries:
#         plt.vlines(
#             boundary,
#             ymin=min(y_coordinate) - 0.2 * (max(y_coordinate) - min(y_coordinate)),
#             ymax=min(y_coordinate) - 0.1 * (max(y_coordinate) - min(y_coordinate)),
#             linestyle="--",
#             color="k",
#         )
#
#     return voronoi_cell_boundaries


def display_state_numbers(boundaries: np.array, x_variable: np.array, y_variable: np.array, digit_width: float) -> None:
    x_min = min(x_variable)
    y_min = min(y_variable)
    y_range = max(y_variable) - min(y_variable)

    def index_label_width(x):
        return digit_width * ceil(np.log10(x + 1))

    for state_index in range(len(boundaries) + 1):
        if state_index == 0:
            plt.text(
                (boundaries[state_index] + x_min) / 2 - index_label_width(state_index),
                y_min - 0.175 * y_range,
                f"{state_index}",
                fontsize=12,
                color="k",
            )
        elif state_index == len(boundaries):
            plt.text(
                (x_min + boundaries[state_index - 1]) / 2
                - index_label_width(state_index),
                y_min - 0.175 * y_range,
                f"{state_index}",
                fontsize=12,
                color="k",
            )
        else:
            plt.text(
                (boundaries[state_index] + boundaries[state_index - 1]) / 2
                - index_label_width(state_index),
                y_min - 0.175 * y_range,
                f"{state_index}",
                fontsize=12,
                color="k",
            )


def plot_free_energy_estimate(potential: Callable, samples: np.array, beta: float, name: str, minimum_counts: int = 50):
    estimated_free_energy, coordinates = free_energy_estimate(
        samples, beta, minimum_counts
    )
    linear_shift = estimated_free_energy[
        floor(len(estimated_free_energy) / 2)
    ] - potential(0)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(coordinates, estimated_free_energy - linear_shift, "k", label="estimated")
    plt.xlabel("Q", fontsize=18)
    plt.ylabel(r"$\mathcal{F}$", fontsize=18)
    x_range = np.arange(
        min(coordinates), max(coordinates), (max(coordinates) - min(coordinates)) / 1000
    )
    plt.plot(x_range, potential(x_range), label="actual")
    plt.legend()
    # plt.title('Free Energy Surface', fontsize=16)
    plt.savefig(name)


def plot_free_energy_slice(samples: np.array, beta: float, slice_centre: list, slice_angle: float, minimum_counts: int = 50) -> list:
    concatenated_samples = np.concatenate(samples)
    projected_samples = project_points_to_line(
        concatenated_samples, np.array(slice_centre), slice_angle
    )
    free_energy, coordinates = free_energy_estimate(
        projected_samples, beta, minimum_counts
    )
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(18, 5)
    axs[0].hist2d(concatenated_samples[:, 0], concatenated_samples[:, 1], bins=300)
    axs[0].plot(slice_centre[0], slice_centre[1], "rx", markersize=12)
    max_x = max(concatenated_samples[:, 0])
    min_x = min(concatenated_samples[:, 0])
    x_range = np.arange(min_x, max_x, (max_x - min_x) / 1000)
    m = np.tan(slice_angle)
    c = slice_centre[1] - m * slice_centre[0]
    y_range = m * x_range + c
    axs[0].plot(x_range, y_range, "r")
    axs[1].hist(projected_samples, bins=100)
    axs[2].plot(coordinates, free_energy)
    plt.show()

    return projected_samples


def plot_free_energy_surface(samples: np.array, beta: float, bins: int = 300) -> None:
    concatenated_samples = np.concatenate(samples)
    free_energy, fig, axs, xedges, yedges = free_energy_estimate_2D(
        samples, beta, bins=bins
    )
    fig.set_size_inches(9, 7)
    clb = axs.contourf(xedges[1:], yedges[1:], free_energy.T)
    plt.colorbar(clb, label="Free Energy", ax=axs)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.show()


def hamiltonian(Q: float, P: float, M: float, U: Callable) -> float:
    return -((P**2) / (2 * M) + U(Q))


def phase_plot(Q: np.array, P: np.array, M: float, U: Callable) -> None:
    min_Q = min(Q)
    max_Q = max(Q)
    range_Q = max_Q - min_Q
    min_P = min(P)
    max_P = max(P)
    range_P = max_P - min_P
    Q_range = np.linspace(min_Q - 0.05 * range_Q, max_Q + 0.05 * range_Q, 100)
    P_range = np.linspace(min_P - 0.05 * range_P, max_P + 0.05 * range_P, 100)
    Q_mesh, P_mesh = np.meshgrid(Q_range, P_range)
    plt.pcolormesh(Q_range, P_range, hamiltonian(Q_mesh, P_mesh, M, U))
    plt.contour(Q_range, P_range, hamiltonian(Q_mesh, P_mesh), levels=15)
    plt.xlabel(r"$Q$", fontsize=20)
    plt.ylabel(r"$P$", fontsize=20)
    plt.plot(Q, P, "k", linewidth=0.7)
    plt.show()


def trajectory_plot(Q0: np.array, Q1: np.array, U: Callable) -> None:
    min_Q0 = min(Q0)
    max_Q0 = max(Q0)
    range_Q0 = max_Q0 - min_Q0
    min_Q1 = min(Q1)
    max_Q1 = max(Q1)
    range_Q1 = max_Q1 - min_Q1
    Q0_range = np.linspace(min_Q0 - 0.05 * range_Q0, max_Q0 + 0.05 * range_Q0, 100)
    Q1_range = np.linspace(min_Q1 - 0.05 * range_Q1, max_Q1 + 0.05 * range_Q1, 100)
    Q0_mesh, Q1_mesh = np.meshgrid(Q0_range, Q1_range)
    plt.pcolormesh(Q0_range, Q1_range, U([Q0_mesh, Q1_mesh]))
    plt.contour(Q0_range, Q1_range, U([Q0_mesh, Q1_mesh]), levels=15)
    plt.xlabel(r"$Q_0$", fontsize=20)
    plt.ylabel(r"$Q_1$", fontsize=20)
    plt.plot(Q0, Q1, "k", linewidth=0.7)
    plt.show()


def potential_contour_plot(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    U: Callable,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    set_min_to_zero: bool = True,
    save: bool = True,
    save_name: str = "test",
) -> None:
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    axs.set_aspect("equal")
    x_range = np.linspace(x_min, x_max, 120)
    y_range = np.linspace(y_min, y_max, 120)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    if set_min_to_zero:
        U_min = np.min(U([x_mesh, y_mesh]))
    else:
        U_min = 0
    im = axs.pcolormesh(x_range, y_range, U([x_mesh, y_mesh]) - U_min)
    # axs.contour(x_range, y_range, U([x_mesh, y_mesh]), levels=15)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.20)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label="Free Energy", fontsize=20)
    axs.xaxis.set_ticks([-2, -1, 0, 1, 2])
    axs.yaxis.set_ticks([-2, -1, 0, 1, 2])
    axs.tick_params(axis="x", labelsize=16)
    axs.tick_params(axis="y", labelsize=16)
    if vmin is not None and vmax is not None:
        im.set_clim(vmin, vmax)

    if save:
        plt.savefig(save_name, format="pdf", bbox_inches="tight")

    plt.show()


def my_draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=None,
        label_pos=0.5,
        font_size=10,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        rotate=True,
        clip_on=True,
        rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform chemicals coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
