#!/usr/bin/env python

"""Utility functions for diffusion coefficient sensitivity analysis and for computing Kramer's rate integrals.

   Author: Dominic Phillips (dominicp6)
"""

from typing import Callable, Union

import numpy as np
import pyemma
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from autograd import grad
import numpy.typing as npt
from pyemma.coordinates.clustering import RegularSpaceClustering, KmeansClustering

from MarkovStateModel import MSM
import general_utils as gutl

# TODO: axs type
def implied_timescale_analysis(
    discrete_traj: npt.NDArray[np.int], lags: npt.NDArray[np.int], axs
) -> None:
    its = pyemma.msm.its(
        discrete_traj, lags=lags, nits=10, reversible=True, connected=True
    )
    pyemma.plots.plot_implied_timescales(its, ylog=False, ax=axs[0, 0])
    axs[0, 0].set_title("Implied Timescale Analysis")


def diffusion_coefficient_sensitivity_analysis(
    cluster_centers: npt.NDArray[np.float64],
    discrete_traj: npt.NDArray[np.int],
    lags: npt.NDArray[np.int],
    time_step: float,
    axs,
) -> None:
    msm = MSM(cluster_centers)
    old_D = None
    diffs = []
    diffusion_coefficients = []
    for lag in lags:
        pyemma_msm = pyemma.msm.estimate_markov_model(dtrajs=discrete_traj, lag=lag)
        msm.set_stationary_distribution(pyemma_msm.stationary_distribution)
        msm.set_transition_matrix(pyemma_msm.transition_matrix)
        D = msm.compute_diffusion_coefficient(time_step, lag)
        diffusion_coefficients.append(D)
        if old_D is not None:
            diffs.append(gutl.vector_rmsd(old_D, D))
        old_D = D

    index_optimal_lag = np.argmin(diffs)
    optimal_lag = lags[index_optimal_lag]

    axs[1, 0].plot(lags[:-1], diffs, c="k")
    axs[1, 0].set_xlabel("Lag", fontsize=16)
    axs[1, 0].set_ylabel(r"RMSD $D(Q)$", fontsize=16)
    axs[1, 0].vlines(optimal_lag, min(diffs), max(diffs), color="r")
    for idx, diff_coeffs in enumerate(diffusion_coefficients):
        axs[0, 1].plot(
            msm.sorted_state_centers, diff_coeffs, label="lag=" + str(lags[idx])
        )
    axs[0, 1].legend()
    axs[0, 1].set_xlabel(r"$Q$", fontsize=16)
    axs[0, 1].set_ylabel(r"$D(Q)$", fontsize=16)
    # Perform Langevin dynamics check
    maxima = []
    for idx, diff_coeffs in enumerate(diffusion_coefficients):
        c4 = msm.calculate_correlation_coefficient(n=4)
        tau = lags[idx] * time_step
        D4 = (1 / (4 * 3 * 2 * tau)) * c4
        error_ratio = D4 / diff_coeffs**2
        # if min(error_ratio) < 0.25:
        axs[1, 1].plot(
            msm.sorted_state_centers,
            D4 / diff_coeffs**2,
            label="lag=" + str(lags[idx]),
        )
        maxima.append(max(D4 / diff_coeffs**2))
    axs[1, 1].set_yscale("log")
    axs[1, 1].hlines(
        0.25, min(msm.sorted_state_centers), max(msm.sorted_state_centers), colors="r"
    )
    axs[1, 1].set_title("Langevin Dynamics Check", fontsize=16)
    axs[1, 1].set_ylabel(r"$D^{(4)}(Q)/D^{(2)}(Q)^2$", fontsize=16)
    axs[1, 1].set_xlabel(r"$Q$", fontsize=16)
    # axs[1, 1].set_ylim(top=np.mean(maxima))
    axs[1, 1].legend()
    # plt.savefig("Kramers_free_energy_eval.pdf")
    plt.show()


def lag_sensitivity_analysis(
    discrete_traj: npt.NDArray[np.int],
    cluster_centers: npt.NDArray[np.float64],
    time_step: float,
) -> None:
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(16, 12)
    lags = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    implied_timescale_analysis(discrete_traj, lags, axs)
    diffusion_coefficient_sensitivity_analysis(
        cluster_centers, discrete_traj, lags, time_step, axs
    )


def cluster_time_series(
    time_series: np.array, cluster_type: str, options: dict
) -> Union[RegularSpaceClustering, KmeansClustering]:
    if cluster_type == "kmeans":
        cluster = pyemma.coordinates.cluster_kmeans(
            time_series,
            k=options["k"],
            stride=options["stride"],
            max_iter=options["max_iter"],
        )
    elif cluster_type == "reg_space":
        cluster = pyemma.coordinates.cluster_regspace(
            time_series, dmin=options["dmin"], max_centers=options["max_centers"]
        )
    else:
        raise ValueError('cluster_type must be either "kmeans" or "reg_space"')

    return cluster


# TODO: np
def compute_well_integrand(free_energy: np.array, beta: float) -> list[float]:
    return [np.exp(-beta * free_energy[x]) for x in range(len(free_energy))]


# TODO: typing evaluator object
def compute_barrier_integrand(
    evaluator_object, free_energy: np.array, beta: float
) -> list[float]:
    return [
        np.exp(beta * free_energy[x])
        / gutl.linear_interp_coordinate_data(
            evaluator_object.smoothed_D_domain,
            evaluator_object.smoothed_D,
            evaluator_object.coordinates[x],
        )
        for x in range(len(free_energy))
    ]


# TODO: numpy
def compute_well_integrand_from_potential(
    potential: Callable, beta: float, x_range: list[float]
) -> list[float]:
    return [np.exp(-beta * potential(x)) for x in x_range]


def compute_barrier_integrand_from_potential(
    potential: Callable, beta: float, diffusion_function: Callable, x_range: list[float]
) -> list[float]:
    return [np.exp(beta * potential(x)) / diffusion_function(x) for x in x_range]


def compute_well_and_barrier_integrals(
    initial_x: float,
    final_x: float,
    mid_x: float,
    well_integrand: list[float],
    barrier_integrand: list[float],
    x_coords: list[float],
) -> (np.array, np.array):
    if final_x > initial_x:
        well_integral = integrate.simpson(
            well_integrand[initial_x : mid_x + 1], x_coords[initial_x : mid_x + 1]
        )
        barrier_integral = integrate.simpson(
            barrier_integrand[initial_x + 1 : final_x],
            x_coords[initial_x + 1 : final_x],
        )
    else:
        well_integral = integrate.simpson(
            well_integrand[mid_x : initial_x + 1], x_coords[mid_x : initial_x + 1]
        )
        barrier_integral = integrate.simpson(
            barrier_integrand[final_x + 1 : initial_x],
            x_coords[final_x + 1 : initial_x],
        )

    return well_integral, barrier_integral


def compute_kramers_rate(
    transition,
    minima: list[int],
    well_integrand: list[float],
    barrier_integrand: list[float],
    x_coords: list[float],
) -> float:
    initial_x = minima[transition[0]]
    final_x = minima[transition[1]]
    mid_x = int(np.floor((initial_x + final_x) / 2))
    well_integral, barrier_integral = compute_well_and_barrier_integrals(
        initial_x, final_x, mid_x, well_integrand, barrier_integrand, x_coords
    )
    kramers_rate = (barrier_integral * well_integral) ** (-1)

    return kramers_rate


def compute_kramers_rate_from_potential(
    potential: Callable,
    diffusion_function: Callable,
    beta: float,
    start_position: int,
    end_position: int,
) -> float:
    x_range = np.arange(
        start_position, end_position, (end_position - start_position) / 5000
    )
    well_integrand = compute_well_integrand_from_potential(potential, beta, x_range)
    barrier_integrand = compute_barrier_integrand_from_potential(
        potential, beta, diffusion_function, x_range
    )
    well_integral, barrier_integral = compute_well_and_barrier_integrals(
        initial_x=0,
        final_x=len(x_range),
        mid_x=int(np.floor(len(x_range) / 2)),
        well_integrand=well_integrand,
        barrier_integrand=barrier_integrand,
        x_coords=x_range,
    )
    kramers_rate = (barrier_integral * well_integral) ** (-1)

    return kramers_rate


# Chapman-Kolmogorov Test
# if self.verbose:
# print('MSM Chapman-Kolmogorov Test')
# ck_test = msm.cktest(min(msm.nstates, 4))
# pyemma.plots.plot_cktest(ck_test)
# plt.show()

# def _compute_counts_matrix(self, traj, lag):
#     num_of_states = self.msm.number_of_states
#     counts = np.zeros((num_of_states, num_of_states))
#     for idx, state in enumerate(traj[:-lag]):
#         counts[state, traj[idx+lag]] += 1
#
#     return counts


def free_energy_estimate(
    samples: np.array, beta: float, minimum_counts: int = 50
) -> (np.array, np.array):
    # histogram
    counts, coordinate = np.histogram(samples, bins=200)
    robust_counts = counts[np.where(counts > minimum_counts)]
    robust_coordinates = coordinate[np.where(counts > minimum_counts)]

    # log normal
    normalised_counts = robust_counts / np.sum(counts)
    with np.errstate(divide="ignore"):
        free_energy = -(1 / beta) * np.log(normalised_counts)

    return free_energy, robust_coordinates


def free_energy_estimate_2D(samples: np.array, beta: float, bins: int = 300):
    fig, axs = plt.subplots(1, 1)
    h, xedges, yedges, quadmesh = axs.hist2d(samples[:, 0], samples[:, 1], bins=bins)
    total_counts = np.sum(h)
    with np.errstate(divide="ignore"):
        # TODO: fix
        free_energy = -(1 / beta) * np.log(h / total_counts + 0.000000000001)
        free_energy = np.nan_to_num(free_energy, nan=0)

    return free_energy - np.min(free_energy), fig, axs, xedges, yedges


def project_points_to_line(
    points: np.array, coords: np.array, theta: float
) -> np.array:
    # coords = (x0, y0), a point that the line goes through
    # theta is the orientation of the line (e.g. theta = 0 is parallel to the x axis, theta = pi/2 is parallel to the y axis)
    a = coords
    b = coords + np.array([np.cos(theta), np.sin(theta)])
    ap = points - a
    ab = b - a
    projected_points = np.dot(ap, ab) / np.dot(ab, ab)

    return projected_points


def relabel_trajectory_by_coordinate_chronology(
    traj: np.array, state_centers: np.array
) -> np.array:
    sorted_indices = np.argsort(np.argsort(state_centers))

    # relabel states in trajectory
    for idx, state in enumerate(traj):
        traj[idx] = sorted_indices[traj[idx]]

    return traj


def compute_discrete_trajectory(
    traj: np.array, k: int = 30
) -> (npt.NDArray[np.int], npt.NDArray[np.float64]):
    cluster = pyemma.coordinates.cluster_kmeans(traj, k=k)
    discrete_traj = cluster.dtrajs[0]
    cluster_centers = cluster.clustercenters.flatten()
    discrete_traj = relabel_trajectory_by_coordinate_chronology(
        discrete_traj, cluster_centers
    )
    cluster_centers = np.sort(cluster_centers)

    return discrete_traj, cluster_centers


def calculate_cni(i: int, X: np.array, n: int, P: np.array) -> np.array:
    return np.sum([(X[j] - X[i]) ** n * P[i, j] for j in range(len(X))])


def calculate_c(X: np.array, n: int, P: np.array) -> np.array:
    return np.array([calculate_cni(i, X, n, P) for i in range(len(X))])


def correlation_coefficients_check(
    beta: float,
    potential: Callable,
    discrete_traj: npt.NDArray[np.int],
    cluster_centers: npt.NDArray[np.float64],
    lag: int,
    time_step: float,
) -> None:
    tau = lag * time_step

    msm = pyemma.msm.estimate_markov_model(discrete_traj, lag)

    x_min = min(cluster_centers)
    x_max = max(cluster_centers)
    x_range = np.arange(x_min, x_max, (x_max - x_min) / 1000)
    grad_potential = grad(potential)

    D1_theory = -beta * np.array([grad_potential(x) for x in x_range])
    D2_theory = np.array([1 for x in x_range])

    C1_theory = tau * D1_theory
    C2_theory = 2 * D2_theory * tau + C1_theory**2

    C1_exp = calculate_c(cluster_centers, 1, msm.transition_matrix)
    C2_exp = calculate_c(cluster_centers, 2, msm.transition_matrix)

    D1_exp = C1_exp / tau
    D2_exp = (C2_exp - C1_exp**2) / (2 * tau)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    axs[0, 0].set_title("C1")
    axs[0, 0].plot(x_range, C1_theory, label="theory")
    axs[0, 0].plot(cluster_centers, C1_exp, label="exp")
    axs[0, 0].legend()

    axs[0, 1].set_title("C2")
    axs[0, 1].plot(x_range, C2_theory, label="theory")
    axs[0, 1].plot(cluster_centers, C2_exp, label="exp")
    axs[0, 1].legend()

    axs[1, 0].set_title("D1")
    axs[1, 0].plot(x_range, D1_theory, label="theory")
    axs[1, 0].plot(cluster_centers, D1_exp, label="exp")
    axs[1, 0].legend()

    axs[1, 1].set_title("D2")
    axs[1, 1].plot(x_range, D2_theory, label="theory")
    axs[1, 1].plot(cluster_centers, D2_exp, label="exp")
    axs[1, 1].legend()

    plt.show()
