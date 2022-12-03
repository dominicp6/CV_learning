#!/usr/bin/env python

"""
   Author: Dominic Phillips (dominicp6)
"""

import os

import numpy as np
import pyemma
import matplotlib.pyplot as plt
import numpy.typing as npt
import networkx as nx
from deeptime.markov import TransitionCountModel
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, MarkovStateModelCollection
from deeptime.markov.tools.analysis import mfpt
from tqdm import tqdm

from utils.openmm_utils import parse_quantity
from utils.plotting_functions import init_plot
from utils.openmm_utils import round_format_unit
from utils.plot_utils import my_draw_networkx_edge_labels


# TODO: transition matrix only defined by specified tau value
# need to add a check to make sure that the calculation of the
# diffusion coefficient is consistent with the transition matrix
# that is provided by the user
class MSM:
    def __init__(self, number_of_states: int, lagstep: int, verbose: bool = True,
                 clustering_init_strategy: str = 'kmeans++', clustering_seed: int = 13,
                 transition_count_mode: str = 'sliding', reversible: bool = True):
        self.number_of_states = number_of_states  # number of discrete states
        self.lagstep = lagstep

        self.timestep = None  # the timestep between frames of the data
        self.lagtime = None  # lagstep * timestep
        self.dimension = None  # the dimension of the trajectory data used to train the MSM
        self.data = None  # the data used to train the MSM

        self.clustering = None
        self.trajectory = None  # a discrete trajectory
        self.transition_counts = None
        self.transition_matrix = None
        self.stationary_distribution = None
        self.msm = None

        # Additional arguments #
        self.verbose = verbose
        self.clustering_init_strategy = clustering_init_strategy
        self.clustering_seed = clustering_seed
        self.transition_count_mode = transition_count_mode
        self.reversible = reversible

    def fit(self, data: np.array, lagtime: str):
        assert len(data.shape) < 3, "The data array must be either 1D or 2D"
        self.dimension = np.shape(data)[1] if len(data.shape) == 2 else 1
        self.lagtime = parse_quantity(lagtime)
        self.trajectory = self._cluster(data)
        self.transition_counts = self._transition_counts(self.trajectory)
        self.msm = self._maximum_likelihood_msm(self.transition_counts)
        self.transition_matrix = self.msm.transition_matrix
        self.stationary_distribution = self.msm.stationary_distribution

    def fetch(self):
        return self.msm

    def fit_fetch(self, data: np.array, lagtime: str) -> MarkovStateModelCollection:
        self.fit(data, lagtime)
        return self.fetch()

    def _cluster(self, data: np.array) -> np.array:
        estimator = KMeans(n_clusters=self.number_of_states, init_strategy='kmeans++', fixed_seed=self.clustering_seed,
                           n_jobs=os.cpu_count() - 1,
                           progress=tqdm)
        self.clustering = estimator.fit(data).fetch_model()
        if self.verbose:
            self.plot_inertia()
            print("Cluster centres", self.clustering.cluster_centers)
        trajectory = self.clustering.transform(data)

        return trajectory

    def _transition_counts(self, trajectory) -> TransitionCountModel:
        estimator = TransitionCountEstimator(lagtime=self.lagstep, count_mode=self.transition_count_mode)
        counts = estimator.fit(trajectory).fetch_model()
        if self.verbose:
            print("Weakly connected sets:", counts.connected_sets(directed=False))
            print("Strongly connected sets:", counts.connected_sets(directed=True))

        return counts

    def _maximum_likelihood_msm(self, transition_counts) -> MarkovStateModelCollection:
        estimator = MaximumLikelihoodMSM(reversible=self.reversible, stationary_distribution_constraint=None)
        msm = estimator.fit(transition_counts).fetch_model()

        return msm

    def calculate_correlation_coefficient(self, n: int):
        assert self.transition_matrix is not None
        return np.sum(
            [
                (self.sorted_state_centers - self.sorted_state_centers[j]) ** n
                * self.transition_matrix[:, j]
                for j in range(self.number_of_states)
            ],
            axis=0,
        )

    # TODO: currently only works for 1D MSMs
    def compute_transition_rate(
            self, state_A: tuple[float, float], state_B: tuple[float, float]
    ):
        # Note lag must be the same as the lag used to define the Markov State Model
        msm = pyemma.msm.estimate_markov_model(self.discrete_trajectory, lag=self.lag)
        initial_states = self.compute_states_for_range(state_A[0], state_A[1])
        final_states = self.compute_states_for_range(state_B[0], state_A[1])
        mfpt = msm.mfpt(A=initial_states, B=final_states) * self.time_step

        return 1 / mfpt

    def plot(self):
        print(
            f"MSM created with {self.number_of_states} states, using lag time {self.lag}."
        )
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.transition_matrix)
        plt.xlabel("j", fontsize=16)
        plt.ylabel("i", fontsize=16)
        plt.title(r"MSM Transition Matrix $\mathbf{P}$", fontsize=16)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.plot(self.stationary_distribution, color="k")
        plt.xlabel("i", fontsize=16)
        plt.ylabel(r"$\pi(i)$", fontsize=16)
        plt.title(r"MSM Stationary Distribution $\mathbf{\pi}$", fontsize=16)
        plt.show()

    def plot_transition_graph(self, threshold_probability: float = 1e-2):
        fig, ax = init_plot(f"Transition matrix with connectivity threshold {threshold_probability:.0e}",
                            figsize=(10, 10))
        msm_graph = nx.DiGraph()
        edge_labels = {}
        self._add_nodes(msm_graph)
        for i in range(self.msm.n_states):
            for j in range(self.msm.n_states):
                if self.msm.transition_matrix[i, j] > threshold_probability:
                    self._add_edge(msm_graph, i, j, edge_labels)

        self._draw_graph(msm_graph, ax, edge_labels)

    def _add_nodes(self, graph):
        for i in range(self.msm.n_states):
            graph.add_node(i, title=f"{i + 1}")

    def _add_edge(self, graph: nx.Graph, i: int, j: int, edge_labels: dict):
        transition_step = mfpt(self.msm.transition_matrix, target=j, origin=i, tau=self.lagstep)
        transition_time = f"{round_format_unit(transition_step * self.timestep, 3)}"
        graph.add_edge(i, j)
        edge_labels[(i, j)] = f"{self.msm.transition_matrix[i, j]:.3e} ({transition_time})"

    def _draw_graph(self, graph, ax, edge_labels):
        pos = nx.fruchterman_reingold_layout(graph)
        nx.draw_networkx_nodes(graph, pos, ax=ax)
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=nx.get_node_attributes(graph, 'title'))
        nx.draw_networkx_edges(graph, pos, ax=ax, arrowstyle='-|>', connectionstyle='arc3, rad=0.3')
        my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels, rotate=False, rad=0.25)

    def plot_inertia(self):
        """
        Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance
        between each data point and its centroid, squaring this distance, and summing these squares across one cluster.
        """
        fig, ax = init_plot("Inertia of KMeans Training", "iteration", "inertia", xscale="log")
        ax.plot(self.clustering.inertias)
        plt.show()


class MSM:
    def __init__(self, state_centers: npt.NDArray[np.float64]):
        self.state_centers = state_centers
        self.number_of_states = len(state_centers)
        self.sorted_state_centers = np.sort(state_centers)
        self.stationary_distribution = None
        self.transition_matrix = None
        self.lag = None
        self.time_step = None
        self.discrete_trajectory = None

    def compute_states_for_range(self, lower_value: float, upper_value: float) -> np.array:
        state_boundaries = [
            (self.sorted_state_centers[i + 1] + self.sorted_state_centers[i]) / 2
            for i in range(len(self.sorted_state_centers) - 1)
        ]
        number_of_lower_states = len(
            [boundary for boundary in state_boundaries if boundary < lower_value]
        )
        number_of_upper_states = len(
            [boundary for boundary in state_boundaries if boundary > upper_value]
        )

        lower_state_index = number_of_lower_states
        upper_state_index = self.number_of_states - number_of_upper_states

        states_in_range = np.arange(lower_state_index, upper_state_index, 1)

        return states_in_range

    def compute_diffusion_coefficient_domain(self):
        diffusion_coeff_domain = []
        for idx in range(len(self.sorted_state_centers) - 1):
            diffusion_coeff_domain.append(
                (self.sorted_state_centers[idx + 1] + self.sorted_state_centers[idx])
                / 2
            )

        return diffusion_coeff_domain

    # def set_stationary_distribution(self, stationary_distribution: np.array):
    #     self.stationary_distribution = stationary_distribution
    #
    # def set_transition_matrix(self, transition_matrix: np.array):
    #     assert transition_matrix.shape[0] == len(self.sorted_state_centers)
    #     self.transition_matrix = transition_matrix
    #
    # def set_lag(self, lag: int):
    #     self.lag = lag
    #
    # def set_time_step(self, time_step: float):
    #     self.time_step = time_step
    #
    # def set_discrete_trajectory(self, discrete_traj: npt.NDArray[np.int]):
    #     self.discrete_trajectory = discrete_traj

    # def calculate_correlation_coefficient(self, n: int):
    #     assert self.transition_matrix is not None
    #     return np.sum(
    #         [
    #             (self.sorted_state_centers - self.sorted_state_centers[j]) ** n
    #             * self.transition_matrix[:, j]
    #             for j in range(self.number_of_states)
    #         ],
    #         axis=0,
    #     )

    def relabel_trajectory_by_coordinate_chronology(self, traj: npt.NDArray[np.int]):
        sorted_indices = np.argsort(np.argsort(self.state_centers))

        # relabel states in trajectory
        for idx, state in enumerate(traj):
            traj[idx] = sorted_indices[traj[idx]]

        return traj

    # def compute_diffusion_coefficient(self, time_step: float, lag: int):
    #     tau = lag * time_step
    #     c1 = self.calculate_correlation_coefficient(n=1)
    #     c2 = self.calculate_correlation_coefficient(n=2)
    #     # space-dependent diffusion coefficient
    #     diffusion_coefficient = (c2 - c1 ** 2) / (2 * tau)
    #
    #     return diffusion_coefficient

    def plot(self):
        print(
            f"MSM created with {self.number_of_states} states, using lag time {self.lag}."
        )
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.transition_matrix)
        plt.xlabel("j", fontsize=16)
        plt.ylabel("i", fontsize=16)
        plt.title(r"MSM Transition Matrix $\mathbf{P}$", fontsize=16)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.plot(self.stationary_distribution, color="k")
        plt.xlabel("i", fontsize=16)
        plt.ylabel(r"$\pi(i)$", fontsize=16)
        plt.title(r"MSM Stationary Distribution $\mathbf{\pi}$", fontsize=16)
        plt.show()

    def compute_transition_rate(
            self, state_A: tuple[float, float], state_B: tuple[float, float]
    ):
        # Note lag must be the same as the lag used to define the Markov State Model
        msm = pyemma.msm.estimate_markov_model(self.discrete_trajectory, lag=self.lag)
        initial_states = self.compute_states_for_range(state_A[0], state_A[1])
        final_states = self.compute_states_for_range(state_B[0], state_A[1])
        mfpt = msm.mfpt(A=initial_states, B=final_states) * self.time_step

        return 1 / mfpt
