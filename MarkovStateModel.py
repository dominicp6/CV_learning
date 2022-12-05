#!/usr/bin/env python

"""
   Author: Dominic Phillips (dominicp6)
"""

import os
import math
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import networkx as nx
import openmm.unit as unit
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


class MSM:
    def __init__(self, number_of_states: int, lagtime: Union[str, float], verbose: bool = True,
                 clustering_init_strategy: str = 'kmeans++', clustering_seed: int = 13,
                 transition_count_mode: str = 'sliding', reversible: bool = True):
        self.number_of_states = number_of_states  # number of discrete states
        self.lagtime = parse_quantity(lagtime)

        self.timestep = None  # the timestep between frames of the data
        self.lagstep = None  # lagtime / timestep
        self.dimension = None  # the dimension of the trajectory data used to train the MSM
        self.data = None  # the data used to train the MSM

        self.clustering = None
        self.state_centres = None  # coordinates of the centres of the MSM states
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

    def fit(self, data: np.array, timestep: unit.Unit):
        assert len(data.shape) < 3, f"The data array must be either 1D or 2D but got shape {data.shape}"
        self.dimension = np.shape(data)[1] if len(data.shape) == 2 else 1
        self.timestep = timestep
        self.lagstep = math.floor(self.lagtime / self.timestep)
        self.lagtime = self.lagstep * self.timestep
        if self.lagstep < 1:
            raise ValueError(
                f"Lagtime provided ({self.lagtime}) is less than the timestep ({self.timestep}).")
        if self.verbose:
            print(f"Initiating MSM model with lagtime {self.lagtime} (lagstep {self.lagstep}).")
        self.trajectory, self.state_centres = self._cluster(data)
        self.transition_counts = self._transition_counts(self.trajectory)
        self.msm = self._maximum_likelihood_msm(self.transition_counts)
        self.transition_matrix = self.msm.transition_matrix
        self.stationary_distribution = self.msm.stationary_distribution

    def _cluster(self, data: np.array) -> np.array:
        estimator = KMeans(n_clusters=self.number_of_states,
                           init_strategy=self.clustering_init_strategy,
                           fixed_seed=self.clustering_seed,
                           n_jobs=os.cpu_count() - 1,
                           progress=tqdm)
        self.clustering = estimator.fit(data).fetch_model()
        if self.verbose:
            self.plot_inertia()
            print("Cluster centres", self.clustering.cluster_centers)
        trajectory = self.clustering.transform(data)
        state_centres = self.clustering.cluster_centers
        if self.dimension == 1:
            trajectory = self.relabel_trajectory_by_coordinate_chronology(trajectory, state_centres)
            state_centres = np.sort(self.state_centres)

        return trajectory, state_centres

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

    # FUNCTIONS FOR 1D MSMs ############################################################
    def compute_transition_rate(
            self, starting_states: tuple[float, float], ending_states: tuple[float, float]
    ):
        assert self.dimension == 1, f"This function requires a 1D MSM and this is a {self.dimension}D MSM."
        initial_states = self.compute_states_for_range(starting_states[0], starting_states[1])
        final_states = self.compute_states_for_range(ending_states[0], starting_states[1])
        transition_time = mfpt(self.msm.transition_matrix, tau=self.lagstep,
                               origin=initial_states, target=final_states) * self.timestep

        return 1 / transition_time

    def compute_states_for_range(self, lower_value: float, upper_value: float) -> list[int]:
        assert self.dimension == 1, f"This function requires a 1D MSM and this is a {self.dimension}D MSM."
        state_boundaries = [(self.state_centres[i + 1] + self.state_centres[i]) / 2
                            for i in range(len(self.state_centres) - 1)]
        number_of_lower_states = len(
            [boundary for boundary in state_boundaries if boundary < lower_value]
        )
        number_of_upper_states = len(
            [boundary for boundary in state_boundaries if boundary > upper_value]
        )

        lower_state_index = number_of_lower_states
        upper_state_index = self.number_of_states - number_of_upper_states

        states_in_range = np.arange(lower_state_index, upper_state_index, 1)

        return list(states_in_range)

    def calculate_correlation_coefficient(self, n: int):
        assert self.dimension == 1, f"This function requires a 1D MSM and this is a {self.dimension}D MSM."
        assert self.transition_matrix is not None
        return np.sum(
            [
                (self.state_centres - self.state_centres[j]) ** n
                * self.transition_matrix[:, j]
                for j in range(self.number_of_states)
            ],
            axis=0,
        )

    def compute_diffusion_coefficient_domain(self):
        diffusion_coeff_domain = []
        for idx in range(len(self.state_centres) - 1):
            diffusion_coeff_domain.append(
                (self.state_centres[idx + 1] + self.state_centres[idx])
                / 2
            )

        return diffusion_coeff_domain

    def compute_diffusion_coefficient(self):
        tau = self.lagtime
        c1 = self.calculate_correlation_coefficient(n=1)
        c2 = self.calculate_correlation_coefficient(n=2)
        # space-dependent diffusion coefficient
        diffusion_coefficient = (c2 - c1 ** 2) / (2 * tau)

        return diffusion_coefficient

    def relabel_trajectory_by_coordinate_chronology(self, traj: npt.NDArray[np.int], state_centres: np.array):
        assert self.dimension == 1, f"This function requires a 1D MSM and this is a {self.dimension}D MSM."
        sorted_indices = np.argsort(np.argsort(state_centres))

        # relabel states in trajectory
        for idx, state in enumerate(traj):
            traj[idx] = sorted_indices[traj[idx]]

        return traj
    ###################################################################################

    def plot(self):
        print(
            f"MSM created with {self.number_of_states} states, using lag time {self.lagtime}."
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
                            figsize=(10, 10), xlabel=None, ylabel=None)
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
        graph.add_edge(i, j, weight=transition_step)
        edge_labels[(i, j)] = f"{self.msm.transition_matrix[i, j]:.3e} ({transition_time})"

    @staticmethod
    def _draw_graph(graph, ax, edge_labels):
        pos = nx.spring_layout(graph, iterations=25, weight='weight')
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

    def plot_timescales(self):
        # TODO: update y-axis to absolute value
        fix, ax = init_plot("MSM State Timescales", "state", f"timescale (x{self.timestep})")
        ax.plot(self.msm.timescales())
        plt.show()
