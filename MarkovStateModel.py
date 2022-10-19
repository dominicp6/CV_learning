#!/usr/bin/env python

"""
   Author: Dominic Phillips (dominicp6)
"""

import numpy as np
import pyemma
import matplotlib.pyplot as plt
import numpy.typing as npt


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

    def compute_states_for_range(self, lower_value: float, upper_value: float):
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

    def set_stationary_distribution(self, stationary_distribution: np.array):
        self.stationary_distribution = stationary_distribution

    def set_transition_matrix(self, transition_matrix: np.array):
        assert transition_matrix.shape[0] == len(self.sorted_state_centers)
        self.transition_matrix = transition_matrix

    def set_lag(self, lag: int):
        self.lag = lag

    def set_time_step(self, time_step: float):
        self.time_step = time_step

    def set_discrete_trajectory(self, discrete_traj: npt.NDArray[np.int]):
        self.discrete_trajectory = discrete_traj

    def calculate_correlation_coefficient(self, n: int):
        assert self.transition_matrix is not None
        return np.sum(
            [
                (self.sorted_state_centers[j] - self.sorted_state_centers) ** n
                * self.transition_matrix[:, j]
                for j in range(self.number_of_states)
            ],
            axis=0,
        )

    def relabel_trajectory_by_coordinate_chronology(self, traj: npt.NDArray[np.int]):
        sorted_indices = np.argsort(np.argsort(self.state_centers))

        # relabel states in trajectory
        for idx, state in enumerate(traj):
            traj[idx] = sorted_indices[traj[idx]]

        return traj

    def compute_diffusion_coefficient(self, time_step: float, lag: int):
        tau = lag * time_step
        print(tau, type(tau))
        c1 = self.calculate_correlation_coefficient(n=1)
        c2 = self.calculate_correlation_coefficient(n=2)
        # space-dependent diffusion coefficient
        diffusion_coefficient = (c2 - c1**2) / (2 * tau)
        print(diffusion_coefficient)

        return diffusion_coefficient

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
        self, state_A: list[float, float], state_B: list[float, float]
    ):
        # Note lag must be the same as the lag used to define the Markov State Model
        msm = pyemma.msm.estimate_markov_model(self.discrete_trajectory, lag=self.lag)
        initial_states = self.compute_states_for_range(state_A)
        final_states = self.compute_states_for_range(state_B)
        mfpt = msm.mfpt(A=initial_states, B=final_states) * self.time_step

        return 1 / mfpt
