#!/usr/bin/env python

"""Module for estimating Kramer's hopping rates in an unknown potential energy surface using long-trajectory samples.

   Author: Dominic Phillips (dominicp6)
"""

import numpy as np
import warnings
import pandas as pd
import pyemma

from autoimpute.imputations import MiceImputer
from itertools import permutations

from MarkovStateModel import MSM
import diffusion_utils as utl
import plot_utils as pltutl
import general_utils as gutl


class KramersRateEvaluator:
    def __init__(self, verbose=True, default_clustering=None):
        self.verbose = verbose
        self.imputer = MiceImputer(strategy={"F": "interpolate"}, n=1, return_list=True)
        self.default_clustering = default_clustering
        self.number_of_default_clusters = (
            None
            if self.default_clustering is None
            else len(default_clustering.clustercenters.flatten())
        )

    def _compute_free_energy(
        self, time_series, beta, bins=200, impute=True, minimum_counts=25
    ):
        counts, coordinates = np.histogram(time_series, bins=bins)
        coordinates = coordinates[:-1]
        with np.errstate(divide="ignore"):
            normalised_counts = counts / np.sum(counts)
            free_energy = (1 / beta) * gutl.replace_inf_with_nan(
                -np.log(normalised_counts)
            )

        if np.isnan(free_energy).any() and impute:
            # If NaNs found, impute
            warnings.warn(
                f"NaN values were found in the free energy calculation. "
                f"Consider using a longer trajectory or rerunning "
                f"with fewer bins (currently bins={bins}). Fixing with imputation for now."
            )
            print(
                f"Note: Of the {len(free_energy)} free energy evaluations, "
                f"{np.count_nonzero(np.isnan(free_energy))} were NaN values."
            )
            df = pd.DataFrame({"F": free_energy})
            free_energy = self.imputer.fit_transform(df)[0][1].F
        else:
            # Else compute free energy of bins with counts > minimum_counts
            robust_counts = counts[np.where(counts > minimum_counts)]
            normalised_counts = robust_counts / np.sum(counts)
            free_energy = -(1 / beta) * np.log(normalised_counts)
            coordinates = coordinates[np.where(counts > minimum_counts)]

        self.coordinates = coordinates
        self.msd_coordinate = gutl.rms_interval(coordinates)
        self.free_energy = free_energy - np.min(free_energy)

    def _fit_msm(
        self, time_series, time_step, lag, cluster_type="kmeans", options=None
    ):

        if options["dmin"] is None:
            options["dmin"] = min(
                10 * self.msd_coordinate,
                (max(self.coordinates) - min(self.coordinates)) / 10,
            )

        # 1) Cluster the coordinate space into discrete states
        if (
            self.default_clustering is None
            or self.number_of_default_clusters != options["k"]
        ):
            print(
                f"Debug: default clustering {self.default_clustering}, num default clusters {self.number_of_default_clusters}, k {options['k']}"
            )
            cluster = utl.cluster_time_series(time_series, cluster_type, options)
            self.default_clustering = cluster
            self.number_of_default_clusters = len(cluster.clustercenters.flatten())
        else:
            print(f"Using default clustering provided.")
            cluster = self.default_clustering

        # 2) Compute the state trajectory
        discrete_traj = cluster.dtrajs[0].copy()
        cluster_centers = cluster.clustercenters.copy().flatten()
        self.msm = MSM(state_centers=cluster_centers)
        discrete_traj = self.msm.relabel_trajectory_by_coordinate_chronology(
            traj=discrete_traj
        )

        if self.verbose:
            utl.lag_sensitivity_analysis(discrete_traj, cluster_centers, time_step)

        # 3) Use the state trajectory to estimate a best-fitting MSM
        msm = pyemma.msm.estimate_markov_model(dtrajs=discrete_traj, lag=lag)
        self.msm.set_stationary_distribution(msm.stationary_distribution)
        self.msm.set_transition_matrix(msm.transition_matrix)
        self.msm.set_lag(lag)
        self.msm.set_time_step(time_step)
        self.msm.set_discrete_trajectory(discrete_traj)
        self.diffusion_coefficients = self.msm.compute_diffusion_coefficient(
            time_step, lag
        )
        self.msm.plot()

        return cluster

    def _compute_kramers_rates(
        self,
        beta: float,
        prominence: float,
        endpoint_minima: bool,
        high_energy_minima: bool,
    ):
        # 1) Compute and plot minima of the free energy landscape
        free_energy_minima = pltutl.get_minima(
            self.smoothed_F, prominence, endpoint_minima, high_energy_minima
        )
        self.free_energy_minima = [
            (self.coordinates[minima], self.smoothed_F[minima])
            for minima in free_energy_minima
        ]
        pltutl.plot_free_energy_landscape(self)

        well_integrand = utl.compute_well_integrand(self.smoothed_F, beta)
        barrier_integrand = utl.compute_barrier_integrand(self, self.smoothed_F, beta)

        # 2) Compute the Kramer's transition rates between every possible pair of minima
        kramers_rates = []
        possible_transitions = permutations(range(len(free_energy_minima)), 2)
        for transition in possible_transitions:
            kramers_rate = utl.compute_kramers_rate(
                transition,
                free_energy_minima,
                well_integrand,
                barrier_integrand,
                self.coordinates,
            )
            kramers_rates.append((transition, kramers_rate))

        if self.verbose is True:
            pltutl.display_kramers_rates(kramers_rates)

        return kramers_rates

    def fit(
        self,
        trajectory,
        beta,
        time_step,
        lag,
        sigmaD,
        sigmaF,
        minimum_counts=25,
        bins=200,
        impute_free_energy_nans=True,
        cluster_type="kmeans",
        k=10,
        ignore_high_energy_minima=False,
        include_endpoint_minima=True,
        minima_prominence=1.5,
        options=None,
    ):

        if options is None:
            options = {
                "k": k,
                "stride": 5,
                "max_iter": 150,
                "max_centers": 1000,
                "metric": "euclidean",
                "n_jobs": None,
                "dmin": None,
            }

        self._compute_free_energy(
            time_series=trajectory,
            beta=beta,
            bins=bins,
            impute=impute_free_energy_nans,
            minimum_counts=minimum_counts,
        )
        self._fit_msm(
            time_series=trajectory,
            lag=lag,
            time_step=time_step,
            cluster_type=cluster_type,
            options=options,
        )
        self.smoothed_D_domain, self.smoothed_D = gutl.gaussian_smooth(
            x=self.msm.sorted_state_centers,
            y=self.diffusion_coefficients,
            dx=self.msd_coordinate,
            sigma=sigmaD,
        )
        self.smoothed_F_domain, self.smoothed_F = gutl.gaussian_smooth(
            x=self.coordinates, y=self.free_energy, dx=self.msd_coordinate, sigma=sigmaF
        )
        kramers_rates = self._compute_kramers_rates(
            beta, minima_prominence, include_endpoint_minima, ignore_high_energy_minima
        )

        return kramers_rates
