#!/usr/bin/env python

"""Module for estimating Kramer's hopping rates in an unknown potential energy surface using long-trajectory samples.

This module can be used to assess the relative quality of collective variables (CVs) by comparing the estimated Kramers
rates corresponding to the different CV free energy surfaces (FESs).

Methodological details: Given a collective variable trajectory between metastable states, KramersRateEvaluator then
estimates the FES (using the histogram method) and estimates the position-dependent diffusion coefficient
(by defining a Markov State Model between the states and computing coefficients of the Kramers-Moyal
expansion via estimated correlation coefficients). The Kramer's transition rate is then evaluated by numerical
integration of the estimated space-dependent diffusion coefficient and FES. For theory details see [LINK].

Motivation: An optimal CV corresponds to a perfect reaction coordinate and should define a coordinate axis that is
perpendicular to the separatrix at the transition state. An optimal CV will thus result in the largest barrier height
and lowest transition rate compared to other possible CVs. The relative performance of CVs can thus be assessed by
comparing their relative Kramer's rates.

   Author: Dominic Phillips (dominicp6)
"""


import warnings
from typing import Optional, Union

import pandas as pd
import numpy as np
import pyemma
#from autoimpute.imputations import MiceImputer
from itertools import permutations
from pyemma.coordinates.clustering import RegularSpaceClustering, KmeansClustering
import matplotlib as plt

from MarkovStateModel import MSM
from utils import diffusion_utils as utl, general_utils as gutl, plot_utils as pltutl

type_kramers_rates = list[tuple[tuple[int, int], float]]

class KramersRateEvaluator:
    def __init__(self,
                 verbose: bool = True,
                 default_clustering: Union[RegularSpaceClustering, KmeansClustering] = None
                 ):
        self.verbose = verbose
        self.imputer = None #MiceImputer(strategy={"F": "interpolate"}, n=1, return_list=True)
        self.default_clustering = default_clustering
        self.number_of_default_clusters = (
            None
            if self.default_clustering is None
            else len(default_clustering.clustercenters.flatten())
        )

    def _compute_free_energy(
        self,
        time_series: np.array,
        beta: float,
        bins: int = 200,
        impute: bool = True,
        minimum_counts: int = 25,
    ) -> None:
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
        self,
        time_series: np.array,
        time_step: float,
        lag: int,
        cluster_type: str = "kmeans",
        options: Optional[dict] = None,
    ) -> Union[RegularSpaceClustering, KmeansClustering]:

        if cluster_type is not "kmeans":
            raise NotImplemented("Only kmeans clustering implemented.")

        if options["dmin"] is None:
            options["dmin"] = min(
                10 * self.msd_coordinate,
                (max(self.coordinates) - min(self.coordinates)) / 10,
            )

        msm = MSM(number_of_states=options['k'], lagtime=lag * time_step)
        msm.fit(data=time_series, timestep=time_step)
        if self.verbose:
            utl.lag_sensitivity_analysis(msm.trajectory, msm.state_centres, msm.timestep)

        self.diffusion_coefficients = msm.compute_diffusion_coefficient()
        msm.plot()
        self.msm = msm

        return msm.clustering

    def _compute_kramers_rates(
        self,
        beta: float,
        prominence: float,
        endpoint_minima: bool,
        high_energy_minima: bool,
    ) -> type_kramers_rates:
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
        barrier_integrand = utl.compute_barrier_integrand(self.smoothed_D_domain,
                                                          self.smoothed_D,
                                                          self.coordinates,
                                                          self.smoothed_F,
                                                          beta)

        # 2) Compute the Kramer's transition rates between every possible pair of minima
        kramers_rates = []
        possible_transitions: iter(tuple[int, int]) = permutations(range(len(free_energy_minima)), 2)
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

    def plot_free_energy_landscape(self):
        fig = plt.figure(figsize=(15, 7))
        ax1 = plt.subplot()

        # Diffusion curve
        (l1,) = ax1.plot(self.smoothed_D_domain, self.smoothed_D, color="red")
        ax1.set_ylim(
            (
                min(self.smoothed_D) - 0.2 * (max(self.smoothed_D) - min(self.smoothed_D)),
                max(self.smoothed_D) + 0.1 * (max(self.smoothed_D) - min(self.smoothed_D)),
            )
        )

        # Free energy curve
        ax2 = ax1.twinx()
        (l2,) = ax2.plot(self.coordinates, self.smoothed_F, color="k")
        digit_width = pltutl.get_digit_text_width(fig, ax2)
        plt.legend([l1, l2], ["diffusion_coefficient", "free_energy"])

        print(f"Free energy profile suggests {len(self.free_energy_minima)} minima.")
        pltutl.plot_minima(self.free_energy_minima, self.smoothed_F)
        state_boundaries = pltutl.display_state_boundaries(self.msm, self.smoothed_F)
        if len(state_boundaries) < 50:
            pltutl.display_state_numbers(
                state_boundaries, self.coordinates, self.smoothed_F, digit_width
            )

        ax1.set_xlabel("Q", fontsize=16)
        ax1.set_ylabel(r"Diffusion Coefficient $D^{(2)}(Q)$ ($Q^2 / s$)", fontsize=18)
        ax2.set_ylabel(r"$\mathcal{F}$ ($kJ/mol$)", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title('Free Energy Landscape', fontsize=16)
        plt.savefig("free_energy_landscape.pdf")
        plt.show()

    def fit(
        self,
        trajectory,
        beta: float,
        time_step: float,
        lag: int,
        sigmaD: float,
        sigmaF: float,
        minimum_counts: int = 25,
        bins: int = 200,
        impute_free_energy_nans: bool = True,
        cluster_type: str = "kmeans",
        k: int = 10,
        ignore_high_energy_minima: bool = False,
        include_endpoint_minima: bool = True,
        minima_prominence: float = 1.5,
        options: Optional[dict] = None,
    ) -> type_kramers_rates:

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
