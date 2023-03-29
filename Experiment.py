#!/usr/bin/env python

"""Defines an Experiment class that enables easy access to the trajectory and metadate stored in a directory.
   Allows systematic calculations (such as applying tICA, VAMP or DF dimensionality reduction) to be easily applied
   to whole or part of the experiment trajectory. Also allows for the automatic creation of PLUMED metadynamics scripts
   based on the trajectory's learnt CVs.

   Author: Dominic Phillips (dominicp6)
"""
import os
import copy
import re
import math
import subprocess
from time import time
from typing import Union, Optional

import deeptime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import openmm.unit as unit
import nglview as nv
from deeptime.util.validation import implied_timescales
from deeptime.plots import plot_implied_timescales
from tqdm import tqdm
from scipy.spatial import Voronoi

# from KramersRateEvaluator import KramersRateEvaluator
# from MarkovStateModel import MSM
from utils.diffusion_utils import my_fit_dm
import pydiffmap.diffusion_map as dfm
from utils.plot_utils import voronoi_plot_2d, find_turning_points_interpolation
from utils.experiment_utils import scatter_fes, get_feature_ids_from_names, \
    reweight_biased_fes, generate_fes, set_fes_cbar_and_axis, check_fes_arguments, init_metadata, init_datafiles, \
    init_biasfiles, init_features, check_feature_is_cv_feature
from utils.general_utils import supress_stdout, assert_kwarg
from utils.openmm_utils import time_to_iteration_conversion
from utils.plotting_functions import init_plot, init_multiplot, save_fig
from utils.feature_utils import compute_best_fit_feature_eigenvector
from utils.biased_experiment_utils import construct_partial_HILLS, create_partial_fes_files, compute_fes_evolution, plot_fes_evolution, plot_fes_slice_evolution


# TODO: fix free energy plot to make it work correctly with collective variables
# TODO: think what is happening to water molecules in the trajectory


class Experiment:
    def __init__(
            self,
            location: str,
            features: Optional[Union[dict, list[str], np.array]] = None,
            cos_sin: bool = False,
            in_progress: bool = False,
    ):
        # ================== DEFAULTS =====================
        self.DM_DEFAULTS = {
            "epsilon": "bgh",
            "alpha": 0.5,
            "k": 64,
            "kernel_type": "gaussian",
            "n_evecs": 2,
            "neighbor_params": {'n_jobs': -1, 'algorithm': 'kd_tree'},
            "metric": "euclidean",
            "metric_params": None,
            "weight_fxn": None,
            "density_fxn": None,
            "bandwidth_type": "-1/(d+2)",
            "bandwidth_normalize": False,
            "oos": "nystroem",
        }
        self.KRE_DEFAULTS = {
            "minimum_counts": 25,
            "bins": 200,
            "impute_free_energy_nans": True,
            "cluster_type": "kmeans",
            "k": 100,
            "ignore_high_energy_minima": False,
            "include_endpoint_minima": True,
            "minima_prominence": 1.5,
            "options": None,
        }
        # ================================================
        self.in_progress = in_progress
        self.location = location
        # Change the working directory to the experiment directory
        os.chdir(self.location)
        (
            self.temperature,
            self.duration,
            self.savefreq,
            self.stepsize,
            self.iterations,
            self.beta,
        ) = init_metadata(location)
        (
            self.pdb,
            self.topology,
            self.traj,
            self.num_frames,
        ) = init_datafiles(self, location)
        self.bias_trajectory = init_biasfiles(location)
        (
            self.feature_dict,
            self.featurizer,
            self.featurized_traj,
            self.feature_means,
            self.feature_stds,
            self.num_features,
            self.features_provided,
        ) = init_features(self, features, cos_sin=cos_sin)

        self.CV_types = ['PCA', 'TICA', 'VAMP', 'DMD', 'DM']
        self.CVs = {"PCA": None, "TICA": None, "VAMP": None, "DMD": None, "DM": None}
        # self.kre = KramersRateEvaluator()
        self.discrete_traj = None
    
    # def hills_reweight(self, stride=6000):
    #     """Reweights the biased trajectory using PLUMED's sum_hills function"""
    #     os.chdir(self.location)
    #     if not os.path.exists("fes_convergence"):
    #         os.mkdir("fes_convergence")
    #         subprocess.call(f"plumed sum_hills --hills HILLS  --stride {stride} --outfile {self.location}/fes_convergence/fes --mintozero", shell=True)
    #     else:
    #         print("FES already calculated")

    def plot_fes_convergence(self, number_of_frames=10):
        """Plots the free energy surface convergence as a function of the number of frames"""

        construct_partial_HILLS(self.location, number_of_frames)
        create_partial_fes_files(self.location)
        # TODO: function to compute reweighted fes evolution
        # NB: this is for unbiased trajectories
        fes_surfaces, fes_trajs = compute_fes_evolution(self.location)
        plot_fes_evolution(fes_surfaces)
        plot_fes_slice_evolution(self.beta, fes_trajs)



    def get_features(self):
        return self.featurizer.describe()

    def get_trajectory(self):
        return self.featurized_traj if self.features_provided else self.traj.xyz

    def free_energy_plot(
            self,
            features: list[str] = None,
            feature_nicknames: Optional[list[str]] = None,
            nan_threshold: int = 50,
            save_name="free_energy_plot",
            data_fraction: float = 1.0,
            bins: int = 100,
            landmark_points: Optional[dict[str, tuple[float, float]]] = None,
            show_turning_points = False,
            save_data: bool = True,
            show_fig: bool = True,
            close_fig: bool = True,
            fig_size=(6, 4),
            ax=None,
            reweight: bool = False,
            plot_type: str = "bezier",    # For reweighted, biased trajectories only "contour", "bezier", "heatmap"
    ):

        feature_nicknames = check_fes_arguments(self, data_fraction, reweight, features, feature_nicknames)
        fig, ax = init_plot("Free Energy Surface", f"{feature_nicknames[0]}", f"{feature_nicknames[1]}", ax=ax, figsize=fig_size)

        x,y,z=None, None, None
        if self.bias_trajectory and reweight:
                ax, im, bias_traj = reweight_biased_fes(self.location, feature_nicknames, ax, self.beta, plot_type)
                x = bias_traj.feat1
                y = bias_traj.feat2
                z = bias_traj.free_energy
        elif self.bias_trajectory and not reweight:
                ax, im = generate_fes(self.beta, ax, self.bias_trajectory, plot_type=plot_type)
                x = self.bias_trajectory.feat1
                y = self.bias_trajectory.feat2
                z = self.bias_trajectory.free_energy
        elif not self.bias_trajectory:
            feature_traj = self.get_feature_trajs_from_names(features)[:int(data_fraction * self.num_frames)]
            ax, im = scatter_fes(self.beta, ax, feature_traj, bins, nan_threshold)
            x = feature_traj[:, 0]
            y = feature_traj[:, 1]
            z = feature_traj[:, 2]

        if show_turning_points:
            turning_points, ax = find_turning_points_interpolation(x, y, z, ax=ax)
            print(turning_points)
        
        cbar = set_fes_cbar_and_axis(im, ax, feature_nicknames)

        if landmark_points:
            for point, coordinates in landmark_points.items():
                ax.plot(coordinates[0], coordinates[1], marker='o', markerfacecolor='w')
                ax.annotate(point, coordinates, color='w')
            vor = Voronoi(np.array(list(landmark_points.values())))
            fig = voronoi_plot_2d(vor, ax=ax, line_colors='red', line_width=2)

        
        save_fig(fig, save_dir=os.getcwd(), name=save_name, save_data=save_data, show_fig=show_fig, close=close_fig)

        return fig, ax

    # def markov_state_model(self, n_clusters: int, lagtime: str, features: list[str],
    #                        feature_nicknames: Optional[list[str]] = None,
    #                        threshold_probability: float = 1e-2):
    #     assert isinstance(lagtime, str), "Lagtime must be a string (e.g. `10ns')."
    #     samples = self.get_trajectory()
    #     # TODO: make clustering work for sin, cos features
    #     msm = MSM(n_clusters, lagtime=lagtime)
    #     fig, axs = init_multiplot(nrows=5, ncols=3, panels=['0:2,0:2', '0,2', '1,2', '2:,:-1', '2,2'], title='MSM')
    #     msm.fit(data=samples, timestep=self.savefreq)
    #     msm.plot_timescales(show=False, ax=axs[1])
    #     msm.plot_transition_matrix(show=False, ax=axs[4])
    #     msm.plot_transition_graph(threshold_probability=threshold_probability, ax=axs[3])
    #     msm.plot_stationary_distribution(show=False, ax=axs[2])
    #
    #     feature_ids = get_feature_ids_from_names(features, self.featurizer)
    #     assert len(feature_ids) == 2
    #     landmark_points = {}
    #     for i in range(msm.number_of_states):
    #         landmark_points[str(i + 1)] = tuple(msm.state_centres[i, feature_ids])
    #
    #     self.free_energy_plot(features, feature_nicknames,
    #                           landmark_points=landmark_points,
    #                           ax=axs[0], save_data=False,
    #                           show_fig=False, close_fig=False)
    #     plt.show()

    # TODO: add PCCA+ function for "coarse graining" the MSM
    def timeseries_analysis(self, contact_threshold: float = 2.0, times: list[str] = None):
        duration_ns = self.duration.in_units_of(unit.nanoseconds)._value

        if times is not None:
            assert len(times) == 3, "Three time snapshots must be provided for contact analysis."
            frames = [time_to_iteration_conversion(t, self.duration, self.num_frames) for t in times]
        else:
            times = ['0.0ns', f'{duration_ns / 2}ns', f'{duration_ns}ns']
            frames = [0, int(len(self.traj) / 2), -1]
        fig, axs = init_multiplot(nrows=6, ncols=3, panels=['0,0', '0,1', '0,2', '1,:', '2,:', '3,:', '4,:', '5,:'])

        self._plot_contact_matrices(fig, axs, frames, times)
        self._plot_trajectory_timeseries(axs, contact_threshold, times, duration_ns)
        plt.show()

    def interact(self):
        view = nv.show_mdtraj(self.traj)

        return view

    def implied_timescale_analysis(self,
                                   max_lag: str,
                                   increment: int = 1,
                                   num_timescales: int = 2,
                                   yscale: str = 'log',
                                   xscale: str = 'linear'):
        """
        Illustrates how TICA implied timescales vary as the lagtime varies.
        For more details on this approach, consult https://docs.markovmodel.org/lecture_implied_timescales.html.
        """
        assert isinstance(max_lag, str), "Max_lag must be a string (e.g. `10ns')."
        max_lag = time_to_iteration_conversion(max_lag, self.duration, self.num_frames)
        lagtimes = np.arange(1, max_lag, increment)

        # save current TICA obj
        TICA_obj = copy.deepcopy(self.CVs['TICA'])

        # learn TICA model for each lagtime
        models = []
        for lagtime in tqdm(lagtimes):
            supress_stdout(self.compute_cv)('TICA', lagtime=lagtime)
            models.append(self.CVs['TICA'])

        # restore original TICA obj
        self.CVs['TICA'] = TICA_obj

        # compute implied timescales and make time unit conversion
        its_data = implied_timescales(models)
        its_data._lagtimes *= self.savefreq
        its_data._its *= self.savefreq

        # plot implied timescales
        fig, ax = init_plot('Implied timescales (TICA)', f'lag time ({self.savefreq.unit})',
                            f'timescale ({self.savefreq.unit})', xscale=xscale, yscale=yscale)
        plot_implied_timescales(its_data, n_its=num_timescales, ax=ax)

        # plt.annotate(f"NB: One step corresponds to a time of {self.savefreq}", xy=(90, 1), xycoords='figure points')
        plt.show()

    def compute_cv(self, CV: str, dim: Optional[int] = None, stride: int = 1, verbose: bool = True, **kwargs):
        """
        Compute a given collective variable (CV) on the trajectory.
        CV options are PCA, TICA, VAMP, DMD and DM.

        :param CV: str, CV to compute.
        :param dim: Number of dimensions to keep.
        :param stride: Stride to use when subsampling the trajectory for computing the CV.
        :param kwargs: Any additional keyword arguments for the decomposition functions.
        :return: None
        """
        assert CV in self.CVs.keys(), f"Method '{CV}' not in {self.CVs.keys()}"
        t0 = time()
        # Trajectory is either featurized or unfeaturized (cartesian coords), depending on object initialisation.
        trajectory = self.get_trajectory()
        if CV == "PCA":
            self.CVs[CV] = PCA(n_components=dim, **kwargs).fit(trajectory[::stride])
        elif CV == "TICA":
            assert_kwarg(kwargs, kwarg="lagtime", obj_name=CV)
            # other kwargs: epsilon, var_cutoff, scaling, observable_transform
            self.CVs[CV] = deeptime.decomposition.TICA(dim=dim, **kwargs).fit_fetch(
                trajectory[::stride]
            )
        elif CV == "VAMP":
            assert_kwarg(kwargs, kwarg="lagtime", obj_name=CV)
            # other kwargs: epsilon, var_cutoff, scaling, epsilon, observable_transform
            self.CVs[CV] = deeptime.decomposition.VAMP(dim=dim, **kwargs).fit_fetch(
                trajectory[::stride]
            )
        elif CV == "DMD":
            # other kwargs: mode, rank, exact
            self.CVs[CV] = deeptime.decomposition.DMD(**kwargs).fit_fetch(
                trajectory[::stride]
            )
        elif CV == "DM":
            dm = dfm.DiffusionMap.from_sklearn(**self.DM_DEFAULTS)
            self.CVs[CV] = my_fit_dm(dm, trajectory, stride, tol=kwargs["tol"] if kwargs is not None else 1e-6)
        # TODO: VAMPnets
        t1 = time()
        if verbose:
            print(f"Computed CV in {round(t1 - t0, 3)}s.")

    def compute_correlations(self, CV: str, features: list[str] = None, stride: int = 1):
        if ":" in CV:
            cv_type = CV.split(':')[0]
            dim = int(CV.split(':')[1])
        else:
            raise ValueError("CV must be of the form 'CV_type:dim'")

        if features is None:
            features = self.featurizer.describe()

        self.compute_cv(cv_type, dim=dim+1, stride=stride, verbose=False)
        cv_data = self._get_cv(cv_type, dim=dim, stride=stride)

        correlations = {}
        for feature in tqdm(features):
            feature_data = self.get_feature_trajs_from_names([feature])[::stride]
            correlations[feature] = np.corrcoef(cv_data, feature_data)[0, 1]

        return correlations

    def compute_best_fit_feature_eigenvector(self,
                                             cv: str,
                                             dimensions_to_keep: int,
                                             stride: int = 1,
                                             features=None):
        list_of_features, feature_coefficients, best_correlations = compute_best_fit_feature_eigenvector(self,
                                                                                                         cv,
                                                                                                         dimensions_to_keep,
                                                                                                         stride,
                                                                                                         features)

        return list_of_features, feature_coefficients, best_correlations

    def analyse_kramers_rate(
            self, CV: str, dimension: int, lag: int, sigmaD: float, sigmaF: float
    ):
        self.kre.fit(
            self._get_cv(CV, dimension),
            beta=self.beta,
            time_step=self.stepsize,
            lag=lag,
            sigmaD=sigmaD,
            sigmaF=sigmaF,
            **self.KRE_DEFAULTS,
        )

    def _get_cv(self, CV, dim, stride: int = 1):
        assert CV in self.CVs.keys(), f"Method '{CV}' not in {self.CVs.keys()}"
        trajectory = self.get_trajectory()
        if self.CVs[CV] is None:
            raise ValueError(f"{CV} CVs not computed.")
        if CV in ["PCA", "TICA", "VAMP", "DMD", "DM"]:
            return self.CVs[CV].transform(trajectory[::stride])[:, dim]
        elif CV == "DM":
            return self.CVs[CV].evecs[:, dim]
        else:
            raise NotImplementedError

    def feature_eigenvalue(self, CV: str, dim: int) -> float:
        if not self.features_provided:
            raise ValueError("Cannot compute a feature eigenvalue for an unfeaturized trajectory.")
        else:
            if CV in ["VAMP", "TICA"]:
                return self.CVs[CV].singular_values[dim]
            elif CV in ["PCA"]:
                return self.CVs[CV].explained_variance_ratio_[dim]
            else:
                raise NotImplementedError

    def feature_eigenvector(self, CV: str, dim: int) -> np.array:
        if not self.features_provided:
            raise ValueError(
                "Cannot computed a feature eigenvector for an unfeaturized trajectory. "
                "Try reinitializing Experiment object with features defined."
            )
        else:
            return self._lstsq_traj_with_features(traj=self._get_cv(CV, dim))

    def _lstsq_traj_with_features(self, traj: np.array, feat_traj = None) -> np.array:
        feat_traj = self.featurized_traj if feat_traj is None else feat_traj
        feat_traj = np.c_[np.ones(self.num_frames), feat_traj]
        coeffs, err, _, _ = np.linalg.lstsq(feat_traj, traj, rcond=None)

        return coeffs[1:]

    def get_feature_trajs_from_names(
            self,
            feature_names: list[str],
    ) -> np.array:

        # If the feature is a CV then it must be of the form "CVdim" e.g. "TICA0" or "PCA2"
        cv_features = [check_feature_is_cv_feature(self, feature_name) for feature_name in feature_names]

        # If any of the features are CVs, then we need to get the CVs from the Experiment object
        if any(cv_features):
            assert all(cv_features), "If any of the features are CVs, then all features must be CVs."
            feature_trajs = []
            for feature_name in feature_names:
                match = check_feature_is_cv_feature(self, feature_name)
                matched_string = match.group(1)
                matched_number = int(re.findall(r"\d+", feature_name)[0])
                print(matched_string, matched_number)
                print(self._get_cv(matched_string, matched_number))
                feature_trajs.append(self._get_cv(CV=matched_string, dim=matched_number))
        # Otherwise, we can just get the features from the featurized trajectory
        else:
            feature_trajs = []
            feature_ids = get_feature_ids_from_names(feature_names, self.featurizer)
            for feature_id in feature_ids:
                feature_trajs.append(self.featurized_traj[:, feature_id])

        return np.array(feature_trajs)
