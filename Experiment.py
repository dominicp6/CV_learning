#!/usr/bin/env python

"""Defines an Experiment class that enables easy access to the trajectory and metadate stored in a directory.
   Allows systematic calculations (such as applying tICA, VAMP or DF dimensionality reduction) to be easily applied
   to whole or part of the experiment trajectory. Also allows for the automatic creation of PLUMED metadynamics scripts
   based on the trajectory's learnt CVs.

   Author: Dominic Phillips (dominicp6)
"""
import os
import json
import copy
import re
import csv
import subprocess
from time import time
from typing import Union, Optional

import mdtraj
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
from Dihedrals import Dihedrals
from utils.diffusion_utils import my_fit_dm
# import pydiffmap.diffusion_map as dfm
from utils.plot_utils import voronoi_plot_2d
from utils.experiment_utils import write_metadynamics_line, get_metadata_file, load_pdb, load_trajectory, get_fe_trajs, \
    BiasTrajectory, scatter_fes, heatmap_fes, bezier_fes, contour_fes, get_feature_ids_from_names, initialise_featurizer, \
    generate_reweighting_file, execute_reweighting_script, load_reweighted_trajectory
from utils.general_utils import supress_stdout, assert_kwarg, print_file_contents
from utils.openmm_utils import parse_quantity, time_to_iteration_conversion
from utils.plotting_functions import init_plot, init_multiplot, save_fig
from utils.feature_utils import compute_best_fit_feature_eigenvector, get_cv_type_and_dim, get_feature_means


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
        ) = self._init_metadata(location)
        (
            self.pdb,
            self.topology,
            self.traj,
            self.num_frames,
        ) = self._init_datafiles(location)
        self.bias_trajectory = self.__init_biasfiles()
        (
            self.feature_dict,
            self.featurizer,
            self.featurized_traj,
            self.feature_means,
            self.feature_stds,
            self.num_features,
            self.features_provided,
        ) = self.__init_features(features, cos_sin=cos_sin)

        self.CV_types = ['PCA', 'TICA', 'VAMP', 'DMD', 'DM']
        self.CVs = {"PCA": None, "TICA": None, "VAMP": None, "DMD": None, "DM": None}
        # self.kre = KramersRateEvaluator()
        self.discrete_traj = None

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
            save_data: bool = True,
            show_fig: bool = True,
            close_fig: bool = True,
            fig_size=(6, 4),
            ax=None,
            reweight: bool = False,
            plot_type: str = "bezier",    # For reweighted, biased trajectories only "contour", "bezier", "heatmap"
    ):
        if not self.bias_trajectory and reweight:
            raise ValueError("Cannot reweight unbiased experiments, please set reweight=False.")

        if self.bias_trajectory:
            if data_fraction != 1.0:
                raise NotImplementedError("Trimming data not implemented for biased experiments, "
                                          "please set data_fraction=1.0.")
            if reweight:
                assert features is not None, "Must provide features for reweighting"
                # TODO: replace by a check looking into the plumed.dat file
                feature_nicknames = features
                fig, ax = init_plot("Free Energy Surface", f"{feature_nicknames[0]}", f"{feature_nicknames[1]}",
                                    ax=ax, figsize=fig_size)
                generate_reweighting_file(os.path.join(self.location, 'plumed.dat'),
                                          os.path.join(self.location, 'plumed_reweight.dat'),
                                          feature1=feature_nicknames[0], feature2=feature_nicknames[1],
                                          stride=50, bandwidth=0.05, grid_bin=50, grid_min=-3.141592653589793,
                                          grid_max=3.141592653589793,
                                          )
                execute_reweighting_script(self.location, 'trajectory.dcd', 'plumed_reweight.dat', kT=1/self.beta._value)
                bias_traj = load_reweighted_trajectory(self.location)
                if plot_type == "contour":
                    ax, im = contour_fes(self.beta, ax, bias_traj)
                elif plot_type == "heatmap":
                    ax, im = heatmap_fes(self.beta, ax, bias_traj)
                elif plot_type == "bezier":
                    ax, im = bezier_fes(self.beta, ax, bias_traj)
            else:
                fig, ax = init_plot("Free Energy Surface", f"CV 1", f"CV 2", ax=ax, figsize=fig_size)
                feature_nicknames = ["CV 1", "CV 2"]
                # biased experiments require contour plots
                ax, im = contour_fes(self.beta, ax, self.bias_trajectory)
        else:
            assert features is not None, "Must provide features for unbiased experiments"
            # unbiased experiments require scatter plots
            feature_nicknames = self._check_fes_arguments(features, feature_nicknames)
            fig, ax = init_plot("Free Energy Surface", f"${feature_nicknames[0]}$", f"${feature_nicknames[1]}$",
                                ax=ax, figsize=fig_size)
            # get traj of features and trim to chemicals fraction
            feature_traj = self.get_feature_trajs_from_names(features)[:int(data_fraction * self.num_frames)]
            ax, im = scatter_fes(self.beta, ax, feature_traj, bins, nan_threshold)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            f"F({feature_nicknames[0]},{feature_nicknames[1]}) / kJ mol$^{{-1}}$"
        )
        plt.gca().set_aspect("equal")
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

    def create_plumed_metadynamics_script(
            self,
            CVs: list[str],
            features: list[list[str]],
            coefficients: list[list[float]],
            filename: str = 'plumed.dat',
            exp_name: str = 'exp',
            gaussian_height: float = 0.2,
            gaussian_pace: int = 1000,
            well_tempered: bool = True,
            bias_factor: float = 8,
            temperature: float = 300,
            sigma_list: Optional[list[float]] = None,
            normalised: bool = True,
            print_to_terminal: bool = True,
            subtract_feature_means: bool = False,
            use_all_features: bool = True,
    ):
        """
        Creates a PLUMED script for metadynamics in the current directory.

        :param CVs: List of CVs to use for metadynamics.
        CV format is CV:dim (dim optional), examples: TICA:0, DM:3, PHI 0 ALA 2, PSI 0 ALA 2.
        :param filename: Name of the PLUMED script. If None, defaults to 'plumed.dat'.
        :param exp_name: Name of the experiment. If None, defaults to 'exp'.
        :param gaussian_height: Height of the Gaussian bias.
        :param gaussian_pace: Number of steps between depositing each Gaussian bias.
        :param well_tempered: Whether to use well-tempered metadynamics.
        :param bias_factor: Bias factor for well-tempered metadynamics.
        :param temperature: Temperature for well-tempered metadynamics.
        :param sigma_list: List of sigmas for each CV. If None, defaults to 0.1 for each CV.
        :param normalised:  Whether to use normalised CVs.
        :param print_to_terminal: Whether to print the script to the terminal.
        :param subtract_feature_means: Whether to subtract the mean of each feature when defining it in the PLUMED script.
        """
        if sigma_list is None:
            sigma_list = [0.1]*len(CVs)
        assert len(sigma_list) == len(CVs), f"Number of sigmas ({len(sigma_list)}) " \
                                            f"must match number of CVs ({len(CVs)})."
        if not self.features_provided:
            raise ValueError(
                "Cannot create PLUMED metadynamics script for an unfeaturized trajectory. "
                "Try reinitializing Experiment object with features defined."
            )
        else:
            # Check if CVs are valid
            self._check_valid_cvs(CVs)

            # Initialise PLUMED script
            file_name = "./plumed.dat" if filename is None else f"{filename}"
            f = open(file_name, "w")
            output = 'RESTART'
            f.write(output + "\n")

            if use_all_features:
                # Use all features in plumed script
                relevant_features = self.featurizer.describe()
            else:
                # Union of all features appearing in the CVs
                relevant_features = list({f for feat in features for f in feat})

            # Save features and coefficients to file
            with open(os.path.join(self.location, 'enhanced_sampling_features_and_coeffs.csv'), 'w') as f2:
                writer = csv.writer(f2, delimiter='\t')
                writer.writerows(zip(features, coefficients))

            if subtract_feature_means:
                offsets = get_feature_means(all_features=self.featurizer.describe(),
                                            all_means=self.feature_means,
                                            selected_features=relevant_features)
            else:
                offsets = None

            # Initialise Dihedrals class (for now only linear combinations of dihedral CVs are supported)
            dihedral_features = Dihedrals(
                topology=self.topology,
                dihedrals=relevant_features,
                offsets=offsets,
                normalised=normalised,
            )

            # Write PLUMED script
            dihedral_features.write_torsion_labels(file=f)
            dihedral_features.write_transform_labels(file=f)

            # Write CVs to PLUMED script
            for idx, CV in enumerate(CVs):
                traditional_cv, cv_type, cv_dim = get_cv_type_and_dim(CV)

                # Only write combined label for traditional CVs
                if traditional_cv:
                    dihedral_features.write_combined_label(CV_name=CV,
                                                           features=features[idx],
                                                           coefficients=coefficients[idx],
                                                           periodic=False,
                                                           file=f)

            # Write metadynamics command to PLUMED script
            write_metadynamics_line(
                well_tempered=well_tempered, bias_factor=bias_factor, temperature=temperature, height=gaussian_height,
                pace=gaussian_pace, sigma_list=sigma_list, CVs=CVs, exp_name=exp_name, file=f, top=self.topology,
            )

            if print_to_terminal:
                print_file_contents(file_name)

    def get_feature_trajs_from_names(
            self,
            feature_names: list[str],
    ) -> np.array:

        # If the feature is a CV then it must be of the form "CVdim" e.g. "TICA0" or "PCA2"
        cv_features = [self._check_feature_is_cv_feature(feature_name) for feature_name in feature_names]

        # If any of the features are CVs, then we need to get the CVs from the Experiment object
        if any(cv_features):
            assert all(cv_features), "If any of the features are CVs, then all features must be CVs."
            feature_trajs = []
            for feature_name in feature_names:
                match = self._check_feature_is_cv_feature(feature_name)
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

    # ====================== INIT HELPERS =================================
    @staticmethod
    def _init_metadata(location: str, keyword="metadata"):
        """
        Reads experiment metadata from file
        """
        metadata_file = get_metadata_file(location, keyword)
        with open(os.path.join(metadata_file)) as metadata_f:
            metadata = json.load(metadata_f)
        values = {}
        for key in ["temperature", "duration", "savefreq", "stepsize"]:
            try:
                values[key] = metadata[key]
            except Exception:
                raise ValueError(f"Key {key} not found in the metadata file.")

        temperature, duration, savefreq, stepsize = (
            parse_quantity(values["temperature"]),
            parse_quantity(values["duration"]),
            parse_quantity(values["savefreq"]),
            parse_quantity(values["stepsize"]),
        )
        iterations = int(duration / stepsize)
        beta = 1 / (temperature * 0.0083144621)
        print("Successfully initialised metadata.")

        return temperature, duration, savefreq, stepsize, iterations, beta

    def _init_datafiles(self, location: str):
        """
        Reads molecular system and trajectory chemicals from file
        """
        pdb = load_pdb(location)
        topology = pdb.topology
        traj = load_trajectory(location, topology)
        num_frames = len(traj)
        if self.in_progress:
            print("[Notice] For in-progress experiments, the save frequency cannot be checked. "
                  "Assuming it is correct in the metadata file.")
            # Set duration based on the number of frames in the trajectory, not based on the duration in the metadata
            self.duration = num_frames * self.savefreq
        else:
            # Check that the number of frames in the trajectory is consistent with the duration and savefreq
            assert np.abs(num_frames - int(self.duration / self.savefreq)) <= 1, (
                f"duration ({self.duration}) and savefreq ({self.savefreq}) incompatible with number of conformations "
                f"found in trajectory (got {num_frames}, expected {int(self.duration / self.savefreq)}). "
                f"Consider re-initialising with in_progress=True."
            )

        return pdb, topology, traj, num_frames

    def __init_biasfiles(self, reweight=False):
        """
        Reads Metadynamics bias potential and weights from file
        """
        if os.path.exists(os.path.join(self.location, "HILLS")):
            HILLS_file = os.path.join(self.location, "HILLS")
            if os.path.exists(os.path.join(self.location, "fes.dat")):
                fes_file = os.path.join(self.location, "fes.dat")
            else:
                print("[Notice] No fes.dat file found in experiment directory. Attempting to generate one from HILLS file.")
                subprocess.call(f"plumed sum_hills --hills {HILLS_file}", shell=True)
                fes_file = os.path.join(self.location, 'fes.dat')

            fe_data = np.genfromtxt(fes_file, autostrip=True)
            feature1_traj, feature2_traj, fe = get_fe_trajs(fe_data, reweight=reweight)
            bias_trajectory = BiasTrajectory(feature1_traj, feature2_traj, fe)
            return bias_trajectory
        else:
            # Not a biased experiment
            return None

    def __init_features(self, features: Optional[Union[dict, list[str], np.array]], cos_sin: bool = False):
        """
        Featurises the trajectory according to specified features.
        Features can be an explicit dictionary of dihedrals, such as {'\phi' : [4, 6, 8 ,14], '\psi' : [6, 8, 14, 16]}.
        Alternatively, you can specify the automatic inclusion of all features of particular classes as follows e.g.
        features = ['dihedrals']                  # adds dihedrals
        features = ['dihedrals', 'carbon_alpha']  # adds dihedrals and distances between all Ca atoms

        Options for features are:
        - dihedrals
        - carbon_alpha
        - sidechain_torsions
        """
        # TODO: check feature names are preserved correctly when adding through dictionary
        if features is not None:
            featurizer = initialise_featurizer(features, self.topology, cos_sin=cos_sin)
            featurized_traj = featurizer.transform(self.traj)
            feature_means = np.mean(featurized_traj, axis=0)
            feature_stds = np.std(featurized_traj, axis=0)
            num_features = len(featurizer.describe())
            features_provided = True
            print(f"Featurized trajectory with {num_features} features.")
        else:
            featurizer, featurized_traj, feature_means, feature_stds, num_features = (
                None,
                None,
                None,
                None,
                None,
            )
            features_provided = False
            print(
                "[Notice] No features provided; trajectory defined with cartesian coordinates (not recommended)."
            )

        return (
            features,
            featurizer,
            featurized_traj,
            feature_means,
            feature_stds,
            num_features,
            features_provided,
        )

    def _load_metad_bias(self, bias_file: str, col_idx: int = 2) -> (list, list):
        colvar = np.genfromtxt(bias_file, delimiter=" ")
        assert (
                col_idx < colvar.shape[1]
        ), "col_idx must not exceed 1 less than the number of columns in the bias file"
        bias_potential_traj = colvar[:, col_idx]  # [::500][:-1]
        weights = np.exp(self.beta * bias_potential_traj)
        return weights, bias_potential_traj

    # ====================== OTHER HELPERS =================================
    def _check_feature_is_cv_feature(self, feature: str):
        regex = r"^(" + "|".join(self.CV_types) + r")\d+$"
        return re.match(regex, feature)

    def _check_valid_cvs(self, CVs: list[str]):
        """
        Checks that the CVs provided are valid.
        """
        for idx, CV in enumerate(CVs):
            _, cv_type, cv_dim = get_cv_type_and_dim(CV)
            # If the CV is neither an atom feature nor a traditional CV (e.g. TICA, PCA, etc.), raise an error
            if cv_type not in self.CVs.keys() and cv_type not in self.featurizer.describe():
                raise ValueError(f"CV '{cv_type}' not in {self.CVs.keys()} or {self.featurizer.describe()}")

            if cv_type in self.CVs.keys() and self.CVs[cv_type] is None:
                raise ValueError(f"{cv_type} CVs not computed.")

    def _check_fes_arguments(
            self,
            features: list[str],
            feature_nicknames: Optional[list[str]],
    ) -> (list[str], str):
        # Check that the experiment is featurized
        if not self.features_provided:
            raise ValueError(
                "Cannot construct ramachandran plot for an unfeaturized trajectory. "
                "Try reinitializing Experiment object with features defined."
            )

        # Check that the required features exist
        for feature in features:
            assert (
                    (feature in self.featurizer.describe())
                    or self._check_feature_is_cv_feature(feature)), \
                f"Feature '{feature}' not found in available features ({self.featurizer.describe()}) " \
                f"or CV types ({self.CV_types})."

        # Automatically set feature nicknames if they are not provided
        if not feature_nicknames:
            feature_nicknames = features

        return feature_nicknames

    def _plot_contact_matrices(self, fig, axs, frames: list[int], times: list[str]):
        contact_data = [mdtraj.compute_contacts(self.traj[frame]) for frame in frames]
        min_dist = 0
        max_dist = np.max(np.hstack((contact_data[0][0], contact_data[1][0], contact_data[2][0])))
        cbar_x_axis_pos = [0.275, 0.602, 0.9275]
        for idx, cbar_pos in enumerate(cbar_x_axis_pos):
            im = axs[idx].imshow(mdtraj.geometry.squareform(contact_data[idx][0], contact_data[idx][1])[0],
                                 vmin=min_dist, vmax=max_dist)
            axs[idx].set_title(f'time = {times[idx]}')
            axs[idx].set_xlabel('Res Number')
            axs[idx].set_ylabel('Res Number')
            cbar_ax = fig.add_axes([cbar_pos, 0.855, 0.02, 0.125])
            fig.colorbar(im, cax=cbar_ax)

    def _plot_trajectory_timeseries(self, axs, contact_threshold: float, times: list[str], duration_ns: float):
        distances, pairs = mdtraj.compute_contacts(self.traj)
        number_of_close_contacts = np.sum((distances < contact_threshold), axis=1)  # sum along the columns (contacts)
        rms_dist = np.sqrt(np.mean(distances ** 2, axis=1))
        rmsd_initial_structure = mdtraj.rmsd(target=self.traj, reference=self.traj, frame=0)
        acylindricity = mdtraj.acylindricity(self.traj)
        radius_of_gyration = mdtraj.compute_rg(self.traj)

        # number of contacts
        x_var = np.arange(0, duration_ns, duration_ns / self.num_frames)
        axs[3].plot(x_var, number_of_close_contacts, linewidth=0.5)
        [axs[3].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[3].set_xlabel('time (ns)')
        axs[3].set_ylabel(f'#contacts < {contact_threshold} $nm$')

        # RMSD of contacts
        axs[4].plot(x_var, rms_dist, linewidth=0.5)
        [axs[4].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[4].set_xlabel('time (ns)')
        axs[4].set_ylabel(f'RMSD Res-Res ($nm$)')

        # RMSD relative to initial structure
        axs[5].plot(x_var, rmsd_initial_structure, linewidth=0.5)
        [axs[5].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[5].set_xlabel('time (ns)')
        axs[5].set_ylabel(f'RMSD Initial Structure ($nm$)')

        # acylindricity
        axs[6].plot(x_var, acylindricity, linewidth=0.5)
        [axs[6].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[6].set_xlabel('time (ns)')
        axs[6].set_ylabel(f'Acylindricity')

        # radius of gyration
        axs[7].plot(x_var, radius_of_gyration, linewidth=0.5)
        [axs[7].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[7].set_xlabel('time (ns)')
        axs[7].set_ylabel(f'Radius of gyration ($nm$)')
