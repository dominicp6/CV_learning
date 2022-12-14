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
from time import time
from typing import Union, Optional

import mdtraj
import pyemma
import deeptime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import openmm.unit as unit
# import nglview as nv
from deeptime.util.validation import implied_timescales
from deeptime.plots import plot_implied_timescales
from tqdm import tqdm
from scipy.spatial import Voronoi

from KramersRateEvaluator import KramersRateEvaluator
from MarkovStateModel import MSM
from Dihedrals import Dihedrals
from utils.diffusion_utils import free_energy_estimate_2D
# import pydiffmap.diffusion_map as dfm
import mdfeature.features as feat
from utils.plot_utils import voronoi_plot_2d
from utils.experiment_utils import write_metadynamics_line, get_metadata_file, load_pdb, load_trajectory
from utils.general_utils import supress_stdout, assert_kwarg, remove_nans, print_file_contents
from utils.openmm_utils import parse_quantity, time_to_iteration_conversion
from utils.plotting_functions import init_plot, init_multiplot, save_fig


# TODO: fix free energy plot to make it work correctly with collective variables
# TODO: think what is happening to water molecules in the trajectory
# TODO: possibly replace with mdtraj featurizer
def initialise_featurizer(
        dihedral_features: Union[dict, str], topology
) -> pyemma.coordinates.featurizer:
    featurizer = pyemma.coordinates.featurizer(topology)
    # If features listed explicitly, add them
    if isinstance(dihedral_features, dict):
        dihedral_indices = feat.create_torsions_list(
            atoms=topology.n_atoms,
            size=0,
            append_to=list(dihedral_features.values()),
            print_list=False,
        )
        featurizer.add_dihedrals(dihedral_indices)  # cossin = True
    elif dihedral_features == "dihedrals":
        # Add all backbone dihedrals
        featurizer.add_backbone_torsions()
    else:
        raise ValueError(
            f"Invalid value for dihedral features: '{dihedral_features}'. "
        )
    featurizer.describe()
    return featurizer


def get_feature_ids_from_names(
        feature_names: list[str],
        featurizer: pyemma.coordinates.featurizer,
):
    all_features = featurizer.describe()
    feature_ids = []
    for feature in feature_names:
        feature_ids.append(all_features.index(feature))

    return feature_ids


def get_feature_mask(feature: str, featurizer: pyemma.coordinates.featurizer):
    all_features = featurizer.describe()
    feature_mask = np.zeros(len(all_features), dtype=int)
    feature_mask[all_features.index(feature)] = 1
    return feature_mask


class Experiment:
    def __init__(
            self,
            location: str,
            features: Optional[Union[dict, str]] = None,
            metad_bias_file=None,
    ):
        # ================== DEFAULTS =====================
        self.DM_DEFAULTS = {
            "epsilon": "bgh",
            "alpha": 0.5,
            "k": 64,
            "kernel_type": "gaussian",
            "n_evecs": 5,
            "neighbor_params": None,
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
        (self.metad_weights, self.bias_potential_traj) = self.__init_biasfiles(
            metad_bias_file
        )
        (
            self.feature_dict,
            self.featurizer,
            self.featurized_traj,
            self.feature_means,
            self.feature_stds,
            self.num_features,
            self.features_provided,
        ) = self.__init_features(features)

        self.CV_types = ['PCA', 'TICA', 'VAMP', 'DM']
        self.CVs = {"PCA": None, "TICA": None, "VAMP": None, "DM": None}
        self.kre = KramersRateEvaluator()
        self.discrete_traj = None

    def get_features(self):
        return self.featurizer.describe()

    def get_trajectory(self):
        return self.featurized_traj if self.features_provided else self.traj.xyz

    def free_energy_plot(
            self,
            features: list[str],
            feature_nicknames: Optional[list[str]] = None,
            nan_threshold: int = 50,
            save_name="free_energy_plot",
            data_fraction: float = 1.0,
            bins: int = 100,
            landmark_points: Optional[dict[str, tuple[float, float]]] = None,
            save_data: bool = True,
            show_fig: bool = True,
            close_fig: bool = True,
            ax=None,
    ):
        feature_nicknames = self._check_fes_arguments(features, feature_nicknames)
        fig, ax = init_plot("Free Energy Surface", f"${feature_nicknames[0]}$", f"${feature_nicknames[1]}$", ax=ax)
        # get traj of features and trim to data fraction
        feature_traj = self.get_feature_trajs_from_names(features)[:int(data_fraction * self.num_frames)]
        if self.bias_potential_traj:
            # biased experiments require contour plots
            ax, im = self._contour_fes(ax, feature_traj)
        else:
            # unbiased experiments require scatter plots
            ax, im = self._scatter_fes(ax, feature_traj, bins, nan_threshold)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            f"$F({feature_nicknames[0]},{feature_nicknames[1]})$ / kJ mol$^{{-1}}$"
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

    def markov_state_model(self, n_clusters: int, lagtime: str, features: list[str],
                           feature_nicknames: Optional[list[str]] = None,
                           threshold_probability: float = 1e-2):
        assert isinstance(lagtime, str), "Lagtime must be a string (e.g. `10ns')."
        samples = self.get_trajectory()
        # TODO: make clustering work for sin, cos features
        msm = MSM(n_clusters, lagtime=lagtime)
        fig, axs = init_multiplot(nrows=5, ncols=3, panels=['0:2,0:2', '0,2', '1,2', '2:,:-1', '2,2'], title='MSM')
        msm.fit(data=samples, timestep=self.savefreq)
        msm.plot_timescales(show=False, ax=axs[1])
        msm.plot_transition_matrix(show=False, ax=axs[4])
        msm.plot_transition_graph(threshold_probability=threshold_probability, ax=axs[3])
        msm.plot_stationary_distribution(show=False, ax=axs[2])

        feature_ids = get_feature_ids_from_names(features, self.featurizer)
        assert len(feature_ids) == 2
        landmark_points = {}
        for i in range(msm.number_of_states):
            landmark_points[str(i + 1)] = tuple(msm.state_centres[i, feature_ids])

        self.free_energy_plot(features, feature_nicknames,
                              landmark_points=landmark_points,
                              ax=axs[0], save_data=False,
                              show_fig=False, close_fig=False)
        plt.show()

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

    def compute_cv(self, CV: str, dim: Optional[int] = None, stride: int = 1, **kwargs):
        """

        :param CV:
        :param dim: Number of dimensions to keep.
        :param stride:
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
            if kwargs is None:
                kwargs = self.DM_DEFAULTS
            dm = dfm.DiffusionMap.from_sklearn(**kwargs)
            self.CVs[CV] = dm.fit(trajectory[::stride])
        # TODO: VAMPnets
        t1 = time()
        print(f"Computed CV in {round(t1 - t0, 3)}s.")

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
        if CV in ["PCA", "TICA", "VAMP"]:
            return self.CVs[CV].transform(trajectory[::stride])[:, dim]
        elif CV == "DM":
            return self.CVs[CV].evecs[:, dim]
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

    def _lstsq_traj_with_features(self, traj: np.array) -> np.array:
        feat_traj = self.featurized_traj
        feat_traj = np.c_[np.ones(self.num_frames), feat_traj]
        coeffs, err, _, _ = np.linalg.lstsq(feat_traj, traj, rcond=None)

        return coeffs[1:]

    def create_plumed_metadynamics_script(
            self,
            CVs: list[str],
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
            atom_features = self._check_valid_cvs(CVs)

            # Initialise PLUMED script
            file_name = "./plumed.dat" if filename is None else f"{filename}"
            f = open(file_name, "w")
            output = 'RESTART'
            f.write(output + "\n")

            # Initialise Dihedrals class
            dihedral_features = Dihedrals(
                topology=self.topology,
                dihedrals=self.featurizer.describe(),
                offsets=self.feature_means,
                normalised=normalised,
            )

            # Write PLUMED script
            dihedral_features.write_torsion_labels(file=f)
            dihedral_features.write_transform_labels(file=f)

            # Write CVs to PLUMED script
            for idx, CV in enumerate(CVs):
                # Only write combined label for traditional CVs
                if ":" in CV:
                    cv_type = CV.split(':')[0]
                    dim = int(CV.split(':')[1])
                    dihedral_features.write_combined_label(CV_name=CV,
                                                           CV_coefficients=self.feature_eigenvector(cv_type, dim=dim),
                                                           file=f)

            # Write metadynamics command to PLUMED script
            write_metadynamics_line(
                well_tempered=well_tempered, bias_factor=bias_factor, temperature=temperature, height=gaussian_height,
                pace=gaussian_pace, sigma_list=sigma_list, CVs=CVs, exp_name=exp_name, file=f
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
        Reads molecular system and trajectory data from file
        """
        pdb = load_pdb(location)
        topology = pdb.topology
        traj = load_trajectory(location, topology)
        num_frames = len(traj)
        assert np.abs(num_frames - int(self.duration / self.savefreq)) <= 1, (
            f"duration ({self.duration}) and savefreq ({self.savefreq}) incompatible with number of conformations "
            f"found in trajectory (got {num_frames}, expected {int(self.duration / self.savefreq)})."
        )

        print("Successfully initialised datafiles.")

        return pdb, topology, traj, num_frames

    def __init_biasfiles(self, metad_bias_file):
        """
        Reads Metadynamics bias potential and weights from file
        """
        if metad_bias_file is not None:
            metad_weights, bias_potential_traj = self._load_metad_bias(metad_bias_file)
            assert len(metad_weights) == len(self.traj), (
                f"metadynamics weights (len {len(metad_weights)}) and trajectory (len {len(self.traj)}) must have the"
                f" same length."
            )
            print("Successfully initialised metadynamics bias files.")
        else:
            metad_weights, bias_potential_traj = None, None
            print(
                "No metadynamics bias files supplied; assuming an unbiased trajectory."
            )

        return metad_weights, bias_potential_traj

    def __init_features(self, features: Optional[Union[dict, str]]):
        """
        Featurises the trajectory according to specified features.
        Features can be an explicit dictionary of dihedrals, such as {'\phi' : [4, 6, 8 ,14], '\psi' : [6, 8, 14, 16]}.
        Alternatively, you can specify the automatic inclusion of *all* backbone dihedrals by setting
        features = "dihedrals".
        """
        # TODO: check feature names are preserved correctly when adding through dictionary
        if features:
            featurizer = initialise_featurizer(features, self.topology)
            featurized_traj = featurizer.transform(self.traj)
            feature_means = np.mean(featurized_traj, axis=0)
            feature_stds = np.std(featurized_traj, axis=0)
            num_features = len(featurizer.describe())
            features_provided = True
            print(f"Successfully featurized trajectory with {num_features} features.")
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
                "No features provided; trajectory defined with cartesian coordinates (not recommended)."
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
        Checks that the CVs provided are valid (either a traditional CV or a single feature)
        """
        atom_features = [False for _ in range(len(CVs))]
        for idx, CV in enumerate(CVs):
            if ":" in CV:
                cv_type = CV.split(':')[0]
            else:
                cv_type = CV
            atom_features[idx] = False
            # Check if the CV is an atom feature (e.g. dihedral angle)
            if cv_type in self.CVs.keys():
                atom_features[idx] = False
            elif cv_type in self.featurizer.describe():
                atom_features[idx] = True
            # If the CV is neither an atom feature nor a traditional CV (e.g. TICA, PCA, etc.), raise an error
            else:
                raise ValueError(f"CV '{cv_type}' not in {self.CVs.keys()} or {self.featurizer.describe()}")

            if not atom_features[idx] and self.CVs[cv_type] is None:
                raise ValueError(f"{cv_type} CVs not computed.")

        return atom_features

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

    def _scatter_fes(self, ax, feature_traj: np.array, bins: int, nan_threshold: int):
        free_energy, xedges, yedges = free_energy_estimate_2D(
            ax,
            remove_nans(feature_traj),
            self.beta,
            bins=bins,
        )
        masked_free_energy = np.ma.array(
            free_energy, mask=(free_energy > nan_threshold)  # true indicates a masked (invalid) value
        )

        im = ax.pcolormesh(np.repeat(yedges[..., None], repeats=len(yedges), axis=1),
                           np.repeat(xedges[None, ...], repeats=len(xedges), axis=0), masked_free_energy)

        return ax, im

    def _contour_fes(self, ax, feature_traj: np.array):
        xyz = remove_nans(
            np.hstack([feature_traj, np.array([self.bias_potential_traj]).T])
        )
        x = xyz[:, 0]
        y = xyz[:, 1]
        bias_potential = xyz[:, 2]
        free_energy = -bias_potential + np.max(
            bias_potential
        )  # free energy is negative bias potential
        # set level increment every unit of kT
        num_levels = int(
            np.floor((np.max(free_energy) - np.min(free_energy)) * self.beta)
        )
        levels = [k * 1 / self.beta for k in range(num_levels + 2)]  # todo: why +2?
        im = ax.tricontourf(x, y, free_energy, levels=levels, cmap="RdBu_r")
        ax.tricontour(x, y, free_energy, levels=levels, linewidths=0.5, colors="k")

        return ax, im

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
