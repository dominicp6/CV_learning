#!/usr/bin/env python

"""Defines an Experiment class that enables easy access to the trajectory and metadate stored in a directory.
   Allows systematic calculations (such as applying tICA, VAMP or DF dimensionality reduction) to be easily applied
   to whole or part of the experiment trajectory. Also allows for the automatic creation of PLUMED metadynamics scripts
   based on the trajectory's learnt CVs.

   Author: Dominic Phillips (dominicp6)
"""
import contextlib
import os
import glob
import json
import copy
import math
import psutil
from time import time
from typing import Union, Optional

import dill
import mdtraj as md
import mdtraj  # todo: remove
import networkx as nx
import pandas as pd
import pyemma
import deeptime
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import openmm.unit as unit
from deeptime.util.validation import implied_timescales
from deeptime.plots import plot_implied_timescales
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.tools.analysis import mfpt
from tqdm import tqdm

from KramersRateEvaluator import KramersRateEvaluator
from Dihedrals import Dihedrals
from analine_free_energy import compute_dihedral_trajectory
from utils.diffusion_utils import free_energy_estimate_2D
# import pydiffmap.diffusion_map as dfm
import mdfeature.features as feat
from utils.msm_utils import msm_clustering, msm_discrete_traj, msm_transition_counts, maximum_likelihood_msm, \
    plot_transition_graph
from utils.openmm_utils import parse_quantity, time_to_iteration_conversion
from utils.plotting_functions import init_plot, init_multiplot, save_fig


# TODO: think what is happening to water molecules in the trajectory
#
# def subsample_trajectory(trajectory, stride):
#     traj = md.Trajectory(trajectory.xyz[::stride], trajectory.topology)
#     return traj.superpose(traj[0])
#
#
# def ramachandran_from_x_y_z(x, y, z, rotate, levels=None):
#     fig, ax = plt.subplots()
#     if rotate is True:
#         cntr2 = ax.tricontourf(y, x, z, levels=levels, cmap="RdBu_r")
#         ax.tricontour(y, x, z, levels=levels, linewidths=0.5, colors='k')
#     else:
#         cntr2 = ax.tricontourf(x, y, z, levels=levels, cmap="RdBu_r")
#         ax.tricontour(x, y, z, levels=levels, linewidths=0.5, colors='k')
#     plt.xlabel(r'$\phi$')
#     plt.ylabel(r'$\psi$')
#     plt.gca().set_aspect('equal')
#     cbar = fig.colorbar(cntr2, ax=ax)
#     cbar.set_label(r'$\mathcal{F}(\phi,\psi)$ / kJmol$^{-1}$')
#     ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
#     plt.xticks(np.arange(-3, 4, 1))
#     plt.subplots_adjust(hspace=0.5)
#
def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def remove_nans(data: np.array, axis: int = 1) -> np.array:
    num_nans = np.count_nonzero(np.isnan(data))
    if num_nans > 0:
        axis_str = "rows" if axis == 1 else "columns"
        print(f"{num_nans} NaNs detected, removing {axis_str} with NaNs.")
        data = data[~np.isnan(data).any(axis=1), :]

    return data


def select_file_option(options: list, file_type: str) -> int:
    if len(options) > 1:
        print(f"{len(options)} {file_type} files found in the given directory:")
        for idx, file in enumerate(options):
            print(f"[{idx + 1}] {file}")
        while True:
            selection = input(f"Which {file_type} file do you want to use? ")
            valid_selections = [str(idx + 1) for idx in range(len(options))]
            if selection not in valid_selections:
                print(f"Input not recognised; must be one of {valid_selections}")
            else:
                break
    else:
        selection = '1'

    return int(selection) - 1  # -1 to allow for correct indexing


def check_if_memory_available(file):
    """
    Checks if the system has available memory for loading the requested file.
    """
    file_stat = os.stat(file)
    file_size = file_stat.st_size
    available_memory = psutil.virtual_memory()[1]
    if file_size > 0.9 * available_memory:
        raise MemoryError(f"Loading the file {file} would use more than 90% of "
                          f"the available memory ({round(file_size / 10 ** 9, 1)}/{round(available_memory / 10 ** 9, 1)}Gb).")
    else:
        pass


def load_pdb(loc: str) -> md.Trajectory:
    pdb_files = glob.glob(os.path.join(loc, "*.pdb"))
    assert len(pdb_files) != 0, f"Read error: no PDB files found in directory."
    selection = select_file_option(pdb_files, 'PDB')
    selected_pdb = pdb_files[selection]
    check_if_memory_available(selected_pdb)

    return md.load_pdb(selected_pdb)


def load_trajectory(
        loc: str, topology: Union[str, md.Trajectory, md.Topology]
) -> md.Trajectory:
    traj_files = glob.glob(os.path.join(loc, "*.dcd"))
    assert len(traj_files) != 0, f"Read error: no traj files found in directory."
    selection = select_file_option(traj_files, 'traj')
    selected_traj = traj_files[selection]
    check_if_memory_available(selected_traj)

    return mdtraj.load(selected_traj, top=topology)


def get_metadata_file(
        loc: str, keyword: str
):
    metadata_files = glob.glob(os.path.join(loc, f"*{keyword}*.json"))
    assert len(metadata_files) != 0, f"Read error: no metadata files found in directory."
    selection = select_file_option(metadata_files, "metadata")
    selected_metadata = metadata_files[selection]

    return selected_metadata


def compute_dihedral_traj(loc: str, dihedrals: list[list]) -> list:
    pdb = glob.glob(os.path.join(loc, "*.pdb"))[0]
    traj = glob.glob(os.path.join(loc, "*.dcd"))[0]
    dihedral_traj = np.array(compute_dihedral_trajectory(pdb, traj, dihedrals)).T
    # correcting order
    dihedral_traj[:, [0, 1]] = dihedral_traj[:, [1, 0]]
    return dihedral_traj


def load_dihedral_trajectory(loc: str, dihedral_pickle_file: str) -> list:
    dihedral_traj = np.array(
        dill.load(open(os.path.join(loc, dihedral_pickle_file), "rb"))
    ).T
    # correcting order
    dihedral_traj[:, [0, 1]] = dihedral_traj[:, [1, 0]]
    return dihedral_traj


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


def flip_dihedral_coords(dihedral_traj):
    temp = dihedral_traj[:, 0].copy()
    dihedral_traj[:, 0] = dihedral_traj[:, 1]
    dihedral_traj[:, 1] = temp


def slice_array(data: np.array, quantity: int) -> Union[np.array, None]:
    if data is None:
        return None
    else:
        return data[:quantity]


def get_feature_ids_from_names(
        feature_names: list[str],
        featurizer: pyemma.coordinates.featurizer,
):
    all_features = featurizer.describe()
    feature_ids = []
    for feature in feature_names:
        feature_ids.append(all_features.index(feature))

    return feature_ids


def get_feature_trajs_from_names(
        feature_names: list[str],
        featurized_traj: np.array,
        featurizer: pyemma.coordinates.featurizer,
) -> np.array:
    feature_ids = get_feature_ids_from_names(feature_names, featurizer)

    feature_trajs = []
    for feature_id in feature_ids:
        feature_trajs.append(featurized_traj[:, feature_id])

    return np.array(feature_trajs)


def assert_kwarg(kwargs: dict, kwarg: str, CV: str):
    try:
        assert kwargs[kwarg] is not None
    except Exception:
        raise ValueError(f"{kwarg} must be provided to {CV}")


def write_metadynamics_line(height: float, pace: int, CV: str, file):
    arg_list = []
    sigma_list = []
    arg_list.append(f"{CV}_%d" % 0)
    sigma_list.append(str(0.1))
    output = (
            "METAD ARG=%s SIGMA=%s HEIGHT=%s FILE=HILLS PACE=%s LABEL=metad"
            % (",".join(arg_list), ",".join(sigma_list), str(height), str(pace))
            + " \\n\\"
    )
    print(output)
    file.writelines(output + "\n")
    output = (
            "PRINT ARG=%s,metad.bias STRIDE=%s FILE=COLVAR"
            % (",".join(arg_list), str(pace))
            + " \\n"
    )
    print(output + '"')
    file.writelines(output + '"' + "\n")
    file.close()


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
    ) -> None:
        feature_nicknames = self._check_fes_arguments(features, feature_nicknames)
        fig, ax = init_plot(
            "Free Energy Surface",
            f"${feature_nicknames[0]}$",
            f"${feature_nicknames[1]}$",
        )
        # get traj of features and trim to data fraction
        feature_traj = get_feature_trajs_from_names(
            features, self.featurized_traj, self.featurizer
        )[: int(data_fraction * self.num_frames)]
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
                plt.plot(coordinates[0], coordinates[1], marker='o', markerfacecolor='k')
                plt.annotate(point, coordinates)
        save_fig(fig, save_dir=os.getcwd(), name=save_name)

        return

    def markov_state_model(self, n_clusters: int, lagtime: str, features: list[str],
                           feature_nicknames: Optional[list[str]] = None):
        assert isinstance(lagtime, str), "Lagtime must be a string (e.g. `10ns')."
        samples = self.get_trajectory()
        lagtime = parse_quantity(lagtime)
        lagstep = math.floor(lagtime / self.savefreq)
        if lagstep < 1:
            raise ValueError(
                f"Lagtime provided ({lagtime}) is less than the saving interval of the trajectory ({self.savefreq}).")
        else:
            print(f"Lagtime {lagtime} corresponds to a lag step of {lagstep}.")

        # Clustering
        # TODO: make clustering work for sin, cos features

        clustering = msm_clustering(samples, n_clusters)
        discrete_traj = msm_discrete_traj(clustering, samples)
        counts = msm_transition_counts(discrete_traj, lagstep)
        msm = maximum_likelihood_msm(counts)
        fix, ax = init_plot("MSM State Timescales", "state", f"timescale (x{self.savefreq})")
        ax.plot(msm.timescales())
        plt.show()
        plot_transition_graph(msm, lagstep, self.savefreq)

        print(clustering.cluster_centers)
        feature_ids = get_feature_ids_from_names(features, self.featurizer)
        assert len(feature_ids) == 2
        print(feature_ids)
        landmark_points = {}
        for i in range(msm.n_states):
            landmark_points[str(i + 1)] = tuple(clustering.cluster_centers[i, feature_ids])

        self.free_energy_plot(features, feature_nicknames, landmark_points=landmark_points)

    # TODO: add PCCA+ function for "coarse graining" the MSM
    def contact_analysis(self, contact_threshold: float = 2.0, times: list[str] = None):
        duration_ns = self.duration.in_units_of(unit.nanoseconds)._value

        if times is not None:
            assert len(times) == 3, "Three time snapshots must be provided for contact analysis."
            frames = [time_to_iteration_conversion(t, self.duration, self.num_frames) for t in times]
        else:
            times = ['0.0ns', f'{duration_ns / 2}ns', f'{duration_ns}ns']
            frames = [0, int(len(self.traj) / 2), -1]
        fig, axs = init_multiplot(nrows=3, ncols=3, title='Contact Analysis', panels=[('0', '0'), ('0', '1'),
                                                                                      ('0', '2'), ('1', 'all'),
                                                                                      ('2', 'all')])

        self._plot_contact_matrices(fig, axs, frames, times)
        self._plot_contacts_over_time(axs, contact_threshold, times, duration_ns)
        plt.show()

    def implied_timescale_analysis(self, max_lag: int = 10, increment: int = 1, yscale: str = 'log'):
        """
        Illustrates how TICA implied timescales vary as the lagtime varies.
        For more details on this approach, consult http://docs.markovmodel.org/lecture_implied_timescales.html.
        """
        lagtimes = np.arange(1, max_lag, increment)
        # save current TICA obj
        TICA_obj = copy.deepcopy(self.CVs['TICA'])
        models = []
        for lagtime in tqdm(lagtimes):
            supress_stdout(self.compute_cv)('TICA', lagtime=lagtime)
            models.append(self.CVs['TICA'])
        self.CVs['TICA'] = TICA_obj
        its_data = implied_timescales(models)
        fig, ax = init_plot('Implied timescales (TICA)', 'lag time (steps)', 'timescale (steps)', yscale=yscale)
        plot_implied_timescales(its_data, n_its=2, ax=ax)
        plt.annotate(f"NB: One step corresponds to a time of {self.savefreq}", xy=(90, 1), xycoords='figure points')
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
            self.CVs[CV] = sklearn.decomposition.PCA(n_components=dim, **kwargs).fit(trajectory[::stride])
        elif CV == "TICA":
            assert_kwarg(kwargs, kwarg="lagtime", CV=CV)
            # other kwargs: epsilon, var_cutoff, scaling, observable_transform
            self.CVs[CV] = deeptime.decomposition.TICA(dim=dim, **kwargs).fit_fetch(
                trajectory[::stride]
            )
        elif CV == "VAMP":
            assert_kwarg(kwargs, kwarg="lagtime", CV=CV)
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

    def _get_cv(self, CV, dim):
        assert CV in self.CVs.keys(), f"Method '{CV}' not in {self.CVs.keys()}"
        if self.CVs[CV] is None:
            raise ValueError(f"{CV} CVs not computed.")
        if CV in ["PCA", "VAMP"]:
            return self.CVs[CV].get_output()[0][:, dim]
        elif CV == "TICA":
            return self.CVs[CV].eigenvectors[:, dim]
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
            CV: str,
            filename: str = None,
            gaussian_height: float = 0.2,
            gaussian_pace: int = 1000,
    ):
        if not self.features_provided:
            raise ValueError(
                "Cannot create PLUMED metadynamics script for an unfeaturized trajectory. "
                "Try reinitializing Experiment object with features defined."
            )
        else:
            assert CV in self.CVs.keys(), f"Method '{CV}' not in {self.CVs.keys()}"
            f = open("./plumed.py" if filename is None else f"./{filename}.py", "w")
            output = 'plumed_script="RESTART ' + "\\n\\"
            f.write(output + "\n")
            print(output)
            dihedral_features = Dihedrals(
                dihedrals=self.featurizer.active_features,
                offsets=self.feature_means,
                coefficients=self.feature_eigenvector(CV, dim=0),
            )
            dihedral_features.write_torsion_labels(file=f)
            dihedral_features.write_transform_labels(file=f)
            dihedral_features.write_combined_label(CV=CV, file=f)
            write_metadynamics_line(
                height=gaussian_height, pace=gaussian_pace, CV=CV, file=f
            )

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
                    feature in self.featurizer.describe()
            ), f"Feature '{feature}' not found in available features ({self.featurizer.describe()})"

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
            free_energy, mask=(free_energy > nan_threshold)
        )
        im = ax.pcolormesh(xedges, yedges, masked_free_energy)

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
        cbar_x_axis_pos = [0.275, 0.602, 0.93]
        for idx, cbar_pos in enumerate(cbar_x_axis_pos):
            im = axs[idx].imshow(mdtraj.geometry.squareform(contact_data[idx][0], contact_data[idx][1])[0],
                                 vmin=min_dist, vmax=max_dist)
            axs[idx].set_title(f'time = {times[idx]}')
            axs[idx].set_xlabel('Res Number')
            axs[idx].set_ylabel('Res Number')
            cbar_ax = fig.add_axes([cbar_pos, 0.66, 0.02, 0.25])
            fig.colorbar(im, cax=cbar_ax)

    def _plot_contacts_over_time(self, axs, contact_threshold: float, times: list[str], duration_ns: float):
        distances, pairs = mdtraj.compute_contacts(self.traj)
        number_of_close_contacts = np.sum((distances < contact_threshold), axis=1)  # sum along the columns (contacts)
        rms_dist = np.sqrt(np.mean(distances ** 2, axis=1))

        # plot of number of contacts in the trajectory as a function of time
        x_var = np.arange(0, duration_ns, duration_ns / self.num_frames)
        axs[3].plot(x_var, number_of_close_contacts, linewidth=0.5)
        [axs[3].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[3].set_xlabel('time (ns)')
        axs[3].set_ylabel(f'#contacts < {contact_threshold} $nm$')

        axs[4].plot(x_var, rms_dist, linewidth=0.5)
        [axs[4].axvline(x=parse_quantity(t)._value, color='r') for t in times]
        axs[4].set_xlabel('time (ns)')
        axs[4].set_ylabel(f'RMSD Res-Res ($nm$)')
