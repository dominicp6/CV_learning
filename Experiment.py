#!/usr/bin/env python

"""Defines an Experiment class that enables easy access to the trajectory and metadate stored in a directory.
   Allows systematic calculations (such as applying tICA, VAMP or DF dimensionality reduction) to be easily applied
   to whole or part of the experiment trajectory. Also allows for the automatic creation of PLUMED metadynamics scripts
   based on the trajectory's learnt CVs.

   Author: Dominic Phillips (dominicp6)
"""

import os
import glob
import json
from time import time
from typing import Union

import dill
import mdtraj as md
import mdtraj  # todo: remove
import pandas as pd
import pyemma
import deeptime
import numpy as np
import matplotlib.pyplot as plt

from KramersRateEvaluator import KramersRateEvaluator
from Dihedrals import Dihedrals
from analine_free_energy import compute_dihedral_trajectory
from utils.diffusion_utils import free_energy_estimate_2D
import pydiffmap.diffusion_map as dfm
import mdfeature.features as feat
from utils.openmm_utils import parse_quantity
from utils.plotting_functions import init_plot


#
# def free_energy_estimate(samples, beta, minimum_counts=50, bins=200):
#     # histogram
#     counts, coordinate = np.histogram(samples, bins=bins)
#     robust_counts = counts[np.where(counts > minimum_counts)]
#     robust_coordinates = coordinate[np.where(counts > minimum_counts)]
#
#     # log normal
#     normalised_counts = robust_counts / np.sum(counts)
#     with np.errstate(divide='ignore'):
#         free_energy = - (1 / beta) * np.log(normalised_counts)
#
#     return free_energy, robust_coordinates
#
# def parse_quantity(s):
#     try:
#         u = s.lstrip('0123456789.')
#         v = s[:-len(u)]
#         return unit.Quantity(
#             float(v),
#             unit_labels[u]
#         )
#     except Exception:
#         raise ValueError(f"Invalid quantity: {s}")
#
#
# def subsample_trajectory(trajectory, stride):
#     traj = md.Trajectory(trajectory.xyz[::stride], trajectory.topology)
#     return traj.superpose(traj[0])
#
#
def check_and_remove_nans(data: np.array, axis: int = 1) -> np.array:
    num_nans = np.count_nonzero(np.isnan(data))
    if num_nans > 0:
        axis_str = "rows" if axis == 1 else "columns"
        print(f"{num_nans} NaNs detected, removing {axis_str} with NaNs.")
        data = data[~np.isnan(data).any(axis=1), :]

    return data


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
#
# def ramachandran(exp, points_of_interest, rotate=True, save_name=None):
#     xyz = check_and_remove_nans(np.hstack([exp.dihedral_traj, np.array([exp.bias_potential_traj]).T]))
#     x = xyz[:, 0]
#     y = xyz[:, 1]
#     z = xyz[:, 2]
#     fe = -z + np.max(z)
#     print(np.min(fe), np.max(fe))
#     # set level increment every unit of kT
#     num_levels = int(np.floor((np.max(fe) - np.min(fe)) / 2.5))
#     levels = [k * 2.5 for k in range(num_levels + 2)]
#     ramachandran_from_x_y_z(x, y, fe, rotate, levels)
#     for point in points_of_interest:
#         plt.text(s=point[0], x=point[1] + 0.12, y=point[2] - 0.1)
#         plt.scatter(point[1], point[2], c='k')
#     # plt.savefig(save_name, format="pdf", bbox_inches="tight")
#     return x, y, -z
#
#
# def ramachandran2(exp, points_of_interest, rotate=True, save_name=None, nan_threshold=50, low_threshold=0,
#                   data_fraction=1, bins=300):
#     dihedral_traj = check_and_remove_nans(exp.dihedral_traj)
#     final_iteration = int(data_fraction * len(dihedral_traj))
#     free_energy, xedges, yedges = free_energy_estimate_2D(dihedral_traj[:final_iteration], exp.beta, bins=bins,
#                                                           weights=exp._slice(exp.metad_weights, final_iteration) / max(
#                                                               exp.metad_weights))
#     fig, ax = plt.subplots()
#     if rotate is True:
#         free_energy = free_energy.T
#     if low_threshold is None:
#         masked_free_energy = np.ma.array(free_energy, mask=(free_energy > nan_threshold))
#     else:
#         masked_free_energy = np.ma.array(free_energy, mask=(
#             np.logical_or((free_energy > nan_threshold), (free_energy < low_threshold))))
#     im = ax.pcolormesh(xedges, yedges, masked_free_energy)
#     cbar = plt.colorbar(im)
#     cbar.set_label(r'$\mathcal{F}(\phi,\psi)$ / kJmol$^{-1}$')
#     plt.xticks(np.arange(-3, 4, 1))
#     plt.xlabel(r'$\phi$')
#     plt.ylabel(r'$\psi$')
#     plt.gca().set_aspect('equal')
#     for point in points_of_interest:
#         plt.text(s=point[0], x=point[1] + 0.12, y=point[2] - 0.1)
#         plt.scatter(point[1], point[2], c='k')
#     plt.savefig(save_name, format="pdf", bbox_inches="tight")
#
#
# def ramachandran_from_file(file, rotate=True):
#     A = np.genfromtxt(file, delimiter=' ')
#     A[A == np.inf] = 50
#     num_levels = int(np.floor((np.max(A[:,2])-np.min(A[:,2])) / 2.5))
#     levels = [k * 2.5 for k in range(num_levels+2)]
#     ramachandran_from_x_y_z(A[:,0],A[:,1],A[:,2]-np.min(A[:,2]),rotate=rotate,levels=levels)
#     print(A.shape)
#     print(A)


def load_pdb(loc: str) -> md.Trajectory:
    pdb_files = glob.glob(os.path.join(loc, "*.pdb"))
    assert (
        len(pdb_files) <= 1
    ), f"Read error: more than one PDB file found in the directory ({pdb_files})."
    assert len(pdb_files) != 0, f"Read error: no PDB files found in directory."
    return md.load_pdb(pdb_files[0])


def load_trajectory(
    loc: str, topology: Union[str, md.Trajectory, md.Topology]
) -> md.Trajectory:
    traj_files = glob.glob(os.path.join(loc, "*.dcd"))
    assert (
        len(traj_files) <= 1
    ), f"Read error: more than one traj file found in the directory ({traj_files})."
    assert len(traj_files) != 0, f"Read error: no traj files found in directory."
    return mdtraj.load(traj_files[0], top=topology)


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


def initialise_featurizer(
    dihedral_features: list, topology
) -> pyemma.coordinates.featurizer:
    # To check
    featurizer = pyemma.coordinates.featurizer(topology)
    dihedral_indices = feat.create_torsions_list(
        atoms=topology.n_atoms,
        size=0,
        append_to=dihedral_features,
        print_list=False,
    )
    featurizer.add_dihedrals(dihedral_indices)  # cossin = True
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
        feature_dict: dict = None,
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
        ) = self.__init_features(feature_dict)

        self.CVs = {"PCA": None, "TICA": None, "VAMP": None, "DM": None}
        self.kre = KramersRateEvaluator()
        self.discrete_traj = None

    def free_energy_plot(
        self,
        features: tuple[str],
        data_fraction: float = 1.0,
        bins: int = 100,
        nan_threshold: int = 50,
        save_name="free_energy_plot.pdf",
    ):
        if not self.features_provided:
            raise ValueError(
                "Cannot construct ramachandran plot for an unfeaturized trajectory. "
                "Try reinitializing Experiment object with features defined."
            )
        else:
            assert (
                len(features) == 2
            ), "Only two features can be used for a free energy plot."
            fig, ax = init_plot(
                "Free Energy Surface", f"${features[0]}$", f"${features[1]}$"
            )
            free_energy, xedges, yedges = free_energy_estimate_2D(
                ax,
                self.featurized_traj[: int(data_fraction * self.num_frames)],
                features,
                self.feature_dict,
                self.beta,
                bins=bins,
            )
            masked_free_energy = np.ma.array(
                free_energy, mask=(free_energy > nan_threshold)
            )
            im = ax.pcolormesh(xedges, yedges, masked_free_energy)
            cbar = plt.colorbar(im)
            cbar.set_label(f"$F({features[0]},{features[1]})$ / kJ mol$^{{-1}}$")
            plt.gca().set_aspect("equal")

            return None

    def implied_timescale_analysis(self, max_lag: int = 10, k: int = 10):
        if self.discrete_traj is None:
            cluster = pyemma.coordinates.cluster_kmeans(self.featurized_traj, k=k)
            self.discrete_traj = cluster.dtrajs[0]
        its = pyemma.msm.its(self.discrete_traj, lags=max_lag)
        pyemma.plots.plot_implied_timescales(its)

    def compute_cv(self, CV: str, dim: int, stride: int = 1, **kwargs):
        assert CV in self.CVs.keys(), f"Method '{CV}' not in {self.CVs.keys()}"
        # TODO: implement stride
        t0 = time()
        # Trajectory is either featurized or unfeaturized (cartesian coords), depending on object initialisation.
        trajectory = self.featurized_traj if self.features_provided else self.traj
        if CV == "PCA":
            self.CVs[CV] = pyemma.coordinates.pca(
                trajectory.xyz, dim=dim, stride=stride
            )
        elif CV == "TICA":
            assert_kwarg(kwargs, kwarg="lagtime", CV=CV)
            # other kwargs: epsilon, var_cutoff, scaling, observable_transform
            self.CVs[CV] = deeptime.decomposition.TICA(dim=dim, **kwargs).fit_fetch(
                trajectory.xyz
            )
        elif CV == "VAMP":
            assert_kwarg(kwargs, kwarg="lagtime", CV=CV)
            # other kwargs: epsilon, var_cutoff, scaling, epsilon, observable_transform
            self.CVs[CV] = deeptime.decomposition.VAMP(dim=dim, **kwargs).fit_fetch(
                trajectory.xyz
            )
        elif CV == "DMD":
            # other kwargs: mode, rank, exact
            self.CVs[CV] = deeptime.decomposition.DMD(**kwargs).fit_fetch(
                trajectory.xyz
            )
        elif CV == "DM":
            if kwargs is None:
                kwargs = self.DM_DEFAULTS
            dm = dfm.DiffusionMap.from_sklearn(**kwargs)
            self.CVs[CV] = dm.fit(trajectory[::stride])
        t1 = time()
        print(f"Computed CV in {round(t1 - t0, 3)}s.")

    def analyse_kramers_rate(
        self, CV: str, dimension: int, lag: int, sigmaD: float, sigmaF: float
    ):
        self.kre.fit(
            self._get_cv(CV, dimension),
            beta=self.beta,
            time_step=self.stepsize.value,
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
    def _init_metadata(location: str, metadata_file="metadata.json"):
        """
        Reads experiment metadata from file
        """
        with open(os.path.join(location, metadata_file)) as metadata_f:
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

    def __init_features(self, features: Union[dict, None]):
        """
        Featurises the trajectory according to specified features
        """
        if features:
            featurizer = initialise_featurizer(list(features.values()), self.topology)
            featurized_traj = featurizer.transform(self.traj)
            feature_means = np.mean(featurized_traj, axis=0)
            feature_stds = np.std(featurized_traj, axis=0)
            num_features = len(features)
            features_provided = True
            print("Successfully featurized trajectory.")
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
