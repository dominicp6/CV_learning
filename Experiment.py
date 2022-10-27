#!/usr/bin/env python

"""Defines an Experiment class that enables easy access to the trajectory and metadate stored in a directory.
   Allows systematic calculations (such as applying tICA, VAMP or DF dimensionality reduction) to be easily applied
   to whole or part of the experiment trajectory. Also allows for the automatic creation of PLUMED metadynamics scripts
   based on the trajectory's learnt CVs.

   Author: Dominic Phillips (dominicp6)
"""

import os, glob
from time import time
from typing import Union

import dill
import mdtraj as md
import mdtraj  # todo: remove
import pandas as pd
import pyemma
import numpy as np
import matplotlib.pyplot as plt
import openmm.unit as unit

from KramersRateEvaluator import KramersRateEvaluator
from Dihedrals import Dihedrals
from analine_free_energy import compute_dihedral_trajectory
from diffusion_utils import free_energy_estimate_2D
import diffusion_map.pydiffmap_weights.diffusion_map as dfm
import mdfeature.features as features


unit_labels = {
    "us": unit.microseconds,
    "ns": unit.nanoseconds,
    "ps": unit.picoseconds,
    "fs": unit.femtoseconds,
}
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


def parse_quantity(s: str):
    try:
        u = s.lstrip("0123456789.")
        v = s[: -len(u)]
        return unit.Quantity(float(v), unit_labels[u])
    except Exception:
        raise ValueError(f"Invalid quantity: {s}")


class Experiment:

    # TODO: fix bug when default torsions = None

    def __init__(
        self,
        location: str,
        temperature: float,
        duration: str,
        savefreq: str,
        stepsize: str,
        default_torsions: list = None,
        dihedral_pickle_file: str = None,
        dihedrals: list[list] = None,
        metad_bias_file=None,
    ):
        self.temperature = temperature
        self.duration = parse_quantity(duration)
        self.savefreq = parse_quantity(savefreq)
        self.stepsize = parse_quantity(stepsize)
        self.iterations = int(self.duration / self.stepsize)
        self.beta = 1 / (self.temperature * 0.0083144621)
        print("Successfully initialised experiment metadata.")

        self.pdb = self.load_pdb(location)
        self.topology = self.pdb.topology
        self.trajectory = self.load_trajectory(location, self.topology)
        if dihedral_pickle_file is not None:
            self.dihedral_traj = self.load_dihedral_trajectory(
                location, dihedral_pickle_file
            )
        else:
            self.dihedral_traj = self._compute_dihedral_traj(location, dihedrals)
        assert len(self.trajectory) > 0, "Trajectory is empty."
        assert len(self.dihedral_traj) > 0, "Dihedral trajectory is empty."
        if metad_bias_file is not None:
            self.metad_weights, self.bias_potential_traj = self._load_metad_bias(
                metad_bias_file
            )
            assert len(self.metad_weights) == len(
                self.trajectory
            ), f"metadynamics weights (len {len(self.metad_weights)}) and trajectory (len {len(self.trajectory)}) must have the same length."
        else:
            self.metad_weights = None
        self.conformations = len(self.trajectory)
        assert (
            np.abs(self.conformations - int(self.duration / self.savefreq)) <= 1
        ), f"duration ({duration}) and savefreq ({savefreq}) incompatible with number of conformations found in trajectory (got {self.conformations}, expected {int(self.duration / self.savefreq)})."
        print("Successfully loaded experiment data.")

        self.featurizer = self.initialise_featurizer(default_torsions)
        self.featurized_trajectory = self.featurizer.transform(self.trajectory)
        self.mean_features = np.mean(self.featurized_trajectory, axis=0)
        self.fluctuations_features = np.std(self.featurized_trajectory, axis=0)
        print("Successfully featurized trajectory.")

        self.PCA = None
        self.TICA = None
        self.VAMP = None
        self.kre = KramersRateEvaluator()
        self.kre_params = {
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
        self.dmap = dfm.DiffusionMap.from_sklearn(
            epsilon="bgh",
            alpha=0.5,
            k=64,
            kernel_type="gaussian",
            n_evecs=5,
            neighbor_params=None,
            metric="euclidean",
            metric_params=None,
            weight_fxn=None,
            density_fxn=None,
            bandwidth_type="-1/(d+2)",
            bandwidth_normalize=False,
            oos="nystroem",
        )
        self.discrete_traj = None

    def flip_dihedral_coords(self):
        print(self.dihedral_traj.shape)
        temp = self.dihedral_traj[:, 0].copy()
        self.dihedral_traj[:, 0] = self.dihedral_traj[:, 1]
        self.dihedral_traj[:, 1] = temp
        print(self.dihedral_traj.shape)

    @staticmethod
    def _compute_dihedral_traj(loc: str, dihedrals: list[list]) -> list:
        pdb = glob.glob(os.path.join(loc, "*.pdb"))[0]
        traj = glob.glob(os.path.join(loc, "*.dcd"))[0]
        dihedral_traj = np.array(compute_dihedral_trajectory(pdb, traj, dihedrals)).T
        # correcting order
        dihedral_traj[:, [0, 1]] = dihedral_traj[:, [1, 0]]
        return dihedral_traj

    def _load_metad_bias(self, bias_file: str, col_idx: int = 2) -> (list, list):
        colvar = np.genfromtxt(bias_file, delimiter=" ")
        assert (
            col_idx < colvar.shape[1]
        ), "col_idx must not exceed 1 less than the number of columns in the bias file"
        bias_potential_traj = colvar[:, col_idx]  # [::500][:-1]
        weights = np.exp(self.beta * bias_potential_traj)
        return weights, bias_potential_traj

    @staticmethod
    def _slice(data: list, quantity: int) -> list:
        if data is None:
            return None
        else:
            print(data[:quantity])
            return data[:quantity]

    @staticmethod
    def load_pdb(loc: str) -> md.Trajectory:
        print(glob.glob(os.path.join(loc)))
        pdb_files = glob.glob(os.path.join(loc, "*.pdb"))
        assert (
            len(pdb_files) <= 1
        ), f"Read error: more than one PDB file found in the directory ({pdb_files})."
        assert len(pdb_files) != 0, f"Read error: no PDB files found in directory."
        return md.load_pdb(pdb_files[0])

    @staticmethod
    def load_trajectory(
        loc: str, topology: Union[str, md.Trajectory, md.Topology]
    ) -> md.Trajectory:
        traj_files = glob.glob(os.path.join(loc, "*.dcd"))
        assert (
            len(traj_files) <= 1
        ), f"Read error: more than one traj file found in the directory ({traj_files})."
        assert len(traj_files) != 0, f"Read error: no traj files found in directory."
        return mdtraj.load(traj_files[0], top=topology)

    @staticmethod
    def load_dihedral_trajectory(loc: str, dihedral_pickle_file: str) -> list:
        dihedral_traj = np.array(
            dill.load(open(os.path.join(loc, dihedral_pickle_file), "rb"))
        ).T
        # correcting order
        dihedral_traj[:, [0, 1]] = dihedral_traj[:, [1, 0]]
        return dihedral_traj

    def initialise_featurizer(
        self, default_torsions: list
    ) -> pyemma.coordinates.featurizer:
        # To check
        featurizer = pyemma.coordinates.featurizer(self.topology)
        dihedral_indices = features.create_torsions_list(
            atoms=self.topology.n_atoms,
            size=0,
            append_to=default_torsions,
            print_list=False,
        )
        featurizer.add_dihedrals(dihedral_indices, cossin=True)
        featurizer.describe()
        return featurizer

    def ramachandran_plot(
        self,
        data_fraction: float = 1.0,
        bins: int = 100,
        nan_threshold: int = 50,
        rotate: bool = False,
        low_threshold: float = None,
        save_fig: bool = False,
        save_name="ramachandran_plot.pdf",
    ):
        if self.metad_weights is None:
            dihedral_traj = check_and_remove_nans(self.dihedral_traj)
            final_iteration = int(data_fraction * len(dihedral_traj))
            free_energy, xedges, yedges = free_energy_estimate_2D(
                dihedral_traj[:final_iteration],
                self.beta,
                bins=bins,
                weights=self._slice(self.metad_weights, final_iteration),
            )
            fig, ax = plt.subplots()
            if rotate is True:
                free_energy = free_energy.T
            if low_threshold is None:
                masked_free_energy = np.ma.array(
                    free_energy, mask=(free_energy > nan_threshold)
                )
            else:
                masked_free_energy = np.ma.array(
                    free_energy,
                    mask=(
                        np.logical_or(
                            (free_energy > nan_threshold), (free_energy < low_threshold)
                        )
                    ),
                )
            im = ax.pcolormesh(xedges, yedges, masked_free_energy)
            cbar = plt.colorbar(im)
            cbar.set_label(r"$\mathcal{F}(\phi,\psi)$ / kJmol$^{-1}$")
            plt.xticks(np.arange(-3, 4, 1))
            plt.xlabel(r"$\phi$")
            plt.ylabel(r"$\psi$")
            plt.gca().set_aspect("equal")
        else:
            xyz = check_and_remove_nans(
                np.hstack([self.dihedral_traj, np.array([self.bias_potential_traj]).T])
            )
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            fe = -z + np.max(z)
            fig, ax = plt.subplots()
            # set level increment every unit of kT
            num_levels = int(np.floor((np.max(fe) - np.min(fe)) / 2.5))
            levels = [k * 2.5 for k in range(num_levels + 2)]
            if rotate is True:
                cntr2 = ax.tricontourf(
                    y, x, -z + np.max(z), levels=levels, cmap="RdBu_r"
                )
                ax.tricontour(y, x, -z, levels=levels, linewidths=0.5, colors="k")
            else:
                cntr2 = ax.tricontourf(
                    x, y, -z + np.max(z), levels=levels, cmap="RdBu_r"
                )
                ax.tricontour(x, y, -z, levels=levels, linewidths=0.5, colors="k")
            plt.xlabel(r"$\phi$")
            plt.ylabel(r"$\psi$")
            plt.gca().set_aspect("equal")
            cbar = fig.colorbar(cntr2, ax=ax)
            cbar.set_label(r"$\mathcal{F}(\phi,\psi)$ / kJmol$^{-1}$")
            ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
            plt.xticks(np.arange(-3, 4, 1))
            plt.subplots_adjust(hspace=0.5)

        if save_fig:
            plt.savefig(save_name, format="pdf", bbox_inches="tight")
        plt.show()

        return masked_free_energy, pd.DataFrame(
            np.vstack(
                [
                    dihedral_traj,
                ]
            ),
            columns=["phi", "psi", "weight"],
        )

    def implied_timescale_analysis(self, max_lag: int = 10, k: int = 10):
        if self.discrete_traj is None:
            cluster = pyemma.coordinates.cluster_kmeans(self.featurized_trajectory, k=k)
            self.discrete_traj = cluster.dtrajs[0]
        its = pyemma.msm.its(self.discrete_traj, lags=max_lag)
        pyemma.plots.plot_implied_timescales(its)

    def compute_PCA(self, dim: int, stride: int = 1, featurized: bool = True):
        t0 = time()
        trajectory = self.featurized_trajectory if featurized else self.trajectory
        self.PCA = pyemma.coordinates.pca(trajectory, dim=dim, stride=stride)
        t1 = time()
        print(f"Computed PCA in {round(t1 - t0, 3)}s.")

    def compute_TICA(
        self,
        dim: int,
        lag: int,
        stride: int = 1,
        featurized: bool = True,
        kinetic_map: bool = True,
    ):
        t0 = time()
        trajectory = self.featurized_trajectory if featurized else self.trajectory
        self.TICA = pyemma.coordinates.tica(
            trajectory, lag=lag, dim=dim, stride=stride, kinetic_map=kinetic_map
        )
        t1 = time()
        print(f"Computed TICA in {round(t1 - t0, 3)}s.")

    def compute_VAMP(
        self, dim: int, lag: int, stride: int = 1, featurized: bool = True
    ):
        t0 = time()
        trajectory = self.featurized_trajectory if featurized else self.trajectory
        self.VAMP = pyemma.coordinates.vamp(trajectory, lag=lag, dim=dim, stride=stride)
        t1 = time()
        print(f"Computed VAMP in {round(t1 - t0, 3)}s.")

    def compute_DMAP(self, stride: int, featurized: bool = True):
        t0 = time()
        trajectory = self.featurized_trajectory if featurized else self.trajectory
        self.DMAP = self.dmap.fit(trajectory[::stride])
        t1 = time()
        print(f"Computed DMAP in {round(t1 - t0, 3)}s.")

    def analyse_PCA(self, dimension: int, lag: int, sigmaD: float, sigmaF: float):
        assert self.PCA is not None, "Run compute_PCA before analyse_PCA."
        self.kre.fit(
            self.PCA.get_output()[0][:, dimension],
            beta=self.beta,
            time_step=self.stepsize,
            lag=lag,
            sigmaD=sigmaD,
            sigmaF=sigmaF,
            **self.kre_params,
        )

    def analyse_TICA(self, dimension: int, lag: int, sigmaD: float, sigmaF: float):
        assert self.TICA is not None, "Run compute_TICA before analyse_TICA."
        self.kre.fit(
            self.TICA.get_output()[0][:, dimension],
            beta=self.beta,
            time_step=self.stepsize,
            lag=lag,
            sigmaD=sigmaD,
            sigmaF=sigmaF,
            **self.kre_params,
        )

    def analyse_VAMP(self, dimension: int, lag: int, sigmaD: float, sigmaF: float):
        assert self.VAMP is not None, "Run compute_VAMP before analyse_VAMP."
        self.kre.fit(
            self.VAMP.get_output()[0][:, dimension],
            beta=self.beta,
            time_step=self.stepsize,
            lag=lag,
            sigmaD=sigmaD,
            sigmaF=sigmaF,
            **self.kre_params,
        )

    def eigenvector(self, CV: str, dim: int) -> np.array:
        self._assert_valid_cv(CV)
        x = None
        if CV == "PCA":
            x = self.PCA.get_output()[0][:, dim]
            print(x)
            print(x.shape)
        elif CV == "VAMP":
            x = self.VAMP.get_output()[0][:, dim]
        elif CV == "DMAP":
            x = self.DMAP.evecs[:, dim]
            print(x)
            print(x.shape)
        elif CV == "TICA":
            return self.TICA.eigenvectors[:, dim]
        return self._lstsq_traj_with_features(traj=x)

    def _lstsq_traj_with_features(self, traj: np.array) -> np.array:
        a = self.featurized_trajectory
        a = np.c_[np.ones(self.conformations), a]
        print("a_shape", a.shape)
        c, err, _, _ = np.linalg.lstsq(a[::2, :], traj, rcond=None)  # TODO: revert back

        return c[1:]

    def _assert_valid_cv(self, CV: str):
        assert CV in [
            "PCA",
            "TICA",
            "VAMP",
            "DMAP",
        ], "CV must be one of PCA, TICA, VAMP, DMAP."
        if CV == "PCA":
            assert self.PCA is not None, "Run compute_PCA first."
        elif CV == "TICA":
            assert self.TICA is not None, "Run compute_TICA first."
        elif CV == "VAMP":
            assert self.VAMP is not None, "Run compute_VAMP first."
        elif CV == "DMAP":
            assert self.DMAP is not None, "Run compute_DMAP first."

    def create_plumed_metadynamics_script(
        self,
        CV: str,
        filename: str = None,
        gaussian_height: float = 0.2,
        gaussian_pace: int = 1000,
    ):
        self._assert_valid_cv(CV)
        f = open("./plumed.py" if filename is None else f"./{filename}.py", "w")
        output = 'plumed_script="RESTART ' + "\\n\\"
        f.write(output + "\n")
        print(output)
        dihedral_features = Dihedrals(
            dihedrals=self.featurizer.active_features,
            offsets=self.mean_features,
            coefficients=self.eigenvector(CV, dim=0),
        )
        dihedral_features.write_torsion_labels(file=f)
        dihedral_features.write_transform_labels(file=f)
        dihedral_features.write_combined_label(CV=CV, file=f)
        self._write_metadynamics_line(
            height=gaussian_height, pace=gaussian_pace, CV=CV, file=f
        )

    def _write_metadynamics_line(self, height: int, pace: int, CV: str, file: str):
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
