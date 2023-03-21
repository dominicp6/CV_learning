#!/usr/bin/env python

"""Utilities for working with chemicals arrays.

   Author: Dominic Phillips (dominicp6)
"""

import os
import glob
import subprocess
from collections import namedtuple

import mdtraj
import pyemma
import openmm.unit as unit
from typing import Union
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline, griddata

import dill
import numpy as np

from utils.general_utils import select_file_option, check_if_memory_available, remove_nans
from Dihedrals import compute_dihedral_label
from utils.feature_utils import get_cv_type_and_dim
from utils.diffusion_utils import free_energy_estimate_2D


BiasTrajectory = namedtuple("BiasTrajectory", "feat1 feat2 free_energy")

# TODO: possibly replace with mdtraj featurizer
def initialise_featurizer(
        features: Union[dict, list[str], np.array], topology, cos_sin=False
) -> pyemma.coordinates.featurizer:
    featurizer = pyemma.coordinates.featurizer(topology)
    # If features is a dictionary, then we need the specified dihedral features
    if isinstance(features, dict):
        dihedral_indices = np.array(features.values())
        featurizer.add_dihedrals(dihedral_indices, cossin=cos_sin)
    # If features is a list of str, then we add all features of the specified type(s)
    elif isinstance(features, list):
        if "dihedrals" in features:
            # Add all backbone dihedrals
            featurizer.add_backbone_torsions(cossin=cos_sin)
        if "carbon_alpha" in features:
            # Add the distances between all carbon alpha atoms to the featurizer
            if len(featurizer.select_Ca()) < 2:
                print(f"WARNING: Not enough carbon alpha atoms found in topology to form carbon_alpha features "
                      f"(needed at least 2, found {len(featurizer.select_Ca())}).")
            else:
                featurizer.add_distances_ca()
        if "sidechain_torsions" in features:
            # Add all sidechain torsions
            try:
                featurizer.add_sidechain_torsions(cossin=cos_sin)
            except ValueError:
                print("WARNING: No sidechain torsions found in topology.")
        if any([feat not in ["dihedrals", "carbon_alpha", "sidechain_torsions"] for feat in features]):
            raise ValueError("Unrecognised feature type(s) provided. Allowed types are: dihedrals, carbon_alpha, "
                             "sidechain_torsions.")
    elif isinstance(features, np.ndarray):
        # If features is a numpy array, then we assume it is a list of dihedral indices
        featurizer.add_dihedrals(features, cossin=cos_sin)
    else:
        raise ValueError(
            f"Invalid value for dihedral features: '{features}'. "
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


def load_pdb(loc: str) -> mdtraj.Trajectory:
    pdb_files = glob.glob(os.path.join(loc, "*.pdb"))
    assert len(pdb_files) != 0, f"Read error: no PDB files (*.pdb) found in directory."
    selection = select_file_option(pdb_files, 'PDB')
    selected_pdb = pdb_files[selection]
    check_if_memory_available(selected_pdb)

    return mdtraj.load_pdb(selected_pdb)


def load_trajectory(
        loc: str, topology: Union[str, mdtraj.Trajectory, mdtraj.Topology]
) -> mdtraj.Trajectory:
    traj_files = glob.glob(os.path.join(loc, "*.dcd"))
    assert len(traj_files) != 0, f"Read error: no traj files (*.dcd) found in directory."
    selection = select_file_option(traj_files, 'traj')
    selected_traj = traj_files[selection]
    check_if_memory_available(selected_traj)

    return mdtraj.load(selected_traj, top=topology)


def get_metadata_file(
        loc: str, keyword: str
):
    metadata_files = glob.glob(os.path.join(loc, f"*{keyword}*.json"))
    assert len(metadata_files) != 0, f"Read error: no metadata files ({keyword}*.json) found in directory."
    selection = select_file_option(metadata_files, "metadata")
    selected_metadata = metadata_files[selection]

    return selected_metadata


def load_dihedral_trajectory(loc: str, dihedral_pickle_file: str) -> list:
    dihedral_traj = np.array(
        dill.load(open(os.path.join(loc, dihedral_pickle_file), "rb"))
    ).T
    # correcting order
    dihedral_traj[:, [0, 1]] = dihedral_traj[:, [1, 0]]
    return dihedral_traj


def write_metadynamics_line(well_tempered: bool,
                            bias_factor: float,
                            temperature: float,
                            height: float,
                            pace: int,
                            sigma_list: list[float],
                            CVs: list[str],
                            exp_name: str,
                            file,
                            top: mdtraj.Topology,):
    arg_list = []
    sigma_list = [str(sigma) for sigma in sigma_list]

    for CV in CVs:
        traditional_cv, cv_type, cv_dim = get_cv_type_and_dim(CV)
        if not traditional_cv:
            arg_list.append(compute_dihedral_label(top, CV))
        else:
            arg_list.append(CV)

    if well_tempered:
        output = (
                f"METAD ARG={','.join(arg_list)} SIGMA={','.join(sigma_list)} HEIGHT={height} "
                f"BIASFACTOR={bias_factor} TEMP={temperature} FILE=HILLS "
                f"PACE={pace} LABEL=metad "
        )
        file.writelines(output + "\n")
    else:
        output = (
                f"METAD ARG={','.join(arg_list)} SIGMA={','.join(sigma_list)} HEIGHT={height} FILE=HILLS "
                f"PACE={pace} LABEL=metad "
        )
        file.writelines(output + "\n")

    output = (
        f"PRINT ARG={','.join(arg_list)},metad.bias STRIDE={pace} FILE=COLVAR "
    )
    file.writelines(output + "\n")
    file.close()

def get_fe_trajs(data, reweight=False):
    """
    Get feature and free energy trajectories from the data array.
    """
    # Different column orders for reweighted and non-reweighted data
    if reweight:
        delta_idx = 1
    else:
        delta_idx = 0
    feature1_traj = data[:, 0+delta_idx]
    feature2_traj = data[:, 1+delta_idx]
    fe = data[:, 2+delta_idx]
    fe = fe - np.min(fe)
    return feature1_traj, feature2_traj, fe


def scatter_fes(beta: unit.Quantity, ax, feature_traj: np.array, bins: int, nan_threshold: int):
    free_energy, xedges, yedges = free_energy_estimate_2D(
        ax,
        remove_nans(feature_traj),
        beta._value,
        bins=bins,
    )
    masked_free_energy = np.ma.array(
        free_energy, mask=(free_energy > nan_threshold)  # true indicates a masked (invalid) value
    )

    im = ax.pcolormesh(np.repeat(yedges[..., None], repeats=len(yedges), axis=1),
                       np.repeat(xedges[None, ...], repeats=len(xedges), axis=0), masked_free_energy)

    return ax, im

def contour_fes(beta: unit.Quantity, ax, bias_traj: BiasTrajectory):
    xyz = remove_nans(np.column_stack((bias_traj.feat1, bias_traj.feat2, bias_traj.free_energy)))
    x = xyz[:, 0]
    y = xyz[:, 1]
    free_energy = xyz[:, 2]
    # set level increment every unit of kT
    num_levels = int(
        np.floor((np.max(free_energy) - np.min(free_energy)) * beta)
    )
    levels = np.array([k * 1 / beta._value for k in range(num_levels + 2)])  # todo: why +2?
    im = ax.tricontourf(x, y, free_energy, levels=levels, cmap="RdBu_r")
    ax.tricontour(x, y, free_energy, levels=levels, linewidths=0.5, colors="k")

    return ax, im

def bezier_fes(beta: unit.Quantity, ax, bias_traj: BiasTrajectory):
    xyz = remove_nans(np.column_stack((bias_traj.feat1, bias_traj.feat2, bias_traj.free_energy)))
    x = xyz[:, 0]
    y = xyz[:, 1]
    free_energy = xyz[:, 2] 
    bz = SmoothBivariateSpline(x, y, free_energy, kx=3, ky=3)
    X,Y = np.meshgrid(np.linspace(x.min(),x.max(),100),np.linspace(y.min(),y.max(),100))
    surface = bz.ev(X.flatten(), Y.flatten())
    surface = surface.reshape(X.shape)
    # large negative values are unphysical
    surface[surface < -0.1 * np.nanmax(free_energy)] = np.nan
    # values above 1.1 * max free energy are unphysical
    surface[surface > 1.1 * np.nanmax(free_energy)] = np.nan
    im = ax.imshow(surface, extent=[x.min(),x.max(),y.min(),y.max()], origin='lower', cmap='RdBu_r')

    return ax, im   

def heatmap_fes(beta: unit.Quantity, ax, bias_traj: BiasTrajectory):
    xyz = remove_nans(np.column_stack((bias_traj.feat1, bias_traj.feat2, bias_traj.free_energy)))
    x = xyz[:, 0]
    y = xyz[:, 1]
    free_energy = xyz[:, 2] 
    X,Y = np.meshgrid(np.linspace(x.min(),x.max(),100),np.linspace(y.min(),y.max(),100))
    Z = griddata((x,y),free_energy,(X,Y),method='cubic')

    # Plot heatmap of Z
    im = ax.imshow(Z, extent=[x.min(),x.max(),y.min(),y.max()], origin='lower', cmap='RdBu_r')

    return ax, im


def generate_reweighting_file(plumed_file, reweighting_file, feature1='DIH_2_5_7_9', feature2='DIH_9_15_17_19',
                              stride=50, bandwidth=0.05, grid_bin=50, grid_min=-3.141592653589793,
                              grid_max=3.141592653589793, reweight_data_file="COLVAR_REWEIGHT", hills_file="HILLS"):
    """
    Generate a reweighting file from a PLUMED file.

    Parameters:
        plumed_file (str): The path to the PLUMED file.
        reweighting_file (str): The path to the reweighting file to be generated.
        stride (int): The stride for the HISTOGRAM action. Default is 50.
        bandwidth (float): The bandwidth for the HISTOGRAM action. Default is 0.05.
        grid_bin (int): The number of bins for the HISTOGRAM action. Default is 50.
    """
    with open(plumed_file, 'r') as f:
        plumed_lines = f.readlines()

    new_lines = []
    metad_args = []

    for line in plumed_lines:
        if ("RESTART" in line) or ("TORSION" in line) or ("CUSTOM" in line) or ("COMBINE" in line):
            new_lines.append(line)
        elif "METAD" in line:
            new_line = ""
            metad_frags = line.split(" ")
            for frag in metad_frags:
                if ("METAD" in frag) or ("ARG" in frag) or ("SIGMA" in frag) or ("BIASFACTOR" in frag) or (
                        "TEMP" in frag) or ("LABEL" in frag):
                    new_line += frag + " "
                if "HEIGHT" in frag:
                    new_line += "HEIGHT=0 "
                if "PACE" in frag:
                    new_line += f"FILE={hills_file} PACE=10000000 "
            new_line.rstrip('\n')
            new_line += "RESTART=YES\n"
            new_lines.append(new_line)
            metad_args = line.split("ARG=")[1].split(" ")[0]
        elif "PRINT" in line:
            new_lines.append(f"PRINT ARG={feature1},{feature2},metad.bias STRIDE=1 FILE={reweight_data_file}\n")

    new_lines.append(f"as: REWEIGHT_BIAS ARG={metad_args}\n")
    new_lines.append(
        f"hh1: HISTOGRAM ARG={feature1} STRIDE={stride} GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin} BANDWIDTH={bandwidth} LOGWEIGHTS=as\n")
    new_lines.append(
        f"hh2: HISTOGRAM ARG={feature2} STRIDE={stride} GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin} BANDWIDTH={bandwidth} LOGWEIGHTS=as\n")
    new_lines.append("ff1: CONVERT_TO_FES GRID=hh1\n")
    new_lines.append("ff2: CONVERT_TO_FES GRID=hh2\n")
    new_lines.append("DUMPGRID GRID=ff1 FILE=ff1.dat\n")
    new_lines.append("DUMPGRID GRID=ff2 FILE=ff2.dat\n")

    with open(reweighting_file, "w") as f:
        for line in new_lines:
            f.write(line)

def execute_reweighting_script(directory: str, trajectory_name: str, plumed_reweight_name: str, kT=2.494339):
    os.chdir(directory)
    #print(directory)
    #print(f"plumed driver --mf_dcd {directory}/{trajectory_name} --plumed {directory}/{plumed_reweight_name} --kt {kT}")
    subprocess.call(f"plumed driver --mf_dcd {directory}/{trajectory_name} --plumed {directory}/{plumed_reweight_name} --kt {kT}", shell=True, stdout=subprocess.DEVNULL)


def load_reweighted_trajectory(directory: str, colvar_file="COLVAR_REWEIGHT"):
    dat_file = os.path.join(directory, colvar_file)
    data = np.genfromtxt(dat_file, autostrip=True)
    feature1_traj, feature2_traj, free_energy = get_fe_trajs(data, reweight=True)

    return BiasTrajectory(feature1_traj, feature2_traj, free_energy - np.min(free_energy))

def plot_average_fes(directory, xlabel, ylabel, xcorrection , ycorrection):
    for subdirectory_tuple in os.walk(directory):
        subdirs = subdirectory_tuple[1]
        feat1s = []
        feat2s = []
        fes = []
        for subdir in subdirs:
            subdir = os.path.join(directory, subdir)
            feat1, feat2, fe = process_free_energy_surface(subdir, plot=False)
            feat1s.append(feat1)
            feat2s.append(feat2)
            fes.append(fe)

        feat1 = np.concatenate(feat1s)
        feat2 = np.concatenate(feat2s)
        fes = np.concatenate(fes)
        contour_fes(feat1, feat2, fes, xlabel, ylabel, xcorrection, ycorrection)
        break

