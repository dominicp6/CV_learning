#!/usr/bin/env python

"""Utilities for working with chemicals arrays.

   Author: Dominic Phillips (dominicp6)
"""

import os
import glob
import subprocess
import csv
import re
import json
from collections import namedtuple
from typing import Optional

import mdtraj
import pyemma
import openmm.unit as unit
from typing import Union
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline, griddata

import numpy as np

from utils.general_utils import select_file_option, check_if_memory_available, remove_nans, print_file_contents
from Dihedrals import Dihedrals, compute_dihedral_label
from utils.feature_utils import get_cv_type_and_dim, get_feature_means
from utils.diffusion_utils import free_energy_estimate_2D
from utils.openmm_utils import parse_quantity
from utils.biased_experiment_utils import BiasTrajectory, get_fe_trajs


# ====================== INIT UTILS =================================
def init_metadata(location: str, keyword="metadata"):
    """
    Reads the experiment metadata from the file [keyword].json
    Returns the temperature, duration, savefreq, stepsize, #iterations, and beta value.
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


def init_datafiles(exp, location: str):
    """
    Reads the trajectory and topology from the specified location.
    Returns the pdb, topology, trajectory, and number of frames.
    """
    pdb = load_pdb(location)
    topology = pdb.topology
    traj = load_trajectory(location, topology)
    num_frames = len(traj)
    if exp.in_progress:
        print("[Notice] For in-progress experiments, the save frequency cannot be checked. "
                "Assuming it is correct in the metadata file.")
        # Set duration based on the number of frames in the trajectory, not based on the duration in the metadata
        exp.duration = num_frames * exp.savefreq
    else:
        # Check that the number of frames in the trajectory is consistent with the duration and savefreq
        assert np.abs(num_frames - int(exp.duration / exp.savefreq)) <= 1, (
            f"duration ({exp.duration}) and savefreq ({exp.savefreq}) incompatible with number of conformations "
            f"found in trajectory (got {num_frames}, expected {int(exp.duration / exp.savefreq)}). "
            f"Consider re-initialising with in_progress=True."
        )

    return pdb, topology, traj, num_frames


def init_biasfiles(location: str):
    """
    Reads the bias trajectory from the fes file in the specified location.
    If no fes file is found, attempts to generate one from the HILLS file (if it exists).
    """
    if os.path.exists(os.path.join(location, "HILLS")):
        HILLS_file = os.path.join(location, "HILLS")
        if os.path.exists(os.path.join(location, "fes.dat")):
            fes_file = os.path.join(location, "fes.dat")
        else:
            print("[Notice] No fes.dat file found in experiment directory. Attempting to generate one from HILLS file.")
            subprocess.call(f"plumed sum_hills --hills {HILLS_file}", shell=True)
            fes_file = os.path.join(location, 'fes.dat')

        fe_data = np.genfromtxt(fes_file, autostrip=True)
        feature1_traj, feature2_traj, fe = get_fe_trajs(fe_data, reweight=False, file_type='fes')
        bias_trajectory = BiasTrajectory(feature1_traj, feature2_traj, fe)
        return bias_trajectory
    else:
        # Not a biased experiment
        return None


def init_features(exp, features: Optional[Union[dict, list[str], np.array]], cos_sin: bool = False):
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
    if features is not None:
        featurizer = initialise_featurizer(features, exp.topology, cos_sin=cos_sin)
        featurized_traj = featurizer.transform(exp.traj)
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


def load_metad_bias(exp, bias_file: str, col_idx: int = 2):
    """
    Loads the bias potential trajectory from a metadynamics bias file.
    """
    colvar = np.genfromtxt(bias_file, delimiter=" ")
    assert (
            col_idx < colvar.shape[1]
    ), "col_idx must not exceed 1 less than the number of columns in the bias file"
    bias_potential_traj = colvar[:, col_idx]  
    weights = np.exp(exp.beta * bias_potential_traj)
    return weights, bias_potential_traj


def initialise_featurizer(
        features: Union[dict, list[str], np.array], topology, cos_sin=False
) -> pyemma.coordinates.featurizer:
    """
    Initialises a featurizer for the specified features.
    """
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

# ====================== HELPER FUNCTIONS =================================
def get_feature_ids_from_names(
        feature_names: list[str],
        featurizer: pyemma.coordinates.featurizer,
):
    """
    Returns the feature IDs corresponding to the specified feature names.
    """
    all_features = featurizer.describe()
    feature_ids = []
    for feature in feature_names:
        feature_ids.append(all_features.index(feature))

    return feature_ids


def get_feature_mask(feature: str, featurizer: pyemma.coordinates.featurizer):
    """
    Returns a feature mask with a 1 in the position of the specified feature.
    """
    all_features = featurizer.describe()
    feature_mask = np.zeros(len(all_features), dtype=int)
    feature_mask[all_features.index(feature)] = 1
    return feature_mask


def load_pdb(location: str) -> mdtraj.Trajectory:
    """
    Loads a PDB file from the specified location.
    """
    pdb_files = glob.glob(os.path.join(location, "*.pdb"))
    assert len(pdb_files) != 0, f"Read error: no PDB files (*.pdb) found in directory."
    selection = select_file_option(pdb_files, 'PDB')
    selected_pdb = pdb_files[selection]
    check_if_memory_available(selected_pdb)

    return mdtraj.load_pdb(selected_pdb)


def load_trajectory(
        location: str, topology: Union[str, mdtraj.Trajectory, mdtraj.Topology]
) -> mdtraj.Trajectory:
    """
    Loads a trajectory from the specified location.
    """
    traj_files = glob.glob(os.path.join(location, "*.dcd"))
    assert len(traj_files) != 0, f"Read error: no traj files (*.dcd) found in directory."
    selection = select_file_option(traj_files, 'traj')
    selected_traj = traj_files[selection]
    check_if_memory_available(selected_traj)

    return mdtraj.load(selected_traj, top=topology)


def get_metadata_file(
        location: str, keyword: str
):
    """
    Returns the metadata file from the specified location.
    """
    metadata_files = glob.glob(os.path.join(location, f"*{keyword}*.json"))
    assert len(metadata_files) != 0, f"Read error: no metadata files ({keyword}*.json) found in directory."
    selection = select_file_option(metadata_files, "metadata")
    selected_metadata = metadata_files[selection]

    return selected_metadata


def check_feature_is_cv_feature(exp, feature: str):
    regex = r"^(" + "|".join(exp.CV_types) + r")\d+$"
    return re.match(regex, feature)


def check_valid_cvs(exp, CVs: list[str]):
    """
    Checks that the provided CVs are valid syntax.
    """
    for idx, CV in enumerate(CVs):
        _, cv_type, cv_dim = get_cv_type_and_dim(CV)
        # If the CV is neither an atom feature nor a traditional CV (e.g. TICA, PCA, etc.), raise an error
        if cv_type not in exp.CVs.keys() and cv_type not in exp.featurizer.describe():
            raise ValueError(f"CV '{cv_type}' not in {exp.CVs.keys()} or {exp.featurizer.describe()}")

        if cv_type in exp.CVs.keys() and exp.CVs[cv_type] is None:
            raise ValueError(f"{cv_type} CVs not computed.")

# ====================== METADYNAMICS HELPERS =================================
def write_metadynamics_line(well_tempered: bool,
                            bias_factor: float,
                            temperature: float,
                            height: float,
                            pace: int,
                            sigma_list: list[float],
                            CVs: list[str],
                            plumed_file,
                            top: mdtraj.Topology,):
    """
    Writes the line defining the given metadynamics protocol to the PLUMED file.
    """
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
        plumed_file.writelines(output + "\n")
    else:
        output = (
                f"METAD ARG={','.join(arg_list)} SIGMA={','.join(sigma_list)} HEIGHT={height} FILE=HILLS "
                f"PACE={pace} LABEL=metad "
        )
        plumed_file.writelines(output + "\n")

    output = (
        f"PRINT ARG={','.join(arg_list)},metad.bias STRIDE={pace} FILE=COLVAR "
    )
    plumed_file.writelines(output + "\n")
    plumed_file.close()


def create_plumed_metadynamics_script(
        exp,
        CVs: list[str],
        features: list[list[str]],
        coefficients: list[list[float]],
        filename: str = 'plumed.dat',
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
    if not exp.features_provided:
        raise ValueError(
            "Cannot create PLUMED metadynamics script for an unfeaturized trajectory. "
            "Try reinitializing Experiment object with features defined."
        )
    else:
        # Check if CVs are valid
        check_valid_cvs(exp, CVs)

        # Initialise PLUMED script
        file_name = "./plumed.dat" if filename is None else f"{filename}"
        f = open(file_name, "w")
        output = 'RESTART'
        f.write(output + "\n")

        if use_all_features:
            # Use all features in plumed script
            relevant_features = exp.featurizer.describe()
        else:
            # Union of all features appearing in the CVs
            relevant_features = list({f for feat in features for f in feat})

        # Save features and coefficients to file
        with open(os.path.join(exp.location, 'enhanced_sampling_features_and_coeffs.csv'), 'w') as f2:
            writer = csv.writer(f2, delimiter='\t')
            writer.writerows(zip(features, coefficients))

        if subtract_feature_means:
            offsets = get_feature_means(all_features=exp.featurizer.describe(),
                                        all_means=exp.feature_means,
                                        selected_features=relevant_features)
        else:
            offsets = None

        # Initialise Dihedrals class (for now only linear combinations of dihedral CVs are supported)
        dihedral_features = Dihedrals(
            topology=exp.topology,
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
            pace=gaussian_pace, sigma_list=sigma_list, CVs=CVs, plumed_file=f, top=exp.topology,
        )

        if print_to_terminal:
            print_file_contents(file_name)


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
    """
    Runs the PLUMED reweighting script.
    """
    os.chdir(directory)
    subprocess.call(f"plumed driver --mf_dcd {directory}/{trajectory_name} --plumed {directory}/{plumed_reweight_name} --kt {kT}", shell=True, stdout=subprocess.DEVNULL)


def load_reweighted_trajectory(directory: str, colvar_file="COLVAR_REWEIGHT"):
    """
    Loads the reweighted trajectory from the COLVAR_REWEIGHT file.
    """
    dat_file = os.path.join(directory, colvar_file)
    data = np.genfromtxt(dat_file, autostrip=True)
    feature1_traj, feature2_traj, free_energy = get_fe_trajs(data, reweight=True, file_type='COLVAR')

    return BiasTrajectory(feature1_traj, feature2_traj, free_energy)

# ====================== PLOTTING HELPERS =================================
def scatter_fes(beta: unit.Quantity, ax, feature_traj: np.array, bins: int, nan_threshold: int):
    """
    Plots a scatter plot of the free energy surface.
    """
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
    """
    Plots a contour plot of the free energy surface.
    """
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
    """
    Plots a bezier plot of the free energy surface.
    """
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
    """
    Plots a heatmap of the free energy surface.
    """
    xyz = remove_nans(np.column_stack((bias_traj.feat1, bias_traj.feat2, bias_traj.free_energy)))
    x = xyz[:, 0]
    y = xyz[:, 1]
    free_energy = xyz[:, 2] 
    X,Y = np.meshgrid(np.linspace(x.min(),x.max(),100),np.linspace(y.min(),y.max(),100))
    Z = griddata((x,y),free_energy,(X,Y),method='cubic')

    # Plot heatmap of Z
    im = ax.imshow(Z, extent=[x.min(),x.max(),y.min(),y.max()], origin='lower', cmap='RdBu_r')

    return ax, im


def generate_fes(beta, ax, bias_traj, plot_type="contour"):
    """
    Generates a free energy surface plot of the given type.
    """
    if plot_type == "contour":
        ax, im = contour_fes(beta, ax, bias_traj)
    elif plot_type == "heatmap":
        ax, im = heatmap_fes(beta, ax, bias_traj)
    elif plot_type == "bezier":
        ax, im = bezier_fes(beta, ax, bias_traj)
    else:
        raise ValueError("plot_type must be either 'contour' or 'heatmap'")

    return ax, im


def reweight_biased_fes(location, feature_nicknames, ax, beta, plot_type="contour"):
    """
    Generates a free energy surface plot for a reweighted, biased trajectory of the given type.
    """
    generate_reweighting_file(os.path.join(location, 'plumed.dat'),
                                          os.path.join(location, 'plumed_reweight.dat'),
                                          feature1=feature_nicknames[0], feature2=feature_nicknames[1],
                                          stride=50, bandwidth=0.05, grid_bin=50, grid_min=-3.141592653589793,
                                          grid_max=3.141592653589793,
                                          )
    execute_reweighting_script(location, 'trajectory.dcd', 'plumed_reweight.dat', kT=1/beta._value)
    bias_traj = load_reweighted_trajectory(location)

    return generate_fes(beta, ax, bias_traj, plot_type=plot_type), bias_traj


def set_fes_cbar_and_axis(im, ax, feature_nicknames):
    """
    Sets the colorbar and axis labels for a free energy surface plot.
    """
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        f"F({feature_nicknames[0]},{feature_nicknames[1]}) / kJ mol$^{{-1}}$"
    )
    plt.gca().set_aspect("equal")

    return cbar


def check_fes_arguments(exp, data_fraction, reweight, features, feature_nicknames):
    """
    Checks the arguments for the free energy surface plot.
    """
    if not exp.bias_trajectory and reweight:
        raise ValueError("Cannot reweight unbiased experiments, please set reweight=False.")

    if exp.bias_trajectory and data_fraction != 1.0:
        raise ValueError("Cannot trim data for biased experiments, please set data_fraction=1.0.")

    if exp.bias_trajectory and not reweight:
        # TODO: replace by a check looking into the plumed.dat file
        feature_nicknames = ['CV 1', 'CV 2']

    if exp.bias_trajectory and reweight:
        assert features is not None, "Must provide features for reweighting biased experiments."
        feature_nicknames = features

    if not exp.bias_trajectory:
        assert features is not None, "Must provide features for unbiased experiments."
        for feature in features:
            assert (
                    (feature in exp.featurizer.describe())
                    or exp._check_feature_is_cv_feature(feature)), \
                f"Feature '{feature}' not found in available features ({exp.featurizer.describe()}) " \
                f"or CV types ({exp.CV_types})."
        # Automatically set feature nicknames if they are not provided
        if not feature_nicknames:
            feature_nicknames = features

    return feature_nicknames


def plot_contact_matrices(exp, fig, axs, frames: list[int], times: list[str]):
    """
    Plots the contact matrices for the given frames and times.
    """
    contact_data = [mdtraj.compute_contacts(exp.traj[frame]) for frame in frames]
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


def plot_trajectory_timeseries(exp, axs, contact_threshold: float, times: list[str], duration_ns: float):
    """
    Plots the timeseries of the number of contacts, RMSD, acylindricity, and radius of gyration.
    """
    distances, pairs = mdtraj.compute_contacts(exp.traj)
    number_of_close_contacts = np.sum((distances < contact_threshold), axis=1)  # sum along the columns (contacts)
    rms_dist = np.sqrt(np.mean(distances ** 2, axis=1))
    rmsd_initial_structure = mdtraj.rmsd(target=exp.traj, reference=exp.traj, frame=0)
    acylindricity = mdtraj.acylindricity(exp.traj)
    radius_of_gyration = mdtraj.compute_rg(exp.traj)

    # number of contacts
    x_var = np.arange(0, duration_ns, duration_ns / exp.num_frames)
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

