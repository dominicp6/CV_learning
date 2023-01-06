#!/usr/bin/env python

"""Utilitys for working with data arrays.

   Author: Dominic Phillips (dominicp6)
"""

import os
import glob
import mdtraj
from typing import Union

import dill
import numpy as np

from utils.general_utils import select_file_option, check_if_memory_available


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
                            file):
    arg_list = []
    sigma_list = [str(sigma) for sigma in sigma_list]
    for CV in CVs:
        arg_list.append("_".join(CV.split(" ")))

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
