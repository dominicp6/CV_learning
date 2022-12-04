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
    assert len(pdb_files) != 0, f"Read error: no PDB files found in directory."
    selection = select_file_option(pdb_files, 'PDB')
    selected_pdb = pdb_files[selection]
    check_if_memory_available(selected_pdb)

    return mdtraj.load_pdb(selected_pdb)


def load_trajectory(
        loc: str, topology: Union[str, mdtraj.Trajectory, mdtraj.Topology]
) -> mdtraj.Trajectory:
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


def load_dihedral_trajectory(loc: str, dihedral_pickle_file: str) -> list:
    dihedral_traj = np.array(
        dill.load(open(os.path.join(loc, dihedral_pickle_file), "rb"))
    ).T
    # correcting order
    dihedral_traj[:, [0, 1]] = dihedral_traj[:, [1, 0]]
    return dihedral_traj


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