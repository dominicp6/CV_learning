#!/usr/bin/env python

"""Wrapper for dihedral features to enable generation of PLUMED code for enhanced sampling.

   Author: Dominic Phillips (dominicp6)
"""

from typing import Optional

import numpy as np
import mdtraj as md
from pyemma.coordinates.data.featurization.angles import BackboneTorsionFeature, SideChainTorsions

def parse_dihedral_string(top, dihedral_string):

    if "SIN" in dihedral_string and "COS" not in dihedral_string:
        sincos = "sin"
    elif "COS" in dihedral_string and "SIN" not in dihedral_string:
        sincos = "cos"
    else:
        sincos = None

    string_parts = dihedral_string.split()

    if sincos is None:
        angle_type = string_parts[0]
    else:
        raise NotImplementedError("SIN and COS dihedrals are not implemented yet")

    standard_dihedrals = ["PHI", "PSI", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]

    if angle_type in standard_dihedrals:
        angle_type, atom_indices = parse_standard_dihedral(top, angle_type, dihedral_string)
    elif angle_type == "DIH:":
        angle_type, atom_indices = parse_generic_dihedral(top, angle_type, dihedral_string)
    else:
        raise ValueError("Unrecognised dihedral type: {}".format(string_parts[0]))

    return angle_type, sincos, atom_indices


def parse_standard_dihedral(top, angle_type, string):
    """Parse a standard dihedral angle from a string.

    Parameters
    ----------
    top : mdtraj.Topology
        The topology of the system.
    angle_type : str
        The type of dihedral angle.
    string : str
        The dihedral string.

    Returns
    -------
    angle_type : str
        The type of dihedral angle.
    atom_indices : np.ndarray
        The indices of the atoms involved in the dihedral angle.

    """

    if angle_type in ["PHI", "PSI"]:
        features = BackboneTorsionFeature(top)
        description = features.describe()
        angle_indices = features.angle_indexes
    elif angle_type in ["CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]:
        features = SideChainTorsions(top, cossin=False, which=angle_type.lower())
        description = features.describe()
        angle_indices = features.angle_indexes
    else:
        raise ValueError(f"Invalid dihedral angle type: {angle_type}")

    # Find the index of the string in the description
    string_index = description.index(string)

    # Find the indices of the atoms involved in the dihedral angle
    atom_indices = angle_indices[string_index]

    return angle_type, atom_indices


def parse_generic_dihedral(angle_type, dihedral_string):
    """Parse a generic dihedral angle from a string.

    Parameters
    ----------
    angle_type : str
        The type of dihedral angle.
    dihedral_string : str
        The dihedral string.

    Returns
    -------
    angle_type : str
        The type of dihedral angle.
    atom_indices : np.ndarray
        The indices of the atoms involved in the dihedral angle.

    Ex string: DIH: ACE 1 C 4 0 - ALA 2 N 6 0 - ALA 2 CA 8 0 - ALA 2 C 14 0
    """

    string_parts = dihedral_string.split()

    if len(string_parts) == 24:
        atom_indices = [string_parts[4], string_parts[10], string_parts[16], string_parts[22]]
    elif len(string_parts) == 20:
        atom_indices = [string_parts[4], string_parts[9], string_parts[14], string_parts[19]]
    else:
        raise ValueError(f"Invalid dihedral string: {dihedral_string}, {len(string_parts)} parts not 20 or 24")

    try:
        atom_indices = [int(i) for i in atom_indices]
    except ValueError:
        raise ValueError(f"Invalid atom indices: {atom_indices}")

    return angle_type[:-1], atom_indices


class Dihedral:
    def __init__(
        self,
        atom_indices: list[int],
        angle_type: str,
        sincos: Optional[str],
        offset: float,
        idx: int,
    ):
        self.atom_indices = atom_indices
        self.sincos = sincos if sincos is not None else ""
        self.offset = offset
        self.idx = idx
        self.dihedral_base_label = "_".join([str(s + 1) for s in atom_indices])
        self.dihedral_label = f'{self.sincos}({angle_type})_{"_".join([str(s + 1) for s in atom_indices])}'

    def torsion_label(self):
        # only output one torsion label per sin-cos pair
        if self.sincos == "sin" or self.sincos == "":
            return (
                "TORSION ATOMS="
                + ",".join(str(i + 1) for i in self.atom_indices)
                + f" LABEL={self.dihedral_base_label} "
            )
        else:
            return None

    def transformer_label(self):
        if self.sincos is not None:
            return f"MATHEVAL ARG={self.dihedral_base_label} FUNC={self.sincos}(x)-{self.offset} LABEL={self.dihedral_label} PERIODIC=NO "
        else:
            return f"MATHEVAL ARG={self.dihedral_base_label} FUNC=x-{self.offset} LABEL={self.dihedral_label} PERIODIC=NO "


class Dihedrals:
    def __init__(
        self,
        topology: md.Topology,
        dihedrals: list[str],
        offsets: np.array,
        normalised: bool,
    ):
        self.dihedral_objs = []
        self.dihedral_labels = []
        self.normalised = normalised
        self.offsets = offsets
        self.topology = topology
        assert len(dihedrals) == len(offsets), \
            f"The number of dihedrals and offsets must be equal, " \
            f"but got lengths {len(dihedrals)} and {len(offsets)} respectively."
        self.initialise_lists(dihedrals, offsets)

    @staticmethod
    def _set_coefficients(coefficients: np.array, normalised: bool):
        if not normalised:
            coefficients = [str(v) for v in coefficients]
        else:
            # Normalised so sum of squares of coefficients is 1
            coefficients = [str(v / np.sqrt(np.sum(coefficients ** 2))) for v in coefficients]

        return coefficients

    def initialise_lists(self, dihedrals: list[str], offsets: list[float]):
        for idx, label in enumerate(dihedrals):
            angle_type, sincos, atom_indices = parse_dihedral_string(self.topology, label)
            dihedral = Dihedral(
                atom_indices, angle_type, sincos, offsets[idx], idx
            )
            self.dihedral_objs.append(dihedral)
            self.dihedral_labels.append(dihedral.dihedral_label)

    def write_torsion_labels(self, file):
        for dihedral in self.dihedral_objs:
            output = dihedral.torsion_label()
            if output is not None:
                file.writelines(output + "\n")

    def write_transform_labels(self, file):
        for dihedral in self.dihedral_objs:
            output = dihedral.transformer_label()
            file.writelines(output + "\n")

    def write_combined_label(self, CV_name: str, CV_coefficients: np.array, file):
        coefficients = self._set_coefficients(CV_coefficients, self.normalised)
        assert len(coefficients) == len(self.dihedral_labels) == len(self.offsets), \
            f"The number of coefficients must equal the number of dihedrals and offsets, " \
            f"but got lengths {len(self.dihedral_labels)} and {len(self.offsets)} respectively."

        output = (
            f"COMBINE LABEL={CV_name}"
            + f" ARG={','.join(self.dihedral_labels)}"
            + f" COEFFICIENTS={','.join(coefficients)}"
            + " PERIODIC=NO "
        )
        file.writelines(output + "\n")
