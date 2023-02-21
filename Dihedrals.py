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
        dihedral_string = dihedral_string.replace(f"{sincos.upper()}", "")
        dihedral_string = dihedral_string.replace("(", "")
        dihedral_string = dihedral_string.replace(")", "")
        angle_type = dihedral_string.split()[0]

    standard_dihedrals = ["PHI", "PSI", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]

    if angle_type in standard_dihedrals:
        angle_type, atom_indices = parse_standard_dihedral(top, angle_type, dihedral_string)
    elif angle_type == "DIH:":
        angle_type, atom_indices = parse_generic_dihedral(angle_type, dihedral_string)
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


def compute_dihedral_label(top = None, dihedral_string = None, atom_indices = None, sincos = None, angle_type = None):
    """Compute the label for a dihedral angle.

    Parameters
    ----------
    top : mdtraj.Topology
        The topology of the system.
    dihedral_string : str
        The dihedral string.

    Returns
    -------
    label : str
        The label for the dihedral angle.

    """

    if angle_type is None or atom_indices is None:
        angle_type, sincos, atom_indices = parse_dihedral_string(top, dihedral_string)

    if sincos is None:
        label = f'{angle_type}_{"_".join([str(s + 1) for s in atom_indices])}'
    else:
        label = f'{sincos}_{angle_type}_{"_".join([str(s + 1) for s in atom_indices])}'

    return label


class Dihedral:
    def __init__(
        self,
        atom_indices: list[int],
        angle_type: str,
        sincos: Optional[str],
        offset: float,
        idx: int,
        periodic: bool = True,
    ):
        self.atom_indices = atom_indices
        self.sincos = sincos if sincos is not None else ""
        self.offset = offset
        self.idx = idx
        self.dihedral_base_label = "_".join([str(s + 1) for s in atom_indices])
        self.dihedral_label = compute_dihedral_label(sincos=sincos, angle_type=angle_type, atom_indices=atom_indices)
        if periodic:
            self.periodic = "YES"
        else:
            self.periodic = "NO"

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
            return f"CUSTOM ARG={self.dihedral_base_label} FUNC={self.sincos}(x)-{self.offset} LABEL={self.dihedral_label} PERIODIC={self.periodic} "
        else:
            return f"CUSTOM ARG={self.dihedral_base_label} FUNC=x-{self.offset} LABEL={self.dihedral_label} PERIODIC={self.periodic} "


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

    def write_combined_label(self, CV_name: str, features: list[str], coefficients: np.array, periodic: bool, file):
        coefficients = self._set_coefficients(coefficients, self.normalised)
        dihedral_labels = []
        for feature in features:
            angle_type, sincos, atom_indices = parse_dihedral_string(self.topology, feature)
            label = compute_dihedral_label(sincos=sincos, angle_type=angle_type, atom_indices=atom_indices)
            dihedral_labels.append(label)
        if periodic:
            periodic = "YES"
        else:
            periodic = "NO"
        assert len(coefficients) == len(features), \
            f"The number of coefficients ({len(coefficients)}) must equal " \
            f"the number of provided features ({len(features)})."

        output = (
            f"COMBINE LABEL={CV_name}"
            + f" ARG={','.join(dihedral_labels)}"
            + f" COEFFICIENTS={','.join(coefficients)}"
            + f" PERIODIC={periodic} "
        )
        file.writelines(output + "\n")
