#!/usr/bin/env python

"""Wrapper for dihedral features to enable generation of PLUMED code for enhanced sampling.

   Author: Dominic Phillips (dominicp6)
"""

import re
from typing import Optional

import numpy as np
import mdtraj as md
from pyemma.coordinates.data.featurization.angles import DihedralFeature
from mdtraj.geometry.dihedral import indices_psi, indices_phi


def parse_dihedral_string(top, dihedral_string):

    if "SIN" in dihedral_string and "COS" not in dihedral_string:
        sincos = "sin"
    elif "COS" in dihedral_string and "SIN" not in dihedral_string:
        sincos = "cos"
    else:
        sincos = None

    parts = dihedral_string.split()

    if sincos is None:
        angle_type = parts[0]
        # res1_index = int(parts[1])    # TODO: this is not a residue index, work out what it is
        res_index = int(parts[3]) 
    else:
        raise NotImplementedError("SIN and COS dihedrals are not implemented yet")

    # Compute the dihedral angle using mdtraj
    if angle_type == "PHI":
        indices = indices_phi(top)
    elif angle_type == "PSI":
        indices = indices_psi(top)
    else:
        raise ValueError(f"Invalid dihedral angle type: {angle_type}")

    # Find the indices of the atoms involved in the dihedral angle
    atom_indices = indices[res_index - 2, :]
    # res_indices = [res1_index, res2_index]

    return sincos, atom_indices, res_index, angle_type


class Dihedral:
    def __init__(
        self,
        label: str,
        atom_indices: list[int],
        # angle_type: str,
        residue_indices: np.array,
        sincos: Optional[str],
        offset: float,
        idx: int,
    ):
        self.atom_indices = atom_indices
        # self.angle_type = angle_type
        self.residue_indices = residue_indices
        self.sincos = sincos
        self.offset = offset
        self.idx = idx
        self.dihedral_base_label = "_".join([str(s + 1) for s in atom_indices])
        self.dihedral_label = "_".join(label.split(" "))
        # if self.sincos:
        #     self.dihedral_label = str(self.sincos) + "_" + self.angle_type + "_" + self.dihedral_base_label
        # else:
        #     self.dihedral_label = self.angle_type + "_" + self.dihedral_base_label

    def torsion_label(self):
        # only output one torsion label per sin-cos pair
        if self.sincos == "sin" or self.sincos is None:
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
            sincos, atom_indices, residue_indices, angle_type = parse_dihedral_string(self.topology, label)
            dihedral = Dihedral(
                label, atom_indices, residue_indices, sincos, offsets[idx], idx
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
