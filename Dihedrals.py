#!/usr/bin/env python

"""Wrapper for dihedral features to enable generation of PLUMED code for enhanced sampling.

   Author: Dominic Phillips (dominicp6)
"""

import re

import numpy as np

from pyemma.coordinates.data.featurization.angles import DihedralFeature


class Dihedral:
    def __init__(
        self,
        atom_indices: list[int],
        residue_indices: np.array,
        sincos: str,
        offset: float,
        idx: int,
    ):
        self.atom_indices = atom_indices
        self.residue_indices = residue_indices
        self.sincos = sincos
        self.offset = offset
        self.idx = idx
        # self.str_index = str(int(np.floor(self.idx/2)))
        self.dihedral_label_trig_removed = "_".join([str(s + 1) for s in atom_indices])
        self.dihedral_label = str(self.sincos) + "_" + self.dihedral_label_trig_removed

    def torsion_label(self):
        # plumed is 1 indexed and mdtraj is not
        if self.sincos == "sin":  # only output one torsion label per sin-cos pair
            return (
                "TORSION ATOMS="
                + ",".join(str(i + 1) for i in self.atom_indices)
                + f" LABEL={self.dihedral_label_trig_removed} \\n\\"
            )
        else:
            return None

    def transformer_label(self):
        return f"MATHEVAL ARG={self.dihedral_label_trig_removed} FUNC={self.sincos}(x)-{self.offset} LABEL={self.dihedral_label} PERIODIC=NO \\n\\"


class Dihedrals:
    def __init__(
        self,
        dihedrals: list[DihedralFeature],
        offsets: list[float],
        coefficients: list[float],
    ):
        # TODO: init with coefficients
        self.dihedral_list = []
        self.dihedral_labels = []
        self.coefficients = [str(v) for v in coefficients]
        self.initialise_lists(dihedrals[0], offsets)

    def parse_dihedral_string(self, txt: str):
        num_seq = np.array([int(s) for s in re.findall(r"\b\d+\b", txt)])
        if "SIN" in txt and "COS" not in txt:
            sincos = "sin"
        elif "COS" in txt and "SIN" not in txt:
            sincos = "cos"
        else:
            raise ValueError(f"Expected either SIN or COS in string, got {txt}.")
        atom_indices = num_seq[1::2]
        residue_indices = num_seq[0::2]
        return atom_indices, residue_indices, sincos

    # todo: inconsistent naming
    def initialise_lists(self, dihedrals: DihedralFeature, offsets: list[float]):
        dihedral_labels = dihedrals.describe()
        assert len(dihedral_labels) == len(
            offsets
        ), "The number of offets must equal the number of dihedrals."
        for idx, label in enumerate(dihedral_labels):
            atom_indices, residue_indices, sincos = self.parse_dihedral_string(label)
            dihedral = Dihedral(
                atom_indices, residue_indices, sincos, offsets[idx], idx
            )
            self.dihedral_list.append(dihedral)
            self.dihedral_labels.append(dihedral.dihedral_label)

    def write_torsion_labels(self, file):
        for dihedral in self.dihedral_list:
            output = dihedral.torsion_label()
            if output is not None:
                print(output)
                file.writelines(output + "\n")

    def write_transform_labels(self, file):
        for dihedral in self.dihedral_list:
            output = dihedral.transformer_label()
            print(output)
            file.writelines(output + "\n")

    def write_combined_label(self, CV: str, file):
        output = (
            f"COMBINE LABEL={CV}_0"
            + f" ARG={','.join(self.dihedral_labels)}"
            + f" COEFFICIENTS={','.join(self.coefficients)}"
            + " PERIODIC=NO"
            + " \\n\\"
        )
        print(output)
        file.writelines(output + "\n")
