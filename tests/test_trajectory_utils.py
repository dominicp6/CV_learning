import mdtraj as md

from utils.trajectory_utils import get_dihedral_atom_and_residue_indices


# Note: all indices are in the convention of PLUMED, which is 1-indexed

def test_get_dihedral_atom_and_residue_indices():
    # Alanine Dipeptide
    string_to_indices_dict = {'PHI 0 ALA 2': [5, 7, 9, 15], 'PSI 0 ALA 2': [7, 9, 15, 17]}

    top = md.load("../exp/data/alanine/alanine.pdb").topology

    for dihedral_string, indices in string_to_indices_dict.items():
        assert get_dihedral_atom_and_residue_indices(top, dihedral_string)[0] == indices
