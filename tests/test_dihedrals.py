import mdtraj as md
import numpy as np
import pytest

from Dihedrals import parse_standard_dihedral, parse_dihedral_string, parse_generic_dihedral

alanine_string_to_indices_dict = {'PHI 0 ALA 2': [4, 6, 8, 14],
                                  'PSI 0 ALA 2': [6, 8, 14, 16]}

chignolin_string_to_indices_dict = {'PHI 0 TYR 2': [2, 9, 10, 11],
                                    'PSI 0 GLY 1': [0, 1, 2, 9],
                                    'PHI 0 ASP 3': [11, 30, 31, 32],
                                    'PSI 0 TYR 2': [9, 10, 11, 30],
                                    'PHI 0 PRO 4': [32, 42, 43, 44],
                                    'PSI 0 ASP 3': [30, 31, 32, 42],
                                    'PHI 0 GLU 5': [44, 56, 57, 58],
                                    'PSI 0 PRO 4': [42, 43, 44, 56],
                                    'PHI 0 THR 6': [58, 71, 72, 73],
                                    'PSI 0 GLU 5': [56, 57, 58, 71],
                                    'PHI 0 GLY 7': [73, 85, 86, 87],
                                    'PSI 0 THR 6': [71, 72, 73, 85],
                                    'PHI 0 THR 8': [87, 92, 93, 94],
                                    'PSI 0 GLY 7': [85, 86, 87, 92],
                                    'PHI 0 TRP 9': [94, 106, 107, 108],
                                    'PSI 0 THR 8': [92, 93, 94, 106],
                                    'PHI 0 GLY 10': [108, 130, 131, 132],
                                    'PSI 0 TRP 9': [106, 107, 108, 130],
                                    'CHI1 0 TYR 2': [9, 10, 13, 14],
                                    'CHI1 0 ASP 3': [30, 31, 34, 35],
                                    'CHI1 0 PRO 4': [42, 43, 46, 47],
                                    'CHI1 0 GLU 5': [56, 57, 60, 61],
                                    'CHI1 0 THR 6': [71, 72, 75, 76],
                                    'CHI1 0 THR 8': [92, 93, 96, 97],
                                    'CHI1 0 TRP 9': [106, 107, 110, 111],
                                    'CHI2 0 TYR 2': [10, 13, 14, 15],
                                    'CHI2 0 ASP 3': [31, 34, 35, 36],
                                    'CHI2 0 PRO 4': [43, 46, 47, 48],
                                    'CHI2 0 GLU 5': [57, 60, 61, 62],
                                    'CHI2 0 TRP 9': [107, 110, 111, 112],
                                    'CHI3 0 GLU 5': [60, 61, 62, 63]}

chignolin_generic_string_to_indices_dict = {'DIH: TRP 9 C 108 - GLY 10 N 130 - GLY 10 CA 131 - GLY 10 C 132':  [108, 130, 131, 132],
                                            'DIH: THR 8 N 92 - THR 8 CA 93 - THR 8 C 94 - TRP 9 N 106':  [92, 93, 94, 106],
                                            'DIH: GLY 1 C 2 - TYR 2 N 9 - TYR 2 CA 10 - TYR 2 C 11':  [2, 9, 10, 11],
                                            'DIH: TRP 9 N 106 - TRP 9 CA 107 - TRP 9 C 108 - GLY 10 N 130':  [106, 107, 108, 130],
                                            'DIH: TYR 2 C 11 - ASP 3 N 30 - ASP 3 CA 31 - ASP 3 C 32':  [11, 30, 31, 32],
                                            'DIH: GLY 1 N 0 - GLY 1 CA 1 - GLY 1 C 2 - TYR 2 N 9':  [0, 1, 2, 9],
                                            'DIH: ASP 3 C 32 - PRO 4 N 42 - PRO 4 CA 43 - PRO 4 C 44':  [32, 42, 43, 44],
                                            'DIH: TYR 2 N 9 - TYR 2 CA 10 - TYR 2 C 11 - ASP 3 N 30':  [9, 10, 11, 30],
                                            'DIH: PRO 4 C 44 - GLU 5 N 56 - GLU 5 CA 57 - GLU 5 C 58':  [44, 56, 57, 58],
                                            'DIH: ASP 3 N 30 - ASP 3 CA 31 - ASP 3 C 32 - PRO 4 N 42':  [30, 31, 32, 42],
                                            'DIH: GLU 5 C 58 - THR 6 N 71 - THR 6 CA 72 - THR 6 C 73':  [58, 71, 72, 73],
                                            'DIH: PRO 4 N 42 - PRO 4 CA 43 - PRO 4 C 44 - GLU 5 N 56':  [42, 43, 44, 56],
                                            'DIH: THR 6 C 73 - GLY 7 N 85 - GLY 7 CA 86 - GLY 7 C 87':  [73, 85, 86, 87],
                                            'DIH: GLU 5 N 56 - GLU 5 CA 57 - GLU 5 C 58 - THR 6 N 71':  [56, 57, 58, 71],
                                            'DIH: GLY 7 C 87 - THR 8 N 92 - THR 8 CA 93 - THR 8 C 94':  [87, 92, 93, 94],
                                            'DIH: THR 6 N 71 - THR 6 CA 72 - THR 6 C 73 - GLY 7 N 85':  [71, 72, 73, 85],
                                            'DIH: THR 8 C 94 - TRP 9 N 106 - TRP 9 CA 107 - TRP 9 C 108':  [94, 106, 107, 108],
                                            'DIH: GLY 7 N 85 - GLY 7 CA 86 - GLY 7 C 87 - THR 8 N 92':  [85, 86, 87, 92],
                                            'DIH: GLY 1 H2 5 - TYR 2 N 9 - GLU 5 O 59 - TRP 9 CZ2 117':  [5, 9, 59, 117],
                                            'DIH: GLY 1 HA3 8 - TYR 2 N 9 - THR 8 OG1 97 - THR 8 HB 101':  [8, 9, 97, 101],
                                            'DIH: TYR 2 HE2 28 - THR 6 HA 79 - THR 6 HG21 82 - THR 8 C 94':  [28, 79, 82, 94],
                                            'DIH: GLY 1 HA3 8 - ASP 3 OD2 37 - THR 8 C 94 - TRP 9 HD1 124':  [8, 37, 94, 124],
                                            'DIH: GLY 1 HA2 7 - TYR 2 HE2 28 - GLU 5 H 65 - GLY 10 N 130':  [7, 28, 65, 130],
                                            'DIH: ASP 3 CG 35 - GLU 5 CA 57 - THR 6 OG1 76 - THR 8 HG21 103':  [35, 57, 76, 103],
                                            'DIH: TYR 2 CG 14 - PRO 4 N 42 - GLU 5 C 58 - TRP 9 CE2 115':  [14, 42, 58, 115],
                                            'DIH: GLU 5 HB2 67 - THR 8 CG2 98 - TRP 9 N 106 - TRP 9 NE1 114':  [67, 98, 106, 114],
                                            'DIH: GLY 1 H3 6 - TYR 2 CD1 15 - PRO 4 HG3 53 - GLU 5 HB3 68':  [6, 15, 53, 68],
                                            'DIH: GLY 1 H3 6 - TYR 2 HH 29 - TRP 9 HA 121 - GLY 10 H 135':  [6, 29, 121, 135],
                                            'DIH: GLY 1 CA 1 - TYR 2 CA 10 - GLU 5 HB3 68 - GLY 10 N 130':  [1, 10, 68, 130],
                                            'DIH: PRO 4 CA 43 - GLU 5 CA 57 - GLU 5 CG 61 - THR 6 O 74':  [43, 57, 61, 74],
                                            'DIH: ASP 3 HB3 41 - THR 6 HG22 83 - THR 8 CG2 98 - TRP 9 CG 111':  [41, 83, 98, 111],
                                            'DIH: TYR 2 HE2 28 - PRO 4 CB 46 - THR 6 CB 75 - GLY 10 N 130':  [28, 46, 75, 130],
                                            'DIH: GLY 1 N 0 - PRO 4 CA 43 - GLY 7 HA3 91 - THR 8 CG2 98':  [0, 43, 91, 98],
                                            'DIH: GLY 1 O 3 - THR 6 OG1 76 - THR 6 HA 79 - THR 6 HB 80':  [3, 76, 79, 80],
                                            'DIH: GLY 1 C 2 - GLY 1 H2 5 - THR 6 HG22 83 - GLY 7 C 87':  [2, 5, 83, 87],
                                            'DIH: GLY 1 H3 6 - PRO 4 HG3 53 - THR 8 HB 101 - TRP 9 C 108':  [6, 53, 101, 108]}

def _test_parse_generic_dihedral(string_to_indices_dict):
    failure_messages = []
    for dihedral_string, expected_indices in string_to_indices_dict.items():
        try:
            angle_type = dihedral_string.split()[0]
            _, obtained_indices = parse_generic_dihedral(angle_type, dihedral_string)
            assert np.array_equal(obtained_indices, expected_indices)
        except ValueError:
            failure_messages.append(f"Failed to parse dihedral string: {dihedral_string}, "
                                    f"expected indices: {expected_indices} "
                                    f"obtained indices: {obtained_indices}")

    return failure_messages

def _test_parse_standard_dihedral(string_to_indices_dict, top):
    failure_messages = []
    for dihedral_string, expected_indices in string_to_indices_dict.items():
        try:
            angle_type = dihedral_string.split()[0]
            _, obtained_indices = parse_standard_dihedral(top, angle_type, dihedral_string)
            assert np.array_equal(obtained_indices, expected_indices)
        except ValueError:
            failure_messages.append(f"Failed to parse dihedral string: {dihedral_string}, "
                                    f"expected indices: {expected_indices} "
                                    f"obtained indices: {obtained_indices}")

    return failure_messages

def _test_parse_dihedral_string(string_to_indices_dict, top):
    failure_messages = []
    for dihedral_string, expected_indices in string_to_indices_dict.items():
        try:
            angle_type, sincos, obtained_indices = parse_dihedral_string(top, dihedral_string)
            assert np.array_equal(obtained_indices, expected_indices)
        except ValueError:
            failure_messages.append(f"Failed to parse dihedral string: {dihedral_string}, "
                                    f"expected indices: {expected_indices} "
                                    f"obtained indices: {obtained_indices}")

    return failure_messages

def test_parse_standard_dihedral_unexpected_angle_type():
    top = md.load("../exp/data/alanine/alanine.pdb").topology

    with pytest.raises(ValueError):
        parse_standard_dihedral(top, 'OMEGA', 'OMEGA 0 ALA 2')

    with pytest.raises(ValueError):
        parse_standard_dihedral(top, 'CHII', 'CHII 0 PRO 4')


def test_parse_dihedral_string_unexpected_angle_type():
    top = md.load("../exp/data/alanine/alanine.pdb").topology

    with pytest.raises(ValueError):
        parse_dihedral_string(top, 'OMEGA 0 ALA 2')

    with pytest.raises(ValueError):
        parse_dihedral_string(top, 'CHII 0 PRO 4')


def test_parse_standard_dihedral_angle_type():
    top = md.load("../exp/data/alanine/alanine.pdb").topology
    angle_type, indices = parse_standard_dihedral(top, 'PHI', 'PHI 0 ALA 2')
    assert angle_type == 'PHI'

def test_parse_dihedral_string_angle_type():
    top = md.load("../exp/data/alanine/alanine.pdb").topology
    angle_type, sincos, indices = parse_dihedral_string(top, 'PHI 0 ALA 2')
    assert angle_type == 'PHI'
    assert sincos is None

def test_parse_standard_dihedral_alanine_dipeptide():
    top = md.load("../exp/data/alanine/alanine.pdb").topology

    failure_messages = _test_parse_standard_dihedral(alanine_string_to_indices_dict, top)
    assert len(failure_messages) == 0, "\n".join(failure_messages)


def test_parse_standard_dihedral_chignolin():
    top = md.load("../exp/data/chignolin/chignolin_1uao-processed.pdb").topology

    failure_messages = _test_parse_standard_dihedral(chignolin_string_to_indices_dict, top)
    assert len(failure_messages) == 0, "\n".join(failure_messages)


def test_parse_dihedral_string_alanine_dipeptide():
    top = md.load("../exp/data/alanine/alanine.pdb").topology

    failure_messages = _test_parse_dihedral_string(alanine_string_to_indices_dict, top)
    assert len(failure_messages) == 0, "\n".join(failure_messages)

def test_parse_dihedral_string_chignolin():
    top = md.load("../exp/data/chignolin/chignolin_1uao-processed.pdb").topology

    failure_messages = _test_parse_dihedral_string(chignolin_string_to_indices_dict, top)
    assert len(failure_messages) == 0, "\n".join(failure_messages)

def test_parse_generic_dihedral_string_chignolin():
    failure_messages = _test_parse_generic_dihedral(chignolin_generic_string_to_indices_dict)
    assert len(failure_messages) == 0, "\n".join(failure_messages)
