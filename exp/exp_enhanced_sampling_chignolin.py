import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')
import mdfeature.features as features
import numpy as np
from EnhancedSamplingExperiments import EnhancedSamplingExperiments

openmm_parameters = {'--duration': '50ns',
                     '--savefreq': '50ps',
                     '--stepsize': '2fs',
                     '--frictioncoeff': '1ps',
                     'precision': 'mixed',
                     '--water': 'tip3p',
                     '--temperature': '300K',
                     '--pressure': '',
                     '--nonbondedcutoff': '1nm',
                     '--solventpadding': '1nm',
                     '--cutoffmethod': 'CutoffPeriodic',
                     '--periodic': True,
                     'forcefield': 'amber14',
                     '--equilibrate': 'NVT',
                     }
meta_d_parameters = {'gaussian_height': 1.2,
                     'gaussian_pace': 500,    # 500 * stepsize = 1ps
                     'well_tempered': True,
                     'bias_factor': 8,
                     'temperature': 300,
                     'sigma_list': [0.35, 0.35],
                     'normalised': True,
                     }

if __name__ == "__main__":
    chignolin_system = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/5us_NPT_chignolin_1uao"

    # Chignolin
    output_dir = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/enhanced_sampling"
    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/one_configuration"
    openmm_parameters['pdb'] = "/home/dominic/PycharmProjects/CV_learning/exp/data/chignolin/minimised.pdb"

    atoms_in_chignolin = 138

    dihedral_features = [[108, 130, 131, 132],
                         [92, 93, 94, 106],
                         [2, 9, 10, 11],
                         [106, 107, 108, 130],
                         [11, 30, 31, 32],
                         [0, 1, 2, 9],
                         [32, 42, 43, 44],
                         [9, 10, 11, 30],
                         [44, 56, 57, 58],
                         [30, 31, 32, 42],
                         [58, 71, 72, 73],
                         [42, 43, 44, 56],
                         [73, 85, 86, 87],
                         [56, 57, 58, 71],
                         [87, 92, 93, 94],
                         [71, 72, 73, 85],
                         [94, 106, 107, 108],
                         [85, 86, 87, 92]]

    sidechain_torsion_features = [[9, 10, 13, 14],
                                  [30, 31, 34, 35],
                                  [42, 43, 46, 47],
                                  [56, 57, 60, 61],
                                  [71, 72, 75, 76],
                                  [92, 93, 96, 97],
                                  [106, 107, 110, 111],
                                  [10, 13, 14, 15],
                                  [31, 34, 35, 36],
                                  [43, 46, 47, 48],
                                  [57, 60, 61, 62],
                                  [107, 110, 111, 112],
                                  [60, 61, 62, 63]]

    dihedral_features.extend(sidechain_torsion_features)

    extended_feature_list = features.create_torsions_list(atoms_in_chignolin,
                                                         size=69,
                                                         print_list=False,
                                                         append_to=list(dihedral_features))
    initial_features = len(extended_feature_list)
    extended_feature_list = np.unique(extended_feature_list, axis=0)
    final_features = len(extended_feature_list)
    print(f"Initial features: {initial_features}, final features: {final_features}")

    with open(f"{output_dir}/100_feature_list.txt", "w") as f:
        for feature in extended_feature_list:
            f.write(f"{feature}")

    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/one_configuration"
    # PHI-PSI Enhanced Sampling (NVT): 50ns
    # CVs = ['DIH: THR 6 C 73 - GLY 7 N 85 - GLY 7 CA 86 - GLY 7 C 87 ', 'DIH: GLY 7 N 85 - GLY 7 CA 86 - GLY 7 C 87 - THR 8 N 92 ']
    # chignolin_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=chignolin_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=np.array(dihedral_features),
    #     feature_dimensions=2,
    # )
    # chignolin_exp.initialise_hills_and_PLUMED()
    # chignolin_exp.run_openmm_experiments()
    # openmm_parameters['--duration'] = '50ns'

    # PHI-PSI Enhanced Sampling (NVT, larger Gaussian height)
    # CVs = ['DIH: THR 6 C 73 - GLY 7 N 85 - GLY 7 CA 86 - GLY 7 C 87 ', 'DIH: GLY 7 N 85 - GLY 7 CA 86 - GLY 7 C 87 - THR 8 N 92 ']
    # meta_d_parameters['gaussian_height'] = 2.5
    # chignolin_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=chignolin_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=extended_feature_list,
    #     feature_dimensions=2,
    # )
    # chignolin_exp.initialise_hills_and_PLUMED()
    # chignolin_exp.run_openmm_experiments()
    # meta_d_parameters['gaussian_height'] = 1.2

    # PCA Enhanced Sampling (NVT) - 5 features

    # TICA Enhanced Sampling (NVT) - 5 features
    CVs = ['TICA:0', 'TICA:1']
    chignolin_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=chignolin_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=extended_feature_list,
        feature_dimensions=3,
        lagtime=5,
    )
    chignolin_exp.initialise_hills_and_PLUMED()
    chignolin_exp.run_openmm_experiments()

    # PCA Enhanced Sampling (NPT) - 5 features
    CVs = ['VAMP:0', 'VAMP:1']
    chignolin_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=chignolin_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=extended_feature_list,
        feature_dimensions=3,
        lagtime=5,
    )
    chignolin_exp.initialise_hills_and_PLUMED()
    chignolin_exp.run_openmm_experiments()

    CVs = ['PCA:0', 'PCA:1']
    chignolin_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=chignolin_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=extended_feature_list,
        feature_dimensions=3
    )
    chignolin_exp.initialise_hills_and_PLUMED()
    chignolin_exp.run_openmm_experiments()

    # Try some with a much reduced Gaussian pace to see if this increases stability
    # md_params['gaussian_pace'] = 2
    # CVs = ['TICA:0', 'TICA:1']
    # chignolin_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=chignolin_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=np.array(dihedral_features),
    #     feature_dimensions=5,
    #     lagtime=5,
    # )
    # chignolin_exp.initialise_hills_and_PLUMED()
    # chignolin_exp.run_openmm_experiments()

