import numpy as np
import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')

import mdfeature.features as features
from EnhancedSamplingExperiments import EnhancedSamplingExperiments

openmm_parameters = {'--duration': '50ns',
                     '--savefreq': '50ps',
                     '--stepsize': '2fs',
                     '--frictioncoeff': '1ps',
                     'precision': 'mixed',
                     '--water': 'tip3p',
                     '--temperature': '300K',
                     '--pressure': '',
                     '--nonbondedcutoff': '0.8nm',
                     '--solventpadding': '1nm',
                     '--cutoffmethod': 'CutoffPeriodic',
                     '--periodic': True,
                     'forcefield': 'amber14',
                     '--equilibrate': 'NVT',
                     }
meta_d_parameters = {'gaussian_height': 1.2,
                     'gaussian_pace': 500,  # 500 * stepsize = 1ps
                     'well_tempered': True,
                     'bias_factor': 8,
                     'temperature': 300,
                     'sigma_list': [0.35, 0.35],
                     'normalised': True,
                     }

if __name__ == "__main__":
    alanine_system = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/5us_NPT_alanine"

    # Alanine dipeptide
    output_dir = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/enhanced_sampling"
    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/one_configuration"
    openmm_parameters[
        'pdb'] = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/5us_NPT_alanine/top_1ps.pdb"

    phi = [4, 6, 8, 14]
    psi = [6, 8, 14, 16]
    zeta = [1, 4, 6, 8]
    theta = [8, 14, 16, 18]
    feature_list = np.array([phi, psi, zeta, theta])

    atoms_in_alanine_dipeptide = 22
    extended_feature_list = features.create_torsions_list(atoms_in_alanine_dipeptide,
                                                                         size=96,
                                                                         print_list=False,
                                                                         append_to=list(feature_list))
    initial_features = len(extended_feature_list)
    extended_feature_list = np.unique(extended_feature_list, axis=0)
    final_features = len(extended_feature_list)
    print(f"Initial features: {initial_features}, final features: {final_features}")

    # write the feature list to a file
    with open(f"{output_dir}/100_feature_list.txt", "w") as f:
        for feature in extended_feature_list:
            f.write(f"{feature}")

    # PHI-PSI Enhanced Sampling (NVT): very long run
    openmm_parameters['--duration'] = '250ns'
    CVs = ['DIH: ACE 1 C 4 0 - ALA 2 N 6 0 - ALA 2 CA 8 0 - ALA 2 C 14 0 ', 'DIH: ALA 2 N 6 0 - ALA 2 CA 8 0 - ALA 2 C 14 0 - NME 3 N 16 0 ']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=feature_list,
        subtract_feature_means=False,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()
    openmm_parameters['--duration'] = '50ns'

    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/five_configurations"

    # PCA Enhanced Sampling (NVT) - 4 features
    CVs = ['PCA:0', 'PCA:1']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=feature_list,
        subtract_feature_means=True,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

    # TICA Enhanced Sampling (NVT) - 4 features
    CVs = ['TICA:0', 'TICA:1']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=feature_list,
        subtract_feature_means=True,
        lagtime=1,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

    # VAMP Enhanced Sampling (NVT) - 4 features
    CVs = ['VAMP:0', 'VAMP:1']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=feature_list,
        subtract_feature_means=True,
        lagtime=1,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

    ###################################################
    # 100 feature experiments
    ###################################################

    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/open_configurations"

    # PCA Enhanced Sampling (NVT) - 100 features
    CVs = ['PCA:0', 'PCA:1']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=extended_feature_list,
        subtract_feature_means=True,
        feature_dimensions=5,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

    # TICA Enhanced Sampling (NVT) - 100 features
    CVs = ['TICA:0', 'TICA:1']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=extended_feature_list,
        subtract_feature_means=True,
        feature_dimensions=5,
        lagtime=1,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

    # VAMP Enhanced Sampling (NVT) - 100 features
    CVs = ['VAMP:0', 'VAMP:1']
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features=extended_feature_list,
        subtract_feature_means=True,
        feature_dimensions=5,
        lagtime=1,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()
