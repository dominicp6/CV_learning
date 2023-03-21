import os
import numpy as np
import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')

import mdfeature.features as features
from EnhancedSamplingExperiments import EnhancedSamplingExperiments

openmm_parameters = {'duration': '50ns',
                     'savefreq': '50ps',
                     'stepsize': '1.5fs',
                     'frictioncoeff': '1ps',
                     'precision': 'mixed',
                     'watermodel': 'tip3p',
                     'temperature': '300K',
                     'pressure': '',
                     'nonbondedcutoff': '0.8nm',
                     'solventpadding': '1nm',
                     'cutoffmethod': 'CutoffPeriodic',
                     'periodic': True,
                     'forcefield': 'amber14',
                     'equilibrate': 'NVT',
                     'integrator': 'LangevinBAOAB',
                     'gpu': '0',
                     'num_water': None,
                     'equilibration_length': '0.1ns',
                     'ionic_strength': '0.0mol',
                     'state_data': True,
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
    openmm_parameters['pdb'] = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/5us_NPT_alanine/top_1ps.pdb"

    phi = [4, 6, 8, 14]
    psi = [6, 8, 14, 16]
    zeta = [1, 4, 6, 8]
    theta = [8, 14, 16, 18]
    feature_list = np.array([phi, psi, zeta, theta])

    atoms_in_alanine_dipeptide = 22
    if os.path.isfile(f"{output_dir}/100_feature_list.txt"):
        with open(f"{output_dir}/100_feature_list.txt", 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            row = [int(x) for x in line.strip().replace('[', '').replace(']', '').split()]
            data.append(row)
        final_features = np.array(data)
    else:
        final_features = features.create_torsions_list(atoms_in_alanine_dipeptide,
                                                       size=96,
                                                       print_list=False,
                                                       append_to=list(feature_list))
        final_features = np.unique(final_features, axis=0)

        # write the feature list to file
        with open(f"{output_dir}/100_feature_list.txt", "w") as f:
            for feature in final_features:
                f.write(f"{feature}\n")

    print(f"Total #features in final features: {len(final_features)}")

    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/one_configuration"

    # PCA Enhanced Sampling (NVT) - 4 features
    # CVs = ['PCA:0', 'PCA:1']
    # alanine_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=alanine_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=feature_list,
    #     cos_sin=True,
    #     subtract_feature_means=False,
    #     num_cv_features=2,
    # )
    # alanine_exp.initialise_hills_and_PLUMED()
    # alanine_exp.run_openmm_experiments()

    # # TICA Enhanced Sampling (NVT) - 4 features
    # CVs = ['TICA:0', 'TICA:1']
    # alanine_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=alanine_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=feature_list,
    #     cos_sin=True,
    #     subtract_feature_means=False,
    #     num_cv_features=2,
    #     lagtime=1,
    # )
    # alanine_exp.initialise_hills_and_PLUMED()
    # alanine_exp.run_openmm_experiments()

    # VAMP Enhanced Sampling (NVT) - 4 features
    # CVs = ['VAMP:0', 'VAMP:1']
    # alanine_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=alanine_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=feature_list,
    #     subtract_feature_means=False,
    #     cos_sin=True,
    #     num_cv_features=2,
    #     lagtime=1,
    # )
    # alanine_exp.initialise_hills_and_PLUMED()
    # alanine_exp.run_openmm_experiments()

    ###################################################
    # 100 feature experiments
    ###################################################

    # PCA Enhanced Sampling (NVT) - 100 features
    # CVs = ['PCA:0', 'PCA:1']
    # alanine_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=alanine_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=final_features,
    #     subtract_feature_means=False,
    #     cos_sin=True,
    #     num_cv_features=4,
    # )
    # alanine_exp.initialise_hills_and_PLUMED()
    # alanine_exp.run_openmm_experiments()

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
        features=final_features,
        subtract_feature_means=False,
        cos_sin=True,
        num_cv_features=4,
        lagtime=1,
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

    # VAMP Enhanced Sampling (NVT) - 100 features
    # CVs = ['VAMP:0', 'VAMP:1']
    # alanine_exp = EnhancedSamplingExperiments(
    #     output_dir=output_dir,
    #     unbiased_exp=alanine_system,
    #     CVs=CVs,
    #     starting_structures=starting_structures,
    #     number_of_repeats=1,
    #     openmm_parameters=openmm_parameters,
    #     meta_d_parameters=meta_d_parameters,
    #     features=final_features,
    #     subtract_feature_means=False,
    #     cos_sin=True,
    #     num_cv_features=4,
    #     lagtime=1,
    # )
    # alanine_exp.initialise_hills_and_PLUMED()
    # alanine_exp.run_openmm_experiments()
