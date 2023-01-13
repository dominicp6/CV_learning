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
                     'forcefield': 'amber',
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
    alanine_system = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/5us_NPT_alanine"

    # Alanine dipeptide
    output_dir = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/enhanced_sampling"
    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/open_configurations"
    CVs = ['PHI 0 ALA 2', 'PSI 0 ALA 2']

    openmm_parameters['pdb'] = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/5us_NPT_alanine/top_1ps.pdb"
    alanine_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=alanine_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features='dihedrals',
    )
    alanine_exp.initialise_hills_and_PLUMED()
    alanine_exp.run_openmm_experiments()

