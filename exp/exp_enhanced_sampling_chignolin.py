import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')
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
    chignolin_system = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/5us_NPT_chignolin_1uao"

    # Chignolin
    output_dir = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/enhanced_sampling"
    starting_structures = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/open_configurations"
    CVs = ['PHI 0 GLY 7', 'PSI 0 GLY 7']
    

    openmm_parameters['pdb'] = "/home/dominic/PycharmProjects/CV_learning/exp/data/chignolin/minimised.pdb"
    chignolin_exp = EnhancedSamplingExperiments(
        output_dir=output_dir,
        unbiased_exp=chignolin_system,
        CVs=CVs,
        starting_structures=starting_structures,
        number_of_repeats=1,
        openmm_parameters=openmm_parameters,
        meta_d_parameters=meta_d_parameters,
        features='dihedrals',
    )
    chignolin_exp.initialise_hills_and_PLUMED()
    chignolin_exp.run_openmm_experiments()

