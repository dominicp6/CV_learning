import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')
from OpenMMSimulation import PDBSimulation, MOL2Simulation

simulation_params = {
    'forcefield': 'amber14',
    'precision': 'mixed',
    'pdb': '/home/dominic/PycharmProjects/CV_learning/exp/data/alanine/alanine-processed.pdb',
    'resume': '/home/dominic/PycharmProjects/CV_learning/exp/outputs/alanine_dipeptide/enhanced_sampling/PCA:0_PCA:1_structure0_repeat_0_total_features_8_feature_dimensions_2',
    'PLUMED': True,
    'gpu': '0',
    'duration': '50ns',
    'savefreq': '50ps',
    'stepsize': '1fs',
    'temperature': '300K',
    'pressure': '',
    'frictioncoeff':  '1ps',
    'solventpadding': '1nm',
    'nonbondedcutoff': '0.8nm',
    'cutoffmethod': 'CutoffPeriodic',
    'periodic': True,
    'minimise': True,
    'watermodel': 'tip3p',
    'seed': None,
    #'name': 'test',
    'directory': '/home/dominic/PycharmProjects/CV_learning/exp/exp',
    'equilibrate':  None,
    'integrator': 'Langevin',
    'state_data': True,
}


if __name__ == '__main__':
    simulation = PDBSimulation().from_args(args=simulation_params)
    simulation.run()


