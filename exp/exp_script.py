import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')
from OpenMMSimulation import PDBSimulation, MOL2Simulation

simulation_params = {
    'forcefield': 'amber14',
    'precision': 'mixed',
    'pdb': '/home/dominic/PycharmProjects/CV_learning/exp/data/alanine/alanine-processed.pdb',
    'resume': None,
    'PLUMED': None,
    'gpu': '0,1',
    'duration': '50ns',
    'savefreq': '0.1ps',
    'stepsize': '1fs',
    'temperature': '300K',
    'pressure': '1bar',
    'frictioncoeff':  '1ps',
    'solventpadding': '1nm',
    'nonbondedcutoff': '0.7nm',
    'cutoffmethod': 'CutoffPeriodic',
    'periodic': True,
    'minimise': True,
    'watermodel': None,
    'seed': None,
    'name': 'test',
    'directory': '/home/dominic/PycharmProjects/CV_learning/exp/exp',
    'equilibrate':  'NPT',
    'integrator': 'Langevin',
}


if __name__ == '__main__':
    simulation = PDBSimulation().from_args(args=simulation_params)
    simulation.run()


