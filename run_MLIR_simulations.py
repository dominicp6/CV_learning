import sys
sys.path.append('/home/dominic/PycharmProjects/CV_learning')
from OpenMMSimulation import PDBSimulation, MOL2Simulation

# simulation_params = {
#     'forcefield': 'amber14',
#     'precision': 'mixed',
#     'pdb': '/home/dominic/PycharmProjects/MLIRspec/pdb/alanine.pdb',
#     'resume': None,
#     'PLUMED': None,
#     'gpu': 0,
#     'duration': '50ns',
#     'savefreq': '0.1ps',
#     'stepsize': '1fs',
#     'temperature': '300K',
#     'pressure': '1bar',
#     'frictioncoeff':  '1ps',
#     'solventpadding': '1nm',
#     'nonbondedcutoff': '0.7nm',
#     'cutoffmethod': 'CutoffPeriodic',
#     'periodic': True,
#     'minimise': True,
#     'watermodel': None,
#     'seed': None,
#     'name': 'test',
#     'directory': '/home/dominic/PycharmProjects/MLIRspec/exp',
#     'equilibrate':  'NPT',
# }

simulation_params = {
    'forcefield': 'amber14',
    'precision': 'mixed',
    'mol2': '/home/dominic/PycharmProjects/MLIRspec/chemicals/ethene/ethene.mol2',
    'xml': '/home/dominic/PycharmProjects/MLIRspec/chemicals/ethene/ethene.xml',
    'resume': None,
    'plumed': None,
    'gpu': 0,
    'duration': '10ns',
    'savefreq': '0.1fs',
    'stepsize': '0.1fs',   # need small stepsize: typical vibrational periods of a molecular bond is 5-20fs
    'temperature': '300K',
    'pressure': '',
    'frictioncoeff':  '1ps',
    'solventpadding': '1nm',
    'nonbondedcutoff': '0.7nm',
    'cutoffmethod': 'NoCutoff',
    'periodic': True,
    'minimise': True,
    'watermodel': None,
    'seed': None,
    'name': 'test',
    'directory': '/home/dominic/PycharmProjects/MLIRspec/exp',
    'equilibrate':  'NVT',
    'integrator': 'Langevin',
}

if __name__ == '__main__':
    simulation = MOL2Simulation().from_args(args=simulation_params)
    # print(simulation.generate_executable_command(args=simulation_params))
    simulation.run()
    
