import argparse

from OpenMMNoPLUMED import MLSimulation, PDBSimulation

simulation_params = {
    'forcefield': 'amber14',
    'precision': 'double',
    'resume': None,
    'PLUMED': None,
    'gpu': '0',
    'duration': '10ns',
    'savefreq': '5ps',
    'stepsize': '0.5fs',
    'temperature': '298K',
    'pressure': '1bar',
    'frictioncoeff':  '1ps',
    'solventpadding': None,
    'nonbondedcutoff': '0.8nm',
    'cutoffmethod': 'CutoffPeriodic',
    'periodic': True,
    'minimise': True,
    'watermodel': 'tip3p',
    'seed': None,
    'directory': '/home/dominic/PycharmProjects/CV_learning/MLIRexp',
    'equilibrate':  'NPT',
    'equilibration_length': '0.05ns',
    'integrator': 'Verlet',
    'num_water': 1000,
    'ionic_strength': '0.15mol',
    'state_data': False,
}

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('system', type=str, default=None, help='System to simulation '
    #                                                            '(Options: BCT, DMF, GDP, GTP, NML)')
    # args = parser.parse_args()
    NUM_WATER = {
        'BCT': 2543,
        'DMF': 2271,
        'GDP': 4966,
        'GTP': 5467,
        'NML': 2661,
    }

    for system in ['DMF', 'NML']:
        simulation_params['pdb'] = f'/home/dominic/PycharmProjects/CV_learning/chemicals/nw_pdb_ideal/{system}ideal.pdb'
        simulation_params['sdf'] = f'/home/dominic/PycharmProjects/CV_learning/chemicals/nw_sdf/{system}ideal.sdf'
        simulation_params['ml_residues'] = f'{system}'
        simulation_params['num_water'] = NUM_WATER[system]
        simulation_params['name'] = f'{system}_10ns_ani2'
        simulation = MLSimulation().from_args(args=simulation_params)
        simulation.run()
