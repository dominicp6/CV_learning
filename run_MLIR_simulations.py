import os.path

from OpenMMNoPLUMED import MLSimulation

absolute_path = os.path.dirname(__file__)

simulation_params = {
    'forcefield': 'amber14',
    'precision': 'double',
    'resume': None,
    'PLUMED': None,
    'gpu': '0',
    'duration': '200ps',
    'savefreq': '1fs',
    'stepsize': '0.1fs',
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
    'directory': os.path.join(absolute_path, 'MLIRexp'),
    'equilibrate':  'NPT',
    'equilibration_length': '0.01ns',
    'integrator': 'Verlet',
    'num_water': 1000,
    'ionic_strength': '0.15mol',
    'state_data': False,
}

if __name__ == '__main__':
    NUM_WATER = {
        'BCT': 2543,  # charged
        'COT': 1112,  # charged (BUGGY PDB/SDF COMBINATION)
        'DMF': 2271,
        'GDP': 4966,  # phosphorus
        'GTP': 5467,  # phosphorus
        'H2P': 2585,  # phosphorus
        'HPO': 2624,  # phosphorus
        'NML': 2661,
    }

    for system in ['DMF', 'NML']:
        simulation_params['pdb'] = os.path.join(absolute_path, f'chemicals/nw_pdb_ideal/{system}ideal.pdb')
        simulation_params['sdf'] = os.path.join(absolute_path, f'chemicals/nw_sdf/{system}ideal.sdf')
        simulation_params['ml_residues'] = f'{system}'
        simulation_params['num_water'] = NUM_WATER[system]
        simulation_params['name'] = f'{system}_200ps_1fs_ani2'
        simulation = MLSimulation().from_args(args=simulation_params)
        simulation.run()
