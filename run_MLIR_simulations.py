import argparse

from OpenMMSimulation import PDBSimulation

simulation_params = {
    'forcefield': 'amber14',
    'precision': 'mixed',
    'resume': None,
    'PLUMED': None,
    'gpu': 0,
    'duration': '50ns',
    'savefreq': '5ps',
    'stepsize': '2fs',
    'temperature': '298K',
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
    'directory': '/home/dominic/PycharmProjects/CV_learning/MLIRexp',
    'equilibrate':  'NPT',
    'integrator': 'LangevinBAOAB',
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('system', type=str, default=None, help='System to simulation '
                                                               '(Options: BCT, COT, DMF, GDP, GTP, H2P, HPO, NML)')
    args = parser.parse_args()

    simulation_params['pdb'] = f'./chemicals/{args.system}.pdb'
    simulation = PDBSimulation().from_args(args=simulation_params)
    simulation.run()

"""
BCT Error:
    system = self.force_field.createSystem(
  File "/home/dominic/miniconda3/envs/diffusion/lib/python3.9/site-packages/openmm/app/forcefield.py", line 1212, in createSystem
    templateForResidue = self._matchAllResiduesToTemplates(data, topology, residueTemplates, ignoreExternalBonds)
  File "/home/dominic/miniconda3/envs/diffusion/lib/python3.9/site-packages/openmm/app/forcefield.py", line 1427, in _matchAllResiduesToTemplates
    raise ValueError('No template found for residue %d (%s).  %s' % (res.index+1, res.name, _findMatchErrors(self, res)))
ValueError: No template found for residue 1 (BCT).  The set of atoms is similar to ASP, but it is missing 7 atoms.

COT Error:
    system = self.force_field.createSystem(
  File "/home/dominic/miniconda3/envs/diffusion/lib/python3.9/site-packages/openmm/app/forcefield.py", line 1212, in createSystem
    templateForResidue = self._matchAllResiduesToTemplates(data, topology, residueTemplates, ignoreExternalBonds)
  File "/home/dominic/miniconda3/envs/diffusion/lib/python3.9/site-packages/openmm/app/forcefield.py", line 1427, in _matchAllResiduesToTemplates
    raise ValueError('No template found for residue %d (%s).  %s' % (res.index+1, res.name, _findMatchErrors(self, res)))
ValueError: No template found for residue 1 (COT).  This might mean your input topology is missing some atoms or bonds, or possibly that you are using the wrong force field.



"""