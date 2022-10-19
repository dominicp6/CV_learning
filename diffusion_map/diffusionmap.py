"""
.. module:: diffusionmap
This module allows computation of the diffusionmaps.
.. moduleauthor:: ZofiaTr
"""

from openmm import *
from openmm.app import *
from simtk.unit import *
from matplotlib.pyplot import cm
import mdtraj as md
import time
import numpy as np
import subprocess
from openmmtools.constants import kB
from sklearn.neighbors import KernelDensity

import pydiffmap_weights.TMDmap_explicit_weights as tmdmap
import pydiffmap_weights.LSD_diffusion_map as lsdmap

def energy(state, simulation):
    """
    Evaluate energy from the current state.

    :param openmm.positions state: the current state
    :param openmm.Simulation simulation: simulation object
    """
    simulation.context.setPositions(state)
    return simulation.context.getState(getEnergy=True).getPotentialEnergy()

def compute_energy(xyz, simulation, positions_unit, energy_unit):
    """
    Compute energy from the trajectory.

    :param ndrarray xyz: trajectory
    :param openmm.Simulation simulation: simulation object
    :param positions_unit: simtk.unit of the positions
    :param energy_unit: simtk.unit of the energy

    :return: numpy ndarray of energy for every trajectory frame.
    """

    Erecompute = np.zeros(len(xyz))

    for i in range(0,len(xyz)):
                Erecompute[i]=energy(xyz[i]*positions_unit, simulation).value_in_unit(energy_unit)

    return  Erecompute

def compute_target_measure(energy, kT, energy_unit):
    """
    Helper function to compute Boltzman density from energy and temperature.

    :param numpy.ndarray energy: energy value,  shape number of frames
    :param numpy.ndarray kT: kT factor, computed as kT = openmmtools.constants.kB * simtk.unit.kelvin * temperature, where temperature is a double
    """

    qTargetDistribution = np.zeros(len(energy))

    for i in range(0, len(energy)):
                betatimesH_unitless = np.abs(energy[i]) / kT.value_in_unit(energy_unit)
                qTargetDistribution[i] = np.exp(-(betatimesH_unitless))

    return qTargetDistribution


import pydiffmap.diffusion_map as dfm
import sys
sys.path.append('/Users/zofiatrst/Code/mdfeature/src/')
import pydiffmap_weights.TMDmap_explicit_weights as tmdmap
from openmmtools.constants import kB

def energy(state, simulation):
    """
    Evaluate energy from the current state.

    :param openmm.positions state: the current state
    :param openmm.Simulation simulation: simulation object
    """
    simulation.context.setPositions(state)
    return simulation.context.getState(getEnergy=True).getPotentialEnergy()

def compute_energy(xyz, simulation, positions_unit, energy_unit):
    """
    Compute energy from the trajectory.

    :param ndrarray xyz: trajectory
    :param openmm.Simulation simulation: simulation object
    :param positions_unit: simtk.unit of the positions
    :param energy_unit: simtk.unit of the energy

    :return: numpy ndarray of energy for every trajectory frame.
    """

    Erecompute = np.zeros(len(xyz))

    for i in range(0,len(xyz)):
                Erecompute[i]=energy(xyz[i]*positions_unit, simulation).value_in_unit(energy_unit)

    return  Erecompute

def compute_target_measure(energy, kT, energy_unit):
    """
    Helper function to compute Boltzman density from energy and temperature.

    :param numpy.ndarray energy: energy value,  shape number of frames
    :param numpy.ndarray kT: kT factor, computed as kT = openmmtools.constants.kB * simtk.unit.kelvin * temperature, where temperature is a double
    """

    qTargetDistribution = np.zeros(len(energy))

    for i in range(0, len(energy)):
                betatimesH_unitless = np.abs(energy[i]) / kT.value_in_unit(energy_unit)
                qTargetDistribution[i] = np.exp(-(betatimesH_unitless))

    return qTargetDistribution


def compute_diffusionmaps(traj_orig, nrpoints=None, epsilon='bgh', nrneigh=64, weights=None, weight_params={}, type=None):
    """
    Compute diffusionmaps using pydiffmap.

    :param mdtraj.Trajectory traj_orig: trajectory for diffusionmap analysis
    :param nrpoints: if None, keep all trajectory, if integer, subsample the trajectory leaving nrpoints of datapoints.
    :param epsilon: epsilon parameter in diffusionmap construction.
    :param int nrneigh: number of neighbors in diffusionmap construction.
    :param str weights:  if None vanilla diffusionmap, if 'compute' then TMDmap correction (requires also weight_params['simulation']=openmm.Simulation and weight_params['temperature']=double). If 'explicit' then ndarray of weights should be passed as a key in the dictionary weight_params['weights'].  If 'explicit' is computed using an old version of pydiffmap which allows for weights to be ndarrays.
    :param str type: options, "noraml", "TMDmap", "explicit", "LSDmap", if None then run vanilla diffusionmap.

    :rtype: pydiffmap.diffusion_map.DiffusionMap, mdtraj.Trajectory
    """

    # subsampling
    if nrpoints is None:
        landmark_indices = np.arange(len(traj_orig.xyz))
        print('Running diffusion maps with no subsampling.')
    else:
        landmark_indices = np.random.choice(np.arange(len(traj_orig.xyz)), size=nrpoints)
        print(traj_orig)
        print('Subsampled to')

    traj = md.Trajectory(traj_orig.xyz[landmark_indices], traj_orig.topology)
    traj = traj.superpose(traj[0])

    print(traj)

    # computation of weights for tmdmap
    if type == "TMDmap":
        simulation = weight_params['simulation']

        positions = simulation.context.getState(getPositions=True).getPositions()
        energy_unit = energy(positions, simulation).unit
        positions_unit = positions.unit

        T = weight_params['temperature']
        kT = kB * T * kelvin

        E = compute_energy(traj.xyz, simulation, positions_unit, energy_unit)
        print('Energy has shape')
        print(E.shape)

        number_of_atoms = traj.xyz.shape[1]

        def compute_boltzmann_onestate(xyz):
            """
            Helper function to compute Boltzman density from energy and temperature from one configuration x: exp(-beta*V(x))

            :param openmm.position x: the current state
            """

            xyz_res = xyz.reshape(number_of_atoms, 3)
            energy_state = energy(xyz_res*positions_unit, simulation).value_in_unit(energy_unit)
            betatimesH_unitless = np.abs(energy_state) / kT.value_in_unit(energy_unit)
            return  np.exp(-(betatimesH_unitless))

        weight_fxn = lambda x: compute_boltzmann_onestate(x)

    Xresh = traj.xyz.reshape(traj.xyz.shape[0], traj.xyz.shape[1]*traj.xyz.shape[2])

    if type is None or type == "normal":
        print('Computing vanilla diffusionmap')
        mydmap = dfm.DiffusionMap.from_sklearn(epsilon = epsilon, alpha = 0.5, k=nrneigh, kernel_type='gaussian', n_evecs=5, neighbor_params=None,
                     metric='euclidean', metric_params=None, weight_fxn=None, density_fxn=None, bandwidth_type="-1/(d+2)",
                     bandwidth_normalize=False, oos='nystroem')

        mydmap.fit(Xresh)

    elif type == "TMDmap":
        # TODO why alpha different here?
        print('Computing TMDmap with target measure exp(-beta(V(x)))')
        mydmap = dfm.DiffusionMap.from_sklearn(epsilon = epsilon, alpha = 1.0, k=nrneigh, kernel_type='gaussian', n_evecs=5, neighbor_params=None,
                     metric='euclidean', metric_params=None, weight_fxn=weight_fxn, density_fxn=None, bandwidth_type="-1/(d+2)",
                     bandwidth_normalize=False, oos='nystroem')
        mydmap.fit(Xresh)

    elif type == 'explicit':
        # TODO work out what explicit weights means
        print('Computing TMDmap with explicit weights')
        mydmap = tmdmap.DiffusionMap(epsilon = epsilon, alpha = 0.5, k=nrneigh, kernel_type='gaussian', n_evecs=5, neighbor_params=None,
                                     metric='euclidean', metric_params=None)
        mydmap.fit(Xresh, weights=weight_params['weights'])

    elif type == "LSDmap":
        # TODO: implement data and LSDMap directory checks
        mydmap = lsdmap.LSDMap(epsilon = epsilon, alpha= 0.5, k=nrneigh, metric='euclidean', metric_params=None)
        dmap_obj = mydmap.fit(traj)
        print('evals')
        print(dmap_obj.evals)
        print(dmap_obj.evals.shape)
        print('evecs')
        print(dmap_obj.evecs)
        print(dmap_obj.evecs.shape)
        print('dmap')
        print(dmap_obj.dmap)
        print(dmap_obj.dmap.shape)
        pass

    else:
        raise

    traj = md.Trajectory(Xresh.reshape(traj.xyz.shape[0], traj.xyz.shape[1], traj.xyz.shape[2]), traj_orig.topology)

    return mydmap, traj
