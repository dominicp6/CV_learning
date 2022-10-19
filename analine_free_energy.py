#!/usr/bin/env python

"""Compute dihedral trajectories from cartesian trajectories.

   Author: Dominic Phillips (dominicp6)
"""

import mdtraj as md
import numpy as np
from multiprocessing import Process, Manager, cpu_count


def compute_torsion_mdraj(dihedral_array, traj, angle_index):
    """
    Compute torsion angle defined by four atoms.

    ARGUMENTS

    traj (mdtraj trajectory frame) - atomic coordinates
    i, j, k, l - four atoms defining torsion angle

    NOTES

    Algorithm of Swope and Ferguson [1] is used.

    [1] Swope WC and Ferguson DM. Alternative expressions for energies and forces due to angle bending and torsional energy.
    J. Comput. Chem. 13:585, 1992.

    """

    torsions = []
    i, j, k, l = angle_index

    for x in traj:
        coordinates = x.xyz.squeeze()

        # Swope and Ferguson, Eq. 26
        rij = coordinates[i] - coordinates[j]
        rkj = coordinates[k] - coordinates[j]
        rlk = coordinates[l] - coordinates[k]
        rjk = coordinates[j] - coordinates[k]

        # Swope and Ferguson, Eq. 27
        t = np.cross(rij, rkj)
        u = np.cross(
            rjk, rlk
        )  # fixed because this didn't seem to match diagram in equation in paper

        # Swope and Ferguson, Eq. 28
        t_norm = np.sqrt(np.dot(t, t))
        u_norm = np.sqrt(np.dot(u, u))

        cos_theta = np.dot(t, u) / (t_norm * u_norm)
        theta = np.arccos(cos_theta) * np.sign(np.dot(rkj, np.cross(t, u)))

        torsions.append(theta)

    dihedral_array.append(torsions)


def compute_dihedral_trajectory(pdb_file: str, trajectory: str, dihedrals: list[list]):
    structure = md.load_pdb(pdb_file)
    raw_traj = md.load_dcd(trajectory, structure.topology)
    initial_frame = raw_traj[0]
    aligned_traj = raw_traj.superpose(initial_frame)

    print(f"Running dihedral trajectory analysis.")

    assert len(dihedrals) <= cpu_count() - 1, "too many dihedrals for available CPUs"
    with Manager() as manager:
        dihedral_trajectories = manager.list()  # <-- can be shared between processes.
        processes = []
        for dihedral in dihedrals:
            p = Process(
                target=compute_torsion_mdraj,
                args=(dihedral_trajectories, aligned_traj, dihedral),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        dihedral_traj = list(dihedral_trajectories)

        print(f"Finished dihedral trajectory analysis.")

        return dihedral_traj


if __name__ == "__main__":
    phi = [4, 6, 8, 14]
    psi = [6, 8, 14, 16]
    compute_dihedral_trajectory(
        pdb_file="./alanine.pdb",
        trajectory="./TICA/outputs/production_alanine_amber_213804_190722/trajectory.dcd",
        dihedrals=[phi, psi],
    )
