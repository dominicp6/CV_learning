import numpy as np
import mdtraj
from pyemma.coordinates.data import CustomFeature


def second_largest_eigenvalue(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    second_largest_eigenvalue = eigenvalues[1]

    return second_largest_eigenvalue


def calc_second_principal_inertia_component(traj: mdtraj.Trajectory):
    inertia_tensors = mdtraj.compute_inertia_tensor(traj)
    f_vec = np.vectorize(second_largest_eigenvalue)

    return f_vec(inertia_tensors)


def inertia_tensor_second_principal_component(traj: mdtraj.Trajectory):
    num_atoms = traj.n_atoms
    dim = 3 * num_atoms
    InertiaTensorSecondPrincipalComponent = CustomFeature(fun=calc_second_principal_inertia_component, dim=dim)

    return InertiaTensorSecondPrincipalComponent

