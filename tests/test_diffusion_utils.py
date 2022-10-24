import numpy as np

from diffusion_utils import compute_well_integrand, compute_well_integrand_from_potential, compute_barrier_integrand_from_potential, compute_well_and_barrier_integrals, project_points_to_line, relabel_trajectory_by_coordinate_chronology, calculate_cni, calculate_c


def test_compute_well_integrand():
    test_array = np.array([-1.0, 0.0, 2.0])
    test_beta = 0.5
    expected_output = np.array([np.exp(-test_beta * x) for x in test_array])
    output = compute_well_integrand(test_array, test_beta)
    assert np.allclose(output, expected_output)


def test_compute_well_integrand_from_potential():
    test_potential = lambda x: x**2
    test_beta = 0.5
    test_array = np.array([-1.0, 0.0, 2.0])
    expected_output = np.array([np.exp(-test_beta * test_potential(x)) for x in test_array])
    output = compute_well_integrand_from_potential(test_potential, test_beta, test_array)
    assert np.allclose(output, expected_output)


def test_compute_barrier_integrand_from_potential():
    test_potential = lambda x: x**2
    test_diffusion_function = lambda x: np.exp(-x**2)
    test_beta = 0.5
    test_array = np.array([-1.0, 0.0, 2.0])
    expected_output = np.array([np.exp(test_beta * test_potential(x)) / test_diffusion_function(x) for x in test_array])
    output = compute_barrier_integrand_from_potential(test_potential, test_beta, test_diffusion_function, test_array)
    assert np.allclose(output, expected_output)


# TODO: complete test
def test_compute_well_and_barrier_integrals():
    pass

# TODO: fix and understand this function
def test_project_points_to_line():
    test_points = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    test_coords = np.array([0.0, 2.0])
    test_theta = np.pi / 4
    expected_output = np.array([[-1.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
    output = project_points_to_line(test_points, test_coords, test_theta)
    print(expected_output)
    print(output)
    assert np.allclose(output, expected_output)


def test_relabel_trajectory_by_coordinate_chronology():
    test_traj = np.array([0, 1, 2, 3, 0, 4])
    test_state_centres = np.array([0.0, -1.0, 2.0, -3.0, 0.5])
    expected_output = np.array([2, 1, 4, 0, 2, 3])
    output = relabel_trajectory_by_coordinate_chronology(test_traj, test_state_centres)
    assert np.allclose(output, expected_output)


def test_calculate_cni():
    test_i = 1
    test_n = 2
    test_X = np.array([-3.0, -1.0, 0.0])
    test_P = np.array([[0.1,  0.6, 0.3],
                      [0.2,  0.4, 0.4],
                      [0.1,  0.6, 0.3]])
    expected_output = (test_X[test_i] - test_X[0])**test_n * test_P[test_i,0] + \
                      (test_X[test_i] - test_X[1])**test_n * test_P[test_i,1] + \
                      (test_X[test_i] - test_X[2])**test_n * test_P[test_i,2]
    output = calculate_cni(test_i, test_X, test_n, test_P)
    assert np.allclose(output, expected_output)


def test_calculate_c():
    test_X = np.array([-3.0, -1.0, 0.0])
    test_n = 2
    test_P = np.array([[0.1,  0.6, 0.3],
                      [0.2,  0.4, 0.4],
                      [0.1,  0.6, 0.3]])
    expected_output = [(test_X[0] - test_X[0])**test_n * test_P[0,0] + \
                      (test_X[0] - test_X[1])**test_n * test_P[0,1] + \
                      (test_X[0] - test_X[2])**test_n * test_P[0,2],
                      (test_X[1] - test_X[0])**test_n * test_P[1,0] + \
                      (test_X[1] - test_X[1])**test_n * test_P[1,1] + \
                      (test_X[1] - test_X[2])**test_n * test_P[1,2],
                      (test_X[2] - test_X[0])**test_n * test_P[2,0] + \
                      (test_X[2] - test_X[1])**test_n * test_P[2,1] + \
                      (test_X[2] - test_X[2])**test_n * test_P[2,2]]
    output = calculate_c(test_X, test_n, test_P)
    assert np.allclose(output, expected_output)