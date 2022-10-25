import numpy as np

from MarkovStateModel import MSM

test_state_centres = np.array([-2.0, -0.5, 1.0, 1.75, 0.0])
msm = MSM(test_state_centres)
test_transition_matrix = np.array([[0.2, 0.3, 0.1, 0.4, 0.0],
                                  [0.05, 0.15, 0.2, 0.3, 0.3],
                                  [0.7, 0.1, 0.1, 0.1, 0.0],
                                  [0.0, 0.2, 0.0, 0.4, 0.4],
                                  [0.2, 0.2, 0.2, 0.2, 0.2]])
msm.set_transition_matrix(test_transition_matrix)


def test_sorted_state_centers():
    expected_output = np.array([-2.0, -0.5, 0.0, 1.0, 1.75])
    output = msm.sorted_state_centers
    assert np.allclose(output, expected_output)


def test_compute_states_for_range():
    test_lower_value = -0.75
    test_upper_value = 1.0
    expected_output = np.array([1,2,3])
    output = msm.compute_states_for_range(test_lower_value, test_upper_value)
    assert np.allclose(output, expected_output)


def test_compute_diffusion_coefficient_domain():
    expected_output = np.array([-1.25, -0.25, 0.5, 1.375])
    output = msm.compute_diffusion_coefficient_domain()
    assert np.allclose(output, expected_output)


def test_calculate_correlation_coefficient():
    test_n = 3
    num_states = len(test_state_centres)
    expected_output = np.array(
        [np.sum([(msm.sorted_state_centers[i]
                  - msm.sorted_state_centers[j]) ** test_n
                * test_transition_matrix[i, j]
                for j in range(num_states)
            ])
         for i in range(num_states)])
    output = msm.calculate_correlation_coefficient(test_n)
    assert np.allclose(output, expected_output)
    test_n = 2
    expected_output = np.array(
        [np.sum([(msm.sorted_state_centers[i]
                  - msm.sorted_state_centers[j]) ** test_n
                * test_transition_matrix[i, j]
                for j in range(num_states)
            ])
         for i in range(num_states)])
    output = msm.calculate_correlation_coefficient(test_n)
    assert np.allclose(output, expected_output)


def test_relabel_trajectory_by_coordinate_chronology():
    test_traj = np.array([2, 3, 1, 4, 0])
    expected_output = np.array([3, 4, 1, 2, 0])
    output = msm.relabel_trajectory_by_coordinate_chronology(test_traj)
    assert np.allclose(output, expected_output)