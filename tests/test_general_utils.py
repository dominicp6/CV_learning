import numpy as np
from scipy.special import erf

from utils.general_utils import (
    replace_inf_with_nan,
    rms_interval,
    vector_rmsd,
    linear_interp_coordinate_data,
    gaussian_smooth,
    select_lowest_minima,
)


def test_replace_inf_with_nan():
    test_array = np.array([0, 0.0, -2, 3.56, np.pi, np.inf, np.nan, -np.inf])
    expected_output = np.array([0, 0.0, -2, 3.56, np.pi, np.nan, np.nan, np.nan])
    output = replace_inf_with_nan(test_array)
    assert np.array_equal(output, expected_output, equal_nan=True)


def test_rms_interval():
    test_array = np.array([0, 1, 2, 3, 4, 0, 2])
    expected_output = np.sqrt((1 + 1 + 1 + 1 + 4**2 + 2**2) / 6)
    output = rms_interval(test_array)
    assert output == expected_output


def test_vector_rmsd():
    test_array_x = np.array([1.0, 0, -1])
    test_array_y = np.array([2.0, -1.0, 2.0])
    expected_output = np.sqrt((1 + 1 + 3**2) / 3)
    output = vector_rmsd(test_array_x, test_array_y)
    assert output == expected_output


def test_linear_interp_coordinate_data():
    test_x_data = np.array([-4.0, -3.0, -2.5, -1.5, -0.5, 0.0, 1.0, 4.0])
    test_y_data = np.array([10.0, 6.0, 3.0, 3.0, 5.0, -1.0, -2.0, 6.0])
    test_x_to_evaluate = 0.0
    expected_output = -1.0
    output = linear_interp_coordinate_data(test_x_data, test_y_data, test_x_to_evaluate)
    assert output == expected_output
    test_x_to_evaluate = 0.5
    expected_output = -1.5
    output = linear_interp_coordinate_data(test_x_data, test_y_data, test_x_to_evaluate)
    assert output == expected_output
    test_x_to_evaluate = -3.5
    expected_output = 8.0
    output = linear_interp_coordinate_data(test_x_data, test_y_data, test_x_to_evaluate)
    assert output == expected_output
    test_x_to_evaluate = 7.0
    expected_output = 14.0
    output = linear_interp_coordinate_data(test_x_data, test_y_data, test_x_to_evaluate)
    assert output == expected_output
    test_x_to_evaluate = -6.0
    expected_output = 18.0
    output = linear_interp_coordinate_data(test_x_data, test_y_data, test_x_to_evaluate)
    assert output == expected_output


def test_gaussian_smooth():
    # TODO: fix/understand gaussian smooth
    test_x = np.array([-1.0, 0.0, 1.0])
    test_y = np.array([1.0, 4.0, 1.0])
    test_dx = 0.2
    test_sigma = 1.0
    expected_interpolated_x = np.array(
        [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    expected_smoothed_y = (
        (
            np.array(
                [
                    2 * erf((1 - np.abs(x)) / (np.sqrt(2) * test_sigma))
                    + (3 * test_sigma / np.sqrt(2 * np.pi))
                    * (np.exp(-((1 - np.abs(x)) ** 2) / (2 * test_sigma**2)) - 1)
                    for x in expected_interpolated_x
                ]
            )
        )
        + (3 * test_sigma / np.sqrt(2 * np.pi))
        * (np.exp(-1 / (2 * test_sigma**2)) - 1)
        + 2 * erf(1 / (np.sqrt(2) * test_sigma))
    )
    interpolated_x, smoothed_y = gaussian_smooth(test_x, test_y, test_dx, test_sigma)
    assert np.allclose(interpolated_x, expected_interpolated_x)
    assert np.allclose(smoothed_y, expected_smoothed_y)


def test_select_lowest_minima():
    test_minima_array = np.array(
        [[0.0, -2.0], [-2.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [0.5, 0.25]]
    )
    test_function = lambda x, y: np.sqrt(x**2 + y**2)
    test_n = 1
    expected_output = np.array([0.5, 0.25])
    output = select_lowest_minima(test_minima_array, test_function, test_n)
    assert np.allclose(output, expected_output)
    test_n = 2
    expected_output = np.array([[0.5, 0.25], [1.0, 1.0]])
    output = select_lowest_minima(test_minima_array, test_function, test_n)
    assert np.allclose(output, expected_output)
