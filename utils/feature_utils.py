import numpy as np
import mdtraj
from pyemma.coordinates.data import CustomFeature
from tqdm import tqdm

from Experiment import Experiment

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


def compute_best_fit_feature_eigenvector(exp: Experiment, cv: str, dimensions_to_keep: int, stride: int = 1, features=None, **kwargs):
    """
    Computes the best fit feature eigenvector for a given CV for a given number of dimensions to keep.
    Uses a greedy algorithm by iteratively adding the feature with the highest correlation to the current eigenvector.

    :param exp: Experiment object
    :param cv: CV name
    :param dimensions_to_keep: Number of dimensions to keep
    :param stride: Stride for reading the trajectory
    :param features: List of features to use. If None, all features are used.
    :return: List of best-fit features and the corresponding best-fit coefficients
    """

    list_of_features = []

    if ":" in cv:
        cv_type = cv.split(':')[0]
        dim = int(cv.split(':')[1])
    else:
        raise ValueError("CV must be of the form 'CV_type:dim'")

    if features is None:
        features = exp.featurizer.describe()

    exp.compute_cv(cv_type, dim=dim + 1, stride=stride, verbose=False, **kwargs)
    cv_data = exp._get_cv(cv_type, dim=dim, stride=stride)

    best_correlations = []

    for dim in tqdm(range(dimensions_to_keep)):

        correlations = {}
        feature_coeffs = {}

        if dim == 0:
            for feature in features:
                # TODO: fix constant term when only keeping one dimension
                feat_traj = exp.get_feature_trajs_from_names([feature])[::stride]
                correlations[feature] = np.corrcoef(cv_data, feat_traj)[0, 1]
                feature_coeffs[feature] = 1

            abs_correlations = {key: abs(value) for key, value in correlations.items()}
            best_feature = max(abs_correlations, key=abs_correlations.get)
            best_correlations.append(correlations[best_feature])
        else:
            for feature in features:
                all_features = list_of_features + [feature]
                feat_data = exp.get_feature_trajs_from_names(all_features)[::stride]
                feat_data = np.c_[np.ones(exp.num_frames), feat_data.T]
                coeffs, err, _, _ = np.linalg.lstsq(feat_data, cv_data, rcond=None)
                feature_coeffs[feature] = coeffs
                feat_traj = np.dot(feat_data, coeffs)
                correlations[feature] = np.corrcoef(cv_data, feat_traj)[0, 1]

            best_feature = max(correlations, key=correlations.get)
            best_correlations.append(correlations[best_feature])

        list_of_features.append(best_feature)

    return list_of_features, feature_coeffs[best_feature], best_correlations





