import numpy as np
import mdtraj
import pyemma
from pyemma.coordinates.data import CustomFeature
from tqdm import tqdm


def get_periodic_labels(featurizer: pyemma.coordinates.featurizer) -> list[bool]:
    """
    Returns a list of booleans indicating whether a given feature is periodic or not.

    :param featurizer: Featurizer object
    :return: List of booleans
    """
    periodic_labels = []
    for feature in featurizer.active_features:
        if feature.periodic:
            periodic_labels.append(True)
        else:
            periodic_labels.append(False)

    return periodic_labels


def get_feature_means(all_features: list[str], all_means: list[float], selected_features: list[str]):
    """
    Returns the means of the selected features.

    :param all_features: List of all features
    :param all_means: List of all means
    :param selected_features: List of selected features
    :return: List of means of selected features
    """
    feature_means = []
    for feature in selected_features:
        feature_means.append(all_means[all_features.index(feature)])

    return feature_means



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


def get_cv_type_and_dim(cv: str):
    if ":" in cv and "DIH:" not in cv:
        cv_type = cv.split(':')[0]
        dim = int(cv.split(':')[1])
        traditional_cv = True
    else:
        cv_type = cv
        dim = 0
        traditional_cv = False

    # Tradition CVs refers to PCA, TICA, etc.
    return traditional_cv, cv_type, dim

def get_features_and_coefficients(exp, cvs: list[str], dimensions_to_keep: int = None, stride: int = 1):
    if dimensions_to_keep is None:
        # Keep all dimensions, i.e. use all features
        features = [exp.featurizer.describe() for _ in cvs]
        coefficients = []
        for cv in cvs:
            traditional_cv, cv_type, cv_dim = get_cv_type_and_dim(cv)
            if traditional_cv:
                coefficients.append(exp.feature_eigenvector(cv_type, dim=cv_dim))
            else:
                # No need to compute coefficients for individual features
                continue
    else:
        # Compute best-fit features
        features = []
        coefficients = []
        for cv in cvs:
            traditional_cv, cv_type, cv_dim = get_cv_type_and_dim(cv)
            if traditional_cv:
                # Then we need to compute the best-fit feature vector
                features_, coefficients_, _ = compute_best_fit_feature_eigenvector(exp,
                                                                                cv,
                                                                                dimensions_to_keep,
                                                                                stride=stride)
                features.append(features_)
                # 1: to get rid of the constant term
                coefficients.append(coefficients_[1:])
            else:
                # The CV itself is a feature
                features.append([cv])
                coefficients.append([1])

    return features, coefficients


def compute_best_fit_feature_eigenvector(exp, cv: str, dimensions_to_keep: int, stride: int = 1, features=None):
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

    traditional_cv, cv_type, cv_dim = get_cv_type_and_dim(cv)

    if not traditional_cv:
        raise ValueError("CV must be of the form 'CV_type:dim'")

    if features is None:
        features = exp.featurizer.describe()

    cv_data = exp._get_cv(cv_type, dim=cv_dim, stride=stride)

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





