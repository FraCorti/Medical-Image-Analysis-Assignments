import numpy as np
from utils.procrustes import align_single_with_procrustes


def update_shape_parameters(P, latent, mean_shape, best_matches):
    """
    :param P: (2*num_points x t) matrix of eigenvectors, where N is the number of points and t is the reduced number of dimensions
    :param latent: (t x 1) vector of eigenvalues of the PCA model, should be the same for each call of the function
    :param mean_shape: (num_points x 2) matrix for mean shape
    :return: b_vector (t x 1) vector of updated shape parameters of the PCA model
    TODO: Task 4: Step 2.2: Update shape parameters
    """

    # TODO: Align the best matches to the mean shape using Procrustes
    # TODO: Hint: Use align_single_with_procrustes()

    # TODO: Follow the formulas in the assignment description to compute b_vector
    # TODO: Hint: Consider reshaping

    u_m = align_single_with_procrustes(best_matches, mean_shape)
    dx = np.reshape(u_m - mean_shape, (36, 1))
    b_vector = np.dot(np.transpose(P), dx)

    eigenvalue_position = 0
    for i in range(len(b_vector)):
        if b_vector[i] < -2 * np.sqrt(latent[eigenvalue_position]):
            b_vector[i] = -2 * np.sqrt(latent[eigenvalue_position])
        if b_vector[i] > 2 * np.sqrt(latent[eigenvalue_position]):
            b_vector[i] = 2 * np.sqrt(latent[eigenvalue_position])
        eigenvalue_position += 1

    return b_vector  # [t, 1]
