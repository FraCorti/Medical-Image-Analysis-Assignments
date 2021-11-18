import numpy as np
from utils.procrustes import align_multiple_with_procrustes


def generalized_procrustes_alignment(training_points, max_iteration=7, max_error=1e-4):
    """
    Aligns all points to an arbitrary chosen 'q' inside training_points using procrustes alignment
    :param training_points: point sets used to compute mean shape / alignment
    :param max_iteration: number of iterations to compute
    :param max_error: maximum error for convergence
    :return: mean_shape, aligned_points
    TODO: Task 1: Generalized Procrustes Alignment
    """
    # TODO: initialize the mean shape according to the assignment description
    mean_shape = training_points[0] - training_points[0].mean(0)
    aligned_points = training_points

    for i in range(10):
        aligned_points = align_multiple_with_procrustes(aligned_points, mean_shape)

        updated_mean_shape = 0
        for point_set in training_points:
            updated_mean_shape += point_set

        mean_shape = np.divide(updated_mean_shape, len(training_points))
        if np.power(np.sum(mean_shape - updated_mean_shape), 2) < max_error:
            break

    # TODO: implement a loop to optimize the mean shape
    # TODO: align all training points to the mean shape using procrustes
    #       hint: use align_multiple_with_procrustes()
    # TODO: update the mean shape as the mean of all aligned training points
    # TODO: calculate the sum of squared differences error between the current and previous mean shape and break
    #       early if the error is below a threshold

    return mean_shape, np.array(aligned_points)  # to plot return np.array(aligned_points)
