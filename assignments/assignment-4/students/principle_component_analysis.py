import numpy as np
from utils.princomp import princomp


def principle_component_analysis(aligned_points):
    """
    Applies a principle component analysis (PCA) and returns the most important eigenvalues and eigenvectors
    :param aligned_points: point sets aligned to the mean shape and used for PCA
    :return: eigenvalues, P (eigenvectors)
    TODO: Task 2: Principle Component Analysis
    """

    # TODO: Use np.reshape to convert aligned_points from [19, 18, 2] matrix to [19, 36] matrix and use princomp() to apply the PCA
    eigenvalues, P = princomp(np.reshape(aligned_points, (19, 36)))

    # TODO: Keep t eigenmodes of PCA, such that 97% of variance in training data is kept (hint: use np.cumsum and np.argmax)
    cumulative_eigenvalues_sum = np.cumsum(eigenvalues)
    maximum = cumulative_eigenvalues_sum[np.argmax(cumulative_eigenvalues_sum)]

    t = -1
    for eigenvalue in cumulative_eigenvalues_sum:
        if eigenvalue / maximum >= 0.97:
            break
        t += 1

    eigenvalues = eigenvalues[0:t]
    P = P[:, 0:t]

    return eigenvalues, P
