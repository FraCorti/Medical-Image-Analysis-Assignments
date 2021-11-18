import numpy as np


def princomp(A):
    """
    Principal component analysis. Returns the Eigenvectors and the Eigenvalues of the covariance matrix of a given matrix A, sorted by Eigenvalues in descending order.
    The column eigenvector[:, i] is the normalized eigenvector corresponding to the eigenvalue[i].
    :param A: The numpy array of size NxK, where N is the number of samples and K is the feature dimension.
    :return: eigenvalues, eigenvectors
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # subtract the mean (along columns)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(M))
    idx = np.argsort(eigenvalues)  # sorting the eigenvalues
    idx = idx[::-1]                # in descending order
    # sorting eigenvectors according to the sorted eigenvalues
    eigenvalues = eigenvalues[idx]  # sorting eigenvalues
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors
