import numpy as np


def make_homogeneous(points):
    """
    Convert Cartesian to homogeneous coordinates.
    :param points: Nx2 numpy array of Cartesian coordinates
    :return: Nx3 numpy array of homogeneous coordinates
    """
    h_points = np.ones((points.shape[0], 3), points.dtype)
    h_points[:, 0:2] = np.copy(points)

    return h_points


def make_cartesian(h_points):
    """
    Convert homogeneous to Cartesian coordinates.
    :param h_points: Nx3 numpy array of homogeneous coordinates
    :return: Nx2 numpy array of Cartesian coordinates
    """
    points = np.zeros((h_points.shape[0], 2), h_points.dtype)
    points[:, 0] = h_points[:, 0] / h_points[:, 2]
    points[:, 1] = h_points[:, 1] / h_points[:, 2]

    return points


def procrustes(p, q):
    """
    Calculates the least-squares best-fit transform between corresponding 2D points p->q
    :param p: Nx2 numpy array of corresponding 2D points
    :param q: Nx2 numpy array of corresponding 2D points
    :return: 3x3 homogeneous transformation matrix
    """
    assert len(p) == len(q)

    # (1) Centre the points
    p_mean = np.mean(p, axis=0)
    q_mean = np.mean(q, axis=0)
    p_cen = p - p_mean
    q_cen = q - q_mean

    # (2) Calculate scale of point sets; get frobenius norm
    p_norm = np.linalg.norm(p_cen)
    q_norm = np.linalg.norm(q_cen)

    # (3) Normalize both data sets with respect to the calculated scale
    p_cen_norm = p_cen / p_norm
    q_cen_norm = q_cen / q_norm

    # (4) rotation matrix of B
    H = p_cen_norm.T @ q_cen_norm
    U, D, Vt = np.linalg.svd(H)
    det = np.linalg.det(U @ Vt)
    R = U @ np.diag([det, det]) @ Vt

    # (5) Determine scale ratio (s)
    #s = det * q_norm / p_norm
    s = 1

    # (6) Determine individual transformation matrices
    # translate points to origin
    T_p_hom = np.eye(3)
    T_p_hom[:2, 2] = -p_mean.T
    # scale points
    S_hom = np.eye(3)
    S_hom[0, 0] = s
    S_hom[1, 1] = s
    # rotate points
    R_hom = np.eye(3)
    R_hom[:2, :2] = R.T
    # translate points to mean q position
    T_q_hom = np.eye(3)
    T_q_hom[:2, 2] = q_mean.T

    # multiply matrices
    T = T_q_hom @ R_hom @ S_hom @ T_p_hom

    return T


def align_single_with_procrustes(src, dst):
    """
    Aligns src points with dst points. Returns the aligned src points.
    :param src: The source points (Nx2 numpy array).
    :param dst: The destination points (Nx2 numpy array).
    :return: The aligned src points (Nx2 numpy array).
    """
    T = procrustes(src, dst)

    src_h = make_homogeneous(src)
    src_transformed_h = (T @ src_h.T).T
    src_transformed = make_cartesian(src_transformed_h)

    return src_transformed


def align_multiple_with_procrustes(sources, dst):
    """
    Aligns a list of src points with dst points. Returns the list of aligned src points.
    :param sources: The list of source points (list of Nx2 numpy array).
    :param dst: The destination points (Nx2 numpy array).
    :return: The aligned src points (list of Nx2 numpy array).
    """
    aligned_points = [align_single_with_procrustes(src, dst) for src in sources]
    return aligned_points
