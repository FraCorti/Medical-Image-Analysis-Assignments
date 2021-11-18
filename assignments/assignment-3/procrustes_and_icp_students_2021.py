import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.io as sio
import scipy.spatial.distance
import skimage.transform


def load_image(filename):
    """
    Loads images as grayscale.
    :param filename: filename to load
    :return: grayscale image as numpy array with type float64
    """
    image = io.imread(filename)
    return image.astype(np.float64)


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
    TODO implement: this function is needed for procrustes_demo() (Task 1) and icp_demo() (Task 2)
         Go through the comments of this function for more details on what has to be implemented.
    """
    # p and q need to have the same length. don't remove this line
    assert len(p) == len(q)

    # (1) Centre the points
    # transform p and q as such, that their center lies an the origin of the coordinate system (see lecture slides)
    p_cen = p - p.mean(0)  # Nx2 numpy array of corresponding centered 2D points
    q_cen = q - q.mean(0)  # Nx2 numpy array of corresponding centered 2D points

    # (2) Calculate scale of point sets; get frobenius norm of each and
    # (3) Normalize both point sets with respect to the calculated scale
    # use the frobenius norm to normalize p_cen and q_cen
    p_cen_norm = p_cen / np.linalg.norm(p_cen, ord='fro')
    q_cen_norm = q_cen / np.linalg.norm(q_cen, ord='fro')

    # (4) Calculate the rotation matrix R
    # first, calculate the prediction matrix H (see assignment description)
    H = np.dot(np.transpose(p_cen_norm), q_cen_norm)
    # then apply a singular value decomposition (SVD) to H to acquire U, Sigma and Vt
    U = np.linalg.svd(H)[0]
    Sigma = np.linalg.svd(H)[1]
    Vt = np.linalg.svd(H)[2]

    # Note: the orthogonal procrustes problem can be solved by calculating R = V dot Ut = U dot Vt yielding an orthogonal matrix
    # as we want to constrain R to a rotation matrix, we are going to solve the constrained procrustes problem
    # to do so, calculate the determinant of U dot Vt
    det = np.linalg.det(np.dot(U, Vt))

    # next, define a diagonal matrix D (see assignment description)
    D = np.diag([1, np.sign(det)])

    # the rotation matrix R can then be calculated by multiplying the matrices V, D and Ut in that order
    R = np.linalg.multi_dot([np.transpose(Vt), D, np.transpose(U)])

    # (5) Determine scale ratio (s)
    # calculate the scale ratio of the frobenius norm of q to the frobenius norm of p
    s = np.linalg.norm(q_cen, ord='fro') / np.linalg.norm(p_cen, ord='fro')

    # (6) Determine individual transformation matrices
    # with R defined, we can define the transformation matrix T to register the points p to q using homogeneous
    # coordinates,
    # define a translation matrix in homogeneous coordinates to translate the points p to the origin
    T_p_origin_hom = np.array([[1.0, 0.0, -p.mean(0)[0]],
                               [0.0, 1.0, -p.mean(0)[1]],
                               [0.0, 0.0, 1.0]])

    # define a scaling matrix in homogeneous coordinates using s as the scaling factors
    S_hom = np.diag([s, s, 1.0])

    # define the rotation matrix R in homogeneous coordinates
    # hint: make sure that the rotation is performed in the correct direction
    R_hom = np.array([[R[0, 0], R[0, 1], 0.0],
                      [R[1, 0], R[1, 1], 0.0],
                      [0.0, 0.0, 1.0]])

    # define a translation matrix in homogeneous coordinates to translate the points from the origin to the mean of q
    T_origin_q_hom = np.array([[1.0, 0.0, q.mean(0)[0]],
                               [0.0, 1.0, q.mean(0)[1]],
                               [0.0, 0.0, 1.0]])

    # define the final transformation matrix T by calculating the matrix multiplication of the homogeneous matrices in
    # the inverse order of their declaration
    T = np.linalg.multi_dot([T_origin_q_hom, R_hom, S_hom, T_p_origin_hom])

    return T


def apply_transformation(src, T):
    """
    Aligns src points by applying a transformation T. Returns the aligned src points.
    :param src: The source points (Nx2 numpy array).
    :param T: A transformation T as returned e.g. from procrustes().
    :return: The aligned src points (Nx2 numpy array).
    TODO implement: this function is required for procrustes_demo() (Task 1) and icp_demo() (Task 2)
         Go through the comments of this function for more details on what has to be implemented.
    """
    # use homogeneous coordinates to apply the matrix multiplication to align the points
    # hint: make_homogeneous() and make_cartesian()
    # hint: consider transposition

    src_homogeneous = make_homogeneous(src)
    src_homogeneous = np.transpose(np.dot(T, np.transpose(src_homogeneous)))
    src = make_cartesian(src_homogeneous)
    return src


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    :param src: Nx2 array of points
    :param dst: Mx2 array of points
    :return (distances, indices): N Euclidean distances and indices in dst of the nearest neighbor
    TODO implement: this function is required for icp_demo() (Task 2)
         Go through the comments of this function for more details on what has to be implemented.
    """
    # calculate the distances between all points in src and all points in dst
    # hint:
    distances = scipy.spatial.distance.cdist(dst, src, metric='euclidean')

    # get the index of the minimal distance for each point in src
    indices = []

    # iterate over the columns and extract the column index
    for column in range(distances.shape[1]):
        min_value = np.where(distances[:, column] == np.min(distances, axis=0)[column])
        row_index = min_value[0][0]
        indices.append(row_index)

    indices = np.array(indices)
    return distances, indices


def translate_and_scale_points(p, translate_x, translate_y, scale_x, scale_y):
    p[:, 0] += translate_x
    p[:, 1] += translate_y
    p[:, 0] *= scale_x
    p[:, 1] *= scale_y
    return p


def procrustes_demo(translate_x, translate_y, scale_x, scale_y):
    """
    Task 1
    Procrustes demo. This is the 'main' function of Task 1.
    :param translate_x: scalar to translate the point set p in x direction
    :param translate_y: scalar to translate the point set p in y direction
    :param scale_x: scalar to scale the point set p in x direction
    :param scale_y: scalar to scale the point set p in y direction
    TODO This is the 'main' function of Task 1.
         You need to implement procrustes() and apply_transformation() to solve Task 1.
         Both functions are required for Task 2 as well.
    """
    f = load_image('Im_00005PP5s.tif')
    shape_points = sio.loadmat('shape_points_5.mat')['ShapePoints'].T
    shape_points_transformed = sio.loadmat('shape_points_transformed_5.mat')['ShapePoints'].T
    shape_points_transformed = translate_and_scale_points(shape_points_transformed, translate_x, translate_y, scale_x,
                                                          scale_y)

    print(shape_points.shape)
    print(shape_points_transformed.shape)

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(f, cmap=plt.cm.gray)
    axarr[1].imshow(f, cmap=plt.cm.gray)
    axarr[0].scatter(shape_points[:, 0], shape_points[:, 1], marker='+', color='blue')
    axarr[0].scatter(shape_points_transformed[:, 0], shape_points_transformed[:, 1], marker='x', color='red')
    plt.draw()
    plt.waitforbuttonpress()

    # calculate final transformation
    # TODO implement procrustes()
    T = procrustes(shape_points_transformed, shape_points)
    print(T)

    # transform shape
    # TODO implement apply_transformation()
    shape_points_transformed = apply_transformation(src=shape_points_transformed, T=T)

    axarr[1].scatter(shape_points[:, 0], shape_points[:, 1], marker='+', color='blue')
    axarr[1].scatter(shape_points_transformed[:, 0], shape_points_transformed[:, 1], marker='x', color='red')
    plt.draw()
    plt.waitforbuttonpress()


def icp_demo(translate_x, translate_y, scale_x, scale_y):
    """
    Task 2
    iterative closest point (ICP) demo. This is the 'main' function of Task 2.
    :param translate_x: scalar to translate the point set p in x direction
    :param translate_y: scalar to translate the point set p in y direction
    :param scale_x: scalar to scale the point set p in x direction
    :param scale_y: scalar to scale the point set p in y direction
    TODO This is the 'main' function of Task 2.
         Go through the comments of this function for more details on what has to be implemented.
    Note: Task 2 depends on the functions you need to implement for Task 1, see procrustes_demo().
    """
    f = load_image('Im_00005PP5s.tif')
    f_shape = load_image('Im_00011PP5s.tif')
    points = sio.loadmat('ground_truth_points_5.mat')['GroundTruthPoints'].T
    shape_points = sio.loadmat('shape_points_11.mat')['ShapePoints'].T
    shape_points = translate_and_scale_points(shape_points, translate_x, translate_y, scale_x, scale_y)

    print(points.shape)
    print(shape_points.shape)

    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(f, cmap=plt.cm.gray)
    axarr[1].imshow(f_shape, cmap=plt.cm.gray)
    axarr[1].scatter(shape_points[:, 0], shape_points[:, 1], marker='x', color='red')
    axarr[2].imshow(f, cmap=plt.cm.gray)

    # src and dst points for icp
    src = shape_points
    dst = points
    f_transformed = f_shape

    # choose plausible values for the maximum number of iterations and the error tolerance to stop early
    max_iterations = 7
    tolerance = 1e-4

    for i in range(max_iterations):

        # implement a loop for the iterative closest point (ICP) algorithm (see lecture slides)
        # (1) in the icp loop, find the nearest neighbors between the current source and destination points
        # TODO implement nearest_neighbor()
        distances, indices = nearest_neighbor(src, dst)

        # plot of the current result and the nearest neighbors
        axarr[0].clear()
        axarr[0].imshow(f, cmap=plt.cm.gray)
        axarr[0].plot(dst[:, 0], dst[:, 1])
        axarr[0].scatter(src[:, 0], src[:, 1], marker='x', color='red')
        for i in range(len(indices)):
            line = np.array([src[i], dst[indices[i]]])
            axarr[0].plot(line[:, 0], line[:, 1], color='red')

        # (2) in the icp loop, compute the transformation between the current source and nearest destination points
        # hint: use the function procrustes() to get the transformation T
        T = procrustes(src, dst[indices])
        f_transformed = skimage.transform.warp(f_transformed, np.linalg.inv(T))

        # plot update step
        axarr[2].clear()
        axarr[2].imshow(f, cmap=plt.cm.gray)
        axarr[2].imshow(f_transformed, cmap=plt.cm.gray, alpha=0.5)
        plt.draw()
        plt.waitforbuttonpress()

        # (3) in the icp loop, update the current source using the transformation T
        # hint: use the function apply_transformation() to update the src points
        src = apply_transformation(src, T)

        # (4) in the icp loop, check the error and break early if the change to the previous iteration is below the tolerance
        if np.linalg.norm(src - dst[indices], ord=2) < tolerance:
            break

    # after the loop, calculate the final transformation
    # hint: use the function procrustes() to get the final transformation T
    _, indices = nearest_neighbor(src, dst)
    T = procrustes(src, dst[indices])
    f_transformed = skimage.transform.warp(f_shape, np.linalg.inv(T))

    # calculate the final shape points
    # hint: use the function apply_transformation() to update the original shape_points
    shape_points = apply_transformation(src, T)
    final_shape_points = shape_points

    axarr[1].clear()
    axarr[1].imshow(f_transformed, cmap=plt.cm.gray)
    axarr[1].scatter(final_shape_points[:, 0], final_shape_points[:, 1], marker='x', color='red')

    axarr[2].clear()
    axarr[2].imshow(f, cmap=plt.cm.gray)
    axarr[2].imshow(f_transformed, cmap=plt.cm.gray, alpha=0.5)
    plt.draw()
    plt.waitforbuttonpress()


def main():
    """
    The main function.
    TODO: Task 1: procurstes_demo()
          Task 2: icp_demo()
          Note: Task 2 is dependent on your implementation of Task 1.
          Either demo function can be executed independently.
    Note: This code uses Matplotlib to plot some intermediate and the final registration results and user input is
          expected to proceed code execution after each plot.
          The default settings of PyCharm Professional have SciView activated, which interferes with Matplotlib and
          consequently also with how this framework is set up.
          This problem does not apply if you are using Pycharm Community Edition as SciView is not available for it.
          SciView can be deactivated by unchecking the following:
          Settings | Tools | Python Scientific | Show Plots in Toolwindow
    """
    # TODO change the following variables and evaluate how they influence the results of Task 2
    #     discuss your findings in the scientific report
    translate_x = 0
    translate_y = 0
    scale_x = 1
    scale_y = 1
    procrustes_demo(translate_x, translate_y, scale_x, scale_y)
    icp_demo(translate_x, translate_y, scale_x, scale_y)


if __name__ == "__main__":
    main()
