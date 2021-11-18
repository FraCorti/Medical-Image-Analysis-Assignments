from utils import diff_ops
import numpy as np
from utils.procrustes import align_single_with_procrustes
from generalized_procrustes_alignment import generalized_procrustes_alignment
from principle_component_analysis import principle_component_analysis
from utils.princomp import princomp
from utils.get_patches import get_patches
from get_best_matches import get_best_matches
from update_shape_parameters import update_shape_parameters
from utils.plot_util import plot_patches
from utils.plot_util import plot_alignments
from utils.plot_util import plot_variation


def evaluate_variation(training_points):
    """
    This function evaluates the variation of the individual dimensions after the PCA.
    :param training_points: (num_images x 2 x num_points) points to train the ASM on
    :return:
    TODO: Evaluation: Use this function to evaluate the variation of the individual dimensions after the PCA
    TODO: Task 1 and 2 are required here. GOTO segment() if you have not finished these Tasks yet.
    """

    mean_shape, aligned_points = generalized_procrustes_alignment(training_points)
    eigenvalues, P = principle_component_analysis(aligned_points)

    # reshape the mean shape to [36 x 1]
    mean_shape = np.reshape(mean_shape, (36, 1))
    v_list = []

    dims = [0, 1]
    vals = [1000, 1]

    for dim in dims:
        for val in vals:
            b = np.zeros((eigenvalues.shape[0]))
            b[dim] = val
            v = mean_shape + np.reshape(np.dot(P, b), (36, 1))
            v = np.reshape(v, (18, 2))
            v_list.append(v)

    mean_shape = np.reshape(mean_shape,(18,2))
    plot_variation(mean_shape, v_list)


def segment(test_image, training_images, training_points, plot_results=True):
    """
    Compute the segmentation points (=shape) for the test_image after training an Iterative Active Shape Model.
    :param test_image: (img_size x img_size) image to be segmented
    :param training_images: (num_images x img_size x image_size) images to train the ASM on
    :param training_points: (num_images x num_points x 2) points to train the ASM on
    :return: segmentation_points: (num_points x 2) points fitting the segmentation of test_image
    TODO: Go through the comments of this function for more details on what has to be implemented.
    """

    # Task 1: Generalized Procrustes Alignment
    mean_shape, aligned_points = generalized_procrustes_alignment(training_points, max_iteration=7, max_error=1e-4)

    # check the plots and verify that everything works
    if plot_results:
        plot_alignments(training_points, aligned_points, mean_shape)

    # Task 2: Principle Component Analysis
    eigenvalues, P = principle_component_analysis(aligned_points)

    # Task 3: Gradient Magnitude and Mean Gradient Patches
    training_gradient_magnitudes = [np.sqrt(
        np.multiply(diff_ops.dx_forward(image), diff_ops.dx_forward(image)) + np.multiply(diff_ops.dy_forward(image),
                                                                                          diff_ops.dy_forward(image)))
        for image in training_images]  # [19, 256, 256]

    test_gradient_magnitude = np.sqrt(
        np.multiply(diff_ops.dx_forward(test_image), diff_ops.dx_forward(test_image)) + np.multiply(
            diff_ops.dy_forward(test_image),
            diff_ops.dy_forward(test_image)))  # [256, 256]

    # Compute mean gradient patches from the training_gradient_magnitudes and training_points
    patch_size = 15
    mean_patches = get_patches(training_points, patch_size, training_gradient_magnitudes)

    # check the plots and verify that everything works
    if plot_results:
        plot_patches(mean_patches)

    current_shape = mean_shape

    matching_area = patch_size * 3
    shape_list = []

    for i in range(15):
        best_matches = get_best_matches(test_gradient_magnitude, mean_patches, matching_area, current_shape)

        b_vector = update_shape_parameters(P, eigenvalues, mean_shape, best_matches)

        x_new = mean_shape + np.reshape(np.dot(P, b_vector), (18, 2))

        new_points = align_single_with_procrustes(x_new, best_matches)

        if (np.sum((np.power(current_shape - new_points, 2)))) < 1e-4:
            pass

        print(np.sum((np.power(current_shape - new_points, 2))))
        matching_area = matching_area - 2
        current_shape = new_points

        if matching_area < patch_size:
            break

        shape_list.append(current_shape)

    return shape_list
