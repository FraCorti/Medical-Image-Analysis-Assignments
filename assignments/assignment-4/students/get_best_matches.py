import numpy as np
from utils.index_of_best_match_ncc import index_of_best_match_ncc


def get_best_matches(test_gradient_magnitude, mean_patches, matching_area, current_shape):
    """
    Get the indices of the best matches between the test_gradient_magnitude in the matching_area and the respective
    mean_patch.
    :param test_gradient_magnitude: (image_size x image_size) gradient of the test image
    :param mean_patches: (num_points x patch_size x patch_size) mean gradient patches of the training set
    :param matching_area: (1) area where matches are searched
    :param current_shape: (num_points x 2) current shape coordinate matrix
    :return: best matches
    TODO: Task 4: Step 2.1: Get the best matches of the ASM for the test gradient magnitude
    """

    # TODO: For each point of the current_shape:
    # TODO: Calculate the minimal and maximal index in x and y that is still inside of the matching area
    # TODO: Get the index of the best matching points of the mean gradient patches in the respective matching area of the test gradient magnitude
    # TODO: Hint: Use index_of_best_match_ncc() to get the index one at a time

    best_matches = []  # [18, 2]
    patch = 0
    for point in current_shape:
        index_x_lower = int(np.round(point[0]) - np.floor(matching_area / 2))
        index_x_upper = int(index_x_lower + matching_area)

        index_y_lower = int(np.round(point[1]) - np.floor(matching_area / 2))
        index_y_upper = int(index_y_lower + matching_area)

        best_match_y, best_match_x = index_of_best_match_ncc(mean_patches[patch], test_gradient_magnitude,
                                                             index_x_lower,
                                                             index_x_upper, index_y_lower, index_y_upper)
        best_matches.append([best_match_x, best_match_y])
        patch += 1

    return np.array(best_matches)
