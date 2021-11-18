
import numpy as np


def get_patches(training_points, patch_size, training_image_gradients):
    """

    :param training_points: (num_images x num_points x 2) matrix of points around which the mean patch will be extracted
    :param patch_size: (1) area around the points
    :param training_image_gradients: (num_images x image_size x image_size) images where the mean patches are extracted
    :return: (num_points x patch_size x patch_size) mean patches of training_image_gradients around training_points
    """
    number_of_points = training_points.shape[1]
    number_of_images = training_points.shape[0]
    mean_patches = []
    for point_index in range(number_of_points):
        patches = []
        for image_index in range(number_of_images):
            current_points = training_points[image_index, point_index, :]
            x_min = int(np.round(current_points[0] - np.floor(patch_size / 2))) - 1
            x_max = int(x_min + patch_size)
            y_min = int(np.round(current_points[1] - np.floor(patch_size / 2))) - 1
            y_max = int(y_min + patch_size)

            # get the current image
            current_image = training_image_gradients[image_index]
            current_patch = current_image[y_min:y_max, x_min:x_max]
            patches.append(current_patch)

        mean_patch = np.mean(patches, axis=0)
        mean_patches.append(mean_patch)

    mean_patches_np = np.asarray(mean_patches)

    return mean_patches_np
