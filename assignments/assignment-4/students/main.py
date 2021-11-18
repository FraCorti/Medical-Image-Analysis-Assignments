import numpy as np
import scipy.io as sio
from skimage import io
import os
import segmentation
from utils.calculate_error import calculate_error


def load_image(filename):
    """
    Loads images as grayscale.
    :param filename: filename to load
    :return: grayscale image as numpy array with type float64
    """
    # image = ndimage.imread(filename, flatten=True)
    image = io.imread(filename)
    return image.astype(np.float64)


def load_images(directory):
    """
    Loads a list of images from a directory as grayscale.
    :param directory: folder where the images are loaded from
    :return: python list of (image_size x image_size) numpy arrays
    """
    images = [load_image(os.path.join(directory, filename))
              for filename in sorted(os.listdir(directory))]
    return images


def load_points(directory):
    """
    Loads groundtruth and contour points from a directory.
    :param directory: folder where groundtruth / corresponding points are loaded from
    :return: groundtruth_points: python list containing point clouds as numpy arrays
             selected_corresponding_points: python list containing point clouds as numpy arrays
    """
    groundtruth_points = []
    selected_corresponding_points = []
    for filename in sorted(os.listdir(directory)):
        contours_and_points = sio.loadmat(os.path.join(directory, filename))
        groundtruth_points.append(contours_and_points['GroundTruthPoints'])
        selected_corresponding_points.append(contours_and_points['SelectedCorrespondingPoints'].T)

    return groundtruth_points, selected_corresponding_points


def main():
    """
    The main function.
    Note: This code uses Matplotlib to plot some intermediate and the final registration results and user input is
          expected to proceed code execution after each plot.
          The default settings of PyCharm Professional have SciView activated, which interferes with Matplotlib and
          consequently also with how this framework is set up.
          This problem does not apply if you are using Pycharm Community Edition as SciView is not available for it.
          SciView can be deactivated by unchecking the following:
          Settings | Tools | Python Scientific | Show Plots in Toolwindow
    """
    groundtruth_points, contour_points = load_points('contoursAndPoints/')
    input_images = load_images('images')

    plot_intermediate_results = True
    plot_final_results = True
    plot_and_evaluate_variation = True

    num_images = len(input_images)
    errors = []

    # leave-one-out cross-validation: choose a different test image for each iteration
    # for test_image_index in range(num_images):
    for test_image_index in range(1):
        print('cross-validation, test_image: ', test_image_index + 1, '/', num_images)

        # extract test image for current cross-val iteration
        test_image = input_images[test_image_index]

        # get a list excluding the chosen test image for the training set
        training_images = list(input_images)
        training_images.pop(test_image_index)

        # also create a list of points for the images excluding the test image
        training_points = list(contour_points)
        training_points.pop(test_image_index)

        # do segmentation
        points_list = segmentation.segment(np.asarray(test_image), np.asarray(training_images),
                                           np.asarray(training_points), plot_results=plot_intermediate_results)
        final_points = points_list[-1]
        error = calculate_error(np.asarray(final_points), np.asarray(groundtruth_points[test_image_index]).T,
                                plot_results=plot_final_results)
        errors.append(error)

        if plot_and_evaluate_variation:
            segmentation.evaluate_variation(np.asarray(training_points))

    mean_error = np.mean(np.asarray(errors))
    print('mean error over cross validation: ', mean_error)


if __name__ == "__main__":
    main()
