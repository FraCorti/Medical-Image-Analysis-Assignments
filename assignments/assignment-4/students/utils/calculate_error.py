
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def calculate_jaccard_index(prediction_mask, groundtruth_mask):
    """
    Calculates the Jaccard index.
    :param prediction_mask: Binary prediction mask.
    :param groundtruth_mask: Binary groundtruth mask.
    :return: The Jaccard index.
    """
    overlap = np.logical_and(prediction_mask, groundtruth_mask)
    union = np.logical_or(prediction_mask, groundtruth_mask)
    overlap_sum = np.sum(overlap)
    union_sum = np.sum(union)
    return overlap_sum.astype(float) / union_sum.astype(float)


def points_to_segmentation_mask(points, max_coordinate=180):
    """
    Creates a binary segmentation mask image for a given polygon
    :param points: The polygon.
    :param max_coordinate: Maximum image coordinate.
    :return: The binary segmentation mask.
    """
    X, Y = np.meshgrid(np.linspace(0, max_coordinate, num=max_coordinate),
                       np.linspace(0, max_coordinate, num=max_coordinate))
    XY = np.dstack((X, Y))
    XY_flat = XY.reshape([-1, 2])
    path = Path(points)  # the vertices of the polygon
    mask_flat = path.contains_points(XY_flat)
    return mask_flat.reshape(X.shape)


def calculate_error(prediction_points, groundtruth_points, plot_results=True):
    """
    Calculates the error of the found segmentation (given by
    found_points) regarding the specified groundtruth_points.
    "Jaccard index" (overlap / union) is used for error calculation.
    Both point sets, overlap and union are plotted.
    :param prediction_points: nx2 matrix of n points fitted to the image. 
                         First column represents the x-,
                         second column the y-coordinates.
    :param groundtruth_points: kx2 matrix of k points representing the
                               ground-truth of the bone-sementation, the
                               contour, resp. (=> loaded from *.csv- or 
                               *.mat-file).
                              (k ~= n)
    :param plot_results: If True, results are plotted.
    :return: error: error of the found segmnetation regarding the groundtruth (1 - Jaccard index)
    """
    #tck, u = scipy.interpolate.splprep(np.concatenate([prediction_points, prediction_points[0:1]]).T, s=10, k=3, per=True)
    #x2, y2 = scipy.interpolate.splev(np.linspace(0, 1, 1000), tck)
    #np.stack([x2, y2]).T

    prediction_mask = points_to_segmentation_mask(prediction_points)
    groundtruth_mask = points_to_segmentation_mask(groundtruth_points)

    # compute "Jaccard Index" and error, respectively
    jaccard_index = calculate_jaccard_index(prediction_mask, groundtruth_mask)
    error = 1.0 - jaccard_index

    if plot_results:
        fig, axarr = plt.subplots(2, 2)

        axarr[0][0].plot(prediction_points[:, 0], prediction_points[:, 1], 'go')
        axarr[0][0].axis('equal')
        axarr[0][0].invert_yaxis()
        axarr[0][0].plot(groundtruth_points[:, 0], groundtruth_points[:, 1], 'r')
        #axarr[0][0].plot(x2, y2, 'g')
        axarr[0][0].set_title('prediction (green) and groundtruth (red) points')

        axarr[0][1].imshow(prediction_mask, cmap=plt.cm.gray, interpolation='none')
        axarr[0][1].set_title('prediction segmentation')

        axarr[1][0].imshow(groundtruth_mask, cmap=plt.cm.gray, interpolation='none')
        axarr[1][0].set_title('groundtruth segmentation')

        overlap = np.logical_and(prediction_mask, groundtruth_mask)
        union = np.logical_or(prediction_mask, groundtruth_mask)
        overlap_and_union = union.astype(float)
        overlap_and_union[overlap] = 0.5
        axarr[1][1].imshow(overlap_and_union, cmap=plt.cm.gray, interpolation='none')
        axarr[1][1].set_title("overlap (grey) and union (error = {:f})".format(error))

        plt.draw()
        plt.waitforbuttonpress()

    return error
