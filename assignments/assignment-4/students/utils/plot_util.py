import matplotlib.pyplot as plt


def plot_alignments(training_points, aligned_points, mean_shape):
    """
    Plots original 2 point clouds (training points, aligned points) in left/right subplot with mean shape overlay
    :param training_points: (num_images x 2 x num_points) points before generalized procrustes alignment
    :param aligned_points: (num_images x 2 x num_points) points after generalized procrustes alignment
    :param mean_shape:  (2 x num_points) mean shape computed from generalized procrustes alignment
    :return: 
    """
    fig, axarr = plt.subplots(1, 2)
    num_images = len(training_points)
    for j in range(num_images):
        axarr[0].plot(training_points[j, :, 0], training_points[j, :, 1])
        axarr[0].plot(training_points[j, :, 0], training_points[j, :, 1], 'rx')

        axarr[1].plot(aligned_points[j, :, 0], aligned_points[j, :, 1])
        axarr[1].plot(aligned_points[j, :, 0], aligned_points[j, :, 1], 'rx')

        axarr[1].plot(mean_shape[:, 0], mean_shape[:, 1], 'bo')


    axarr[0].axis('equal')
    axarr[1].axis('equal')
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.draw()
    plt.waitforbuttonpress()



def plot_variation(mean_shape, variation_shape_list):
    """
    Plots each element of the variation_shape_list with the mean_shape as a subplot side by side.
    :param mean_shape:  (2 x num_points) mean shape computed from generalized procrustes alignment
    :param variation_shape_list: a list of (2 x num_points) points to visualize the variation of the mean shape when modifying individual values
    :return:
    """
    fig, axarr = plt.subplots(1, len(variation_shape_list))

    for i, variation_shape in enumerate(variation_shape_list):
        axarr[i].plot(mean_shape[:, 0], mean_shape[:, 1], color='blue')
        axarr[i].plot(mean_shape[:, 0], mean_shape[:, 1], 'bo')
        axarr[i].plot(variation_shape[:, 0], variation_shape[:, 1], color='green')
        axarr[i].plot(variation_shape[:, 0], variation_shape[:, 1], 'gx')
        axarr[i].axis('equal')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.draw()
    plt.waitforbuttonpress()



def plot_patches(patches):
    """
    Plots given patches in a combined plot
    :param patches: (18 x patch_size x patch_size) python list of image patches
    :return: 
    """
    num_patches = patches.shape[0]
    fig, axarr = plt.subplots(3, 6)
    x = 0
    y = 0
    for patch_index in range(num_patches):
        axarr[y][x].imshow(patches[patch_index], cmap=plt.cm.gray, interpolation='none')
        axarr[y][x].set_title(patch_index)
        x = x + 1
        if x % 6 == 0:
            x = 0
            y = y + 1
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.draw()
    plt.waitforbuttonpress()



