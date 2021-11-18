import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import numpy.fft as fft
from skimage.transform import rotate
import scipy.fftpack
import scipy.interpolate
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.ndimage.interpolation
import skimage.io
from sklearn.metrics import mean_squared_error


def plot_filter_sinogram(sinogram, ramp_filter, filtered_sinogram):
    pass


def plot_pst_with_interpolation(sinogram, sinogram_fft, sx, sy, sinogram_fft2d, fft2d_original):
    """
    Plot sinogram, sinogram with slice-wise fft, singoram fft points put into 2d fft and original 2d fft for comparison.
    :param sinogram: The sinogram.
    :param sinogram_fft: The sinogram with slice-wise fft.
    :param sx: The x-coordinates of the 2d fft sinogram.
    :param sy: The y-coordinates of the 2d fft sinogram.
    :param sinogram_fft2d: The interpolated 2d fft sinogram.
    :param fft2d_original: The 2d fft ofr the original image for comparison.
    """
    pass


def plot_results(backprojection, filtered_backprojection, pst_with_interpolation, image):
    """
    Plots the results of the three methods for reconstruction.
    :param backprojection: The results from backprojection.
    :param filtered_backprojection: The result from filtered backprojection.
    :param pst_with_interpolation: The result from the projection slice theorem with interpolating.
    :param image: The original image for comparison.
    """
    fig, axarr = plt.subplots(2, 2)
    fig.tight_layout(pad=1)

    fontsize = 10
    axarr[0, 0].set_title('input', fontsize=fontsize)
    axarr[0, 1].set_title('backprojection', fontsize=fontsize)
    axarr[1, 0].set_title('filtered backprojection', fontsize=fontsize)
    axarr[1, 1].set_title('projection slice theorem with interpolation', fontsize=fontsize)
    if image is not None:
        axarr[0, 0].imshow(image, cmap='gray')
        axarr[0, 0].set_axis_off()
    if backprojection is not None:
        axarr[0, 1].imshow(backprojection, cmap='gray')
        axarr[0, 1].set_axis_off()
    if filtered_backprojection is not None:
        axarr[1, 0].imshow(filtered_backprojection, cmap='gray')
        axarr[1, 0].set_axis_off()
    if pst_with_interpolation is not None:
        axarr[1, 1].imshow(pst_with_interpolation, cmap='gray')
        axarr[1, 1].set_axis_off()


def load_image(filename):
    """
    Load the image at filename and return an np array.
    :param filename: The filename of the image to load.
    :return: The np.array of the loaded image.
    """
    image = skimage.io.imread(filename, as_gray=True)
    image = image.astype(np.float64)

    print(image.shape)
    print('image: min = ', np.min(image), ' max = ', np.max(image))

    return image


def pad_image(image):
    """
    Pad the image such that every rotation will fit into the padded image.
    :param image: The image to pad.
    :return: The padded image, the padding width.
    """
    # we assume image has same width and height
    padded_image_size = int(np.floor(image.shape[0] * np.sqrt(2)))
    # make sure padded image has odd dimensions
    if np.remainder(padded_image_size, 2) == 0:
        padded_image_size += 1

    padding_width = int((padded_image_size - image.shape[0]) / 2)

    # place the input image into a padded version, extending with zeros
    padded_image = np.zeros((padded_image_size, padded_image_size), dtype=np.float64)
    padded_image[padding_width:image.shape[0] + padding_width,
    padding_width:image.shape[1] + padding_width] = image

    return padded_image, padding_width


def crop_image(image, crop_width):
    """
    Crop an image by the given width. Inverse of pad_image.
    :param image: The image to crop.
    :param crop_width: The crop width.
    :return: The cropped image.
    """
    if image is None:
        return None
    cropped_image = image[crop_width:image.shape[0] - crop_width,
                    crop_width:image.shape[1] - crop_width]

    return cropped_image


def calculate_sinogram(image, num_angles):
    """
    Calculate a sinogram from the given image for the given number of angles.
    :param image: The image.
    :param num_angles: The number of angles.
    :return: The sinogram.
    """
    image_size = image.shape[0]
    sinogram = np.zeros([num_angles, image_size], dtype=np.float64)
    # go over all angles to calculate one projection line of the sinogram
    # rotate image by angle, counterclockwise (scipy.ndimage.rotate)
    # compute projection by summing up over the 0th axis
    # put projection lines into columns (axis 0)

    if num_angles == 180:
        sinogram = np.array([
            np.sum(scipy.ndimage.interpolation.rotate(image, theta, order=1, reshape=False, mode='constant', cval=0.0),
                   axis=0) for theta in list(range(num_angles))
        ])
    else:
        angles = np.linspace(0, 180, num_angles)
        current_angle = 0
        for i in range(num_angles):
            sinogram[i] = rotate(image, angles[current_angle]).sum(axis=0)
            current_angle = current_angle + 1

    return sinogram


def filter_sinogram(sinogram):
    """
    Filter the sinogram with a ramp filter.
    :param sinogram: The sinogram.
    :return: The filtered sinogram.
    """
    # create ramp filter in frequency domain (np.linspace)
    # normalize ramp filter (divide by sum)
    # go over all angles to receive one projection line of the sinogram
    # to apply discrete fft we need to shift to have zero in the center
    # multiply ramp filter in the frequency domain
    # inverse fft, extract real part, undo fftshift -> filtered line in the spatial domain
    # store filtered lines in output sinogram

    filtered_sinogram = np.zeros([sinogram.shape[0], sinogram.shape[1]], dtype=np.float64)
    ramp_filter = np.concatenate([np.linspace(182, 1, 182, endpoint=True), np.linspace(1, 181, 181)])
    ramp_filter = ramp_filter / np.sum(ramp_filter)

    for angle in range(sinogram.shape[0]):
        projection_line = fft.fft(sinogram[angle])
        frequency_domain_line = fft.fftshift(projection_line)
        frequency_domain_line = frequency_domain_line * ramp_filter
        filtered_sinogram[angle] = np.real(fft.ifft(fft.ifftshift(frequency_domain_line)))

    filtered_sinogram = np.real(filtered_sinogram)

    plot_filter_sinogram(sinogram, ramp_filter, filtered_sinogram)

    return filtered_sinogram


def backproject(sinogram):
    """
    Backproject a (filtered) sinogram to create a reconstruction.
    :param sinogram: The (filtered) sinogram.
    :return: The reconstructed image.
    """
    # create result zero image create image that contains a single sinogram line go through every angle i in the
    # filtered sinogram and calculate the filtered backprojection put projection line into image that contains a
    # single sinogram line, use np broadcasting, e.g., im[:, :] = sinogram[i, :] rotate image according to angle (
    # scipy.ndimage.rotate) add to result image

    backprojection = np.zeros([sinogram.shape[1], sinogram.shape[1]], dtype=np.float64)

    for angle in range(sinogram.shape[0]):
        image = np.zeros([sinogram.shape[1], sinogram.shape[1]], dtype=np.float64)

        # create image[363,363] by broadcasting the current angle stored in sinogram
        image[:, :] = sinogram[angle, :]
        image = rotate(image, angle)
        backprojection += image

    return np.flipud(backprojection)


def calculate_sinogram_fft(sinogram):
    """
    Perform slice-wise 1d fft for a given sinogram.
    :param sinogram: The sinogram.
    :return: Slice-wise 1d fft of the sinogram.
    """
    # go over all angles and calculate 1D fft
    # hint: use fft.fftshift, fft.fft, and fft.ifftshift
    sinogram_fft = np.array(scipy.fftpack.ifftshift(scipy.fftpack.fft(scipy.fftpack.fftshift(sinogram))))
    return sinogram_fft


def get_sx_sy(sinogram_fft):
    """
    Return the sinogram x and y coordinates when putting it into the 2d fft image.
    :param sinogram_fft: The sinogram 1d fft.
    :return: The x coordinates of where to put back the sinogram_fft, the y coordinates of where to put back the sinogram_fft.
    """
    # x coordinates are: int(image_size / 2) + r * np.cos(a)
    # y coordinates are: int(image_size / 2) + r * np.sin(a)
    # hint: use linspace for angles and radii
    # hint: either use np.meshgrid, or iterate over all angles and radii

    r = np.linspace(0, sinogram_fft.shape[1], sinogram_fft.shape[1], endpoint=False) - sinogram_fft.shape[1] / 2
    a = (math.pi / sinogram_fft.shape[0]) * np.linspace(0, sinogram_fft.shape[0], sinogram_fft.shape[0], endpoint=False)
    r, a = np.meshgrid(r, a)
    r = r.flatten()
    a = a.flatten()
    sx = int(sinogram_fft.shape[1] / 2) + r * np.cos(a)
    sy = int(sinogram_fft.shape[1] / 2) + r * np.sin(a)
    return sx, sy


def interpolate_sinogram_fft_for_fft2d(sinogram_fft, sx, sy):
    """
    Interpolates the sinogram 1d fft from the given x and y coordinates into a 2d fft.
    :param sinogram_fft: The sinogram 1d fft.
    :param sx: The x coordinates of where to put back the sinogram_fft.
    :param sy: The y coordinates of where to put back the sinogram_fft.
    :return: The interpolated 2d fft.
    """
    # hint: use scipy.interpolate.griddata
    # hint: use np.meshgrid to create the points at which to interpolate data.
    # hint: use np.flatten() and np.reshape()

    x_destination, y_destination = np.meshgrid(np.arange(sinogram_fft.shape[1]), np.arange(sinogram_fft.shape[1]))
    x_destination = x_destination.flatten()
    y_destination = y_destination.flatten()

    fft2d = scipy.interpolate.griddata(
        (sy, sx),
        sinogram_fft.flatten(),
        (y_destination, x_destination),
        method='linear',
        fill_value=0.0
    )
    return np.reshape(fft2d, (sinogram_fft.shape[1], sinogram_fft.shape[1]))


def perform_fft2d(image):
    """
    Returns the 2d fft of the given image.
    :param image: The image.
    :return: The 2d fft of the image.
    """
    return fft.fftshift(fft.fft2(fft.ifftshift(image)))


def perform_ifft2d(image_fft2d):
    """
    Returns the 2d ifft of the given fourier image.
    :param image_fft2d: The fft2d image.
    :return: The original image.
    """
    # hint: use fft.fftshift, fft.ifft2, and fft.ifftshift
    return np.real(fft.fftshift(fft.ifft2(fft.ifftshift(image_fft2d))))


def calculate_pst_with_interpolation(sinogram, fft2d_original):
    """
    Calculate the projection slice theorem by generating a 1d fft for each line of the sinogram, putting it
    back into an image to get the 2d fft of the image, performing an interpolation, and a final inverse 2d fft.
    :param sinogram: The sinogram.
    :param fft2d_original: The original 2d fft of the input image. Only used for comparison.
    :return: The reconstruction.
    """
    # calculate slice-wise 1d fft
    sinogram_fft = calculate_sinogram_fft(sinogram)
    # get sinogram coordinates in 2d fft
    sx, sy = get_sx_sy(sinogram_fft)
    # put slice-wise 1d fft into 2d fft
    sinogram_fft2d = interpolate_sinogram_fft_for_fft2d(sinogram_fft, sx, sy)
    # reconstruct the original image with the inverse 2d fft
    recon = perform_ifft2d(sinogram_fft2d)

    plot_pst_with_interpolation(sinogram, sinogram_fft, sx, sy, sinogram_fft2d, fft2d_original)

    return recon


def main():
    filename = 'CTThoraxSlice257.png'

    # load and pad image
    image = load_image(filename)

    padded_image, padding_width = pad_image(image)

    for num_angles in [180]:
        """
        The main function.
        """

        # num_angles = 180

        # calculate and save sinogram
        sinogram = calculate_sinogram(padded_image, num_angles)

        # calculate and crop backprojection
        backprojection = backproject(sinogram)
        cropped_backprojection = crop_image(backprojection, padding_width)

        print('cropped_backprojection: min = ', np.min(cropped_backprojection), ' max = ',
              np.max(cropped_backprojection))

        # calculate and crop filtered backprojection
        filtered_sinogram = filter_sinogram(sinogram)
        filtered_backprojection = backproject(filtered_sinogram)
        cropped_filtered_backprojection = crop_image(filtered_backprojection, padding_width)
        print('cropped_filtered_backprojection: min = ', np.min(cropped_filtered_backprojection), ' max = ',
              np.max(cropped_filtered_backprojection))

        # calculate and crop with the projection slice theorem
        # 2D fft from original image, only used for comparison
        fft2d_original = perform_fft2d(image)

        pst_with_interpolation = calculate_pst_with_interpolation(sinogram, fft2d_original)
        cropped_pst_with_interpolation = crop_image(pst_with_interpolation, padding_width)

        print('cropped_pst: min = ', np.min(cropped_pst_with_interpolation), ' max = ',
              np.max(cropped_pst_with_interpolation))
        plot_results(cropped_backprojection, cropped_filtered_backprojection, cropped_pst_with_interpolation, image)
        plt.show()

        # RMSE of reconstructed images

        rms1_1 = cropped_backprojection - image

        rmse_backprojection = np.sqrt(sum(rms1_1.flatten() ** 2 / len(rms1_1.flatten())))

        rms2_1 = cropped_filtered_backprojection - image
        rmse_cropped_filtered_backprojection = np.sqrt(sum(rms2_1.flatten() ** 2 / len(rms2_1.flatten())))

        rms3_1 = cropped_pst_with_interpolation - image
        rmse_pst_with_interpolation = np.sqrt(sum(rms3_1.flatten() ** 2 / len(rms3_1.flatten())))


        print(
            'Number of angles: {} RMSE Backprojection: {} RMSE Filtered-Backprojection: {} RMSE Projection Slice Theorem with interpolation:{}'.format(
                num_angles, rmse_backprojection, rmse_cropped_filtered_backprojection, rmse_pst_with_interpolation))


if __name__ == '__main__':
    main()
