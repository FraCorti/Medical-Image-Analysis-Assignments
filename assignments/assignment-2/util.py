from scipy import ndimage
import imageio
import numpy as np


def load_image(filename):
    image = imageio.imread(filename)
    # input image is color png depicting grayscale, just use first plane from here on
    image = image[:, :, 1].astype(np.float64)
    # print(image.shape)
    # print('image: min = ', np.min(image), ' max = ', np.max(image))

    return image


def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)

    return np.sqrt(mse)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0

    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def add_gaussian_noise(image, mean=0, var=20):
    return image + np.random.standard_normal(image.shape) * var + mean


def add_salt_pepper_noise(image, s_vs_p=0.5, amount=0.004):
    row, col = image.shape
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]

    out[coords] = np.max(image)

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]

    out[coords] = np.min(image)

    return out


def compute_normalization(image):
    return np.linalg.norm(image)
