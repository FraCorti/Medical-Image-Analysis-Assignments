import numpy as np
from scipy.signal import correlate2d


def index_of_best_match_ncc(template, image, indices_col_lower, indices_col_upper, indices_row_lower, indices_row_upper):
    """
    Computes the row and column index within the specified image, representing the center of the image are which best
    matches the specified template
    :param template: template for which the best match is searched
    :param image: image (region) in which the best match is searched
    :param indices_col_lower: lower bound of image region column index
    :param indices_col_upper: upper bound of image region column index
    :param indices_row_lower: lower bound of image region row index
    :param indices_row_upper: upper bound of image region row index
    :return: tuple of index_row, index_col, which describe the point in the full image where the best match was found
    """

    # handle/crop regions out of image bounds
    indices_row_lower = np.maximum(indices_row_lower, 0)
    indices_col_lower = np.maximum(indices_col_lower, 0)

    indices_row_upper = np.minimum(indices_row_upper, image.shape[0])
    indices_col_upper = np.minimum(indices_col_upper, image.shape[1])

    #check if template is smaller than image crop
    if (indices_row_upper - indices_row_lower) < template.shape[0] or \
       (indices_col_upper - indices_col_lower) < template.shape[1]:
        print('image crop smaller than template when computing "normalized cross correlation"')

    norm_image = (image - image.mean()) / image.std()
    norm_template = (template - template.mean()) / template.std()

    NCC = correlate2d(norm_image[indices_row_lower:indices_row_upper, indices_col_lower:indices_col_upper], norm_template)

    # don't use correlation values which are (partly) based on values outside the image clip, set them to -1

    num_rows_out_of_bounds = template.shape[0] - 1
    num_cols_out_of_bounds = template.shape[1] - 1

    NCC[0:num_rows_out_of_bounds, :] = -1
    NCC[(NCC.shape[0] - num_rows_out_of_bounds):, :] = -1
    NCC[:, 0:num_cols_out_of_bounds] = -1
    NCC[:, (NCC.shape[1] - num_cols_out_of_bounds):] = -1

    # Find indices of best match

    y, x = np.unravel_index(np.argmax(NCC), NCC.shape)

    i_ncc_max = np.asarray([y, x])

    # Compute index in clip and whole image
    indices_row = list(range(indices_row_lower.astype(int), indices_row_upper.astype(int)))
    indices_col = list(range(indices_col_lower.astype(int), indices_col_upper.astype(int)))

    i_clip_center = (i_ncc_max - np.floor(np.asarray(template.shape) / 2)).astype(int)

    index_row = indices_row[i_clip_center[0]]
    index_col = indices_col[i_clip_center[1]]

    return index_row, index_col
