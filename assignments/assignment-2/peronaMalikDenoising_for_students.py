import numpy as np
import matplotlib.pyplot as plt
from util import load_image
import diff_ops
import util
import skimage.filters


def main():
    filename = 'heart_axial_slice.png'

    # load image as float type image
    f_orig = load_image(filename)
    f = util.add_gaussian_noise(f_orig, mean=0, var=20)
    print(f.shape)
    print('f: min = ', np.min(f), ' max = ', np.max(f))

    # fixed parameters
    lambda_param = 0.1
    tau_param = 0.125
    max_iterations = 100
    convergence_epsilon = 1e-4
    K = 30
    display_step = 2

    # calculate edge related function of f
    # blur image f with gaussian filter sigma=(2,2) (tip: use skimage function)
    blurred_f = skimage.filters.gaussian(f, sigma=(2, 2))
    # calculate gradient and gradient magnitude of blurred image f (tip: use util function)
    grad_f_x = diff_ops.dx_forward(blurred_f)
    grad_f_y = diff_ops.dy_forward(blurred_f)
    grad_mag_f = np.sqrt(np.multiply(grad_f_x, grad_f_x) + np.multiply(grad_f_y, grad_f_y))
    # calculate edge related function of f
    c = 1 / (1 + np.multiply(grad_mag_f / K, grad_mag_f / K))

    # generate the figure
    fig, axarr = plt.subplots(1, 3)
    fig.set_size_inches(15, 7.5)
    axarr[0].imshow(f, cmap=plt.cm.gray)
    axarr[1].imshow(c, cmap=plt.cm.gray)
    plt.ion()

    # initialization
    u = f
    # gradient optimization loop
    for iteration in range(max_iterations):

        # store old u for convergence check
        u_old = u

        # compute the gradient of u
        grad_u_x = diff_ops.dx_forward(u)
        grad_u_y = diff_ops.dy_forward(u)

        # calculate laplace of u taking into account diffusion tensor (Perona-Malik) (tip: multiply gradient of u with edge related function)
        laplace_u = (diff_ops.dx_backward(grad_u_x) + diff_ops.dy_backward(grad_u_y)) * c

        # Gradient Decent Optimization iteration of u
        u = u - tau_param * (-laplace_u + lambda_param * (u - f))

        # calculate the normalized change of u
        re = util.compute_normalization(u_old) - util.compute_normalization(u)

        # compare the change of u with the stopping criteria
        if re < convergence_epsilon:
            break

        # display, print infos
        if iteration % display_step == 0:
            current_psnr = util.psnr(f_orig, u)
            print('Iteration: ', iteration, ' relative: ', re, 'psnr: ', current_psnr)


    print('Computation finished, final PSNR: ', util.psnr(u, f_orig))
    axarr[2].imshow(u, cmap=plt.cm.gray)
    # keep in mind that drawing like this is pretty inefficient...
    # but it works like that
    plt.draw()
    plt.pause(0.001)

    # this is so the image stays shown at the end
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
