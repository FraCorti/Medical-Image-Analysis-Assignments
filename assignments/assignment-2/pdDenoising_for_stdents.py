import numpy as np
import matplotlib.pyplot as plt
from util import load_image
import util
import diff_ops


def main():
    filename = 'heart_axial_slice.png'

    # load image as float type image
    f_orig = load_image(filename)
    f = util.add_gaussian_noise(f_orig, mean=0, var=20)
    # check the method performance when there is also salt and pepper noise
    #f = util.add_salt_pepper_noise(f_orig, amount = 0.01)

    # important to scale to [0,1]!
    f /= 255.0

    print('f: min = ', np.min(f), ' max = ', np.max(f))

    # fixed parameters
    lambda_param = 2
    max_iterations = 200
    convergence_epsilon = 1e-4
    display_step = 1

    # generate the figure
    fig, axarr = plt.subplots(1, 2)
    fig.set_size_inches(12, 7.5)
    axarr[0].imshow(f, cmap=plt.cm.gray)

    # initialization
    u = f
    ones_mask = np.ones(u.shape)
    p_x = np.zeros(f.shape)
    p_y = np.copy(p_x)
    # gradient optimization loop
    for iteration in range(max_iterations):

        # store old u for convergence check
        u_old = u

        # calculate the time steps (Zhu and Chan, 2008)
        tau_dual = (0.2 + 0.08 * iteration)
        tau_primal = (0.5 - (5 / (15 + iteration))) / tau_dual

        # divergence of p
        div_p = diff_ops.dx_backward(p_x) + diff_ops.dy_backward(p_y)

        # calculate the next iteration of u
        u = u - tau_primal * (-div_p + lambda_param * (u - f))

        # compute gradient of u
        grad_u_x = diff_ops.dx_forward(u)
        grad_u_y = diff_ops.dy_forward(u)

        # iteration for p_tilde
        p_tilde_x = p_x + tau_dual * grad_u_x
        p_tilde_y = p_y + tau_dual * grad_u_y

        # calculate the 2-norm of p_tilde_abs
        p_tilde_abs = np.sqrt(np.multiply(p_tilde_x, p_tilde_x) + np.multiply(p_tilde_y, p_tilde_y))

        # estimated the maximum value of p_tilde_abs
        max_p_tilde = np.maximum(p_tilde_abs, ones_mask)

        # calculate p values through normalization
        # with the maximum value
        p_x = np.divide(p_tilde_x, max_p_tilde)
        p_y = np.divide(p_tilde_y, max_p_tilde)

        # calculate the normalized change of u
        re = util.compute_normalization(u_old) - util.compute_normalization(u)

        # compare the change of u with the stopping criteria
        if iteration > 1 and re < convergence_epsilon:
            break

        # display, print infos
        if iteration % display_step == 0:
            current_rmse = util.rmse(f_orig, 255 * u)
            print('Iteration: ', iteration, ' relative: ', re, 'rmse: ', current_rmse)

    print('Computation finished, final RMSE: ', util.rmse(f_orig, 255 * u))
    # this is so the image stays shown at the end
    axarr[1].imshow(u, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    main()
