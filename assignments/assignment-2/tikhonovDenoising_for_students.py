import numpy as np
import matplotlib.pyplot as plt
from util import load_image
import diff_ops
import util


def main():
    filename = 'heart_axial_slice.png'

    # load image as float type image
    f_orig = load_image(filename)
    f = util.add_gaussian_noise(f_orig, mean=0, var=20)
    print(f.shape)
    print('f: min = ', np.min(f), ' max = ', np.max(f))

    # fixed parameters
    lambda_param = 0.5
    tau_param = 0.001
    max_iterations = 600
    convergence_epsilon = 1e-4
    display_step = 5

    # generate the figure
    fig, axarr = plt.subplots(1, 2)
    fig.set_size_inches(12, 7.5)
    axarr[0].imshow(f, cmap=plt.cm.gray)
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

        # compute the laplace of u
        laplace_u = diff_ops.dx_backward(grad_u_x) + diff_ops.dy_backward(grad_u_y)

        # Gradient Descent Optimization
        u = u - tau_param * (-laplace_u + lambda_param * (u - f))

        # calculate the normalized change of u
        re = util.compute_normalization(u_old) - util.compute_normalization(u)

        # compare the change of u with the stopping criteria
        if re < convergence_epsilon:
            break

            # display, print infos
        if iteration % display_step == 0:
            current_rmse = util.rmse(f_orig, u)
            print('Iteration: ', iteration, ' relative: ', re, 'RMSE: ', current_rmse)

    axarr[1].imshow(u, cmap=plt.cm.gray)

    # keep in mind that drawing like this is pretty inefficient...
    # but it works like that
    plt.draw()
    plt.pause(0.1)
    print('Computation finished after iteration', iteration, ', final RMSE: ', util.rmse(u, f_orig))

    # this is so the image stays shown at the end
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
