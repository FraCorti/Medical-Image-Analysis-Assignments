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
    lambda_param = 0.05
    tau_param = 0.125
    max_iterations = 200
    rof_epsilon = 1e-8
    convergence_epsilon = 1e-4
    display_step = 2

    # generate the figure
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(f, cmap=plt.cm.gray)
    plt.ion()

    # initialization
    u = f
    for iteration in range(max_iterations):

        # store old u for convergence check
        u_old = u

        # compute the gradients, magnitude of gradients and normalized gradients of u
        # take into account the rof_epsilon when normalizing
        grad_u_x = diff_ops.dx_forward(u)
        grad_u_y = diff_ops.dy_forward(u)
        mag_grad_u = np.sqrt(np.multiply(grad_u_x, grad_u_x) + np.multiply(grad_u_y, grad_u_y))
        grad_u_x = grad_u_x / (mag_grad_u + rof_epsilon)
        grad_u_y = grad_u_y / (mag_grad_u + rof_epsilon)

        # compute the laplace of u
        modified_laplace_u = (diff_ops.dx_backward(grad_u_x) + diff_ops.dy_backward(grad_u_y))

        # Gradient Decent Optimization iteration of u
        u = u - tau_param * (-modified_laplace_u + lambda_param * (u - f))

        # calculate the normalized change of u
        re = util.compute_normalization(u_old) - util.compute_normalization(u)

        # compare the change of u with the stopping criteria
        if re < convergence_epsilon:
            break

        # display, print infos
        if iteration % display_step == 0:
            print('Iteration: ', iteration, ' relative: ', re)

    axarr[1].imshow(u, cmap=plt.cm.gray)
    # keep in mind that drawing like this is pretty inefficient...
    # but it works like that
    plt.draw()
    plt.pause(0.01)

    # this is so the image stays shown at the end
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
