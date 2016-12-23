
import numpy as np
import sys
sys.path.insert(0,'../linear_models')

from common import binarithate


def simple_optimizer(func, init_w, jac):
    """The simplest minimither function that could ever be. For test only."""
    for i in range(200):
        init_w -= 1 * jac(init_w)
    return type("", (), {'x': init_w, 'success': True})()


def numerical_dj(x, w, y, indexes, alpha, loss_func):
    """Compute the gradient of the w numericaly,
    only for elements witch index in indexes."""
    WALPHA = 10**-4
    dj = np.zeros(w.shape)

    for inx in indexes:
        w_original = w[inx]
        w[inx] += WALPHA
        j_up_w = loss_func(x, w, y, alpha=0)
        w[inx] -= 2 * WALPHA
        j_down_w = loss_func(x, w, y, alpha=0)
        w[inx] = w_original
        dj[inx] = (j_up_w - j_down_w) / (2 * WALPHA)

    return dj


def compare_analitic_and_numerical_dj(x, y, alpha, loss_func, analitical, numerical):
    """Returns relative error between numerical and analitical gradient.
        Should be used when makes any changes in dj appears to check correctnes."""

    inx_to_check = None
    w = None
    nfeatures = x.shape[1]
    class_size = np.unique(y).size
    if class_size == 2:
        w = np.random.randn(nfeatures) / np.sqrt(nfeatures)
        inx_to_check = np.random.randint(w.size, size=10)
        paired_indexes = inx_to_check
    else:
        w = np.random.uniform(-1, 1, nfeatures *
                              class_size).reshape((nfeatures, class_size))
        inx_to_check = zip(np.random.randint(nfeatures, size=10),
                           np.random.randint(class_size, size=10))
        paired_indexes = zip(*inx_to_check)
        y = binarithate(y)

    dj_analitical = analitical(x, w, y, alpha)
    dj_numerical = numerical(x, w, y, inx_to_check, alpha, loss_func)

    dj_analitical = dj_analitical[paired_indexes]
    dj_numerical = dj_numerical[paired_indexes]

    absolute_error = dj_numerical - dj_analitical
    relative_error = np.abs(absolute_error / dj_numerical)
    return np.mean(relative_error)
