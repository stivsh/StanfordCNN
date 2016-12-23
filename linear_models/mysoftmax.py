"""
This module contains implementation of SoftMax multy class classification algorithm.
"""

import numpy as np
from scipy.misc import logsumexp
from common import BaseMultyClassClassifier, bias_to_0


def j(x, w, y, alpha=0):
    """The loss function, the aim is to minimize it."""
    prodact = x.dot(w)
    lg_of_h = prodact - \
        logsumexp(prodact, axis=1).reshape((prodact.shape[0], 1))
    lg_of_h[y == 0] = 0
    trimed_biases = bias_to_0(w)
    return (-np.sum(lg_of_h)) / x.shape[0] + 0.5 * alpha * np.sum(trimed_biases * trimed_biases)


def dj(x, w, y, alpha=0):
    """The gradient of the loss function, dj/dw."""
    return -x.T.dot(y - h(x, w)) / x.shape[0] + bias_to_0(w) * alpha


def h(x, w):
    """The scores, this scores is logs of probability
    of belonging to class."""
    prodact = x.dot(w)
    # this subtraction makes it numerical stable
    max_classes_val = np.max(prodact, axis=1)
    normalized = prodact - max_classes_val.reshape((max_classes_val.size, 1))
    normalized = np.exp(normalized)
    rows_sum = np.sum(normalized, axis=1)
    return normalized / rows_sum.reshape((rows_sum.size, 1))


class SoftMaxClassifier(BaseMultyClassClassifier):
    """SoftMaxClassifier multi class classifier."""

    def h(self, x, w):
        """The scores, this scores is logs of probability
        of belonging to sertain class."""
        return h(x, w)

    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimize."""
        return j(x, w, y, alpha)

    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw."""
        return dj(x, w, y, alpha)
