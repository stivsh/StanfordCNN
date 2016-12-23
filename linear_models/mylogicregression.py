"""
This module contains implementation of binary LogisticRegression and
One Versus All LogisticRegression.
"""

import numpy as np
from common import BaseBinaryLinearClassifier, BaseOneVersusAllLinearClassifier, bias_to_0


def h(x, w):
    """The scores, this scores should be interpret
    as probability to belongs of sertain class."""
    z = x.dot(w)
    for_up_zero = 1.0 / (np.exp(-z) + 1)
    z_exp = np.exp(z)
    for_below_zero = z_exp / (1 + z_exp)
    for_up_zero[z < 0] = 0
    for_below_zero[z >= 0] = 0
    return for_up_zero + for_below_zero


def j(x, w, y, alpha=0):
    """The loss function, the aim of LogisticRegression is to minimize it. """
    scores = h(x, w)
    log_score = np.log(scores)
    log_one_min_h = np.log(1 - scores)
    log_score[y == 0] = 0
    log_one_min_h[y == 1] = 0
    trimed_biases = bias_to_0(w)
    data_loss = (-np.sum(log_score + log_one_min_h)) / x.shape[0]
    weights_loss = 0.5 * alpha * np.sum(trimed_biases * trimed_biases)
    return data_loss + weights_loss


def dj(x, w, y, alpha=0):
    """The gradient of the loss function, dj/dw"""
    scores = h(x, w)
    scores_error = (scores - y)
    d_data = np.sum(
        x * scores_error.reshape((len(scores_error), 1)), axis=0) / x.shape[0]
    d_weights = alpha * bias_to_0(w)
    return d_data + d_weights


class LogisticClassifier(BaseBinaryLinearClassifier):
    """LogisticRegression Binary Classifier"""

    def h(self, x, w):
        """Return scores, this scores should be interpret
        as probability to belongs of sertain class."""
        return h(x, w)

    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimize."""
        return j(x, w, y, alpha)

    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw"""
        return dj(x, w, y, alpha)

    def get_treshhold(self):
        """Standart treshhold for lr. In parent class 0."""
        return 0.5


class OneVersusAllLogisticClassifier(BaseOneVersusAllLinearClassifier):
    """SVM Multivariable classifier"""

    def h(self, x, w):
        """Return scores, scores > 0 means data probably belongs to the class."""
        return h(x, w)

    @BaseOneVersusAllLinearClassifier.j_transform_fo_multivariabl
    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimize."""
        return j(x, w, y, alpha)

    @BaseOneVersusAllLinearClassifier.dj_transform_fo_multivariabl
    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw"""
        return dj(x, w, y, alpha)
