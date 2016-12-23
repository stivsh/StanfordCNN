"""
This module contains binary SVM and One Versus All SVM implementation.
"""

import numpy as np
from common import BaseBinaryLinearClassifier, BaseOneVersusAllLinearClassifier, bias_to_0


def h(x, w):
    """The scores of data, if the score > 0 probable the data x belong
    to the class with weights w """
    return x.dot(w)


def j(x, w, y, alpha=0):
    """The loss function, the aim of SVM is to minimize it. """
    scores = h(x, w)
    j = 1 - y * scores
    j[j < 0] = 0
    w_without_bias = bias_to_0(w)

    data_loss = np.sum(j) / len(j)
    weight_loss = 0.5 * alpha * np.sum(w_without_bias * w_without_bias)
    return data_loss + weight_loss


def dj(x, w, y, alpha=0):
    """The gradient of the loss function, the aim of SVM is to
    minimize the loss function."""
    scores = h(x, w)
    j = 1 - y * scores

    false_predicted_indexes = np.where(j >= 0)[0]
    false_predicted_x = x[false_predicted_indexes]
    false_predicted_y = y[false_predicted_indexes]
    false_predicted_y = false_predicted_y.reshape((len(false_predicted_y), 1))
    data_grad = (np.sum(-false_predicted_x *
                        false_predicted_y, axis=0) / x.shape[0])

    reg_grad = bias_to_0(w) * alpha
    return data_grad + reg_grad


class SvmClassifier(BaseBinaryLinearClassifier):
    """SVM Binary Classifier"""

    def h(self, x, w):
        """The scores, if scores > 0 data probably belongs to the class."""
        return h(x, w)

    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimized."""
        return j(x, w, y, alpha)

    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw"""
        return dj(x, w, y, alpha)

    def transform_y(self, y):
        """Transform y before fit."""
        return y * 2 - 1


class OneVersusAllSVM(BaseOneVersusAllLinearClassifier):
    """SVM Multivariable classifier"""

    def h(self, x, w):
        """The scores, if scores > 0 the data probably belongs to the class."""
        return h(x, w)

    @BaseOneVersusAllLinearClassifier.j_transform_fo_multivariabl
    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimized."""
        return j(x, w, y, alpha)

    @BaseOneVersusAllLinearClassifier.dj_transform_fo_multivariabl
    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw"""
        return dj(x, w, y, alpha)

    def transform_y(self, y):
        """Transform y before fit."""
        y = super(OneVersusAllSVM, self).transform_y(y)
        return y * 2 - 1
