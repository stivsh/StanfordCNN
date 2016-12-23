"""
This module contains implementation of multiclass svm algorithm.
"""

import numpy as np
from common import BaseMultyClassClassifier, bias_to_0


def h(x, w):
    """The scores, this scores proportional to probability
    of belonging to sertain class."""
    return x.dot(w)


def confusion_matrix(x, w, y):
    scores = h(x, w)
    true_classes_score = scores[xrange(scores.shape[0]), np.argmax(
        y, axis=1)].reshape((scores.shape[0], 1))
    scores += -true_classes_score + 1
    scores[y.astype('bool')] = 0
    scores[scores < 0] = 0
    return scores


def j(x, w, y, alpha=0):
    """The loss function, the aim is to minimize it."""
    conf_mat = confusion_matrix(x, w, y)
    trimed_biases = bias_to_0(w)
    data_loss = np.sum(conf_mat) / conf_mat.shape[0]
    weights_loss = 0.5 * alpha * np.sum(trimed_biases * trimed_biases)
    return data_loss + weights_loss


def dj(x, w, y, alpha=0):
    """The gradient of the loss function, dj/dw."""
    conf_mat = confusion_matrix(x, w, y)
    conf_mat[conf_mat > 0] = 1
    conf_mat = conf_mat.astype('bool')

    dw = np.zeros(w.shape)

    # part of dw for negative classes
    for i in xrange(w.shape[1]):
        dw[:, i] = np.sum(x[np.where(conf_mat[:, i])], axis=0)

    # part of dw for positive classes
    rows_with_errors = np.nonzero(conf_mat)[0]
    w_of_true_class = np.argmax(y, axis=1)[rows_with_errors]
    for i in xrange(len(w_of_true_class)):
        dw[:, w_of_true_class[i]] -= x[rows_with_errors[i]]

    dw /= x.shape[0]
    return dw + bias_to_0(w) * alpha


class SVMMultiClassClassifier(BaseMultyClassClassifier):
    """SVM multi class classifier"""

    def h(self, x, w):
        """The scores, this scores proportional to probability
        of belonging to sertain class."""
        return h(x, w)

    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimized."""
        return j(x, w, y, alpha)

    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw."""
        return dj(x, w, y, alpha)
