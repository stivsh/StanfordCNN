"""
"""

import numpy as np


def bias_to_0(w):
    """Set weight column for constant data Xn = 1 to 0"""
    copy = w.copy()
    copy[0] = 0
    return copy


def add_bias_column(x):
    """Add bias column to the data."""
    bias_column = np.ones((x.shape[0], 1))
    return np.hstack((bias_column, x))


def binarithate(y):
    """Standart OneHotEncoding of labels """
    binarithate_y = np.zeros((y.size, np.unique(y).size))
    binarithate_y[np.arange(y.size), y] = 1
    return binarithate_y


class LinearClassifierInterface(object):
    """This class provides description of interface of every classifier"""

    def __init__(self, alpha=0, optimizer=None):
        """alpha - regularization parameter, optimazer - function that should takes:
        loss func with one parameter(weights), initial weights,
        jacobian(gradient dj/dw) and return optimal parameters(w) of model."""
        raise NotImplemented(
            "Pure virtual class? Just for Interfase standartization")

    def fit(self, x, y):
        """In this method fitting the data should be implemented. All needed work for
        finding paraeters for predicting and transfoming data."""
        raise NotImplemented(
            "Pure virtual class? Just for Interfase standartization")

    def predict(self, x):
        """Predicts labels by data"""
        raise NotImplemented(
            "Pure virtual class? Just for Interfase standartization")


class BaseLinearClassifier(LinearClassifierInterface):
    """This class add to the outer Interface description of functions that should be
    implemented in every classifier, this functions in the hard of classification algorithms."""

    def __init__(self, alpha=0, optimizer=None):
        """alpha - regularization parameter, optimazer - function that should takes:
        loss func with one parameter(weights), initial weights,
        jacobian(gradient dj/dw) and return optimal parameters(w) of model."""
        self.alpha = alpha
        if not optimizer:
            from scipy.optimize import minimize
            optimizer = minimize
        self.optimizer = optimizer

    def h(self, x, w):
        """Return scores, scores proportional to probability to belong to a class"""
        raise NotImplemented("Pure virtual")

    def j(self, x, w, y, alpha):
        """Train error, this function shuld be minimize."""
        raise NotImplemented("Pure virtual")

    def dj(self, x, w, y, alpha):
        """The gradient. dj/dw"""
        raise NotImplemented("Pure virtual")

    def transform_x(self, x):
        """Transform x before fit and predict, By deffault add biasian term."""
        return add_bias_column(x)

    def transform_y(self, y):
        """Transform y before fit. By deffault do nothing.
        In case of SVM use y = y * 2 - 1."""
        return y

    def get_init_w(self, x, y):
        """Get initial value for hipotize parameters w"""
        raise NotImplemented("Pure virtual")

    def fit(self, x, y):
        """Fit the data, find optimal parameters w"""
        y = self.transform_y(y)
        x = self.transform_x(x)
        init_w = self.get_init_w(x, y)

        def objective(step_w):
            return self.j(x, step_w, y, self.alpha)

        def jacobian(step_w):
            return self.dj(x, step_w, y, self.alpha)

        self.w = self.optimizer(objective, init_w, jac=jacobian).x
        return self


class BaseBinaryLinearClassifier(BaseLinearClassifier):
    """ Simple Base class for binary classifier, provide implementation of fit
     and predict methods, you should implement h, j, dj methods.
     transform_x, transform_y, get_init_w is optional."""

    def get_init_w(self, x, y):
        """Get initial value for hipotize parameters w"""
        return np.random.randn(x.shape[1]) / np.sqrt(x.shape[1])

    def get_treshhold(self):
        """return treshhold, if score > treshhold data belong to the class.
        depends of method(logicreg, svm) and level of desired confidence."""
        return 0

    def predict(self, x):
        """Predicts labels by data, if scores > trashhold data belongs to the class."""
        x = self.transform_x(x)
        scores = self.h(x, self.w)
        return (scores > self.get_treshhold()).astype("int")


class BaseMultyClassClassifier(BaseLinearClassifier):
    """Base class for all multy class classification, provides common
    functionality such as label binarithation, prediction by chusing maximum score, get initial parameters of model,"""

    def __binarithate__(self, y):
        """Perform One Hot Encoding of y."""
        self.nlabels = np.unique(y).size
        return binarithate(y)

    def transform_y(self, y):
        """Transforms each label into binary array."""
        return self.__binarithate__(y)

    def get_init_w(self, x, y):
        """Get initial value for hipotize parameters w."""
        nfeatures = x.shape[1]
        return np.random.uniform(-1, 1, nfeatures * self.nlabels).reshape((nfeatures, self.nlabels))

    def predict(self, x):
        """Predicts labels by data."""
        x = self.transform_x(x)
        scores = self.h(x, self.w)
        predictions = np.argmax(scores, axis=1)
        return predictions


class BaseOneVersusAllLinearClassifier(BaseMultyClassClassifier):
    """Makes as much classifaers as uniqe labels, all you need is to provide
    standart functions(h, j, dj) as for binary classifier but j, dj with spetial
    decorator that transforms it for multi domential classification."""

    @staticmethod
    def dj_transform_fo_multivariabl(dj_function):
        """Transform one domentional dj function to multi domentional."""

        def multivariable_dj(self, x, w, y, alpha):
            dw = np.zeros(w.shape)
            for i in xrange(self.nlabels):
                dw[:, i] = dj_function(self, x, w[:, i], y[:, i], self.alpha)
            return dw
        return multivariable_dj

    @staticmethod
    def j_transform_fo_multivariabl(j_function):
        """Transform one domentional j function to multi domentional."""

        def multivariable_j(self, x, w, y, alpha):
            per_class_errors = [j_function(self, x, w[:, i], y[:, i], self.alpha)
                                for i in xrange(self.nlabels)]
            return np.mean(per_class_errors)
        return multivariable_j
