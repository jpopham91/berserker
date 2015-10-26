"""This module contains the objects which can be added to layers"""
#from abc import ABCMeta, abstractmethod
import numpy as np
import arrow
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.preprocessing import scale
import dis, sys, io
#from typing import Callable

# todo: find elegant way of enabling typing for pre python 3.5 setups
#from mypy.types import CallableType as Callable

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _get_bytecode(func):
    tmp = sys.stdout
    stream = io.StringIO()
    sys.stdout = stream
    dis.dis(func)
    sys.stdout = tmp
    return stream.getvalue()


# not an sklearn estimator, but a transformer
class Node(TransformerMixin, BaseEstimator):
    """
    A base node object which all inherit from

    Warning: Do not use this class directly. Use derived classes instead.
    """

    def __init__(self, estimator, name=None, # todo: have nodes contain core/wrapped/base object?
                 suffix = '',
                 target_transform=(lambda x: x),
                 inverse_transform=(lambda x: x),
                 baggs = 0,
                 scale_x = False):
        """
        :param estimator: classifier or regressor compatible with the scikit-learn api
        :type estimator: sklearn estimator
        :param target_transform:
        :type target_transform: (np.array -> np.array)
        :param inverse_transform:
        :type inverse_transform: (np.array -> np.array)
        :return:
        """

        if name:
            self.name = name
        else:
            self.name = str(estimator).split('(')[0]

        self.name = '{} {}'.format(self.name, suffix)

        if baggs:
            self.estimator = BaggingRegressor(estimator, baggs)
        else:
            self.estimator = estimator
        self.target_transform = np.vectorize(target_transform)
        self.inverse_transform = np.vectorize(inverse_transform)
        self.baggs = baggs
        self.predictions = []
        self.is_fit = False
        self.scale_x = scale_x

    def _fingerprint(self):
        return str([self.estimator, self.scale_x,
                    _get_bytecode(self.target_transform),
                    _get_bytecode(self.inverse_transform)]).encode('ascii')

    def fit(self, X: np.ndarray, y: np.array):
        print('[{}] Training {}...'.format(arrow.utcnow().to('EST').format('HH:mm:ss'), self.name))
        if self.scale_x:
            X = scale(X)
        self.estimator.fit(X, self.target_transform(y))
        self.is_fit = True
        return self

    def predict(self, X: np.ndarray, y: np.array=None):
        print('[{}] Predicting {}...'.format(arrow.utcnow().to('EST').format('HH:mm:ss'), self.name))
        if self.scale_x:
            X = scale(X)
        pred = self.estimator.predict(X)
        return self.inverse_transform(pred)


    def transform(self, X, y=None):
        transformed = self.predict(X).reshape(-1, 1)
        return transformed

    def score(self, X, y, metric):
        return metric(y, self.predict(X))


class File(Node):
    """
    A means for injecting external predictions into a model.
    """


class Model(Node):
    """
    Base node for regression and classification models.

    Warning: Do not use this class directly. Use derived classes instead.
    """


class Regressor(Model):
    """
    Standard node for predicting continuous values.
    """


class BaggingRegressor(Regressor):
    """
    A regression node that uses bootstrap aggregation when training.
    This results in less variance than the standard regressor
    """


class BoostingRegressor(Regressor):
    """
    A regression node that fits multiple copies of itself which focus on samples with high error
    """


class Classifier(Model):
    """
    Standard node for predicting binary or categorical values.
    """


class BaggingClassifier(Classifier):
    """
    A classifier node that uses bootstrap aggregation when training.
    This results in less variance than the standard classifier
    """


class BoostingClassifier(Classifier):
    """
    A classifier node that fits multiple copies of itself which focus on samples with high error
    """


class Transformer(Node):
    """
    A node which can be used to transform (scale, decompose, vectorize, etc) feature vectors
    Does not have a predict method
    """


# todo: might change the via to an estimator/tranformer to be wrapped in Transformer
class Via(Node):
    """A node which passes feature vectors through to the next layer unaltered"""

    def __init__(self):
        return

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):
        return X

    def transform(self, X, *args, **kwargs):
        return X
