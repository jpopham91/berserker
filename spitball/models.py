"""This module contains the objects which can be added to layers"""
#from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable

# todo: find elegant way of enabling typing for pre python 3.5 setups
#from mypy.types import CallableType as Callable

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# not an sklearn estimator, but a transformer
# todo: maybe abstract as a 'node' or 'element' class

class Model(TransformerMixin, BaseEstimator):

    def __init__(self, estimator,
                 target_transform: Callable[[float], float]=(lambda x: x),
                 inverse_transform: Callable[[float], float]=(lambda x: x)):
        """
        :param estimator: classifier or regressor compatible with the scikit-learn api
        :type estimator: sklearn estimator
        :param target_transform:
        :type target_transform: (np.array -> np.array)
        :param inverse_transform:
        :type inverse_transform: (np.array -> np.array)
        :return:
        """
        self.estimator = estimator
        self.target_transform = np.vectorize(target_transform)
        self.inverse_transform = np.vectorize(inverse_transform)
        self.predictions = []

    # todo: add support for bagging and folds
    def fit(self, X: np.ndarray, y: np.array):
        print("Fitting {}".format(str(self.estimator)))
        self.estimator.fit(X, self.target_transform(y))
        return self

    def predict(self, X: np.ndarray, y: np.array=None):
        pred = self.estimator.predict(X)
        return self.inverse_transform(pred)

    def transform(self, X, *args, **kwargs):
        return self.predict(X, *args, **kwargs).reshape(-1,1)

    def score(self, X, y, metric):
        return metric(y, self.predict(X))


# todo: might change the via to an estimator
class Via(Model):
    """A node which passes feature vectors through to the next layer unaltered"""

    def __init__(self):
        return

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):
        return X

    def transform(self, X, *args, **kwargs):
        return X


# todo: separate wrappers for regression / classification?
# class Regressor(Model):
#
#     def predict(self, X: np.ndarray) -> np.array:
#         return self.estimator.predict(X)
#
# class Classifier(Model):
#
#     def predict(self, X: np.ndarray) -> np.array:
#         return self.estimator.predict_proba(X)[:, 1]
