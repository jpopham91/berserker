"""This module contains the objects which can be added to layers"""
#from abc import ABCMeta, abstractmethod
import dis
from glob import glob
import io
import os
import sys
import hashlib

import numpy as np
import arrow
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import scale, StandardScaler

# todo: find elegant way of enabling typing for pre python 3.5 setups
#from typing import Callable
#from mypy.types import CallableType as Callable

import warnings
from berserker import PREDICTIONS_DIR#, log
import logging

#log = logging.getLogger()
#logging.setLevel(logging.DEBUG)

# todo: handle logging globally
logging.basicConfig(format='[%(asctime)s] [%(levelname)8s] %(message)s',
                    level=logging.DEBUG,
                    datefmt='%I:%M:%S')

warnings.filterwarnings("ignore", category=DeprecationWarning)

def _get_bytecode(func):
    # todo: this could use something like
    # [(i.opname, i.argval) for i in dis.Bytecode(f)]
    tmp = sys.stdout
    stream = io.StringIO()
    sys.stdout = stream
    dis.dis(func)
    sys.stdout = tmp
    raw = stream.getvalue()

    lines = list(map(lambda x: x[12:], raw.splitlines()))
    return '\n'.join(lines)


def check_cache(node, X_trn, X_prd):
    """
    checks model cache for existing prediction on this train-predict-node combo
    :param node:
    :param X:
    :return:
    """
    # create directory if it doesn't exist
    md5 = generate_hash(node, X_trn, X_prd)
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)

    fname = '{}/{}.npy'.format(PREDICTIONS_DIR, md5)
    if fname in glob('{}/*.npy'.format(PREDICTIONS_DIR)):
        return True, fname
    else:
        return False, fname


def generate_hash(node, X_trn, X_prd):
    """
    Generates the md5 hash for a given model, trained on X_trn, predicting X_prd.
    Predictions can be stored using this hash in the filename,
    so that repeat fitting, predictions can be avoided an improve speed.
    """
    # todo: make utils/io module?
    # todo: does this need to incorporate y_trn?
    md5 = hashlib.md5()
    model_hash = str(node).encode('utf')
    trn_hash = b'\x00' + X_trn.data.tobytes() + str(X_trn.shape).encode('utf')
    tst_hash = b'\x01' + X_prd.data.tobytes() + str(X_prd.shape).encode('utf')

    md5.update(model_hash)
    md5.update(trn_hash)
    md5.update(tst_hash)

    return md5.hexdigest()


# not an sklearn estimator, but a transformer
class Node(TransformerMixin, BaseEstimator):
    """
    A base node object which all inherit from

    Warning: Do not use this class directly. Use derived classes instead.
    """

    def __init__(self, estimator, name=None, # todo: have nodes contain core/wrapped/base object?
                 tag = '',
                 target_transform=(lambda x: x),
                 inverse_transform=(lambda x: x),
                 baggs = 0,
                 scale_x = False,
                 fillna=-1,
                 columns=np.s_[:]):
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

        self.name = '{} {}'.format(self.name, tag)

        if baggs:
            self.estimator = sklearn.ensemble.BaggingRegressor(estimator, baggs)
        else:
            self.estimator = estimator
        self.target_transform = np.vectorize(target_transform)
        self.inverse_transform = np.vectorize(inverse_transform)
        self.baggs = baggs
        self.predictions = []
        self.is_fit = False
        self.scale_x = scale_x
        self.fillna = fillna
        self.columns = columns
        self.cached_preds = 0
        self.total_preds = 0

    def __str__(self):
        return str([self.estimator, self.scale_x,
                    _get_bytecode(self.target_transform),
                    _get_bytecode(self.inverse_transform)])

    def fit_predict(self, X_trn, y_trn, X_prd, refit=False):
        self.total_preds += 1

        # arrays are mutable
        X_trn = X_trn.copy()[:,self.columns]
        X_prd = X_prd.copy()[:,self.columns]
        y_trn = y_trn.copy()

        if self.fillna is not None:
            X_trn[np.isnan(X_trn)] = self.fillna
            X_prd[np.isnan(X_prd)] = self.fillna

        if self.scale_x:
            scl = StandardScaler()
            #scl.fit(np.vstack([X_trn, X_prd]))
            scl.fit(X_trn)
            X_trn = scl.transform(X_trn)
            X_prd = scl.transform(X_prd)

        exists, fname = check_cache(self, X_trn, X_prd)
        if exists:
            logging.info('{} re-using existing predictions'.format(self.name))
            self.cached_preds += 1
            return np.load(fname)

        if not self.is_fit or refit:
            self._fit(X_trn, y_trn)

        preds = self._predict(X_prd)
        np.save(fname, preds)
        return preds

    def _fit(self, X: np.ndarray, y: np.array):
        logging.info('Training {}...'.format(self.name))
        #if self.scale_x:
        #    X = scale(X)
        y_tfd = self.target_transform(y)
        self.estimator.fit(X, y_tfd)
        self.is_fit = True
        return self

    def _predict(self, X: np.ndarray, y: np.array=None):
        if not self.is_fit:
            raise AttributeError('Model has not been fit. Don\'t call predict directly. Use fit_predict instead.')
        logging.info('Predicting {}...'.format(self.name))
        # todo: needs handling for classifiers / proba
        pred = self.estimator.predict(X).reshape(X.shape[0], -1)
        pred_tfd = self.target_transform(pred)
        pred_itfd = self.inverse_transform(pred)
        return pred_itfd


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


