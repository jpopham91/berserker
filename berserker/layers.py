from berserker import PREDICTIONS_DIR
from berserker.nodes import Node
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.pipeline import make_union
from glob import glob
import hashlib
import os

class Layer(object):

    def __init__(self, X, y, validation_split=None, validation_set=None, folds=1):
        """
        :param X:
        :param y:
        :param validation_split:
        :param validation_set:
        :return:
        """
        assert not (validation_split and validation_set), \
               "Choose EITHER validation split or holdout"

        if validation_split:
            cutoff = int(len(X)*validation_split)
            self.X_trn = X[:-cutoff]
            self.y_trn = y[:-cutoff]
            self.X_val = X[cutoff:]
            self.y_val = X[cutoff:]

        elif validation_set:
            self.X_trn = X
            self.y_trn = y
            self.X_val = validation_set[0]
            self.y_val = validation_set[1]

        else:
            self.X_trn = X
            self.y_trn = y
            self.X_val = None
            self.y_val = None

        self.nodes = []
        self.folds = folds

    @staticmethod
    def _generate_hash(model, X_trn, X_prd):
        """
        Generates the md5 hash for a given model, trained on X_trn, predicting X_prd.
        Predictions can be stored using this hash in the filename,
        so that repeat fitting, predictions can be avoided an improve speed.
        """

        model_hash = model._fingerprint()
        train_data_hash = X_trn.data.tobytes()
        test_data_hash = X_prd.data.tobytes()

        md5 = hashlib.md5()
        md5.update(model_hash)
        md5.update(train_data_hash)
        md5.update(test_data_hash)

        return md5.hexdigest()

    def add(self, thing):
        """appends a node to the current layer"""
        if isinstance(thing, Node):
            self.nodes.append(thing)
        elif isinstance(thing, BaseEstimator):
            self.nodes.append(Node(thing))
        else:
            print('Warning, unfamiliar type: {}'.format(type(thing)))

    def _fit_one(self, node, X=None, y=None, force=False):
        """
        fits a single node if not already fit
        :param node:
        :param force:
        :return:
        """
        if not (X or y):
            X = self.X_trn
            y = self.y_trn
        if not node.is_fit or force:
            node.fit(X, y)

    def _predict_one(self, node, X, refit=False):
        """
        obtains a single node's prediction.
        checks for an existing result in the cache first.
        :param node:
        :param X:
        :return:
        """
        md5 = Layer._generate_hash(node, self.X_trn, X)
        if not os.path.exists(PREDICTIONS_DIR):
            os.makedirs(PREDICTIONS_DIR)
        fname = '{}/{}.npy'.format(PREDICTIONS_DIR, md5)
        if fname in glob('{}/*.npy'.format(PREDICTIONS_DIR)):
            print('Using previously trained model.')
            return np.load(fname)
        else:
            self._fit_one(node)
            preds = node.transform(X)
            np.save(fname, preds)
            return preds

    def _predict_out_of_fold(self, node, X):
        """
        :param node:
        :param X:
        :param folds:
        :return:
        """
        preds = np.empty_like(X.shape[0])
        for trn_idx, prd_idx in KFold(len(self.y_trn), self.folds):
            self._predict_one()


    def _predict_all(self, X):
        """
        predict helper function
        :param X:
        :return:
        """
        for node in self.nodes:
            yield self._predict_one(node, X)

    def predict(self, X):
        all_preds = [pred for pred in self._predict_all(X)]
        #print(all_preds)
        return np.hstack(all_preds)

    transform = predict