from berserker import PREDICTIONS_DIR
from berserker.nodes import Node
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.pipeline import make_union
from glob import glob
from hashlib import md5
import os


def generate_hash(node, X_trn, X_prd):
    """
    Generates the md5 hash for a given model, trained on X_trn, predicting X_prd.
    Predictions can be stored using this hash in the filename,
    so that repeat fitting, predictions can be avoided an improve speed.
    """
    # todo: make utils/io module
    model_hash = node._fingerprint()
    train_data_hash = X_trn.data.tobytes()
    test_data_hash = X_prd.data.tobytes()

    return md5(model_hash).hexdigest() + md5(train_data_hash).hexdigest() + md5(test_data_hash).hexdigest()

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

class Layer(object):

    def __init__(self, X, y, folds=1, validation_split=None, validation_set=None, pass_features=False):
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

        elif isinstance(validation_set, np.ndarray):
            self.X_trn = X
            self.y_trn = y
            self.X_val = validation_set[0]
            self.y_val = validation_set[1]

        else:
            self.X_trn = self.X_val = X
            self.y_trn = self.y_val = y

        self.nodes = []
        self.trn_preds = []
        self.val_preds = []
        self.folds = folds
        self.pass_features = pass_features

    def add(self, thing, **kwargs):
        """appends a node to the current layer"""
        if isinstance(thing, Node):
            self.nodes.append(thing)
        elif isinstance(thing, BaseEstimator):
            self.nodes.append(Node(thing, **kwargs))
        else:
            print('Warning, unfamiliar type: {}'.format(type(thing)))

    def _fit_one(self, node, X=None, y=None, force=False):
        """
        fits a single node if not already fit
        :param node:
        :param force:
        :return:
        """
        if not (X.any() or y.any()):
            X = self.X_trn
            y = self.y_trn
        if not node.is_fit or force:
            node.fit(X, y)

    # def _predict_one(self, node, X, refit=False):
    #     """
    #     obtains a single node's prediction.
    #     checks for an existing result in the cache first.
    #     :param node:
    #     :param X:
    #     :return:
    #     """
    #     self.trn_preds.append(np.empty_like(self.X_trn.shape[0]))
    #
    #     exists, fname = check_cache(node, self.X_trn, X)
    #     if exists:
    #         return np.load(fname)
    #     else:
    #         self._fit_one(node)
    #         preds = node.transform(X)
    #         np.save(fname, preds)
    #         return preds

    def _predict_one(self, node, X, refit=False):
        """
        Fits training data on K folds and predicts the out of fold indices.
        Test data is averaged over all folds.
        :param node:
        :param X:
        :param folds:
        :return:
        """
        preds = np.zeros((X.shape[0], 1))
        self.val_preds.append(np.empty((self.X_val.shape[0], 1)))

        if self.folds == 1:
            fold_list = [(np.arange(self.X_trn.shape[0]), np.arange(self.X_val.shape[0]))]
        else:
            fold_list = KFold(len(self.y_trn), self.folds)

        for trn_idx, val_idx in fold_list:
            fold_X_trn = self.X_trn[trn_idx]
            fold_X_val = self.X_val[val_idx]
            fold_y_trn = self.y_trn[trn_idx]

            # make predictions on input data
            exists, fname = check_cache(node, fold_X_trn, X)
            if exists:
                #print('Found existing predictions,', preds.shape, '<-', np.load(fname).shape)
                preds += np.load(fname)
            else:
                self._fit_one(node, fold_X_trn, fold_y_trn, force=True)
                fold_preds = node.transform(X)
                np.save(fname, fold_preds)
                preds += fold_preds

            # make predictions on out-of-fold training data
            exists, fname = check_cache(node, fold_X_trn, fold_X_val)
            if exists:
                #print('Found existing predictions,', self.val_preds[-1][val_idx].shape, '<-', np.load(fname).shape)
                self.val_preds[-1][val_idx] = np.load(fname)
            else:
                self._fit_one(node, fold_X_trn, fold_y_trn, force=True) # todo: inefficient if it trains twice
                fold_preds = node.transform(fold_X_val)
                np.save(fname, fold_preds)
                self.val_preds[-1][val_idx] = fold_preds

        preds /= self.folds
        return preds

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
        if self.pass_features:
            print(np.hstack(all_preds).shape, X.shape)
            print(np.hstack(self.val_preds).shape, self.X_val.shape)
            return np.hstack([np.hstack(all_preds), X]), np.hstack([np.hstack(self.val_preds), self.X_val])
        else:
            return np.hstack(all_preds), np.hstack(self.val_preds)

    transform = predict