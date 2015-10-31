import numpy as np
from sklearn.cross_validation import KFold

from sklearn.base import BaseEstimator

from berserker.nodes import Node, check_cache
from berserker.logger import log


class Layer(object):

    def __init__(self, X: np.ndarray, y: np.array, folds: int=1, validation_split: float=0.,
                 validation_set: (np.ndarray, np.array)=(), pass_features: bool=False):
        """
        :param X: feature vectors, shape = (n_samples, n_features)
        :param y: target values, shape = (n_samples, 1) or (n_samples,)
        :param folds: k value for kfold validation option (stacking)
        :param validation_split: fraction of input data to use as validation set or holdout (blending)
        :param validation_set: explicit choice of (X, y) validation data or holdout (blending)
        :param pass_features: determines whether predict returns the feature vectors it was trained on.
        (Must be False for the final layer / meta-estimator)
        """
        assert not (validation_split and validation_set), \
            "Choose EITHER validation split or holdout"

        # initialized input data
        self.X_trn = None
        self.y_trn = None
        self.X_val = None
        self.y_val = None

        # parameters for splitting input data
        self.folds = folds
        self.validation_split = validation_split
        self.validation_set = validation_set

        # calculate trn/val based on split params
        self._calculate_inputs(X, y)

        # pred/output params
        self.nodes = []
        self.val_preds = []
        self.pass_features = pass_features

    def _calculate_inputs(self, X: np.ndarray, y: np.array) -> None:
        """
        Divides input data into training and validation sets.
        Used in predict when training data is altered, such as for any layer beyond the base pool.
        :param X: feature vectors
        :param y: target values
        """
        if self.validation_split:
            cutoff = -int(len(X)*self.validation_split)
            self.X_trn = X[:cutoff]
            self.y_trn = y[:cutoff]
            self.X_val = X[cutoff:]
            self.y_val = y[cutoff:]

        elif self.validation_set:
            self.X_trn = X
            self.y_trn = y
            self.X_val = self.validation_set[0]
            self.y_val = self.validation_set[1]

        else:
            self.X_trn = self.X_val = X
            self.y_trn = self.y_val = y

    def __len__(self) -> int:
        """Overridden to return the number of nodes in the layer"""
        return self.size()

    def size(self) -> int:
        """Number of nodes in this layer"""
        return len(self.nodes)

    def add(self, thing, **kwargs) -> None:
        """
        appends a node to the current layer
        :param thing: the entity to be added to the layer
        :type thing: Node|BaseEstimator
        :param kwargs: if passing a BaseEstimator, the args are used to create a new node containing it
        :type kwargs: dict
        """
        # todo: might use the new @overload to separate add estimator vs add node
        if isinstance(thing, Node):
            self.nodes.append(thing)
        elif isinstance(thing, BaseEstimator):
            self.nodes.append(Node(thing, **kwargs))
        else:
            log.warning('Warning, unfamiliar type: {}'.format(type(thing)))

    def _predict_one(self, node, X):
        """
        Fits a single node and makes predictions on the validation and test sets
        If layer uses kfold, it fits training data and predicts the out of fold indices.
        Test data is averaged over all folds.
        :param node:
        :param X:
        :return: array of predictions for X
        """
        # create new empty arrays for the validation and test predictions
        preds = np.zeros((X.shape[0], 1))
        self.val_preds.append(np.empty((self.X_val.shape[0], 1)))

        # when using the same data for training and validation
        # train/val indices are determined via kfold
        if np.array_equal(self.X_trn, self.X_val) and self.folds > 1:
            fold_list = KFold(len(self.X_trn), self.folds)
        # otherwise a holdout validation set is used
        else:
            fold_list = [(np.arange(self.X_trn.shape[0]), np.arange(self.X_val.shape[0]))]

        for trn_idx, val_idx in fold_list:
            fold_X_trn = self.X_trn[trn_idx]
            fold_y_trn = self.y_trn[trn_idx]
            fold_X_val = self.X_val[val_idx]

            # always force refit to override previous folds
            self.val_preds[-1][val_idx] = node.fit_predict(fold_X_trn, fold_y_trn, fold_X_val, refit=True)
            preds += node.fit_predict(fold_X_trn, fold_y_trn, X, refit=False)

        # average preds over all folds
        preds /= self.folds
        return preds

    def _predict_all(self, X):
        """
        simple generator function used as a helper for predict()
        :param X:
        :return:
        """
        for node in self.nodes:
            yield self._predict_one(node, X)

    # todo: consider removing input data from init, and call predict with X_trn, y_trn, X?
    def predict(self, X: np.ndarray, new_data: (np.ndarray, np.array)=()):
        """
        Trains all nodes in the layer and collects their predictions for validation and test sets.
        :param X: the feature vectors to predict on.
        :type X: np.ndarray
        :param new_data: (X, y) to train/validate on.
        :type new_data: (np.ndarray, np.array)
        Used when called by Ensemble so base predictions can be added to the feature space.
        :return: test_preds, (val_preds, y_val)
        test_preds shape = (n_samples, n_nodes) or (n_samples, n_nodes+n_features) if pass_features is True.
        """
        if new_data:
            self._calculate_inputs(*new_data)
        self.val_preds = []
        all_preds = np.hstack([pred for pred in self._predict_all(X)])
        val_preds = np.hstack(self.val_preds)
        # optionally add the input data to our predictions
        if self.pass_features:
            all_preds = np.hstack([all_preds, X])
            val_preds = np.hstack([val_preds, self.X_val])
        return all_preds, (val_preds, self.y_val)
