__author__ = 'jake'
from sklearn.base import BaseEstimator
import numpy as np


class Averager(BaseEstimator):
    """
    Simple meta-estimator which averages predictions
    May use any of the pythagorean means
    """


class StepwiseRegressor(Averager):
    """
    An averager which iteratively adds predictions which optimize a metric
    """


class FeatureWeightedEstimator(BaseEstimator):
    """
    Expands the feature space by taking the outer product of the features and predictions at each sample
    This is then fit using some estimator (log/lin regression)
    """
    def __init__(self, estimator):
        self.estimator = estimator

    @staticmethod
    def _combine_features(X, y_pred):
        Xy = np.empty_like((X.shape[0], X.shape[1]*y_pred.shape[1]))
        for i in X.shape[1]:
            for j in y_pred.shape[1]:
                Xy[:, i*X.shape[0]+j] = X[i]*y_pred[j]
        return Xy

    def fit(self, X, y_pred, y_true):
        """Takes the feature vectors AND predictions as training data"""
        assert X.shape[0] == y_pred.shape[0] == len(y_true)
        Xy = self._combine_features(X, y_pred)
        self.estimator.fit(Xy, y_true)

    def predict(self, X, y_pred):
        assert X.shape[0] == y_pred.shape[0]
        Xy = self._combine_features(X, y_pred)
        return self.estimator.predict(Xy)