import sys
from berserker.layers import Layer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline, make_union
import numpy as np
from time import time, sleep
from fn import _

class Ensemble(object):
    """
    vanilla ensemble object
    contains N layers, where layers 0..N-1 are collections of models and transformations
    and the Nth layer is a single (meta)estimator for making final predictions
    """
    def __init__(self, X, y, metric, holdout=()):
        self.X_trn = X
        self.y_trn = y

        self.holdout = holdout

        self.metric = metric
        self.layers = []

    def add_node(self, node, **kwargs):
        self.layers[-1].add(node, **kwargs)
        return self

    def add_meta_estimator(self, node, **kwargs):
        self.add_layer(folds=1)
        self.add_node(node, **kwargs)
        return self

    def add_layer(self, **kwargs):
        self.layers.append(Layer(self.X_trn, self.y_trn, **kwargs))
        return self

    def _predict_all(self, X):
        """recursively trace through each layer, using the previous layer's output as training data"""
        def _predict_layer(layers, X, new_data):
            head, *tail = layers
            preds, val_data = head.predict(X, new_data)
            if not tail:
                return preds
            else:
                return _predict_layer(tail, preds, val_data)

        return _predict_layer(self.layers, X, (self.X_trn, self.y_trn))

    def predict(self, X, y=None):
        start = time()
        preds = self._predict_all(X)
        elapsed = time() - start

        sleep(1)

        print('\n' + '='*53)
        print('R E S U L T S'.center(53, '-'))
        print('-'*53)
        print('Elapsed time:                         {:.3g} seconds'.format(elapsed))
        print('Total models in ensemble:             {:d}'.format(sum([layer.size() for layer in self.layers])))
        print('Cached predictions used:              {:d} / {:d}'.format(sum([node.cached_preds for layer in self.layers for node in layer.nodes]),
                                                             sum([node.total_preds for layer in self.layers for node in layer.nodes])))
        print('-'*53)
        for n, layer in enumerate(self.layers):
            print('{: <36}  {: <16}'.format('\nLevel {:d} Estimators ({} features)'.format(n+1, layer.X_trn.shape[1]),  'Validation Score'))
            print('-'*53)
            for node, pred in zip(layer.nodes, layer.val_preds):
                print('{: <36}  {:.4g}'.format(node.name, self.metric(layer.y_val, pred)))

        if y is not None:
            print('{: <36}  {: <16}'.format('\nFull Ensemble'.format(n+1),  'Holdout Score'))
            print('-'*53)
            print('\033[1m{: <36}  {:.4g}\033[0m'.format('', self.metric(y, preds)))

        print('\n' + '='*53)
        return preds
