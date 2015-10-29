from berserker.layers import Layer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline, make_union
import numpy as np
from fn import _

class Ensemble(object):
    """
    vanilla ensemble object
    contains N layers, where layers 0..N-1 are collections of models and transformations
    and the Nth layer is a single (meta)estimator for making final predictions
    """
    def __init__(self, X, y, metric, validation_split=0.2):
        cutoff = int(len(X)*validation_split)
        self.X_trn = X#[:-cutoff]
        self.y_trn = y#[:-cutoff]

        # todo: look into how keras deals with the existence of validation data
        #self.X_val = X[-cutoff:]
        #self.y_val = y[-cutoff:]

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

    def predict(self, X):
        self.layers[0].X_trn = self.layers[0].X_val = self.X_trn
        preds, val_preds = self.layers[0].transform(X)
        if len(self.layers) == 0:
            return preds
        for layer in self.layers[1:]:
            layer.X_trn = layer.X_val = val_preds
            #layer.y_trn = self.y_val
            preds, val_preds = layer.transform(preds)
        return preds

    def scores(self, X, y=None):
        preds = self.predict(X)
        for n, layer in enumerate(self.layers):
            #print('\nLayer', n+1)
            print('{: <36}  {: <16}'.format('\nLevel {:d} Estimators ({} features)'.format(n+1, layer.X_trn.shape[1]),  'Validation Score'))
            print('-'*53)
            for node, pred in zip(layer.nodes, layer.val_preds):
                print('{: <36}  {:.4f}'.format(node.name, self.metric(layer.y_val, pred)))

        #print('{: <36}  {: <16}'.format('\nFull Ensemble'.format(n+1),  'Holdout Score'))
        #print('-'*53)
        #print('\033[1m{: <36}  {:.4f}\033[0m'.format('', self.metric(y, scr_preds)))
        return preds

    # todo: some of the string formatting here is ugly and may need to get factored out
    def report(self, sort=False):
        val_scores = []
        for model in self.layers[0].models:
            if hasattr(model, 'estimator'):
                val_scores.append((str(model.estimator).split('(')[0],
                                   model.score(self.X_trn, self.y_trn, self.metric),
                                   model.score(self.X_val, self.y_val, self.metric)))

        if sort: val_scores = sorted(val_scores, key=lambda x: x[2])
        width = max(map(lambda x: len(x[0]), val_scores))
        print('\n{: <{w}}  {: <6}  {: <6}'.format('Model', 'Train', 'Val', w=width))
        print('-'*(width+16))
        for score in val_scores:
            print('{: <{w}}  {:.4f}  {:.4f}'.format(*score, w=width))
        print('\033[1m{: <{w}}  {:.4f}  {:.4f}\033[0m'.format('Full Ensemble', self.score(self.X_trn, self.y_trn), self.score(self.X_val, self.y_val), w=width))

class Stacker(Ensemble):
    """
    splits the data into k folds
    for each, the base models train on the training fold
    and predict the validation fold
    concatenate these predictions to match the original input
    train each model again on the entire training set,
    predict the test set with these models
    finally use a second stage meta-estimator to fit the weights of each prediction
    predict the test set with this meta estimator
    """

    def __init__(self):
        super.__init__(self)

# began as a cut an paste of ensemble
class Blender(object):
    """
    creates a hold-out set up front
    base models are fit to training set
    base models predict validation and test sets
    meta-model is fit to base validation predictions
    meta-model predicts base test predictions
    """
    def __init__(self, X, y, metric, validation_split=0.2, meta_estimator=LinearRegression()):
        cutoff = int(len(X)*validation_split)
        self.X_trn = X[:-cutoff]
        self.y_trn = y[:-cutoff]
        self.trn_preds = []
        self.trn_score = None

        # todo: look into how keras deals with the existence of validation data
        self.X_val = X[-cutoff:]
        self.y_val = y[-cutoff:]
        self.val_preds = []
        self.val_score = None

        self.is_fit = False
        self.meta_estimator = meta_estimator
        self.metric = metric
        self.base_models = Layer(self.X_trn, self.y_trn)
        self.pipe = None

    # todo: add handling for single node layer (i.e. final meta-estimator)
    def add(self, model):
        """
        adds models to the base layer
        :param layer:
        :return:
        """
        self.base_models.add(model)

    def _fit(self):
        for model in self.base_models.nodes:
            model.fit(self.X_trn, self.y_trn)
            self.val_preds.append(model.transform(self.X_val))
        self.meta_estimator.fit(np.hstack(self.val_preds), self.y_val)

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            self._fit()

        preds = []
        for model in self.base_models.nodes:
            preds.append(model.transform(X))

        return self.meta_estimator.predict(np.hstack(preds))

    def score(self, X, y):
        return self.metric(self.predict(X), y)

    # todo: some of the string formatting here is ugly and may need to get factored out
    # maybe break into get_scores and report?
    def report(self, sort=False):
        val_scores = []
        for model in self.base_models.nodes:
            if hasattr(model, 'estimator'):
                val_scores.append((model.name,
                                   model.score(self.X_trn, self.y_trn, self.metric),
                                   model.score(self.X_val, self.y_val, self.metric)))

        self.trn_score = self.score(self.X_trn, self.y_trn)
        self.val_score = self.score(self.X_val, self.y_val)

        if sort: val_scores = sorted(val_scores, key=lambda x: x[2])
        width = max(map(lambda x: len(x[0]), val_scores))

        print('\n{: <{w}}  {: <6}  {: <6}'.format('Model', 'Train', 'Val', w=width))
        print('-'*(width+16))
        for score in val_scores:
            print('{: <{w}}  {:.4f}  {:.4f}'.format(*score, w=width))
        print('\033[1m{: <{w}}  {:.4f}  {:.4f}\033[0m'.format('Blend Ensemble', self.trn_score, self.val_score, w=width))
