from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

__author__ = 'jake'


class Ensemble:
    def __init__(self, X, y, metric, val_split=0.8, layers=[]):
        self.X_trn = X
        self.y_trn = y
        self.metric = metric
        self.layers = layers
        self.pipe = None

    # todo: add handling for single node layer (i.e. final meta-estimator)
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        if not self.pipe:
            steps = list([layer.make_union() for layer in self.layers])
            self.pipe = make_pipeline(*steps)
            self.pipe.fit(self.X_trn, self.y_trn)
        return self.pipe.predict(X)

    def score(self, X, y):
        return self.metric(self.predict(X), y)

    def report(self):
        print('Base Model Scores:')
        for model in self.layers[0].models:
            print('{:.4f}'.format(model.score(self.X_trn, self.y_trn, self.metric)))
        print('Meta Estimator Score:')
        print('{:.4f}'.format(self.score(self.X_trn, self.y_trn)))
