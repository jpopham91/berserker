from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

__author__ = 'jake'


class Ensemble:
    def __init__(self, X, y, metric, validation_split=0.2, layers=[]):
        cutoff = int(len(X)*validation_split)
        self.X_trn = X[:-cutoff]
        self.y_trn = y[:-cutoff]

        # todo: look into how keras deals with the existence of validation data
        self.X_val = X[-cutoff:]
        self.y_val = y[-cutoff:]

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

    # todo: allow models to be named to improve readability
    def report(self):
        print('Base Model Validation Scores:')
        for model in self.layers[0].models:
            print('{:.4f}'.format(model.score(self.X_val, self.y_val, self.metric)))
        print('Meta Estimator Validation Score:')
        print('{:.4f}'.format(self.score(self.X_val, self.y_val)))
