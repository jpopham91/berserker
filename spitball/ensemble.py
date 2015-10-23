from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from fn import _

class Ensemble(object):
    """
    vanilla ensemble object
    contains N layers, where layers 0..N-1 are collections of models and transformations
    and the Nth layer is a single (meta)estimator for making final predictions
    """
    def __init__(self, X, y, metric, validation_split=0.2):
        cutoff = int(len(X)*validation_split)
        self.X_trn = X[:-cutoff]
        self.y_trn = y[:-cutoff]

        # todo: look into how keras deals with the existence of validation data
        self.X_val = X[-cutoff:]
        self.y_val = y[-cutoff:]

        self.metric = metric
        self.layers = []
        self.pipe = None

    # todo: add handling for single node layer (i.e. final meta-estimator)
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        steps = [layer.make_union() for layer in self.layers]
        self.pipe = make_pipeline(*steps)
        self.pipe.fit(self.X_trn, self.y_trn)
        return self.pipe.predict(X)

    def score(self, X, y):
        return self.metric(self.pipe.predict(X), y)

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

