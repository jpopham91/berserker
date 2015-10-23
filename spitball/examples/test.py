__author__ = 'jake'

from layers import Layer, Ensemble
from models import Model, Via
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

X = np.random.rand(2000,50)
X2 = np.random.rand(2000,50)
y = np.random.rand(2000)

est = make_pipeline(make_union(PCA(1), PCA(2)), LogisticRegression())

est.fit(X,y)

ensemble = Ensemble(X, y, mean_squared_error)

layer = Layer()
layer.add(Model(LinearRegression()))
layer.add(Model(LinearRegression()))
layer.add(Via())
ensemble.add(layer)

layer = Layer()
layer.add(Model(LogisticRegression()))
layer.add(Model(LogisticRegression()))
ensemble.add(layer)

ensemble.predict(X2)