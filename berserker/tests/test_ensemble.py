import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from berserker.layers import Layer
from berserker.nodes import Node
from berserker.ensemble import Ensemble


class TestBase(unittest.TestCase):
    def setUp(self):
        #np.random.seed(42)
        self.X = np.random.rand(400).reshape((100, 4))
        self.Xt = np.random.rand(200).reshape((50, 4))
        self.y = np.random.rand(100)
        self.ensemble = Ensemble(self.X, self.y, mean_squared_error)

class TestEnsembles(TestBase):
    def test_add_layer(self):
        self.ensemble.add_layer()
        self.assertEqual(len(self.ensemble.layers), 1)

    def test_add_node(self):
        self.ensemble.add_layer()
        self.ensemble.add_node(LinearRegression())
        self.assertEqual(len(self.ensemble.layers[0]), 1)

    def test_predict(self):
        self.ensemble.add_layer(folds=5)
        self.ensemble.add_node(LinearRegression())
        self.ensemble.add_node(SVR())
        self.ensemble.add_meta_estimator(LinearRegression())
        self.ensemble.predict(self.Xt)
        Xt2 = np.random.rand(200).reshape((50, 4))
        self.ensemble.predict(Xt2)