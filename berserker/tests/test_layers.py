import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from berserker.layers import Layer
from berserker.nodes import Node


class TestBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.rand(100).reshape((25, 4))
        self.Xt = np.random.rand(200).reshape((-1, 4))
        self.y = np.random.rand(25)
        self.layer = Layer(self.X, self.y)
        self.node = Node(LinearRegression())


class TestLayers(TestBase):
    def test_no_val_split(self):
        layer = Layer(self.X, self.y)
        self.assertEqual(self.X.shape, layer.X_trn.shape)

    def test_val_split(self):
        split = 0.2
        layer = Layer(self.X, self.y, validation_split=split)
        self.assertAlmostEqual(self.X.shape[0]*(1-split), layer.X_trn.shape[0])
        self.assertEqual(self.X.shape[1], layer.X_trn.shape[1])

    def test_add_node(self):
        self.assertEqual(len(self.layer.nodes), 0)
        self.layer.add(self.node)
        self.assertEqual(len(self.layer.nodes), 1)

    def test_pred_shape(self):
        self.layer.add(LinearRegression())
        self.layer.add(SVR())
        tpreds, vpreds = self.layer.predict(self.Xt)
        self.assertEqual(vpreds.shape[1], 2)