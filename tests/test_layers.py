import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from berserker.layers import Layer
from berserker.nodes import Node


class TestBase(unittest.TestCase):
    def setUp(self):
        #np.random.seed(42)
        self.X = np.random.rand(100).reshape((25, 4))
        self.Xt = np.random.rand(200).reshape((50, 4))
        self.y = np.random.rand(25)
        self.stack = Layer(self.X, self.y, folds=3)
        self.blend = Layer(self.X, self.y, validation_split=0.2)
        self.stack_clf = Layer(self.X, (self.y+0.5).astype(int), folds=3)
        self.node = Node(LinearRegression())

# todo refactor out base classes for stacks, blends
class TestLayers(TestBase):
    def test_no_val_split(self):
        layer = Layer(self.X, self.y)
        self.assertEqual(self.X.shape, layer.X_trn.shape)

    def test_val_split(self):
        split = 0.2
        layer = Layer(self.X, self.y, validation_split=split)
        self.assertAlmostEqual(self.X.shape[0]*(1-split), layer.X_trn.shape[0])
        self.assertEqual(self.X.shape[1], layer.X_trn.shape[1])
        self.assertTrue(np.array_equal(self.stack.X_trn, self.stack.X_val))

    def test_val_set(self):
        X_val = np.random.rand(*self.X.shape)
        y_val = np.random.rand(*self.y.shape)
        stack = Layer(self.X, self.y, validation_set=(X_val, y_val))
        self.assertTrue(np.array_equal(stack.X_trn, self.X))
        self.assertTrue(np.array_equal(stack.X_val, X_val))

    def test_add_node(self):
        self.assertEqual(len(self.stack.nodes), 0)
        self.stack.add(self.node)
        self.assertEqual(len(self.stack.nodes), 1)

    def test_reg_pred_shape(self):
        self.stack.add(LinearRegression())
        self.stack.add(SVR())
        #preds, val_data = self.stack.predict(self.Xt)
        preds, (X_val, y_val) = self.stack.predict(self.Xt)
        self.assertEqual(X_val.shape[1], 2)
        self.assertEqual(preds.shape[0], self.Xt.shape[0])
        self.assertEqual(X_val.shape[0], self.X.shape[0])

    def test_blend_reg_pred_shape(self):
        blend = Layer(self.X, self.y, validation_split=0.2)
        blend.add(LinearRegression())
        blend.add(SVR())
        preds, (X_val, y_val) = blend.predict(self.Xt)
        self.assertEqual(X_val.shape[1], 2)
        self.assertEqual(preds.shape[0], self.Xt.shape[0])
        self.assertAlmostEqual(X_val.shape[0], self.X.shape[0]*0.2)

    def test_clf_pred_shape(self):
        self.stack_clf.add(LogisticRegression())
        self.stack_clf.add(SVC())
        preds, (X_val, y_val)  = self.stack_clf.predict(self.Xt)
        self.assertEqual(X_val.shape[1], 2)
        self.assertEqual(preds.shape[0], self.Xt.shape[0])
        self.assertEqual(X_val.shape[0], self.X.shape[0])

    def test_multipred(self):
        self.stack.add(LinearRegression())
        self.stack.add(SVR())
        preds0, (X_val0, y_val0)  = self.stack.predict(self.Xt)
        self.stack.add(SVR('linear'))
        preds, (X_val, y_val)  = self.stack.predict(self.Xt)
        self.assertEqual(X_val.shape[1], 3)
        self.assertTrue(np.array_equal(preds[:, :2], preds0[:, :2]))
