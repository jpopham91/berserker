import unittest

from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import hashlib

from berserker.nodes import Node, _get_bytecode, generate_hash, check_cache


class TestBase(unittest.TestCase):
    def setUp(self):
        self.f1 = lambda x: x**2 - x
        self.f2 = lambda x: x**2 - x
        self.f3 = lambda x: x + x**3

        self.n1 = Node(LogisticRegression(), target_transform=self.f1)
        self.n2 = Node(LogisticRegression(), target_transform=self.f2)
        self.n3 = Node(LinearRegression())
        self.n4 = Node(LogisticRegression('l1'), target_transform=self.f1)

        self.a1 = np.random.rand(100000).reshape((5000, 20))
        self.a2 = np.random.rand(100000).reshape((5000, 20))
        self.a3 = self.a1.reshape((50, 2000))
        self.a4 = self.a1[::-1]

    def assert_nodes_equal(self, a, b):
        self.assertEqual(str(a), str(b))
        self.assertEqual(generate_hash(a, self.a1, self.a2),
                         generate_hash(b, self.a1, self.a2))

    def assert_nodes_not_equal(self, a, b):
        self.assertNotEqual(str(a), str(b))
        self.assertNotEqual(generate_hash(a, self.a1, self.a2),
                            generate_hash(b, self.a1, self.a2))

    def assert_data_equal(self, a, b):
        self.assertEqual(generate_hash(self.n1, a, self.a2),
                         generate_hash(self.n1, b, self.a2))

    def assert_data_not_equal(self, a, b):
        self.assertNotEqual(generate_hash(self.n1, a, self.a2),
                            generate_hash(self.n1, b, self.a2))


class TestNodes(TestBase):

    def test_str(self):
        self.assert_nodes_equal(self.n1, self.n2)

    def test_transforms(self):
        self.assertNotEqual(self.f1, self.f2)
        self.assertEqual(_get_bytecode(self.f1), _get_bytecode(self.f2))
        self.assertNotEqual(_get_bytecode(self.f1), _get_bytecode(self.f3))

    def test_hashgen_lambdas(self):
        self.assert_nodes_equal(self.n1, self.n2)
        self.assert_nodes_not_equal(self.n1, Node(LogisticRegression, target_transform=lambda x: 2 + x/2))

    def test_hashgen_swap_data(self):
        self.assert_data_equal(self.a1, self.a1)
        self.assert_data_not_equal(self.a1, self.a2)
        self.assertNotEqual(generate_hash(self.n1, self.a1, self.a2),
                            generate_hash(self.n1, self.a2, self.a1))

    def test_hashgen_reshape_data(self):
        self.assert_data_equal(self.a1.reshape((-1, 1)), self.a1.reshape((-1, 1)))
        self.assert_data_not_equal(self.a1, self.a1.reshape((-1, 1)))

    def test_hashgen_reorder_data(self):
        self.assert_data_equal(self.a1[::-1], self.a1[::-1])
        self.assert_data_not_equal(self.a1, self.a1[::-1])

    def test_hashgen_estimator_args(self):
        self.assert_nodes_not_equal(Node(LogisticRegression('l1')),
                                    Node(LogisticRegression('l2')))
        self.assert_nodes_equal(Node(LogisticRegression('l1', C=5)),
                                Node(LogisticRegression(C=5, penalty='l1')))

    def test_predict_shape(self):
        X = np.random.rand(100).reshape((25, 4))
        Xt = np.random.rand(200).reshape((-1, 4))
        y = np.random.rand(25)
        preds = self.n3.fit_predict(X, y, Xt)
        self.assertEqual(preds.shape, (len(Xt),))

    def test_transform_shape(self):
        X = np.random.rand(100).reshape((25, 4))
        Xt = np.random.rand(200).reshape((-1, 4))
        y = np.random.rand(25)
        self.n3.fit_predict(X, y, Xt)
        preds = self.n3.transform(Xt)
        self.assertEqual(preds.shape, (len(Xt), 1))

    def test_saving(self):
        X = np.random.rand(100).reshape((25, 4))
        Xt = np.random.rand(200).reshape((-1, 4))
        y = np.random.rand(25)
        self.n3.fit_predict(X, y, Xt)
        exists, _ = check_cache(self.n3, X, Xt)
        self.assertTrue(exists)


if __name__ == '__main__':
    unittest.main()
