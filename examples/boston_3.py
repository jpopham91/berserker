from berserker.ensemble import Ensemble
from berserker.layers import Layer
from berserker.nodes import Node
from sklearn.datasets import load_boston
from sklearn.linear_model import *
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

data = load_boston().data
n_samples = data.shape[0]
test_split = int(0.2*n_samples)
np.random.seed(42)
shuffled_data = data[np.random.permutation(n_samples)]

#print(data)

train = shuffled_data[:-test_split]
test = shuffled_data[-test_split:]
X_trn = train[:, :-1]
y_trn = train[:, -1]

X_tst = test[:, :-1]
y_tst = test[:, -1]

model = Ensemble(X_trn, y_trn, mean_squared_error)

model.add_layer(folds=5, pass_features=True)
for linear_model in [LinearRegression, Ridge, Lasso, ElasticNet, Lars, BayesianRidge, RANSACRegressor]:
    model.add_node(linear_model(), scale_x=True)
for tree_model in [RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor]:
    for n_trees in [5, 50, 500]:
        model.add_node(tree_model(n_estimators=n_trees), suffix='{:d} trees'.format(n_trees))
for kernel in ['linear', 'rbf']:
    for c in [.1, 1, 10]:
        model.add_node(SVR(kernel, C=c), suffix='{} kernel, C={:.1f}'.format(kernel, c), scale_x=True)

model.add_layer(folds=3, pass_features=True)
model.add_node(RandomForestRegressor(500), name='RF Meta Estimator')
model.add_node(GradientBoostingRegressor(n_estimators=500), name='GBR Meta Estimator')
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

model.add_layer(folds=1)
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

model.scores(X_tst, y_tst)



