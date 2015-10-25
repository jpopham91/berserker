from berserker.ensemble import Ensemble
from berserker.layers import Layer
from berserker.nodes import Node
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

data = load_boston().data
n_samples = data.shape[0]
test_split = int(0.2*n_samples)
np.random.seed(42)
shuffled_data = data#[np.random.permutation(n_samples)]

print(data)

train = shuffled_data[:-test_split]
test = shuffled_data[test_split:]

X_trn = train[:, :-1]
y_trn = train[:, -1]

X_tst = train[:, :-1]
y_tst = train[:, -1]

model = Ensemble(X_trn, y_trn, mean_squared_error)

model.add_node(LinearRegression())
model.add_node(RandomForestRegressor())
model.add_node(RandomForestRegressor(50))
model.add_node(ExtraTreesRegressor())

model.add_layer(folds=1)
model.add_node(LinearRegression())

preds = model.predict(X_trn)
model.scores()
print(mean_squared_error(y_tst, preds))


