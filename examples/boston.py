from berserker.ensemble import Ensemble
from berserker.layers import Layer
from berserker.nodes import Node
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

data = load_boston().data
n_samples = data.shape[0]
test_split = int(0.2*n_samples)
np.random.seed(43)
shuffled_data = data[np.random.permutation(n_samples)]

train = shuffled_data[:-test_split]
test = shuffled_data[-test_split:]
X_trn = train[:, :-1]
y_trn = train[:, -1]

X_tst = test[:, :-1]
y_tst = test[:, -1]

log_transform = {'target_transform'  : lambda x: np.log(x+1),
                 'inverse_transform' : lambda x: np.exp(x)-1}

model = Ensemble(X_trn, y_trn, mean_squared_error)

model.add_layer(pass_features=False, validation_split=0.2)
model.add_node(RandomForestRegressor(50), name='50 Tree Random Forest')
model.add_node(GradientBoostingRegressor(n_estimators=250), name='Gradient Boosted Trees')
model.add_node(SVR('linear'), name='Linear SVR Log(y)', scale_x=True)
model.add_node(SVR('linear'), name='Linear SVR', scale_x=True, **log_transform)

model.add_layer(folds=1)
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

preds = model.predict(X_tst, y_tst)


