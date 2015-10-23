from .layers import Layer
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import *
from sklearn.svm import SVC, SVR
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.metrics import mean_squared_error

print("load and shuffle training data")
train = pd.read_csv('/home/jake/Downloads/rossman/train_transformed.csv')
train = train[train.Sales > 0]
test = pd.read_csv('/home/jake/Downloads/rossman/test_transformed.csv')
#np.random.seed(42)
#train = train.iloc[np.random.permutation(len(train))]

train.Open.fillna(1, inplace=True)
test.Open.fillna(1, inplace=True)


train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

ids = test.Id.values
y = train['Sales'].values

other_drops = ['Avg_Customers']

X = train.drop(['Store', 'Sales', 'Customers'] + other_drops, axis=1).values
X_tst = test.drop(['Id', 'Store'] + other_drops, axis=1).values

#assert np.array_equal(train.columns, test.columns)

#X = np.array(train.values, dtype=np.float32)
#X_tst = np.array(test.values, dtype=np.float32)

assert X.shape[1] == X_tst.shape[1]

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true-y_pred)/y_true)))

est = xgb.XGBRegressor(n_estimators=50, max_depth=10, learning_rate=0.25,
                              subsample=0.9, colsample_bytree=0.8)

est.fit(X, y)
y_pred = est.predict(X)

train['pred'] = y_pred

train.to_csv('/home/jake/Downloads/rossman/train_preds.csv', index=False)

ensemble = Layer(X, y, rmspe, .8)

print("building ensemble")

ensemble.add(Lasso())
ensemble.add(ElasticNet())
ensemble.add(Ridge())
#ensemble.add(SVR(kernel='linear'))
ensemble.add(xgb.XGBRegressor(n_estimators=10, max_depth=10, learning_rate=0.25,
                              subsample=0.9, colsample_bytree=0.8))
ensemble.add(RandomForestRegressor(n_estimators=10, n_jobs=-1))
#ensemble.add(ExtraTreesRegressor(n_estimators=50, n_jobs=-1))


ensemble.fit_predict(X_tst, 4)

print('\nBlending with meta-estimator')
preds = ensemble.blend(LinearRegression())
pd.DataFrame({'Id' : ids, 'Sales' : preds}).to_csv('/home/jake/Downloads/rossman/blend.csv', index=False)

print('\nAveraging predictions with stepwise regression')
preds = ensemble.stepwise_regression()
pd.DataFrame({'Id' : ids, 'Sales' : preds}).to_csv('/home/jake/Downloads/rossman/stepwise.csv', index=False)

print('\nAveraging predictions with stepwise regression')
preds = ensemble.stepwise_regression()
pd.DataFrame({'Id' : ids, 'Sales' : preds}).to_csv('/home/jake/Downloads/rossman/stepwise.csv', index=False)