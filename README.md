# Spitball
## A Simple Model Ensembling Framework

Spitball is a python module used for streamlining the creation of complex machine learning ensembles.  Standing on the shoulders of scikit-learn, spitball allows multiple estimators to easily be combined into a single model that is more powerful than its constituents. If your top priority is predictive accuracy, spitball's simple api gives everything you need to quickly build a powerful black-box model that will easily outperform more traditional approaches. You don't need statistical rigor or sound methodology - you have results.

*This module is very much in active development. While exciting new features will be added, extensive refactoring and breaking changes are basically inevitable.*

## Key components

 - __Node:__ The fundemental unit in spitball ensembles, nodes are containers that adds extra functionality to your estimator. Their killer feature is allowing predictors to mimic transformers so that they can be chained together.  Nodes feature more advanced configurability than vanilla sklearn models, allowing transformations, scaling, bagging, etc. to be specified as unique hyperparameters for every estimator.
 
 - __Meta-Estimators:__ This is the component that forms a final prediction.  It is trained on the predictions of other estimators. Any estimator may currently be used as the meta-estimator, and specialized algorithms like voters, feature-weighted stackers, and greedy stepwise regressors will be added shortly.
 
 - __Ensemble:__ A generic model built on top of scikit-learn's pipeline and feature union classes. A meta-estimator is trained on the predictions of a pool of base estimators to find an optimal combination. This class is currently the only option for connecting any arbitrary number of layers.

   - __Blender:__ A specific ensemble implementation which uses a hold-out training set to train the meta-estimator, preventing leakage and overfitting.

   - __Stacker:__ A similar implementation which uses out-of-fold predictions to train the meta-estimator, allowing the full training set to be used.

## Example

###Incredibly simple regression model blending:
```python
from spitball.ensemble import Blender
from spitball.models import Model, Via

blender = Blender(X, y, mean_squared_error)

estimators = [Ridge, ElasticNet, Lasso, LinearRegression, SVR, KNeighborsRegressor,
              XGBRegressor, RandomForestRegressor]
              
for estimator in estimators:
    blender.add(Model(estimator()))

blender.add(Via())

preds = blender.predict(X_tst)
```
<pre>
Model                  Train   Val   
-------------------------------------
Ridge                  0.0273  0.0281
ElasticNet             0.0523  0.0541
Lasso                  0.0523  0.0541
LinearRegression       0.0272  0.0280
SVR                    0.0191  0.1406
KNeighborsRegressor    0.0302  0.0470
XGBRegressor           0.0227  0.0227
RandomForestRegressor  0.0356  0.0351
<b>Blend Ensemble         0.0146  0.0168</b>
</pre>
