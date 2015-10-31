# Berserker
<!---ᛒΣᚱᛊΣᚱᛕΣᚱ-->
## Ensembles for your ensembles so you can predict on your predictions ![blah](http://s3-ak.buzzfeed.com/static/user_images/web02/2010/9/3/10/xzibit-13834-1283523581-22_large.jpg "yodawg")

**Berserker** is a python module used for streamlining the creation of complex machine learning ensembles.  These aren't your dad's ensembles, they're the entire damn orchestra.  Ever wondered what would happen if you combined 100 different regression models into a single super-estimator?  Because now's your chance to find out.  Berserker provides you with the duct tape and glue needed to stack and blend your way to the most frightening(ly effective) predictive models imaginable.

**Warning: If you are concerned with such topics as *"statistical rigor"* and *"sound methodology"*, you may find the content herein disturbing. Proceed with caution. You have been warned.**

---
*This module is very much in active development. While exciting new features will be added, extensive refactoring and breaking changes are basically inevitable.*

## Features

 - __Scikit-Learn Compatibility:__ Berserker nodes interact with their base estimators assuming a scikit-learn api, so there is no need to reinvent the wheel.  Any scikit-learn estimator or transformer can be added to a your ensemble, as well as models from many third party packages like xgboost and keras.
  
 - __Intelligent Model Persistence:__ A unique 384-bit hash id is generated any time a node is asked to make a prediction which is used to cache the model's output to disk.  Berserker automatically checks the cache for the current model/train/test combination, so it won't waste time retraining a model it already has predictions for.
 
 - __Everything is a Hyperparameter:__ Feature scaling, target data transformations, splits, and folds are now variables which can be optimized.  But rather than searching for the ideal combination, just feed everything into the model and most useful ones are automatically given the most weight.
 
 - __Positively Pythonic:__ Berserker ensembles are constructed using a compact, human-readable syntax at the highest level of abstraction that is practical.  Rapid development would be an understatement - the most basic ensembles can be set up in about five lines of code, and nodes/layers can generated algorithmically.

## Key Components

 - __Node:__ The fundamental unit in berserker ensembles, nodes are containers that adds extra functionality to your estimator. Their killer feature is allowing predictors to mimic transformers so that they can be chained together.  Nodes feature more advanced configurability than vanilla sklearn models, allowing transformations, scaling, bagging, etc. to be specified as unique hyperparameters for every estimator.
 
 - __Layer:__ A level of abstraction above the node is the layer.  Layers are collections of one or more nodes which share a common input (training and test data).  Layers inherit the same scikit-learn api is nodes, and their predict function combines the predictions of every node as columns in a matrix.  The predictions of one layer can be piped into another as training data.
 
 - __Ensemble:__ A generic model containing a sequence of layers.  Ensembles are initialized with a labeled training dataset, and some performance metric to optimize. A single 'predict' call orchestrates transformations and movement of data between the ensemble's components to report its performance and produce an estimation. Ensembles will typically at least two layers:
    - A pool of base estimators which make initial predictions based on the training data.
    - A meta-estimator to find an the optimal weights for each prediction from the previous layer.
 
## Example
Below is a few toy examples using the Boston housing prices dataset.

#### A bare-minimum stacking ensemble:
```python
from berserker.ensemble import Ensemble
from berserker.layers import Layer
from berserker.nodes import Node

model = Ensemble(X_trn, y_trn, mean_squared_error)

# base estimator pool
model.add_layer(folds=5)
model.add_node(RandomForestRegressor(50), name='50 Tree Random Forest')
model.add_node(GradientBoostingRegressor(n_estimators=250), name='250 Gradient Boosted Trees')

# meta-estimator
model.add_layer()
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

preds = model.predict(X_tst)
```
<pre>
Level 1 Estimators (12 features)     Validation Score
-----------------------------------------------------
50 Tree RF                            16.1368
Gradient Boosted Trees                18.4357

Level 2 Estimators (14 features)      Validation Score
-----------------------------------------------------
<b>Lin Reg Meta Estimator                15.5071</b>
</pre>

Even with this simple example, the ensemble managed to achieve a lower error than either of the two individual models.  Now lets try scaling up a tiny bit and add another base estimator.  Remember, the original random forest and gradient boosting models to not need to be recomputed, so playing around with adding new nodes is fast and computationally inexpensive.

#### Diversifying the base model pool:
```python
model = Ensemble(X_trn, y_trn, mean_squared_error)

# base estimator pool
model.add_layer(folds=5, pass_features=True)
model.add_node(RandomForestRegressor(50), name='50 Tree Random Forest')
model.add_node(GradientBoostingRegressor(n_estimators=250), name='250 Gradient Boosted Trees')
model.add_node(SVR('linear'), name='Linear SVR', scale_x=True)

# meta-estimator
model.add_layer()
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

preds = model.predict(X_tst)
```
<pre>
Level 1 Estimators (12 features)     Validation Score
-----------------------------------------------------
50 Tree Random Forest                 16.1368
Gradient Boosted Trees                18.4357
Linear SVR                            19.5079

Level 2 Estimators (15 features)     Validation Score
-----------------------------------------------------
<b>Lin Reg Meta Estimator                15.4028</b>
</pre>

Incredible, right?  Even adding a model which is worse individually than any of the existing ones, the performance of the model *improves*.  This is one of key advantages to using ensemble methods - weaker models can still be useful as long as they add sufficient diversity. (It may be wrong more often, but it is unlikely to be wrong in the exact same ways)

Now lets go a little crazy (or berserk, if you will).  We can algorithmically generate a slew of base estimators with varying parameters.  We'll add another layer too, while we're at it.

```python
model = Ensemble(X_trn, y_trn, mean_squared_error)

# base estimator pool
model.add_layer(folds=5, pass_features=True)

for linear_model in [LinearRegression, Ridge, Lasso, ElasticNet, Lars, BayesianRidge, RANSACRegressor]:
    model.add_node(linear_model(), scale_x=True)

for tree_model in [RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor]:
    for n_trees in [5, 50, 500]:
        model.add_node(tree_model(n_estimators=n_trees), suffix='{:d} trees'.format(n_trees))

for kernel in ['linear', 'rbf']:
    for c in [.1, 1, 10]:
        model.add_node(SVR(kernel, C=c), suffix='{} kernel, C={:.1f}'.format(kernel, c), scale_x=True)

# second level meta-estimators
model.add_layer(folds=3, pass_features=True)
model.add_node(RandomForestRegressor(500), name='RF Meta Estimator')
model.add_node(GradientBoostingRegressor(n_estimators=500), name='GBR Meta Estimator')

# final meta-estimator
model.add_layer()
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

preds = model.predict(X_tst)
```
<pre>
Level 1 Estimators (12 features)     Validation Score
-----------------------------------------------------
LinearRegression                      19.3009
Ridge                                 19.2869
Lasso                                 21.1333
ElasticNet                            21.3602
Lars                                  19.8137
BayesianRidge                         19.2145
RANSACRegressor                       30.0788
RandomForestRegressor 5 trees         18.5954
RandomForestRegressor 50 trees        15.4131
RandomForestRegressor 500 trees       15.5900
ExtraTreesRegressor 5 trees           14.8087
ExtraTreesRegressor 50 trees          14.0941
ExtraTreesRegressor 500 trees         14.4438
GradientBoostingRegressor 5 trees     32.4673
GradientBoostingRegressor 50 trees    18.0031
GradientBoostingRegressor 500 trees   18.6556
SVR linear kernel, C=0.1              19.7266
SVR linear kernel, C=1.0              19.5079
SVR linear kernel, C=10.0             19.7403
SVR rbf kernel, C=0.1                 37.2206
SVR rbf kernel, C=1.0                 20.9449
SVR rbf kernel, C=10.0                16.9735

Level 2 Estimators (34 features)     Validation Score
-----------------------------------------------------
RF Meta Estimator                     15.4941
GBR Meta Estimator                    17.5497

Level 3 Estimators (36 features)     Validation Score
-----------------------------------------------------
<b>Lin Reg Meta Estimator                11.2712</b>
</pre>

You can see that is very simple to churn out a massive amount of base estimators, and doing so will generally improve the ensembles performance.

## Todo
- Tests
- Include more metaclassifiers
- Prettier reports
- Tests
- Classification examples
- Tests
- Finish documentation, type hints
- Tests