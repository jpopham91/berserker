__author__ = 'jake'

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.pipeline import make_union

# todo: maybe let stacking vs blending be determined at the layer level?
# then the model can just fit predict each layer and feed the results onward
# the blend only passes forth the validation holdout
# stacks pass everything
# most of the work being done in ensemble could be done here
# print a report of estimator performace AT EACH LEVEL
class Layer(object):

    def __init__(self):#, X, y, metric, val_split=.8):
        self.models = []

    # todo cast estimators to models
    def add(self, model):
        """appends a model to the current layer"""
        self.models.append(model)

    def fit_all(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def make_union(self):
        """creates a sklean.FeatureUnion object from the models in this layer"""
        if len(self.models) == 1:
            return self.models[0]
        return make_union(*self.models)


    # def fit_predict(self, X_tst, n_fold: int=3):
    #     xgb_eval_metric = 'auc'
    #     for model in self.models:
    #         print('\nTraining', str(model).split('(')[0])
    #         fold_scores = []
    #         for n, fold in enumerate(KFold(len(self.X_trn), n_fold)):
    #             if 'xgboost' in str(type(model)):
    #                 model.fit(self.X_trn[fold[0]], self.y_trn[fold[0]],
    #                           eval_metric=xgb_eval_metric, verbose=True)
    #             else:
    #                 model.fit(self.X_trn[fold[0]], self.y_trn[fold[0]])
    #
    #             # fold_trn_preds = model.predict_proba(self.X_trn[fold[1]])[:,1]
    #             # self.val_preds.append(model.predict_proba(self.X_val)[:,1])
    #             # self.tst_preds.append(model.predict_proba(X_tst)[:,1])
    #
    #             fold_trn_preds = model.predict(self.X_trn[fold[1]])
    #             fold_val_preds = model.predict(self.X_val)
    #             fold_tst_preds = model.predict(X_tst)
    #
    #             # print(fold_val_preds)
    #             # print(fold_tst_preds)
    #
    #             self.val_preds.append(fold_val_preds)
    #             self.tst_preds.append(fold_tst_preds)
    #
    #             fold_cv_score = self.cost(self.y_trn[fold[1]], fold_trn_preds)
    #             fold_val_score = self.cost(self.y_val, fold_val_preds)
    #
    #             print('Fold {}/{}: Validation Scores: {:.4f} {:.4f}'.format(n+1, n_fold, fold_cv_score, fold_val_score))
    #             #print('Fold {}/{}: Holdout Score {:.4f}'.format(n+1, n_fold, fold_val_score))
    #
    #             #print('Fold {}/{}: Validation Score {:.4f}'.format(n+1, n_fold, mean_squared_error(self.y_trn[fold[1]], fold_trn_preds)))
    #             #print('Fold {}/{}: Holdout Score {:.4f}'.format(n+1, n_fold, mean_squared_error(self.y_val, fold_val_preds)))
    #
    #             #print(np.array(self.y_val, dtype=int))
    #             #print(np.array(self.val_preds[-1], dtype=int))
    #             fold_scores.append(fold_cv_score)
    #         self.scores.append(fold_scores)
    #
    #     self.is_fit = True
    #
    # def blend(self, meta_est) -> np.array:
    #
    #     #self.fit_predict(X_tst)
    #
    #     # blend
    #     meta_est.fit(np.array(self.val_preds).T, self.y_val)
    #     #blended_preds = meta_est.predict_proba(np.array(self.val_preds).T)[:,1]
    #     blended_val_preds = meta_est.predict(np.array(self.val_preds).T)
    #     blended_tst_preds = meta_est.predict(np.array(self.tst_preds).T)
    #
    #     # print("\nMean Estimator Scores:")
    #     # for score_list in self.scores:
    #     #     print('{:.4f}'.format(np.mean(score_list)))
    #
    #     print('\033[1m' + 'Ensemble Score:' + '{:.4f}\033[0m'.format(self.cost(self.y_val, blended_val_preds)))
    #
    #     #print('\n' + '\033[1m' + 'Averaged Score:')
    #     #print('{:.4f}'.format(self.cost(self.y_val, np.array(self.val_preds).mean(axis=0))))
    #
    #
    #     # print(self.y_val)
    #     # print(blended_val_preds)
    #     # print(blended_tst_preds)
    #     #print(meta_est.coef_)
    #     #print(sum(meta_est.coef_))
    #     return blended_tst_preds



# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.info('this is an info message')
#
# print("load and shuffle training data")
# train = pd.read_csv('/home/jake/Downloads/train_clean.csv')
# np.random.seed(42)
# train = train.iloc[np.random.permutation(len(train))]
#
# y = train['target'].values
# train.drop(['ID', 'target'], axis=1, inplace=True)
# X = np.array(train.values, dtype=np.float32)
# del train
#
# ensemble = Ensemble(X, y, roc_auc_score)
#
# print("building ensemble")
# ensemble.add(Pipeline([('scl', StandardScaler()), ('clf', SVC(kernel='linear'))]))
# ensemble.add(xgb.XGBClassifier(n_estimators=750, max_depth=7, learning_rate=0.1,
#                                subsample=0.7, colsample_bytree=0.8))
# ensemble.add(RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='gini'))
# ensemble.add(RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy'))
# ensemble.add(ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini'))
# ensemble.add(ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='entropy'))
# ensemble.add(AdaBoostClassifier(n_estimators=250))
# ensemble.add(GradientBoostingClassifier(n_estimators=250, loss='deviance'))
#
# print("training models")
# #ensemble.fit()
# del X
#
# test = pd.read_csv('/home/jake/Downloads/test_clean.csv')
# ids = test.ID.values
# test.drop(['ID'], axis=1, inplace=True)
# X2 = np.array(test.values, dtype=np.float32)
#
# preds = ensemble.blend(X2)
#
# pd.DataFrame({'ID' : ids, 'target' : preds}).to_csv('/home/jake/mega/preds.csv', index=False)