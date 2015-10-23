from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

__author__ = 'jake'


class RandomForest:

    def __init__(self, **kwargs):
        self.clf = RandomForestClassifier(warm_start=True, **kwargs)

    def fit(self, X, y, validation_data):
        X_val, y_val = validation_data
        step_size = self.clf.n_estimators
        early_stopping_rounds = 5
        margin = .001

        score_hist = [-1]
        while max(score_hist) in score_hist[-early_stopping_rounds:]:
            self.clf.fit(X, y)
            round_score = roc_auc_score(y_val, self.clf.predict_proba(X_val)[:,1])
            score_hist.append(round_score)
            print('{:4d} {:.4f}'.format(self.clf.n_estimators, round_score))
            self.clf.n_estimators += step_size