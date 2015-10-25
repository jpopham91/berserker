from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

__author__ = 'jake'


class RandomForest:
    """
    Iterative random forest with the ability to stop early.
    """
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

    # todo: make this a class
    def stepwise_regression(self):

        #self.fit_predict(X_tst)

        pred_list = self.val_preds[:]

        # blend
        selection = []
        best_indices = []
        score_list = []

        while score_list == sorted(score_list, reverse=True):
            best_score = 100
            best_pred = None
            best_index = 0
            for i, pred in enumerate(pred_list):
                candidate = list(selection)
                candidate.append(pred)
                this_avg = sum(candidate) / len(candidate)

                this_score = self.cost(self.y_val, this_avg)
                #print('Candidate score: {:.4f}'.format(this_score))

                if this_score < best_score:
                    best_score = this_score
                    best_pred = pred
                    best_index = i

            score_list.append(best_score)
            selection.append(best_pred)
            pred_list.pop(best_index)
            best_indices.append(best_index)

        selection.pop()
        blended_preds = sum(selection) / len(selection)

        # print("\nMean Estimator Scores:")
        # for score_list in self.scores:
        #     print('{:.4f}'.format(np.mean(score_list)))

        print('\033[1m' + 'Ensemble Score:' + '{:.4f}\033[0m'.format(self.cost(self.y_val, blended_preds)))

        return sum(list(np.array(self.tst_preds)[best_indices])) / len(list(np.array(self.tst_preds)[best_indices]))
