from model.searcher import QuantileSearcher, MeanSearcher
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

GBM_PARAM_GRID = {'n_estimators': [50],
                  'num_leaves': [10, 50, 100],
                  'min_child_samples': [3, 9, 15],
                  'min_child_weight': [1e-2, 1e-1, 1],
                  'subsample': [0.4, 0.6, 0.8],
                  'colsample_bytree': [0.4, 0.6],
                  'reg_alpha': [1e-1, 1, 5],
                  'reg_lambda': [1e-1, 1, 5]}
GBM_NUM_GRIDS = 1
for param in GBM_PARAM_GRID.values():
    GBM_NUM_GRIDS *= len(param)


class QuantileLightGBM(QuantileSearcher):
    def __init__(self,
                 num_iters,
                 num_folds,
                 quantile=0.5,
                 num_jobs=-1,
                 rand_seed=111):
        self.base_model = LGBMRegressor
        if GBM_NUM_GRIDS > num_iters:
            self.searcher = RandomizedSearchCV(estimator=self.base_model(objective='quantile', metric='quantile',
                                                                         alpha=quantile, n_jobs=-1),
                                               param_distributions=GBM_PARAM_GRID,
                                               n_iter=num_iters,
                                               cv=num_folds,
                                               random_state=rand_seed,
                                               n_jobs=num_jobs)
        else:
            self.searcher = GridSearchCV(estimator=self.base_model(objective='quantile', metric='quantile',
                                                                   alpha=quantile, n_jobs=-1),
                                         param_grids=GBM_PARAM_GRID,
                                         cv=num_folds,
                                         n_jobs=num_jobs)
        self.quantile = quantile

    def predict(self, x_data, quantile=None):
        return self.searcher.predict(x_data).reshape(-1, 1)

    def get_init_model(self):
        return self.base_model(max_depth=-1, objective='quantile', metric='quantile',
                               alpha=self.quantile, n_jobs=-1, **self.searcher.best_params_)

