from model.searcher import QuantileSearcher, MeanSearcher
from model.forests import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

TREE_PARAM_GRID = {'n_estimators': [50],
                   'min_samples_split': [8, 16, 64],
                   'min_samples_leaf': [8, 16, 64]}

TREE_NUM_GRIDS = 1
for param in TREE_PARAM_GRID.values():
    TREE_NUM_GRIDS *= len(param)


class QuantileRandomForest(QuantileSearcher):
    def __init__(self,
                 num_iters,
                 num_folds,
                 quantile=0.5,
                 num_jobs=-1,
                 rand_seed=111):
        self.base_model = RandomForestQuantileRegressor
        if TREE_NUM_GRIDS > num_iters:
            self.searcher = RandomizedSearchCV(estimator=self.base_model(n_jobs=-1),
                                               param_distributions=TREE_PARAM_GRID,
                                               n_iter=num_iters,
                                               cv=num_folds,
                                               random_state=rand_seed,
                                               n_jobs=num_jobs)
        else:
            self.searcher = GridSearchCV(estimator=self.base_model(n_jobs=-1),
                                         param_grid=TREE_PARAM_GRID,
                                         cv=num_folds,
                                         n_jobs=num_jobs)
        self.quantile = quantile

    def full_predict(self, x_data, quantile_list):
        return self.searcher.best_estimator_.predict(x_data, quantile_list)

    def get_init_model(self):
        return self.base_model(**self.searcher.best_params_, n_jobs=-1)

