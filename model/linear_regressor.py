import statsmodels.api as sm
from model.searcher import QuantileSearcher
from sklearn.base import BaseEstimator, RegressorMixin


class SMQRmodel(BaseEstimator, RegressorMixin):
    def __init__(self,
                 quantile,
                 fit_intercept=True):
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = sm.QuantReg(y, X)
        self.results_ = self.model_.fit(q=self.quantile, max_iter=10000000)

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')
        return self.results_.predict(X)


class QuantileRegressor(QuantileSearcher):
    def __init__(self,
                 quantile=0.5,
                 **kwargs):
        self.searcher = SMQRmodel(quantile=quantile)
        self.quantile = quantile

    def fit(self, x_train, y_train):
        self.searcher.fit(x_train, y_train.reshape(-1))

    def predict(self, x_data, quantile=None):
        return self.searcher.predict(x_data).reshape(-1, 1)

    def get_init_model(self):
        return SMQRmodel(quantile=self.quantile)

