import numpy as np


class QuantileSearcher:
    quantile = 0.5
    searcher = None
    base_model = None

    def fit(self, x_train, y_train):
        self.searcher.fit(x_train, y_train.reshape(-1))

    def predict(self, x_data, quantile=None):
        if quantile is None:
            quantile = self.quantile
        return self.searcher.best_estimator_.predict(x_data, int(quantile * 100)).reshape(-1, 1)

    def eval_loss(self, x_data, y_data, quantile=None):
        if quantile is None:
            quantile = self.quantile
        error_data = y_data - self.predict(x_data, quantile)
        loss_data = np.maximum(quantile * error_data, (quantile - 1) * error_data)
        return loss_data.mean()

    def get_init_model(self):
        return self.base_model(**self.searcher.best_params_)


class MeanSearcher:
    searcher = None
    base_model = None

    def fit(self, x_train, y_train):
        self.searcher.fit(x_train, y_train.reshape(-1))

    def predict(self, x_data):
        return self.searcher.best_estimator_.predict(x_data).reshape(-1, 1)

    def eval_loss(self, x_data, y_data):
        error_data = y_data - self.predict(x_data)
        loss_data = error_data * error_data
        loss_data = np.sqrt(loss_data)
        return loss_data.mean()

    def get_init_model(self):
        return self.base_model(**self.searcher.best_params_)

