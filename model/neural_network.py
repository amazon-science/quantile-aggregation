import copy
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from util.metric import pinball_loss, huber_loss, margin_loss, pinball_loss_np
from util.misc import fix_crossing, pava_forward, sort_forward
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterSampler
EVAL_STEPS = 10
STOP_STEPS = EVAL_STEPS * 50
QUANTILE_LOSS_PARAM_GRID = {'alpha': [0.0, 0.01],
                            'lr': [1e-3, 3e-4],
                            'wd': [1e-5, 1e-7],
                            'weight': [0.0, 1.0]}
QUANTILE_MARGIN_PARAM_GRID = {'margin': [0.0, 0.5, 1.0, 5.0],
                              'scale': [1e-3, 1e-4]}

QUANTILE_MARGIN_PARAM_GRID_FIX = {'margin': [0.0, 0.5, 1.0, 5.0],
                                  'margin_delta': [0.0001, 0.001, 0.005, 0.0075, 0.0099]}

MEAN_LOSS_PARAM_GRID = {'lr': [1e-3, 1e-4],
                        'wd': [1e-5, 1e-7]}
NETWORK_PARAM_GRID = {'hidden_size': [64, 128],
                      'num_layers': [2, 3],
                      'dropout': [0.0, 0.05, 0.1]}


class NeuralJointQuantileRegressor(nn.Module):
    def __init__(self,
                 quantile_list,
                 input_size,
                 hidden_size=64,
                 num_layers=3,
                 dropout=0.0,
                 activation='elu',
                 use_grad=True,
                 trans_type='mono',
                 use_margin=True,
                 margin_type=''):
        super(NeuralJointQuantileRegressor, self).__init__()
        # quantile list to handle
        self.num_quantiles = len(quantile_list)
        self.register_buffer('quantile_list', torch.Tensor(quantile_list).float())

        # activation
        act_fn = nn.ELU()
        if activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()

        # network with predicting quantiles
        layers = [nn.Linear(input_size, hidden_size), act_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_size, self.num_quantiles))
        self.network = nn.Sequential(*layers)

        # post-process
        self.use_grad = use_grad
        if trans_type is None:
            self.use_grad = False
        self.trans_type = trans_type
        self.use_margin = use_margin
        self.margin_type = margin_type

        #print('----------------')
        #print('NeuralJointQuantileRegressor')
        #print('use_grad: ', self.use_grad)
        #print('trans_type: ', self.trans_type)
        #print('use_margin: ', self.use_margin)
        #print('margin_type: ', self.margin_type)
        #print('----------------')

    def forward(self, input_data):
        # get output values
        output_data = self.network(input_data)
        return output_data

    def compute_loss(self,
                     input_data,
                     target_data,
                     aux_data=None,
                     margin=0.0,
                     scale=0.0,
                     alpha=0.0,
                     weight=0.0,
                     margin_delta=0.0):
        # train mode
        self.train()

        # margin loss
        if margin > 0.0 and self.use_margin:
            # include aux-data
            if aux_data is not None:
                batch_size = input_data.size()[0]
                full_input_data = torch.cat([input_data, aux_data], 0)
                predict_data = self(full_input_data)
                m_loss = margin_loss(predict_data, self.quantile_list * scale)
                predict_data = predict_data[:batch_size].contiguous()
            else:
                predict_data = self(input_data)
                if self.margin_type == 'single':
                    m_loss = margin_loss(predict_data, margin_delta)

                else:
                    m_loss = margin_loss(predict_data, self.quantile_list * scale)
        else:
            predict_data = self(input_data)
            m_loss = 0.0

        # fix crossing
        if self.use_grad:
            h_loss = weight * huber_loss(predict_data, target_data, self.quantile_list, alpha=alpha)
            if self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data)

            else:
                NotImplementedError()
            h_loss += (1 - weight) * huber_loss(predict_data, target_data, self.quantile_list, alpha=alpha)

        else:
            h_loss = huber_loss(predict_data, target_data, self.quantile_list, alpha=alpha)

        # combine
        return h_loss + margin * m_loss

    def eval_loss(self, input_data, target_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data)

            if self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data)#torch.sort(predict_data, -1)[0]

            return pinball_loss(predict_data, target_data, self.quantile_list).item()

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data)
            if self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data)#torch.sort(predict_data, -1)[0]

            return predict_data.data.cpu().numpy()

class NeuralSingleQuantileRegressor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=3,
                 dropout=0.0,
                 activation='elu'):
        super(NeuralSingleQuantileRegressor, self).__init__()
        # activation
        act_fn = nn.ELU()
        if activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()

        # network with predicting quantiles
        layers = [nn.Linear(input_size + 1, hidden_size), act_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, input_data, quantile_data):
        quantile_data = quantile_data.reshape(-1, 1)
        assert quantile_data.size()[0] == input_data.size()[0]

        # get output values
        output_data = self.network(torch.cat([input_data, quantile_data - 0.5], 1))
        return output_data

    def compute_loss(self, input_data, target_data, quantile_data, alpha=0.01):
        # train mode
        self.train()

        # prediction
        predict_data = self(input_data, quantile_data)

        # compute huber loss
        return huber_loss(predict_data, target_data, quantile_data, alpha)

    def eval_loss(self, input_data, target_data, quantile_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data, quantile_data)
            return pinball_loss(predict_data, target_data, quantile_data).item()

    def predict(self, input_data, quantile_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data, quantile_data)
            return predict_data.data.cpu().numpy()


class NeuralCondtionalGaussian(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=3,
                 dropout=0.0,
                 activation='elu'):

        super(NeuralCondtionalGaussian, self).__init__()
        # activation
        act_fn = nn.ELU()
        if activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()

        # network with predicting quantiles
        layers = [nn.Linear(input_size, hidden_size), act_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, input_data):
        output_data = self.network(input_data)
        mean_data = output_data[:, 0].reshape(-1, 1)
        var_data = F.softplus(output_data[:, 1].reshape(-1, 1)) + 1e-6
        return mean_data, var_data

    def compute_loss(self, input_data, target_data):
        # train mode
        self.train()

        # prediction
        mean_data, var_data = self(input_data)

        # compute negative log-likelihood
        nll_loss  = torch.pow(target_data - mean_data, 2).div(2 * var_data)
        nll_loss += var_data.log().div(2)
        return nll_loss.mean()

    def eval_loss(self, input_data, target_data, quantile_data=None):
        if quantile_data is None:
            self.eval()
            with torch.no_grad():
                mean_data, var_data = self(input_data)
                error_data = target_data - mean_data
                return torch.sqrt(torch.mean(error_data * error_data)).item()
        else:
            predict_data = self.predict(input_data, quantile_data)
            target_data = target_data.data.cpu().numpy()
            return pinball_loss_np(predict_data, target_data, quantile_data, True)

    def predict(self, input_data, quantile_data=None):
        self.eval()
        with torch.no_grad():
            mean_data, var_data = self(input_data)
            if quantile_data is None:
                return mean_data.data.cpu().numpy()
            else:
                std_data = torch.sqrt(var_data).data.cpu().numpy()
                ppf_data = norm.ppf(quantile_data).reshape(1, -1)
                predict_data = mean_data.data.cpu().numpy() + std_data * ppf_data
                return predict_data


class QuantileJointNeuralNetwork:
    def __init__(self,
                 quantile_list,
                 num_iters,
                 cv_split,
                 batch_size=64,
                 use_grad=True,
                 trans_type='mono',
                 use_margin=True,
                 rand_seed=1,
                 device=-1,
                 margin_type='',
                 **kwargs):

        self.num_iters = num_iters
        self.cv_split = cv_split
        self.quantile_list = quantile_list
        self.use_grad = use_grad
        self.trans_type = trans_type
        self.use_margin = use_margin
        self.rand_seed = rand_seed

        self.input_size = None
        self.batch_size = batch_size

        self.best_model = None
        self.best_params = None

        self.margin_type = margin_type

        print('----------------')
        print('QuantileJointNeuralNetwork')
        print('use_grad: ', self.use_grad)
        print('trans_type: ', self.trans_type)
        print('use_margin: ', self.use_margin)
        print('margin_type: ', self.margin_type)
        print('num_iters: ', self.num_iters)
        print('----------------')

        if torch.cuda.is_available() and device > -1:
            self.device = torch.device("cuda:{}".format(device))
        else:
            self.device = torch.device("cpu")

    def fit(self, x_train, y_train, ax_train=None):
        # train models
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        if ax_train is not None:
            ax_train = torch.FloatTensor(ax_train).to(self.device)

        # get input size
        self.input_size = x_train.size()[1]

        # build params list
        full_param_grid = {**NETWORK_PARAM_GRID,
                           **QUANTILE_LOSS_PARAM_GRID}

        if self.use_margin == True and self.margin_type == 'single':
            full_param_grid = {**full_param_grid,
                               **QUANTILE_MARGIN_PARAM_GRID_FIX}

        elif self.use_margin:
            full_param_grid = {**full_param_grid,
                               **QUANTILE_MARGIN_PARAM_GRID}

        if not self.use_grad or self.trans_type in ['pava', 'sort']:
            full_param_grid['weight'] = [0.0]
        
        params_list = list(ParameterSampler(param_distributions=full_param_grid,
                                            n_iter=self.num_iters,
                                            random_state=self.rand_seed))

        # set data loader
        train_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][0]],
                                                y_train[self.cv_split[0][0]]),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        valid_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][1]],
                                                y_train[self.cv_split[0][1]]),
                                  shuffle=False,
                                  batch_size=1024,
                                  drop_last=False)

        if ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(torch.cat([ax_train, x_train[self.cv_split[0][1]]], 0)),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))
        else:
            aux_loader = None

        # for each param
        best_eval_loss = np.inf
        best_eval_step = 0
        for p, params in enumerate(params_list):
            print('iter:',p, ' ', params)
            # fit model with given data and params
            eval_loss, eval_step = self.fit_model(train_loader, valid_loader, aux_loader, **params)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = eval_step
                self.best_params = params

            print('eval_loss : %.4f, best_eval_sofar: %.4f, eval_step: %d' %(eval_loss, best_eval_loss, eval_step))
            print()

        self.best_params['num_steps'] = best_eval_step

        # retrain model with train split with best hyper-params
        train_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][0]],
                                                y_train[self.cv_split[0][0]]),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        if ax_train is not None:
            aux_loader = DataLoader(TensorDataset(ax_train),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))
        else:
            aux_loader = None

        print('best_params:', self.best_params)
        self.best_model = self.fit_model(train_loader, None, aux_loader, **self.best_params)

    def fit_model(self,
                  train_loader,
                  valid_loader=None,
                  aux_loader=None,
                  hidden_size=64,
                  num_layers=3,
                  dropout=0.5,
                  activation='elu',
                  lr=1e-3,
                  wd=1e-5,
                  num_steps=None,
                  margin=0.0,
                  scale=0.0,
                  alpha=0.0,
                  weight=0.0,
                  margin_delta=0.0):

        # init model
        random.seed(self.rand_seed)
        if self.device == torch.device("cpu"):
            torch.manual_seed(self.rand_seed)

        else:
            torch.cuda.manual_seed_all(self.rand_seed)

        model = NeuralJointQuantileRegressor(quantile_list=self.quantile_list,
                                             input_size=self.input_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             dropout=dropout,
                                             activation=activation,
                                             use_grad=self.use_grad,
                                             trans_type=self.trans_type,
                                             use_margin=self.use_margin,
                                             margin_type=self.margin_type)
        model = model.to(self.device)

        # init optimizer
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr, weight_decay=wd, amsgrad=True)

        # init aux_loader
        if aux_loader is None:
            aux_loader_iterator = None
        else:
            aux_loader_iterator = iter(aux_loader)

        # for each update
        steps = 0
        best_valid_loss = np.inf
        best_step = 0
        while True:
            # for each batch (update)
            for x_batch, y_batch in train_loader:
                # aux data
                if aux_loader_iterator is None:
                    aux_batch = None
                else:
                    try:
                        aux_batch = next(aux_loader_iterator)[0]
                    except StopIteration:
                        aux_loader_iterator = iter(aux_loader)
                        aux_batch = next(aux_loader_iterator)[0]

                # compute loss
                weight = weight * (np.cos(min((steps / float(STOP_STEPS)), 1.0) * np.pi) + 1) * 0.5
                batch_loss = model.compute_loss(input_data=x_batch,
                                                target_data=y_batch,
                                                aux_data=aux_batch,
                                                margin=margin,
                                                scale=scale,
                                                alpha=alpha,
                                                weight=weight,
                                                margin_delta=margin_delta)

                # backprop and update
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # step up
                steps += 1

                # validate
                if steps % 100 == 0 and valid_loader is not None:
                    valid_loss = 0.0
                    valid_size = 0.0

                    for x_batch, y_batch in valid_loader:
                        batch_size = x_batch.size()[0]
                        batch_loss = model.eval_loss(input_data=x_batch, target_data=y_batch)
                        valid_loss += batch_loss * batch_size
                        valid_size += batch_size
                    valid_loss /= valid_size

                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_step = steps
                    elif steps - best_step >= STOP_STEPS:
                        return best_valid_loss, best_step
                elif num_steps is not None and steps >= num_steps:
                    assert valid_loader is None
                    return copy.deepcopy(model)

    def predict(self, x_data):
        x_data = torch.FloatTensor(x_data).to(self.device)
        y_pred = self.best_model.predict(x_data)
        return y_pred

    def refit_model(self, x_train, y_train, ax_train=None):
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        if ax_train is not None:
            ax_train = torch.FloatTensor(ax_train).to(self.device)

        train_loader = DataLoader(TensorDataset(x_train, y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)
        if ax_train is not None:
            aux_loader = DataLoader(TensorDataset(ax_train),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False)
        else:
            aux_loader = None
        self.best_model = self.fit_model(train_loader, None, aux_loader, **self.best_params)


class QuantileSingleNeuralNetwork:
    def __init__(self,
                 num_iters,
                 cv_split,
                 quantile_list,
                 batch_size=64,
                 rand_seed=111,
                 device=-1,
                 **kwargs):

        self.num_iters = num_iters
        self.cv_split = cv_split
        self.quantile_list = quantile_list
        self.rand_seed = rand_seed

        self.input_size = None
        self.batch_size = batch_size

        self.best_model = None
        self.best_params = None

        if torch.cuda.is_available() and device > -1:
            self.device = torch.device("cuda:{}".format(device))
        else:
            self.device = torch.device("cpu")

    def fit(self, x_train, y_train, ax_train=None):
        # train models
        x_train = torch.FloatTensor(x_train).float().to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).float().to(self.device)

        # get input size
        self.input_size = x_train.size()[1]

        # build params list
        full_param_grid = {**NETWORK_PARAM_GRID,
                           **QUANTILE_LOSS_PARAM_GRID}
        del full_param_grid['weight']
        params_list = list(ParameterSampler(param_distributions=full_param_grid,
                                            n_iter=self.num_iters,
                                            random_state=self.rand_seed))

        # set data loader
        train_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][0]],
                                                y_train[self.cv_split[0][0]]),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)

        valid_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][1]],
                                                y_train[self.cv_split[0][1]]),
                                  shuffle=False,
                                  batch_size=1024,
                                  drop_last=False)

        # for each param
        best_eval_loss = np.inf
        best_eval_step = 0
        for p, params in enumerate(params_list):
            # fit model with given data and params
            eval_loss, eval_step = self.fit_model(train_loader, valid_loader, **params)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = eval_step
                self.best_params = params
        self.best_params['num_steps'] = best_eval_step

        # retrain model train split not full data
        #train_loader = DataLoader(TensorDataset(x_train, y_train),
        #                          shuffle=True, batch_size=self.batch_size, drop_last=True)

        train_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][0]],
                                                y_train[self.cv_split[0][0]]),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)

        self.best_model = self.fit_model(train_loader, None, **self.best_params)

    def fit_model(self, train_loader, valid_loader=None,
                  hidden_size=64, num_layers=3, dropout=0.5, activation='elu',
                  lr=1e-3, wd=1e-5, num_steps=None, alpha=0.01):
        # init model
        random.seed(self.rand_seed)
        if self.device == torch.device("cpu"):
            torch.manual_seed(self.rand_seed)
        else:
            torch.cuda.manual_seed_all(self.rand_seed)

        model = NeuralSingleQuantileRegressor(input_size=self.input_size,
                                              hidden_size=hidden_size,
                                              num_layers=num_layers,
                                              dropout=dropout,
                                              activation=activation)
        model = model.to(self.device)

        # init optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

        # for each update
        steps = 0
        best_valid_loss = np.inf
        best_step = 0
        while True:
            # for each batch (update)
            for x_batch, y_batch in train_loader:
                # sample quantile
                batch_size = x_batch.size()[0]
                q_batch = torch.rand(batch_size, 1).to(self.device)
                q_batch = torch.clamp(q_batch, 0.001, 0.999)

                # compute loss
                batch_loss = model.compute_loss(input_data=x_batch, target_data=y_batch,
                                                quantile_data=q_batch, alpha=alpha)

                # backprop and update
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # step up
                steps += 1

                # validate
                if steps % 100 == 0 and valid_loader is not None:
                    valid_loss = 0.0
                    valid_size = 0.0

                    for x_batch, y_batch in valid_loader:
                        batch_size = x_batch.size()[0]
                        for q in self.quantile_list:
                            q_batch = q * torch.ones(batch_size, 1).to(self.device)
                            batch_loss = model.eval_loss(input_data=x_batch, target_data=y_batch, quantile_data=q_batch)
                            valid_loss += batch_loss * batch_size
                            valid_size += batch_size
                    valid_loss /= valid_size

                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_step = steps
                    elif steps - best_step >= STOP_STEPS:
                        return best_valid_loss, best_step
                elif num_steps is not None and steps >= num_steps:
                    assert valid_loader is None
                    return copy.deepcopy(model)

    def predict(self, x_data):
        x_data = torch.FloatTensor(x_data).to(self.device)
        batch_size = x_data.size()[0]
        y_pred_list = []
        for q in self.quantile_list:
            q_data = q * torch.ones(batch_size, 1).to(self.device)
            y_pred = self.best_model.predict(x_data, q_data)
            y_pred_list.append(y_pred)
        return np.concatenate(y_pred_list, 1)

    def refit_model(self, x_train, y_train, ax_train=None):
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        train_loader = DataLoader(TensorDataset(x_train, y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)

        self.best_model = self.fit_model(train_loader, None, **self.best_params)


class QuantileConditionalGaussianNetwork:
    def __init__(self,
                 num_iters,
                 cv_split,
                 quantile_list,
                 batch_size=64,
                 rand_seed=111,
                 device=-1,
                 **kwargs):
        self.num_iters = num_iters
        self.cv_split = cv_split
        self.quantile_list = quantile_list
        self.rand_seed = rand_seed

        self.input_size = None
        self.batch_size = batch_size

        self.best_model = None
        self.best_params = None

        if torch.cuda.is_available() and device > -1:
            self.device = torch.device("cuda:{}".format(device))
        else:
            self.device = torch.device("cpu")

    def fit(self, x_train, y_train, ax_train=None):
        # train models
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        # get input size
        self.input_size = x_train.size()[1]

        # build params list
        full_param_grid = NETWORK_PARAM_GRID
        params_list = list(ParameterSampler(param_distributions=full_param_grid,
                                            n_iter=self.num_iters,
                                            random_state=self.rand_seed))

        # set data loader
        train_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][0]],
                                                y_train[self.cv_split[0][0]]),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)

        valid_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][1]],
                                                y_train[self.cv_split[0][1]]),
                                  shuffle=False,
                                  batch_size=1024,
                                  drop_last=False)

        # for each param
        best_eval_loss = np.inf
        best_eval_step = 0
        for p, params in enumerate(params_list):
            # fit model with given data and params
            eval_loss, eval_step = self.fit_model(train_loader, valid_loader, **params)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = eval_step
                self.best_params = params
        self.best_params['num_steps'] = best_eval_step

        # retrain model with only train split not full data
        # train_loader = DataLoader(TensorDataset(x_train, y_train),
        #                          shuffle=True, batch_size=self.batch_size, drop_last=True)
        train_loader = DataLoader(TensorDataset(x_train[self.cv_split[0][0]],
                                                y_train[self.cv_split[0][0]]),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)

        print('best_params:', self.best_params)
        self.best_model = self.fit_model(train_loader, None, **self.best_params)

    def fit_model(self,
                  train_loader,
                  valid_loader=None,
                  hidden_size=64,
                  num_layers=3,
                  dropout=0.5,
                  activation='elu',
                  lr=1e-3,
                  wd=1e-5,
                  num_steps=None):

        # init model
        random.seed(self.rand_seed)
        if self.device == torch.device("cpu"):
            torch.manual_seed(self.rand_seed)
        else:
            torch.cuda.manual_seed_all(self.rand_seed)

        model = NeuralCondtionalGaussian(input_size=self.input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         dropout=dropout,
                                         activation=activation)
        model = model.to(self.device)

        # init optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

        # for each update
        steps = 0
        best_valid_loss = np.inf
        best_step = 0
        while True:
            # for each batch (update)
            for x_batch, y_batch in train_loader:
                # compute loss
                batch_loss = model.compute_loss(input_data=x_batch, target_data=y_batch)

                # backprop and update
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # step up
                steps += 1

                # validate
                if steps % 100 == 0 and valid_loader is not None:
                    valid_loss = 0.0
                    valid_size = 0.0
                    for x_batch, y_batch in valid_loader:
                        batch_size = x_batch.size()[0]
                        batch_loss = model.eval_loss(input_data=x_batch, target_data=y_batch,
                                                     quantile_data=self.quantile_list)
                        valid_loss += batch_loss * batch_size
                        valid_size += batch_size
                    valid_loss /= valid_size

                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_step = steps
                    elif steps - best_step >= STOP_STEPS:
                        return best_valid_loss, best_step
                elif num_steps is not None and steps >= num_steps:
                    assert valid_loader is None
                    return copy.deepcopy(model)

    def predict(self, x_data):
        x_data = torch.FloatTensor(x_data).to(self.device)
        return self.best_model.predict(x_data, self.quantile_list)

    def refit_model(self, x_train, y_train, ax_train=None):
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        train_loader = DataLoader(TensorDataset(x_train, y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False)

        self.best_model = self.fit_model(train_loader, None, **self.best_params)

