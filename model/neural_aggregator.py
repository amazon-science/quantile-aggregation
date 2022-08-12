import copy
import torch
import numpy as np
import torch.nn as nn
from util.metric import pinball_loss, huber_loss, margin_loss
from util.misc import fix_crossing, pava_forward, sort_forward
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterSampler
EVAL_STEPS = 10
STOP_STEPS = EVAL_STEPS * 50
QUANTILE_LOSS_PARAM_GRID = {'alpha': [0.0],
                            'lr': [1e-3, 5e-4],
                            'wd': [1e-7],
                            'margin_weight': [0.5, 1.0, 2.0, 5.0, 10.0],
                            'margin_scale': [1e-1, 5e-2, 1e-2, 1e-3, 1e-4]}

NETWORK_PARAM_GRID = {'hidden_size': [64, 128],
                      'num_layers': [2, 3],
                      'dropout': [0.0, 0.05, 0.1]
                      }

QUANTILE_LOSS_PARAM_GRID_FIX = {'alpha': [0.0],
                                'lr': [1e-3],
                                'wd': [1e-7],
                                'margin_weight': [0.5, 1.0, 2.0, 5.0, 10.0],
                                'margin_delta': [0.0001, 0.001, 0.005, 0.0075, 0.0099]
                                }

class QuantileGlobalAggregator(nn.Module):
    def __init__(self,
                 num_models,  # number of base models
                 quantile_list,  # list of quantile levels
                 normalize=True,  # normalize weights
                 margin_list=None,  # using margin
                 trans_type=None,  # apply non-crossing
                 use_grad=True,  # using non-crossing training
                 share=False,  # share between base models
                 cross=False,  # cross between quantile levels
                 margin_type='',
                 regularization_strength=1.0,
                 ):
        super(QuantileGlobalAggregator, self).__init__()
        # model size
        self.num_models = num_models

        # quantile list to handle
        self.num_quantiles = len(quantile_list)
        self.register_buffer('quantiles', torch.FloatTensor(quantile_list))

        # normalize weights
        self.normalize = normalize

        # post-process (monotnoizer)
        self.trans_type = trans_type
        self.use_grad = use_grad
        self.margin_list = margin_list
        self.margin_type = margin_type
        self.regularization_strength = regularization_strength

        # set weight
        self.share = share
        self.cross = cross
        self.model_type = None
        if self.share:
            self.weights = nn.Parameter(torch.zeros([1, self.num_models, 1]))
            self.model_type = 'Coarse'
        else:
            if self.cross:
                self.weights = nn.Parameter(torch.zeros([1, self.num_models * self.num_quantiles, self.num_quantiles]))
                self.model_type = 'Fine'
            else:
                self.weights = nn.Parameter(torch.zeros([1, self.num_models, self.num_quantiles]))
                self.model_type = 'Medium'

    # aggregate estimates
    def forward(self, input_data):
        # get convex weight (normalize over weights)
        if self.normalize:
            convex_weights = self.weights.softmax(1)
        else:
            convex_weights = self.weights

        # weight sum
        if self.share:
            output_data = input_data * convex_weights
        else:
            if self.cross:
                output_data = input_data.reshape(-1, self.num_models * self.num_quantiles, 1) * convex_weights
            else:
                output_data = input_data * convex_weights

        # aggregate
        output_data = torch.sum(output_data, 1)
        return output_data

    def compute_loss(self,
                     input_data,
                     target_data,
                     aux_data=None,
                     margin_weight=0.0,
                     margin_scale=0.0,
                     alpha=0.0,
                     margin_delta=0):
        # train mode
        self.train()
        # get prediction and margin loss
        if margin_weight > 0.0 and (self.margin_list is not None or self.margin_type == 'single'):
            if aux_data is not None:
                batch_size = input_data.size()[0]
                full_input_data = torch.cat([input_data, aux_data], 0)
                predict_data = self(full_input_data)
                m_loss = margin_weight * margin_loss(predict_data, self.margin_list * margin_scale)
                predict_data = predict_data[:batch_size].contiguous()

            else:
                predict_data = self(input_data)
                if self.margin_type == 'single':
                   m_loss = margin_weight * margin_loss(predict_data, margin_delta)

                else:
                    m_loss = margin_weight * margin_loss(predict_data, self.margin_list * margin_scale)
        else:
            predict_data = self(input_data)
            m_loss = 0

        # back-prop through non-crossing
        if self.use_grad:
            if self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data, self.regularization_strength)

        # pinball loss
        h_loss = huber_loss(predict_data, target_data, self.quantiles, alpha)
        return h_loss + m_loss

    def eval_loss(self, input_data, target_data):
        # evaluation mode
        self.eval()

        with torch.no_grad():
            # get aggregated prediction
            predict_data = self(input_data)

            # monotonize
            if self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data, self.regularization_strength)

            # compute pinball loss
            return pinball_loss(predict_data, target_data, self.quantiles).item()

    def predict(self, input_data):
        # evaluation mode
        self.eval()

        with torch.no_grad():
            # get aggregated prediction
            predict_data = self(input_data)

            # monotonize
            if self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data, self.regularization_strength)

            return predict_data.data.cpu().numpy()


class QuantileLocalAggregator(nn.Module):
    def __init__(self,
                 num_models,  # number of base models
                 quantile_list,  # list of quantile levels
                 input_size,  # input feature data size
                 hidden_size=64,  # hidden size
                 num_layers=3,  # number of layers
                 dropout=0.0,  # drop out ratio
                 activation='elu',  # activation
                 normalize=True,  # normalize weights
                 margin_list=None,  # using margin
                 trans_type=None,  # apply non-crossing
                 use_grad=True,  # using non-crossing training
                 share=False,  # share between base models
                 cross=False,  # cross between quantile levels
                 margin_type='',
                 regularization_strength=1,
                 ):

        super(QuantileLocalAggregator, self).__init__()
        # model size
        self.num_models = num_models

        # quantile list to handle
        self.num_quantiles = len(quantile_list)
        self.register_buffer('quantiles', torch.FloatTensor(quantile_list))

        # normalize weights
        self.normalize = normalize

        # post-process (monotnoizer)
        self.trans_type = trans_type
        self.use_grad = use_grad
        self.margin_list = margin_list
        self.margin_type = margin_type
        self.regularization_strength = regularization_strength

        # set output size
        self.share = share
        self.cross = cross
        self.model_type = None
        if self.share:
            num_outputs = self.num_models
            self.model_type = 'Coarse'
        else:
            if self.cross:
                num_outputs = self.num_models * self.num_quantiles * self.num_quantiles
                self.model_type = 'Fine'
            else:
                num_outputs = self.num_models * self.num_quantiles
                self.model_type = 'Medium'

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

        layers.append(nn.Linear(hidden_size, num_outputs))
        self.network = nn.Sequential(*layers)
        '''
        if num_layers == 2:
            self.network = nn.Sequential(
                                nn.Linear(input_size, hidden_size),
                                act_fn,
                                nn.BatchNorm1d(hidden_size, affine=False),
                                nn.Linear(hidden_size, hidden_size),
                                act_fn,
                                nn.BatchNorm1d(hidden_size, affine=False),
                                nn.Linear(hidden_size, num_outputs),
                                )

        elif num_layers == 3:
            self.network = nn.Sequential(
                                nn.Linear(input_size, hidden_size),
                                act_fn,
                                nn.BatchNorm1d(hidden_size, affine=False),
                                nn.Linear(hidden_size, hidden_size),
                                act_fn,
                                nn.BatchNorm1d(hidden_size, affine=False),
                                nn.Linear(hidden_size, hidden_size),
                                act_fn,
                                nn.BatchNorm1d(hidden_size, affine=False),
                                nn.Linear(hidden_size, num_outputs),
                                )
        else:
            self.network = nn.Sequential(
                            nn.Linear(input_size, hidden_size),
                            act_fn,
                            nn.BatchNorm1d(hidden_size, affine=False),
                            nn.Linear(hidden_size, hidden_size),
                            act_fn,
                            nn.BatchNorm1d(hidden_size, affine=False),
                            nn.Linear(hidden_size, hidden_size),
                            act_fn,
                            nn.BatchNorm1d(hidden_size, affine=False),
                            nn.Linear(hidden_size, hidden_size),
                            act_fn,
                            nn.BatchNorm1d(hidden_size, affine=False),
                            nn.Linear(hidden_size, num_outputs),
                            )
        '''
        #print(self.network)
    # aggregate estimates
    def forward(self, cond_data, input_data):
        # combination weight
        convex_weights = self.network(cond_data)

        # reshape weights
        if self.share:
            convex_weights = convex_weights.reshape(-1, self.num_models, 1)
            input_data = input_data.reshape(-1, self.num_models, self.num_quantiles)
        else:
            if self.cross:
                convex_weights = convex_weights.reshape(-1, self.num_models * self.num_quantiles, self.num_quantiles)
                input_data = input_data.reshape(-1, self.num_models * self.num_quantiles, 1)
            else:
                convex_weights = convex_weights.reshape(-1, self.num_models, self.num_quantiles)
                input_data = input_data.reshape(-1, self.num_models, self.num_quantiles)

        # normalize (sum to 1)
        if self.normalize:
            convex_weights = convex_weights.softmax(1)

        # aggregate
        output_data = torch.sum(input_data * convex_weights, 1)
        return output_data

    def compute_loss(self,
                     cond_data,
                     input_data,
                     target_data,
                     aux_cond_data=None,
                     aux_input_data=None,
                     margin_weight=0.0,
                     margin_scale=0.0,
                     alpha=0.0,
                     margin_delta=0.0):

        # train mode
        self.train()

        if margin_weight > 0.0 and (self.margin_list is not None or self.margin_type == 'single'):
            if aux_cond_data is not None and aux_input_data is not None:
                batch_size = input_data.size()[0]
                predict_data = self(torch.cat([cond_data, aux_cond_data], 0),
                                    torch.cat([input_data, aux_input_data], 0))
                m_loss = margin_weight * margin_loss(predict_data, self.margin_list * margin_scale)
                predict_data = predict_data[:batch_size].contiguous()

            else:
                predict_data = self(cond_data, input_data)
                if self.margin_type == 'single':
                    m_loss = margin_weight * margin_loss(predict_data, margin_delta)

                else:
                    m_loss = margin_weight * margin_loss(predict_data, self.margin_list * margin_scale)

        else:
            predict_data = self(cond_data, input_data)
            m_loss = 0

        # back-prop through non-crossing
        if self.use_grad:
            if self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data, self.regularization_strength)

        # pinball loss
        h_loss = huber_loss(predict_data, target_data, self.quantiles, alpha)
        return h_loss + m_loss

    def eval_loss(self, cond_data, input_data, target_data):
        # evaluation mode
        self.eval()

        with torch.no_grad():
            # get aggregated prediction
            predict_data = self(cond_data, input_data)

            # monotonize
            if self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data, self.regularization_strength)

            # compute pinball loss
            return pinball_loss(predict_data, target_data, self.quantiles).item()

    def predict(self, cond_data, input_data):
        # evaluation mode
        self.eval()

        with torch.no_grad():
            # get aggregated prediction
            predict_data = self(cond_data, input_data)

            # monotonize
            if self.trans_type == 'pava':
                predict_data = pava_forward(predict_data)

            elif self.trans_type == 'mono':
                predict_data = fix_crossing(predict_data)

            elif self.trans_type == 'sort':
                predict_data = sort_forward(predict_data, self.regularization_strength)

            return predict_data.data.cpu().numpy()

class QuantileGlobalAggregatorTrainer:
    def __init__(self,
                 num_searches,  # number of searching
                 cv_split,  # cross-validation splitting
                 quantile_list,  # list of quantile levels
                 batch_size=64,  # mini batch size
                 normalize=True,  # normalize weights
                 margin_list=None,  # using margin
                 trans_type=None,  # apply non-crossing
                 use_grad=True,  # using non-crossing training
                 share_weight=False,  # share weight over models
                 cross_weight=False,  # cross quantiles
                 rand_seed=111,  # random seed
                 device=-1,  # device id,
                 margin_type='',
                 regularization_strength=1,
                 ):
        # training setting
        self.num_searches = num_searches
        self.cv_split = cv_split

        # model setting
        self.quantile_list = quantile_list
        self.normalize = normalize
        self.margin_list = margin_list
        self.trans_type = trans_type
        self.use_grad = use_grad
        self.share_weight = share_weight
        self.cross_weight = cross_weight
        self.rand_seed = rand_seed
        self.num_models = None
        self.num_quantiles = len(quantile_list)
        self.batch_size = batch_size
        self.margin_type = margin_type
        self.regularization_strength = regularization_strength

        # best model after training
        self.best_model = None
        self.best_params = None

        # set device
        if torch.cuda.is_available() and device > -1:
            self.device = torch.device("cuda:{}".format(device))
        else:
            self.device = torch.device("cpu")

    # fit model by model selection
    def fit(self,
            x_train,
            y_train,
            x_val,
            y_val,
            ax_train=None):
        # convert data
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        x_val = torch.FloatTensor(x_val).to(self.device)
        y_val = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)

        if ax_train is not None:
            ax_train = torch.FloatTensor(ax_train).to(self.device)

        # get number of base models
        self.num_models = x_train.size()[1]
        assert self.num_quantiles == x_train.size()[2]

        # build params list
        if self.margin_type == 'single':
            full_param_grid = {**QUANTILE_LOSS_PARAM_GRID_FIX}

        else:
            full_param_grid = {**QUANTILE_LOSS_PARAM_GRID}
            if self.margin_list is None:
                del full_param_grid['margin_weight']
                del full_param_grid['margin_scale']


        params_list = list(ParameterSampler(param_distributions=full_param_grid,
                                            n_iter=self.num_searches,
                                            random_state=self.rand_seed))

        # set data loader
        train_loader = DataLoader(dataset=TensorDataset(x_train,
                                                        y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        valid_loader = DataLoader(dataset=TensorDataset(x_val,
                                                        y_val),
                                  shuffle=False,
                                  batch_size=1024,
                                  drop_last=False)

        if ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(torch.cat([ax_train, x_train], 0)),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))

        else:
            aux_loader = None

        # starting model selection
        best_eval_loss = np.inf
        best_eval_step = 0

        # for each param
        for p, params in enumerate(params_list):
            # fit model with given data and params
            print('iter:',p, ' ', params)
            eval_loss, eval_step = self.fit_model(train_loader=train_loader,
                                                  valid_loader=valid_loader,
                                                  aux_loader=aux_loader,
                                                  **params)

            # if best in terms of validation
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = eval_step
                self.best_params = params

            print('eval_loss : %.4f, best_eval_sofar: %.4f, eval_step: %d' %(eval_loss, best_eval_loss, eval_step))
            print()
        self.best_params['num_steps'] = best_eval_step

        # retrain model with full data
        train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        if ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(ax_train),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))

        else:
            aux_loader = None

        # retrain model and set as best model
        print('best_params:', self.best_params)
        print('best_eval_loss: %.4f' % best_eval_loss)
        self.best_model = self.fit_model(train_loader, None, aux_loader, **self.best_params)

    # fit single model based on given params
    def fit_model(self,
                  train_loader,
                  valid_loader=None,
                  aux_loader=None,
                  num_steps=None,
                  lr=1e-3,
                  wd=1e-7,
                  margin_weight=0.0,
                  margin_scale=0.0,
                  alpha=0.0,
                  margin_delta=0.0):
        # init model
        model = QuantileGlobalAggregator(num_models=self.num_models,
                                         quantile_list=self.quantile_list,
                                         normalize=self.normalize,
                                         margin_list=self.margin_list,
                                         trans_type=self.trans_type,
                                         use_grad=self.use_grad,
                                         share=self.share_weight,
                                         cross=self.cross_weight,
                                         margin_type=self.margin_type,
                                         regularization_strength=self.regularization_strength)
        model = model.to(self.device)

        # init optimizer
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr, weight_decay=wd, amsgrad=True)

        # init results
        steps = 0
        best_valid_loss = np.inf
        best_step = 0

        # init aux_loader
        if aux_loader is None:
            aux_loader_iterator = None
        else:
            aux_loader_iterator = iter(aux_loader)

        # for each epoch
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
                batch_loss = model.compute_loss(input_data=x_batch,
                                                target_data=y_batch,
                                                aux_data=aux_batch,
                                                margin_weight=margin_weight,
                                                margin_scale=margin_scale,
                                                alpha=alpha,
                                                margin_delta=margin_delta)

                # backprop and update
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # step up
                steps += 1

                # evaluation over validation set
                if steps % 100 == 0 and valid_loader is not None:
                    valid_loss = 0.0
                    valid_size = 0.0

                    # compute validation loss
                    for x_batch, y_batch in valid_loader:
                        batch_size = x_batch.size()[0]
                        batch_loss = model.eval_loss(input_data=x_batch, target_data=y_batch)
                        valid_loss += batch_loss * batch_size
                        valid_size += batch_size
                    valid_loss /= valid_size

                    # update best validation loss
                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_step = steps
                    # if no improvement seen
                    elif steps - best_step >= STOP_STEPS:
                        return best_valid_loss, best_step
                # if number of steps is reached
                elif num_steps is not None and steps >= num_steps:
                    assert valid_loader is None
                    return copy.deepcopy(model)

    # prediction
    def predict(self, x_data):
        x_data = torch.FloatTensor(x_data).to(self.device)
        y_pred = self.best_model.predict(x_data)
        return y_pred

    def refit_model(self,
                    x_train, y_train,
                    ax_train=None):
        self.best_model = None

        # convert data
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        if ax_train is not None:
            ax_train = torch.FloatTensor(ax_train).to(self.device)

        # retrain model with full data
        train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        if ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(ax_train),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))

        else:
            aux_loader = None

        print('best_params:', self.best_params)
        # retrain model and set as best model
        self.best_model = self.fit_model(train_loader, None, aux_loader, **self.best_params)


class QuantileLocalAggregatorTrainer:
    def __init__(self,
                 num_searches,  # number of searching
                 cv_split,  # cross-validation splitting
                 quantile_list,  # list of quantile levels
                 batch_size=64,  # mini batch size
                 normalize=True,  # normalize weights
                 margin_list=None,  # using margin
                 trans_type=None,  # apply non-crossing
                 use_grad=True,  # using non-crossing training
                 share_weight=False,  # share weight over models
                 cross_weight=False,  # cross quantiles
                 rand_seed=111,  # random seed
                 device=-1,  # device id,
                 margin_type='',
                 regularization_strength=1.0,
                 ):
        # training setting
        self.num_searches = num_searches
        self.cv_split = cv_split

        # model setting
        self.quantile_list = quantile_list
        self.normalize = normalize
        self.margin_list = margin_list
        self.trans_type = trans_type
        self.use_grad = use_grad
        self.share_weight = share_weight
        self.cross_weight = cross_weight
        self.rand_seed = rand_seed
        self.input_size = None
        self.num_models = None
        self.num_quantiles = len(quantile_list)
        self.batch_size = batch_size
        self.margin_type = margin_type
        self.regularization_strength = regularization_strength

        # best model after training
        self.best_model = None
        self.best_params = None

        # set device
        if torch.cuda.is_available() and device > -1:
            self.device = torch.device("cuda:{}".format(device))
        else:
            self.device = torch.device("cpu")

    # fit model by model selection
    def fit(self,
            c_train,
            x_train,
            y_train,
            c_val,
            x_val,
            y_val,
            ac_train=None,
            ax_train=None):

        # convert data
        c_train = torch.FloatTensor(c_train).to(self.device)
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        c_val = torch.FloatTensor(c_val).to(self.device)
        x_val = torch.FloatTensor(x_val).to(self.device)
        y_val = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)

        if ac_train is not None and ax_train is not None:
            ac_train = torch.FloatTensor(ac_train).to(self.device)
            ax_train = torch.FloatTensor(ax_train).to(self.device)

        # get input size and number of base models
        self.input_size = c_train.size()[1]
        self.num_models = x_train.size()[1]
        assert self.num_quantiles == x_train.size()[2]

        # build params list
        if self.margin_type == 'single':
            full_param_grid = {**NETWORK_PARAM_GRID,
                               **QUANTILE_LOSS_PARAM_GRID_FIX}

        else:
            full_param_grid = {**NETWORK_PARAM_GRID,
                               **QUANTILE_LOSS_PARAM_GRID}

            if self.margin_list is None:
                del full_param_grid['margin_weight']
                del full_param_grid['margin_scale']

        params_list = list(ParameterSampler(param_distributions=full_param_grid,
                                            n_iter=self.num_searches,
                                            random_state=self.rand_seed))

        # set data loader
        train_loader = DataLoader(dataset=TensorDataset(c_train,
                                                        x_train,
                                                        y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        valid_loader = DataLoader(dataset=TensorDataset(c_val,
                                                        x_val,
                                                        y_val),
                                  shuffle=False,
                                  batch_size=1024,
                                  drop_last=False)

        if ac_train is not None and ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(torch.cat([ac_train, c_val], 0),
                                                          torch.cat([ax_train, x_val], 0)),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))

        else:
            aux_loader = None


        # starting model selection
        best_eval_loss = np.inf
        best_eval_step = 0

        # for each param
        for p, params in enumerate(params_list):
            # fit model with given data and params
            print('iter:',p, ' ', params)
            eval_loss, eval_step = self.fit_model(train_loader=train_loader,
                                                  valid_loader=valid_loader,
                                                  aux_loader=aux_loader,
                                                  **params)

            # if best in terms of validation
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = eval_step
                self.best_params = params

            print('eval_loss : %.4f, best_eval_sofar: %.4f, eval_step: %d' %(eval_loss, best_eval_loss, eval_step))
            print()
        self.best_params['num_steps'] = best_eval_step

        # retrain model with full data
        #train_loader = DataLoader(dataset=TensorDataset(c_train, x_train, y_train),
        #                          shuffle=True, batch_size=self.batch_size, drop_last=True,
        #                          worker_init_fn=np.random.seed(self.rand_seed))
        train_loader = DataLoader(dataset=TensorDataset(c_train,
                                                        x_train,
                                                        y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))


        if ac_train is not None and ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(ac_train, ax_train),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))

        else:
            aux_loader = None

        print('best_params:', self.best_params)
        print('best_eval_loss: %.4f' % best_eval_loss)
        # retrain model and set as best model
        self.best_model = self.fit_model(train_loader, None, aux_loader, **self.best_params)

    # fit single model based on given params
    def fit_model(self,
                  train_loader,
                  valid_loader=None,
                  aux_loader=None,
                  num_steps=None,
                  hidden_size=64,
                  num_layers=3,
                  dropout=0.1,
                  activation='elu',
                  lr=1e-3,
                  wd=1e-7,
                  margin_weight=0.0,
                  margin_scale=0.0,
                  alpha=0.0,
                  margin_delta=0.0):

        # init model
        model = QuantileLocalAggregator(num_models=self.num_models,
                                        quantile_list=self.quantile_list,
                                        input_size=self.input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        dropout=dropout,
                                        activation=activation,
                                        normalize=self.normalize,
                                        margin_list=self.margin_list,
                                        trans_type=self.trans_type,
                                        use_grad=self.use_grad,
                                        share=self.share_weight,
                                        cross=self.cross_weight,
                                        margin_type=self.margin_type,
                                        regularization_strength=self.regularization_strength)

        model = model.to(self.device)

        # init optimizer
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr, weight_decay=wd, amsgrad=True)

        # init results
        steps = 0
        best_valid_loss = np.inf
        best_step = 0

        # init aux_loader
        if aux_loader is None:
            aux_loader_iterator = None
        else:
            aux_loader_iterator = iter(aux_loader)

        # for each epoch
        while True:
            # for each batch (update)
            for c_batch, x_batch, y_batch in train_loader:
                # aux data
                if aux_loader_iterator is None:
                    ac_batch, ax_batch = None, None
                else:
                    try:
                        ac_batch, ax_batch = next(aux_loader_iterator)
                    except StopIteration:
                        aux_loader_iterator = iter(aux_loader)
                        ac_batch, ax_batch = next(aux_loader_iterator)

                # compute loss
                batch_loss = model.compute_loss(cond_data=c_batch,
                                                input_data=x_batch,
                                                target_data=y_batch,
                                                aux_cond_data=ac_batch,
                                                aux_input_data=ax_batch,
                                                margin_weight=margin_weight,
                                                margin_scale=margin_scale,
                                                alpha=alpha,
                                                margin_delta=margin_delta)

                # backprop and update
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # step up
                steps += 1

                # evaluation over validation set
                if steps % 100 == 0 and valid_loader is not None:
                    valid_loss = 0.0
                    valid_size = 0.0

                    # compute validation loss
                    for c_batch, x_batch, y_batch in valid_loader:
                        batch_size = x_batch.size()[0]
                        batch_loss = model.eval_loss(cond_data=c_batch, input_data=x_batch, target_data=y_batch)
                        valid_loss += batch_loss * batch_size
                        valid_size += batch_size
                    valid_loss /= valid_size

                    # update best validation loss
                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_step = steps
                    # if no improvement seen
                    elif steps - best_step >= STOP_STEPS:
                        return best_valid_loss, best_step
                # if number of steps is reached
                elif num_steps is not None and steps >= num_steps:
                    assert valid_loader is None
                    return copy.deepcopy(model)

    # prediction
    def predict(self, c_data, x_data):
        c_data = torch.FloatTensor(c_data).to(self.device)
        x_data = torch.FloatTensor(x_data).to(self.device)
        y_pred = self.best_model.predict(c_data, x_data)
        return y_pred

    def refit_model(self,
                    c_train,
                    x_train,
                    y_train,
                    ac_train=None,
                    ax_train=None):

        self.best_model = None

        # convert data
        c_train = torch.FloatTensor(c_train).to(self.device)
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        if ac_train is not None and ax_train is not None:
            ac_train = torch.FloatTensor(ac_train).to(self.device)
            ax_train = torch.FloatTensor(ax_train).to(self.device)

        # retrain model with full data
        train_loader = DataLoader(dataset=TensorDataset(c_train, x_train, y_train),
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  drop_last=False,
                                  worker_init_fn=np.random.seed(self.rand_seed))

        if ac_train is not None and ax_train is not None:
            aux_loader = DataLoader(dataset=TensorDataset(ac_train, ax_train),
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    drop_last=False,
                                    worker_init_fn=np.random.seed(self.rand_seed))

        else:
            aux_loader = None

        print('best_params:', self.best_params)
        # retrain model and set as best model
        self.best_model = self.fit_model(train_loader, None, aux_loader, **self.best_params)
