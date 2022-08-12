import random
import torch
import argparse
import numpy as np
import pickle as pkl
from model.linear_regressor import QuantileRegressor
from model.random_forest import QuantileRandomForest
from model.extra_trees import QuantileExtraTrees
from model.light_gbm import QuantileLightGBM
from model.neural_network import QuantileJointNeuralNetwork
from model.neural_network import QuantileSingleNeuralNetwork
from model.neural_network import QuantileConditionalGaussianNetwork
from util.misc import make_dir
from sklearn.model_selection import KFold
from data.data_loader import load_uci, load_openml, UCI_LIST, OPENML_DICT
from util.others import dump_to_json
import os
import copy

# quantile list (from 1% to 99%)
QUANTILE_LIST = np.arange(1, 100, 1) / 100.0
CV_RATIO = 0.8

# run neural networks
def run_neural(model,
               x_train, y_train,
               x_test, y_test,
               cv_split, kfolder,
               num_iters, rand_seed=1, device=0,
               eparams=None):
    # model name
    model_name = model.__name__
    print('Train start: ', model_name)

    # set model learner
    batch_size = int(2 ** (3 + np.floor(np.log10(x_train.shape[0] + x_test.shape[0]))))
    model_learner = model(quantile_list=QUANTILE_LIST,
                          num_iters=num_iters,
                          cv_split=cv_split,
                          batch_size=batch_size,
                          rand_seed=rand_seed,
                          device=device,
                          use_grad=eparams.use_grad,
                          trans_type=eparams.trans_type,
                          use_margin=eparams.use_margin,
                          margin_type=eparams.margin_type
                          )
    #########
    # train/val split happens inside model_learner based on cv_split
    #########
    # fit model where x_train and y_train are split into val and train according to cv_plsit
    model_learner.fit(x_train, y_train, None)

    # compute test prediction
    z_test = model_learner.predict(x_test)
    print('z_test', z_test.shape)

    # model prediction on validation
    z_val = model_learner.predict(x_train[cv_split[0][1]])
    print('z_val', z_val.shape)

    #########################################
    #### remove validation from train set ###
    #########################################
    print('Full training size', x_train.shape, y_train.shape)
    x_train = x_train[cv_split[0][0]].copy()
    y_train = y_train[cv_split[0][0]].copy()
    print('Training size after removing validation', x_train.shape, y_train.shape)

    # get nested out-of-fold predictions
    full_index = np.arange(x_train.shape[0])
    # fold_list[5,2] where [:,0] is training and [:,1] validation
    fold_list  = list(kfolder.split(x_train))
    num_folds  = len(fold_list)

    # for oof
    oof_x_train = {}
    oof_y_train = {}
    oof_z_train = {}
    for k0 in range(num_folds):

        # compute oof for k0 and train
        oof_index0 = fold_list[k0][1]
        #train_index = np.setdiff1d(full_index, oof_index0)
        train_index  = np.setdiff1d(fold_list[k0][0], oof_index0)

        # split train / valid
        cv_x_train, cv_y_train   = x_train[train_index], y_train[train_index]
        cv_x_valid0, cv_y_valid0 = x_train[oof_index0], y_train[oof_index0]

        # fit on cv
        model_learner.refit_model(cv_x_train, cv_y_train, None)

        # obtain prediction over quantiles
        cv_z_valid0 = model_learner.predict(cv_x_valid0)

        oof_x_train['{}'.format(k0)] = cv_x_valid0
        oof_y_train['{}'.format(k0)] = cv_y_valid0
        oof_z_train['{}'.format(k0)] = cv_z_valid0

    return oof_x_train, oof_y_train, oof_z_train, z_test, z_val

# run tree models
def run_tree(model,
             x_train, y_train,
             x_test, y_test,
             cv_split, kfolder,
             num_iters, rand_seed=1,
             eparams=None,
             **kwargs):
    # model name
    model_name = model.__name__
    print('Train start: ', model_name)

    # set model learner
    model_learner = model(num_iters=num_iters, num_folds=cv_split, rand_seed=rand_seed)
    model_learner.fit(x_train, y_train)

    # compute test prediction on best hyper-params
    z_test = model_learner.full_predict(x_test, list(QUANTILE_LIST))
    print('z_test', z_test.shape)

    # model prediction on validation
    z_val = model_learner.full_predict(x_train[cv_split[0][1]], list(QUANTILE_LIST))
    print('z_val', z_val.shape)

    #########################################
    #### remove validation from train set ###
    #########################################
    print('Full training sizes:', x_train.shape, y_train.shape)
    x_train = x_train[cv_split[0][0]].copy()
    y_train = y_train[cv_split[0][0]].copy()
    print('Training size after removing validation', x_train.shape, y_train.shape)

    # get nested out-of-fold predictions
    full_index = np.arange(x_train.shape[0])
    fold_list = list(kfolder.split(x_train))
    num_folds = len(fold_list)

    # for oof
    oof_x_train = {}
    oof_y_train = {}
    oof_z_train = {}
    for k0 in range(num_folds):

        # compute oof for k0 and train
        oof_index0 = fold_list[k0][1]
        #train_index = np.setdiff1d(full_index, oof_index0)
        train_index = np.setdiff1d(fold_list[k0][0], oof_index0)

        # split train / valid
        cv_x_train, cv_y_train = x_train[train_index], y_train[train_index]
        cv_x_valid0, cv_y_valid0 = x_train[oof_index0], y_train[oof_index0]

        # fit on cv
        cv_model = model_learner.get_init_model()
        cv_model.fit(cv_x_train, cv_y_train.reshape(-1))

        # obtain prediction over quantiles
        cv_z_valid0 = cv_model.predict(cv_x_valid0, list(QUANTILE_LIST))

        oof_x_train['{}'.format(k0)] = cv_x_valid0
        oof_y_train['{}'.format(k0)] = cv_y_valid0
        oof_z_train['{}'.format(k0)] = cv_z_valid0

    return oof_x_train, oof_y_train, oof_z_train, z_test, z_val

# run other models
def run_others(model,
               x_train, y_train,
               x_test, y_test,
               cv_split, kfolder,
               num_iters, rand_seed=1,
               eparams=None,
               **kwargs):
    # model name
    model_name = model.__name__
    print('Train start: ', model_name)

    #########################################
    #### remove validation from train set ###
    #########################################
    x_train_org = x_train.copy()
    y_train_org = y_train.copy()

    x_val = x_train[cv_split[0][1]].copy()
    print('x_val', x_val.shape)

    print('Full training sizes:', x_train.shape, y_train.shape)
    x_train = x_train[cv_split[0][0]].copy()
    y_train = y_train[cv_split[0][0]].copy()
    print('Training size after removing validation', x_train.shape, y_train.shape)

    # get nested out-of-fold predictions
    full_index = np.arange(x_train.shape[0])
    fold_list = list(kfolder.split(x_train))
    num_folds = len(fold_list)

    # for each quantile
    oof_x_train = {}
    oof_y_train = {}
    oof_z_train = {}
    z_test = []
    z_val  = []

    for quantile in QUANTILE_LIST:
        # find best model
        model_learner = model(quantile=quantile, num_iters=num_iters, num_folds=cv_split, rand_seed=rand_seed)
        model_learner.fit(x_train_org, y_train_org)

        # compute test prediction
        z_test.append(model_learner.predict(x_test).reshape(-1, 1))

        # val prediction
        z_val.append(model_learner.predict(x_val).reshape(-1, 1))

        # get out-of-fold predictions
        for k0 in range(num_folds):

            # and compute oof for k0 and k1
            oof_index0 = fold_list[k0][1]
            #train_index = np.setdiff1d(full_index, oof_index0)
            train_index = np.setdiff1d(fold_list[k0][0], oof_index0)

            # split train / valid
            cv_x_train, cv_y_train   = x_train[train_index], y_train[train_index]
            cv_x_valid0, cv_y_valid0 = x_train[oof_index0], y_train[oof_index0]

            # fit on cv
            cv_model = model_learner.get_init_model()
            cv_model.fit(cv_x_train, cv_y_train.reshape(-1))

            # get oof
            cv_z_valid0 = cv_model.predict(cv_x_valid0).reshape(-1, 1)
            if '{}'.format(k0) not in oof_x_train:
                oof_x_train['{}'.format(k0)] = cv_x_valid0
            if '{}'.format(k0) not in oof_y_train:
                oof_y_train['{}'.format(k0)] = cv_y_valid0
            if '{}'.format(k0) not in oof_z_train:
                oof_z_train['{}'.format(k0)] = cv_z_valid0
            else:
                cv_z_valid0 = np.concatenate([oof_z_train['{}'.format(k0)], cv_z_valid0], -1)
                oof_z_train['{}'.format(k0)] = cv_z_valid0

    z_test = np.concatenate(z_test, 1)
    z_val  = np.concatenate(z_val, 1)

    return oof_x_train, oof_y_train, oof_z_train, z_test, z_val

def prepare_data(task_id,
                num_iters=20,
                num_folds=5,
                rand_seed=1,
                eparams=None):

    print('Data preparation')
    # set seed
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    # load dataset
    if task_id in UCI_LIST:
        task_name = task_id
        dataset = load_uci(task_id, random_seed=rand_seed, data_loc=eparams.data_loc)

    else:
        assert task_id in OPENML_DICT
        task_name = OPENML_DICT[task_id]
        dataset = load_openml(task_id, random_seed=rand_seed, data_loc=eparams.data_loc)

    dataset_size = dataset.size
    x_train, y_train = dataset.x_train, dataset.y_train
    x_test, y_test = dataset.x_test, dataset.y_test
    feature_size = x_train.shape[1]
    print('Data: {} (seed {}, dataset size {}, feature size {})'.format(task_name, rand_seed, dataset_size, feature_size))

    # save data split
    exp_name = [task_id, num_folds, num_iters, rand_seed]
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_nested_base_x_test_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), x_test)
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_nested_base_y_test_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), y_test)

    # set cv splits
    train_size = x_train.shape[0]
    train_idx_list = np.arange(train_size)
    np.random.shuffle(train_idx_list)
    cv_split = [[train_idx_list[:int(train_size * CV_RATIO)], train_idx_list[int(train_size * CV_RATIO):]]]
    ########
    # save the train and val split
    ########
    print(eparams.log_id +'quantile_all_train_idx_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_all_train_idx_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), train_idx_list)
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_train_val_idx_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), cv_split)
    print(eparams.log_id +'quantile_train_val_idx_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))

    #########################################
    #### validation is removed from training set
    #########################################
    print('Train data:', x_train[cv_split[0][0]].shape, y_train[cv_split[0][0]].shape)
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_nested_base_x_train_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), x_train[cv_split[0][0]])
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_nested_base_y_train_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), y_train[cv_split[0][0]])

    print('Val data:', x_train[cv_split[0][1]].shape, y_train[cv_split[0][1]].shape)
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_nested_base_x_val_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), x_train[cv_split[0][1]])
    np.save(eparams.DATA_PATH + eparams.log_id +'quantile_nested_base_y_val_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), y_train[cv_split[0][1]])

    print('Test data:', x_test.shape, y_test.shape)

    output = {'x_train':x_train,
              'y_train':y_train,
              'x_test' :x_test,
              'y_test' :y_test,
              'cv_split':cv_split,
             }
    return output

def run_exp(task_id,
            model_name,
            num_iters=20,
            num_folds=5,
            rand_seed=1,
            device=-1,
            cleaned_data=None,
            eparams=None):

    # set seed
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    if torch.cuda.is_available() and device > -1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    exp_name = [task_id, num_folds, num_iters, rand_seed]

    # set k-Folder for out of fold prediction (no need shuffle)
    kfolder = KFold(n_splits=num_folds)

    model, exp_fn = None, None
    if model_name == 'cgn':
        model = QuantileConditionalGaussianNetwork
        exp_fn = run_neural

    elif model_name == 'sqr':
        model = QuantileSingleNeuralNetwork
        exp_fn = run_neural

    elif model_name == 'mqr':
        model = QuantileJointNeuralNetwork
        exp_fn = run_neural

    elif model_name == 'rf':
        model = QuantileRandomForest
        exp_fn = run_tree

    elif model_name == 'xt':
        model = QuantileExtraTrees
        exp_fn = run_tree

    elif model_name == 'lgbm':
        model = QuantileLightGBM
        exp_fn = run_others

    elif model_name == 'qr':
        model = QuantileRegressor
        exp_fn = run_others

    # run exp
    oof_x_train, oof_y_train, oof_z_train, z_test, z_val = exp_fn(model=model,
                                                                  x_train=cleaned_data['x_train'].copy(),
                                                                  y_train=cleaned_data['y_train'].copy(),
                                                                  x_test=cleaned_data['x_test'].copy(),
                                                                  y_test=cleaned_data['y_test'].copy(),
                                                                  cv_split=copy.deepcopy(cleaned_data['cv_split']),
                                                                  kfolder=kfolder,
                                                                  num_iters=num_iters,
                                                                  rand_seed=rand_seed,
                                                                  device=device,
                                                                  eparams=eparams)

    # save
    print('Saving')
    print('z_test', z_test.shape)
    np.save(eparams.DATA_PATH  + eparams.log_id +'quantile_nested_{}_z_test_{}_cv{}_iter{}_seed{}.npy'.format(model.__name__, *exp_name), z_test)
    print(eparams.log_id +'quantile_nested_{}_z_test_{}_cv{}_iter{}_seed{}.npy'.format(model.__name__, *exp_name))

    print('z_val', z_val.shape)
    np.save(eparams.DATA_PATH  + eparams.log_id +'quantile_nested_{}_z_val_{}_cv{}_iter{}_seed{}.npy'.format(model.__name__, *exp_name), z_val)
    print(eparams.log_id +'quantile_nested_{}_z_val_{}_cv{}_iter{}_seed{}.npy'.format(model.__name__, *exp_name))

    with open(eparams.DATA_PATH + eparams.log_id +'quantile_nested_{}_oof_x_train_{}_cv{}_iter{}_seed{}.pkl'.format(model.__name__, *exp_name), 'wb') as handle:
        pkl.dump(oof_x_train, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(eparams.log_id +'quantile_nested_{}_oof_x_train_{}_cv{}_iter{}_seed{}.pkl'.format(model.__name__, *exp_name))

    with open(eparams.DATA_PATH  + eparams.log_id +'quantile_nested_{}_oof_y_train_{}_cv{}_iter{}_seed{}.pkl'.format(model.__name__, *exp_name), 'wb') as handle:
        pkl.dump(oof_y_train, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(eparams.log_id +'quantile_nested_{}_oof_y_train_{}_cv{}_iter{}_seed{}.pkl'.format(model.__name__, *exp_name))

    with open(eparams.DATA_PATH  + eparams.log_id +'quantile_nested_{}_oof_z_train_{}_cv{}_iter{}_seed{}.pkl'.format(model.__name__, *exp_name), 'wb') as handle:
        pkl.dump(oof_z_train, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(eparams.log_id +'quantile_nested_{}_oof_z_train_{}_cv{}_iter{}_seed{}.pkl'.format(model.__name__, *exp_name))

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()

    # parser
    parser.add_argument('--task-id', type=str, help='task id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--cv', type=int, default=5, help='folds for out of fold predictions')
    parser.add_argument('--iter', type=int, default=20, help='number of iterations for grid search')
    parser.add_argument('--trans_type', default='pava')
    parser.add_argument('--use_grad', default=False, action='store_true')
    parser.add_argument('--use_margin', default=False, action='store_true')
    parser.add_argument('--DATA_PATH', default='./output/data/')
    parser.add_argument('--log_id', default='mylogid')
    parser.add_argument('--data_loc', default='./data/dataset/')
    parser.add_argument('--margin_type', type=str, default='single', help='margin type')


    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    make_dir(args.DATA_PATH)
    fname_json = os.path.join(args.DATA_PATH, args.log_id + '-t' + \
                                              str(args.task_id) + '_cv' + str(args.cv) + \
                                              '_i' + str(args.iter) + \
                                              '_S' + str(args.seed) + '.json' )
    dump_to_json(fname_json, {'args': args.__dict__})
    print(fname_json)

    # first prepare data
    cleaned_dt = prepare_data(task_id=args.task_id,
                              num_iters=args.iter,
                              num_folds=args.cv,
                              rand_seed=args.seed,
                              eparams=args)
    print('---------------------------------------------------')
    print()
    # run

    model_list = ['cgn', 'sqr', 'mqr', 'rf', 'xt', 'lgbm']

    for model in model_list:
        print(model , '...')
        run_exp(task_id=args.task_id,
                model_name=model,
                num_iters=args.iter,
                num_folds=args.cv,
                rand_seed=args.seed,
                device=args.gpu,
                cleaned_data=cleaned_dt,
                eparams=args,
                )
        print('---------------------------------------------------')

    print('Done.')

