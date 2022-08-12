import random
import torch
import argparse
import numpy as np
import pickle as pkl
from model.neural_aggregator import QuantileLocalAggregatorTrainer, QuantileGlobalAggregatorTrainer
from util.misc import make_dir
from data.data_loader import UCI_LIST, OPENML_DICT
import warnings
from util.others import dump_to_json
import os

warnings.filterwarnings('ignore')

# quantile list (from 1% to 99%)
QUANTILE_LIST = np.arange(1, 100, 1) / 100.0
NUM_QUANTILES = len(QUANTILE_LIST)
#CV_RATIO = 0.8

def model_prediction(model, x_data, z_data):
    batch_size = 512
    data_size = x_data.shape[0]
    num_batches = int(np.ceil(float(data_size) / float(batch_size)))

    e_data = []
    for i in range(num_batches):
        e_data.append(model.predict(x_data[i * batch_size:(i + 1) * batch_size], z_data[i * batch_size:(i + 1) * batch_size]))

    return np.concatenate(e_data, 0)

def run_exp(task_id, use_local=True,
            share_weight=False, cross_weight=True,
            normalize=True, margin_type=None,
            trans_type=None, use_grad=False,
            num_searches=20, num_folds=5, rand_seed=1, device=-1,
            eparams=None):

    print('--------------------')
    print('use_local: ', use_local)
    print('share_weight: ', share_weight)
    print('cross_weight: ', cross_weight)
    print('margin_type: ', margin_type)
    print('trans_type: ', trans_type)
    print('use_grad: ', use_grad)
    print('num_searches: ', num_searches)
    print('num_folds: ', num_folds)
    print('rand_seed: ', rand_seed)
    print('eparams.use_mean_pt: ', eparams.use_mean_pt)
    print('device: ', device)
    print('regularization_strength:', eparams.regularization_strength)
    print('--------------------')

    # set seed
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    if torch.cuda.is_available() and device > -1:
        torch.cuda.manual_seed(rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # exp setting
    task_name = task_id
    if task_id not in UCI_LIST:
        task_name = OPENML_DICT[task_id]

    exp_name = [task_id, num_folds, num_searches, rand_seed]

    # load dataset
    with open(eparams.DATA_PATH + eparams.log_id_base + 'quantile_nested_base_oof_x_train_{}_cv{}_iter{}_seed{}.pkl'.format(*exp_name), 'rb') as handle:
        oof_x_train = pkl.load(handle)

    with open(eparams.DATA_PATH + eparams.log_id_base + 'quantile_nested_base_oof_y_train_{}_cv{}_iter{}_seed{}.pkl'.format(*exp_name), 'rb') as handle:
        oof_y_train = pkl.load(handle)

    with open(eparams.DATA_PATH + eparams.log_id_base + 'quantile_nested_base_oof_z_train_{}_cv{}_iter{}_seed{}.pkl'.format(*exp_name), 'rb') as handle:
        oof_z_train = pkl.load(handle)

    x_train, y_train, z_train = [], [], []
    for k in range(num_folds):
        x_train.append(oof_x_train['{}'.format(k)])
        y_train.append(oof_y_train['{}'.format(k)])
        z_train.append(oof_z_train['{}'.format(k)])

    x_train = np.concatenate(x_train, 0)
    y_train = np.concatenate(y_train, 0)
    z_train = np.concatenate(z_train, 0)

    x_test = np.load(eparams.DATA_PATH  + eparams.log_id_base +  'quantile_nested_base_x_test_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))
    y_test = np.load(eparams.DATA_PATH  + eparams.log_id_base +  'quantile_nested_base_y_test_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))
    z_test = np.load(eparams.DATA_PATH  + eparams.log_id_base +  'quantile_nested_base_z_test_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))

    x_val = np.load(eparams.DATA_PATH  + eparams.log_id_base +  'quantile_nested_base_x_val_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))
    y_val = np.load(eparams.DATA_PATH  + eparams.log_id_base +  'quantile_nested_base_y_val_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))
    z_val = np.load(eparams.DATA_PATH  + eparams.log_id_base +  'quantile_nested_base_z_val_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))

    # get empirical quantile-margin
    if margin_type == 'non':
         margin_list = None

    elif margin_type == 'single':
         margin_list = None

    elif margin_type == 'adapt':
        # compute error (from estimated median)
        if eparams.use_mean_pt == True:
            print('using mean for adapt..')
            e_train = y_train.reshape(-1) - np.mean(z_train[..., NUM_QUANTILES // 2], 1).reshape(-1).astype(np.float32)

        else:
            print('using median..')
            e_train = y_train.reshape(-1) - np.median(z_train[..., NUM_QUANTILES // 2], 1).reshape(-1).astype(np.float32)

        margin_list = np.quantile(e_train.reshape(-1), QUANTILE_LIST, 0).astype(np.float32)

    elif margin_type == 'vec':
        margin_list = (QUANTILE_LIST.reshape(-1) * 1e-2).astype(np.float32)

    else:
        raise ValueError('%s margin not supported' % margin_type)

    # data size
    feature_size = x_train.shape[1]
    print('Data: {} (seed {}, train size {}, feature size {}, val size {}, test size {})'.format(
                    task_name, rand_seed, x_train.shape[0], feature_size, x_val.shape[0], x_test.shape[0]))

    #train_size = x_train.shape[0]
    train_idx_list = np.load(eparams.DATA_PATH + eparams.log_id_base +'quantile_all_train_idx_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name))
    cv_split = np.load(eparams.DATA_PATH + eparams.log_id_base +'quantile_train_val_idx_{}_cv{}_iter{}_seed{}.npy'.format(*exp_name), allow_pickle=True)

    # set cv splits
    #train_size = x_train.shape[0]
    #train_idx_list = np.arange(train_size)
    #np.random.shuffle(train_idx_list)
    #cv_split = [[train_idx_list[:int(train_size * CV_RATIO)], train_idx_list[int(train_size * CV_RATIO):]]]

    # num of base models and quantiles
    num_models, num_quantiles = z_train.shape[1], z_train.shape[2]
    assert num_quantiles == NUM_QUANTILES

    # model name
    if use_local:
        model_name = 'Local'

    else:
        model_name = 'Global'

    if share_weight:
        model_name += '-Coarse'

    elif cross_weight:
        model_name += '-Fine'

    else:
        model_name += '-Medium'

    print('Train start: ', model_name)

    # full experiment name
    if trans_type is not None:
        output_name = 'DQA_{}_norm{}_grad{}_{}_{}_margin_results_{}_cv{}_iter{}_seed{}'.format(model_name,
                                                                                               int(normalize),
                                                                                               int(use_grad),
                                                                                               trans_type,
                                                                                               margin_type,
                                                                                               *exp_name)
    else:
        output_name = 'DQA_{}_norm{}_{}_margin_results_{}_cv{}_iter{}_seed{}'.format(model_name,
                                                                                     int(normalize),
                                                                                     margin_type,
                                                                                     *exp_name)

    print(eparams.run_id + output_name)

    # set model learner
    batch_size = int(2 ** (3 + np.floor(np.log10(y_train.shape[0] + y_test.shape[0]))))
    model_trainer = QuantileLocalAggregatorTrainer if use_local else QuantileGlobalAggregatorTrainer
    model = model_trainer(quantile_list=QUANTILE_LIST, num_searches=num_searches, cv_split=cv_split,
                          share_weight=share_weight, cross_weight=cross_weight,
                          normalize=normalize, margin_list=margin_list,
                          trans_type=trans_type, use_grad=use_grad,
                          batch_size=batch_size, rand_seed=rand_seed, device=device,
                          margin_type=margin_type,
                          regularization_strength=eparams.regularization_strength)

    # fit model
    if use_local:
        model.fit(c_train=x_train, x_train=z_train, y_train=y_train,
                  c_val=x_val, x_val=z_val, y_val=y_val,
                  ac_train=None, ax_train=None)

    else:
        #model.fit(z_train, y_train, None)
        model.fit(x_train=z_train, y_train=y_train,
                  x_val=z_val, y_val=y_val,
                  ax_train=None)


    # compute test prediction
    if use_local:
        org_e_test = model_prediction(model, x_test, z_test)

    else:
        org_e_test = model.predict(z_test)

    # save results
    np.save(eparams.RESULT_PATH + eparams.run_id  + eparams.log_id_out + output_name + '_org_e_test.npy', org_e_test)
    np.save(eparams.RESULT_PATH + eparams.run_id  + output_name + '_org_e_test.npy', org_e_test)

    print('Done.')

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()

    # parser
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--fold', type=int, default=5, help='folds for out of fold predictions')
    parser.add_argument('--iter', type=int, default=20, help='number of iterations for grid search')
    parser.add_argument('--task-id', type=str, default='boston', help='task id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--local', type=int, default=1)
    parser.add_argument('--share', type=int, default=0, help='share combination')
    parser.add_argument('--cross', type=int, default=1, help='full combination')
    parser.add_argument('--grad', type=int, default=0, help='use grad')
    parser.add_argument('--trans', type=str, default='sort', help='non-crossing type')
    parser.add_argument('--margin', type=str, default='single', help='margin type')

    parser.add_argument('--norm', type=int, default=1, help='normalize weight')

    parser.add_argument('--DATA_PATH', default='./output/data/')
    parser.add_argument('--RESULT_PATH', default='./output/result/')
    parser.add_argument('--log_id_base', default='mylogid')
    parser.add_argument('--log_id_out', default='exNone')
    parser.add_argument('--run_id', default='p1_')
    parser.add_argument('--use_mean_pt', default=False, action='store_true')
    parser.add_argument('--regularization_strength', type=float, default=0.1)

    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    make_dir(args.DATA_PATH)
    make_dir(args.RESULT_PATH)

    fname_json = os.path.join(args.RESULT_PATH, args.log_id_out + '_' +
                              args.log_id_base +'_' + args.run_id + args.task_id +'_args_aggr.json' )
    print(fname_json)
    dump_to_json(fname_json, {'args': args.__dict__})

    # run
    run_exp(task_id=args.task_id,use_local=bool(args.local),
                share_weight=args.share, cross_weight=args.cross,
                normalize=bool(args.norm), margin_type=args.margin,
                trans_type=args.trans, use_grad=bool(args.grad),
                num_searches=args.iter, num_folds=args.fold,
                rand_seed=args.seed, device=args.gpu,
                eparams=args,
                )
