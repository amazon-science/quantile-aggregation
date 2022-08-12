import os
import argparse
import numpy as np
import pickle as pkl
MODEL_LIST = \
    ['QuantileConditionalGaussianNetwork',
     'QuantileSingleNeuralNetwork',
     'QuantileJointNeuralNetwork',
     'QuantileRandomForest',
     'QuantileExtraTrees',
     'QuantileLightGBM',
    ]

def merge_results(dataset, seed, eparams):
    merge_z_test = []
    merge_z_val  = []

    merge_oof_x_train = {}
    merge_oof_y_train = {}
    merge_oof_z_train = {}
    for model in MODEL_LIST:
        file_path = eparams.DATA_PATH + eparams.log_id + 'quantile_nested_{}_z_test_{}_cv5_iter20_seed{}.npy'.format(model, dataset, seed)
        file_path_val = eparams.DATA_PATH + eparams.log_id + 'quantile_nested_{}_z_val_{}_cv5_iter20_seed{}.npy'.format(model, dataset, seed)

        if os.path.exists(file_path):
            merge_z_test.append(np.load(file_path))

        if os.path.exists(file_path_val):
            merge_z_val.append(np.load(file_path_val))


        file_path = eparams.DATA_PATH + eparams.log_id + 'quantile_nested_{}_oof_x_train_{}_cv5_iter20_seed{}.pkl'.format(model, dataset, seed)
        if os.path.exists(file_path) and len(merge_oof_x_train) == 0:
            with open(file_path, 'rb') as handle:
                merge_oof_x_train = pkl.load(handle)

        file_path = eparams.DATA_PATH + eparams.log_id + 'quantile_nested_{}_oof_y_train_{}_cv5_iter20_seed{}.pkl'.format(model, dataset, seed)
        if os.path.exists(file_path) and len(merge_oof_y_train) == 0:
            with open(file_path, 'rb') as handle:
                merge_oof_y_train = pkl.load(handle)

        file_path = eparams.DATA_PATH + eparams.log_id + 'quantile_nested_{}_oof_z_train_{}_cv5_iter20_seed{}.pkl'.format(model, dataset, seed)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as handle:
                oof_z_train = pkl.load(handle)

            if len(merge_oof_z_train) == 0:
                merge_oof_z_train = oof_z_train
                for pair_key in merge_oof_z_train.keys():
                    if '-' in pair_key:
                        tmp0, tmp1 = merge_oof_z_train[pair_key]
                        merge_oof_z_train[pair_key] = [[tmp0], [tmp1]]
                    else:
                        tmp0 = merge_oof_z_train[pair_key]
                        merge_oof_z_train[pair_key] = [tmp0]
            else:
                for pair_key in merge_oof_z_train.keys():
                    if '-' in pair_key:
                        tmp0, tmp1 = oof_z_train[pair_key]
                        merge_oof_z_train[pair_key][0].append(tmp0)
                        merge_oof_z_train[pair_key][1].append(tmp1)
                    else:
                        tmp0 = oof_z_train[pair_key]
                        merge_oof_z_train[pair_key].append(tmp0)

    merge_z_test = np.stack(merge_z_test, 1)
    merge_z_val = np.stack(merge_z_val, 1)

    for pair_key in merge_oof_z_train.keys():
        if '-' in pair_key:
            tmp0, tmp1 = merge_oof_z_train[pair_key]
            merge_oof_z_train[pair_key] = [np.stack(tmp0, 1), np.stack(tmp1, 1)]
        else:
            tmp0 = merge_oof_z_train[pair_key]
            merge_oof_z_train[pair_key] = np.stack(tmp0, 1)

    np.save(eparams.DATA_PATH  + eparams.log_id + 'quantile_nested_base_z_test_{}_cv5_iter20_seed{}.npy'.format(dataset, seed), merge_z_test)
    np.save(eparams.DATA_PATH  + eparams.log_id + 'quantile_nested_base_z_val_{}_cv5_iter20_seed{}.npy'.format(dataset, seed), merge_z_val)

    file_path = eparams.DATA_PATH  + eparams.log_id + 'quantile_nested_base_oof_x_train_{}_cv5_iter20_seed{}.pkl'.format(dataset, seed)
    with open(file_path, 'wb') as handle:
        pkl.dump(merge_oof_x_train, handle, protocol=pkl.HIGHEST_PROTOCOL)

    file_path = eparams.DATA_PATH  + eparams.log_id + 'quantile_nested_base_oof_y_train_{}_cv5_iter20_seed{}.pkl'.format(dataset, seed)
    with open(file_path, 'wb') as handle:
        pkl.dump(merge_oof_y_train, handle, protocol=pkl.HIGHEST_PROTOCOL)

    file_path = eparams.DATA_PATH  + eparams.log_id + 'quantile_nested_base_oof_z_train_{}_cv5_iter20_seed{}.pkl'.format(dataset, seed)
    with open(file_path, 'wb') as handle:
        pkl.dump(merge_oof_z_train, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()

    # parser
    parser.add_argument('--task-id', type=str, help='task id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--DATA_PATH', default='./output/data/')
    parser.add_argument('--log_id', default='mylogid')

    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    merge_results(args.task_id, args.seed, args)
    print('Done')
