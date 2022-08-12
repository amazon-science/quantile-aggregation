import numpy as np
import pickle as pkl
import openml
import collections
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace
from process_data import processData
import os.path
import argparse

OPENML_DICT = \
    {
     '359934': 'tecator',
     '359931': 'sensory',
     '359947': 'MIP-2016-regression',
     '359932': 'socmob',
     '167210': 'Moneyball',
     '359951': 'house_prices_nominal',
     '359945': 'us_crime',
     '359930': 'quake',
     '359933': 'space_ga',
     '359944': 'abalone',
     '233215': 'Mercedes_Benz_Greener_Manufacturing',
     '359948': 'SAT11-HAND-runtime-regression',
     '233214': 'Santander_transaction_value',
     '13854': 'QSAR-TID-11',
     '14097': 'QSAR-TID-10980',
     '359935': 'wine_quality',
     '359942': 'colleges',
     '359939': 'topo_2_1',
     '359940': 'yprop_4_1',
     '317612': 'Brazilian_houses',
     '359946': 'pol',
     '359936': 'elevators',
     '359949': 'house_sales',
     '359952': 'house_16H',
     '359941': 'OnlineNewsPopularity',
     '233211': 'diamonds',
     }

def load_openml(task_id, random_seed=1):
    data_path = "./dataset/openml_{}_seed{}.pkl".format(task_id, random_seed)
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            return pkl.load(f)

    # otherwise load
    task = openml.tasks.get_task(task_id)

    # get label
    label = task.target_name

    # get full pd_frame
    full_data = task.get_dataset().get_data()[0]
    full_size = full_data.shape[0]

    # split data
    train_data, test_data = train_test_split(full_data, test_size=0.1, random_state=random_seed)
    if hasattr(train_data, 'sparse'):
        train_data = train_data.sparse.to_dense()
    if hasattr(test_data, 'sparse'):
        test_data = test_data.sparse.to_dense()

    # preprocess x_train
    x_train, y_train, x_transformer = processData(data=train_data, label_column=label, problem_type='regression')
    x_test, y_test, _ = processData(data=test_data, label_column=label, ag_predictor=x_transformer)

    # convert to numpy
    x_train = x_train.values
    y_train = y_train.values.reshape(-1, 1)
    x_test = x_test.values
    y_test = y_test.values.reshape(-1, 1)

    # y normalizer based on train data
    y_transformer = StandardScaler().fit(y_train)

    # transform data
    y_train = y_transformer.transform(y_train)
    y_test = y_transformer.transform(y_test)

    # dataset
    dataset = Namespace(size=full_size, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    info = {'task_id':  task_id, 
            'x_train.shape':x_train.shape,
            'y_train.shape': y_train.shape,
            'x_test.shape':x_test.shape,
            'y_test.shape':y_test.shape,
            'seed':random_seed
           }
    print('-------------------------')
    print('task_id', task_id)
    print('train-->', x_train.shape, y_train.shape)
    print('test-->', x_test.shape, y_test.shape)
    print('-------------------------')

    # pickle
    with open(data_path, 'wb') as f:
        pkl.dump(dataset, f)

    # return dataset
    return dataset, info

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)

if __name__ == "__main__":

    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    all_info = []
    for i in OPENML_DICT.keys():
        print('task', i)
        _, info = load_openml(i, random_seed=args.seed)
        all_info.append(info)
        print('*******')
    print('Done')
    print('---------------------------------------')
    print('all_info', all_info)
