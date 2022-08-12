import numpy as np
import pickle as pkl
import openml
import collections
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace
from data.process_data import processData
import os.path

NUM_RI_DATASETS = 13

UCI_LIST = \
    ['yacht',
     'boston',
     'energy',
     'concrete',
     'kin8nm',
     'power',
     'naval',
     'protein']

OPENML_DICT = \
    {'359949': 'house_sales',
     '359945': 'us_crime',
     '359943': 'nyc-taxi-green-dec-2016',
     '359942': 'colleges',
     '359944': 'abalone',
     '359941': 'OnlineNewsPopularity',
     '359926': 'Airlines_DepDelay_1M',
     '317614': 'Yolanda',
     '317612': 'Brazilian_houses',
     '233214': 'Santander_transaction_value',
     '233212': 'Allstate_Claims_Severity',
     '233215': 'Mercedes_Benz_Greener_Manufacturing',
     '359951': 'house_prices_nominal',
     '233211': 'diamonds',
     '359948': 'SAT11-HAND-runtime-regression',
     '359947': 'MIP-2016-regression',
     '168891': 'black_friday',
     '167210': 'Moneyball',
     '233213': 'Buzzinsocialmedia_Twitter',
     '14097': 'QSAR-TID-10980',
     '13854': 'QSAR-TID-11',
     '359952': 'house_16H',
     '359930': 'quake',
     '359931': 'sensory',
     '359932': 'socmob',
     '4857': 'boston',
     '359933': 'space_ga',
     '359934': 'tecator',
     '359939': 'topo_2_1',
     '359940': 'yprop_4_1',
     '359935': 'wine_quality',
     '359936': 'elevators',
     '359946': 'pol'}


def load_uci(dataset_name, random_seed=111, data_loc='./data/dataset/'):
    if dataset_name not in UCI_LIST:
        raise NotImplementedError('not available dataset')
    # load data
    data = np.loadtxt(os.path.join(data_loc ,"{}.txt".format(dataset_name)))
    x_full = data[:, :-1]
    y_full = data[:, -1].reshape(-1, 1)

    # split into train / test
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.1, random_state=random_seed)

    # normalizer based on train data
    x_transformer = StandardScaler().fit(x_train)
    y_transformer = StandardScaler().fit(y_train)

    # transform data
    x_train = x_transformer.transform(x_train)
    y_train = y_transformer.transform(y_train)
    x_test = x_transformer.transform(x_test)
    y_test = y_transformer.transform(y_test)

    return Namespace(size=x_full.shape[0], x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def load_openml(task_id, random_seed=1, data_loc='./data/dataset/'):
    data_path = os.path.join(data_loc ,"openml_{}_seed{}.pkl".format(task_id, random_seed))
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

    # pickle
    with open(data_path, 'wb') as f:
        pkl.dump(dataset, f)

    # return dataset
    return dataset

