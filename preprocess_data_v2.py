from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import random
import h5py
import json
import os
from tqdm import tqdm
import wang_to_varcnn
from tensorflow.keras.utils import to_categorical

def main(config):
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    data_dir = config['data_dir']
    mon_data_loc = data_dir + 'all_closed_world.npz'
    unmon_data_loc = data_dir + 'all_open_world.npz'

    print('Starting %d_%d_%d_%d.h5 (RAM OPTIMIZED)' % (num_mon_sites, num_mon_inst,
                                       num_unmon_sites_train,
                                       num_unmon_sites_test))
    start = time.time()

    train_seq_and_labels = []
    test_seq_and_labels = []

    print('reading monitored data')
    mon_dataset = np.load(mon_data_loc)
    mon_dir_seq = mon_dataset['dir_seq']
    mon_labels = mon_dataset['labels']

    mon_site_data = {}
    mon_site_labels = {}
    for dir_seq, site_name in tqdm(zip(mon_dir_seq, mon_labels)):
        if site_name not in mon_site_data:
            if len(mon_site_data) >= num_mon_sites:
                continue
            else:
                mon_site_data[site_name] = []
                mon_site_labels[site_name] = len(mon_site_labels)

        mon_site_data[site_name].append([dir_seq, mon_site_labels[site_name]])

    for instances in tqdm(mon_site_data.values()):
        random.shuffle(instances)
        for inst_num, all_data in enumerate(instances):
            if inst_num < num_mon_inst_train:
                train_seq_and_labels.append(all_data)
            elif inst_num < num_mon_inst:
                test_seq_and_labels.append(all_data)
            else:
                break

    del mon_dataset, mon_dir_seq, mon_labels, mon_site_data, mon_site_labels

    print('reading unmonitored data')
    unmon_dataset = np.load(unmon_data_loc)
    unmon_dir_seq = unmon_dataset['dir_seq']

    unmon_site_data = [[dir_seq, num_mon_sites] for dir_seq in unmon_dir_seq]
    random.shuffle(unmon_site_data)
    
    for inst_num, all_data in tqdm(enumerate(unmon_site_data)):
        if inst_num < num_unmon_sites_train:
            train_seq_and_labels.append(all_data)
        elif inst_num < num_unmon_sites:
            test_seq_and_labels.append(all_data)
        else:
            break

    del unmon_dataset, unmon_dir_seq, unmon_site_data

    random.shuffle(train_seq_and_labels)
    random.shuffle(test_seq_and_labels)

    train_dir = []
    train_labels = []
    test_dir = []
    test_labels = []

    for dir_seq, label in train_seq_and_labels:
        train_dir.append(dir_seq)
        train_labels.append(label)
    for dir_seq, label in test_seq_and_labels:
        test_dir.append(dir_seq)
        test_labels.append(label)

    del train_seq_and_labels, test_seq_and_labels

    train_dir = np.array(train_dir)
    test_dir = np.array(test_dir)

    train_dir = np.reshape(train_dir, (train_dir.shape[0], train_dir.shape[1], 1))
    test_dir = np.reshape(test_dir, (test_dir.shape[0], test_dir.shape[1], 1))

    num_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    print('Training labels:', train_labels.shape)

    with h5py.File('%s%d_%d_%d_%d.h5' %
                   (data_dir, num_mon_sites, num_mon_inst,
                    num_unmon_sites_train, num_unmon_sites_test), 'w') as f:
        f.create_group('training_data')
        f.create_group('validation_data')
        f.create_group('test_data')
        
        f.create_dataset('training_data/dir_seq', data=train_dir[:int(0.95 * len(train_dir))])
        f.create_dataset('training_data/labels', data=train_labels[:int(0.95 * len(train_labels))])
        
        f.create_dataset('validation_data/dir_seq', data=train_dir[int(0.95 * len(train_dir)):])
        f.create_dataset('validation_data/labels', data=train_labels[int(0.95 * len(train_labels)):])
        
        f.create_dataset('test_data/dir_seq', data=test_dir)
        f.create_dataset('test_data/labels', data=test_labels)

    end = time.time()
    print('Finished in %f seconds' % (end - start))

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
    main(config)
