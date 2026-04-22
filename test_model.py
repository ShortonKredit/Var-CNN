from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import time
import numpy as np

import var_cnn
import df
import evaluate
import preprocess_data
import data_generator


def predict(config, model, mixture_num, sub_model_name):
    """Compute and save final predictions on test set."""
    print('generating predictions for %s %s model'
          % (model_name, sub_model_name))

    if model_name == 'var-cnn':
        if os.path.exists('model_weights.weights.h5'):
            print("Loading weights from model_weights.weights.h5")
            model.load_weights('model_weights.weights.h5')
        else:
            print("WARNING: model_weights.weights.h5 not found!")

    test_size = num_mon_sites * num_mon_inst_test + num_unmon_sites_test
    test_steps = test_size // batch_size

    test_time_start = time.time()
    predictions = model.predict(
        data_generator.generate(config, 'test_data', mixture_num),
        steps=test_steps if test_size % batch_size == 0 else test_steps + 1,
        verbose=1)
    test_time_end = time.time()

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    np.save(file='%s%s_model' % (predictions_dir, sub_model_name),
            arr=predictions)

    print('Total test time: %f' % (test_time_end - test_time_start))


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    data_dir = config['data_dir']
    model_name = config['model_name']
    mixture = config['mixture']
    batch_size = config['batch_size']
    predictions_dir = config['predictions_dir']

    if not os.path.exists('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                                num_mon_inst,
                                                num_unmon_sites_train,
                                                num_unmon_sites_test)):
        print("Preprocessing data for evaluation...")
        preprocess_data.main(config)

    for mixture_num, inner_comb in enumerate(mixture):
        model, callbacks = var_cnn.get_model(config, mixture_num) \
            if model_name == 'var-cnn' else df.get_model(config)

        sub_model_name = '_'.join(inner_comb)
        predict(config, model, mixture_num, sub_model_name)

    print('evaluating mixture on test data...')
    evaluate.main(config)
