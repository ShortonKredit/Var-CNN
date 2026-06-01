from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import shutil

# --- TAT LOG RAC TENSORFLOW ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import argparse
import numpy as np
import h5py

import var_cnn
import evaluate
import data_generator


def update_config(config, updates):
    """Updates config dict and config file with updates dict."""
    config.update(updates)
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)


def is_valid_mixture(mixture):
    """Check if mixture is a 2D array with strings representing the models."""
    assert isinstance(mixture, list) and len(mixture) > 0
    for inner_comb in mixture:
        assert isinstance(inner_comb, list) and len(inner_comb) > 0
        for model in inner_comb:
            assert model in ['dir', 'time', 'metadata', 'dir_iat', 'dir_iat_log', 'dir_x_iat', 'dir_iat_raw']


def train_and_val(config, model, callbacks, mixture_num=None, sub_model_name=None):
    """Train and validate model."""
    is_flat = "sequence_dataset" in config
    batch_size = config['batch_size']
    model_name = config['model_name']
    epochs = config['var_cnn_max_epochs'] if model_name == 'var-cnn' else config['df_epochs']

    if is_flat:
        model_id = config.get("model_id", "default_model")
        print('training flat model %s' % model_id)

        data_file = config['processed_h5']
        with h5py.File(data_file, 'r') as f:
            train_size = len(f['training_data/labels'])
            val_size = len(f['validation_data/labels'])

        train_steps = int(np.ceil(train_size / batch_size))
        val_steps = int(np.ceil(val_size / batch_size))

        output_dir = config.get("output_dir", "/kaggle/working/outputs")
        if not output_dir.endswith(model_id):
            target_dir = os.path.join(output_dir, model_id)
        else:
            target_dir = output_dir

        weights_file = os.path.join(target_dir, f"{model_id}.weights.h5")
    else:
        num_mon_sites = config['num_mon_sites']
        num_mon_inst_train = config['num_mon_inst_train']
        num_unmon_sites_train = config['num_unmon_sites_train']
        print('training %s %s model' % (model_name, sub_model_name))

        train_size = int((num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.95)
        train_steps = train_size // batch_size
        val_size = int((num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.05)
        val_steps = val_size // batch_size

        weights_dir = config.get('data_dir', '.')
        weights_file = os.path.join(weights_dir, f"{model_name}_{sub_model_name}.weights.h5")

    # Resume/Load weights logic
    if os.path.exists(weights_file):
        print(f'---> Found existing weights file, making backup copy and resuming training...')
        try:
            shutil.copy(weights_file, weights_file + ".bak")
        except Exception:
            pass
        model.load_weights(weights_file)

    # Note: callbacks already contains ModelCheckpoint from get_model
    # We also append ModelCheckpoint as in the original code for root save
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(weights_file, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    if 'callbacks' not in locals() or callbacks is None:
        callbacks = []
    callbacks.append(checkpoint)

    train_time_start = time.time()
    model.fit(
        data_generator.generate(config, 'training_data', mixture_num),
        steps_per_epoch=train_steps if (not is_flat and train_size % batch_size == 0) else train_steps,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=data_generator.generate(config, 'validation_data', mixture_num),
        validation_steps=val_steps if (not is_flat and val_size % batch_size == 0) else val_steps,
        shuffle=False)
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))


def predict(config, model, mixture_num=None, sub_model_name=None):
    """Compute and save final predictions on test set."""
    is_flat = "sequence_dataset" in config
    batch_size = config['batch_size']
    model_name = config['model_name']

    if is_flat:
        model_id = config.get("model_id", "default_model")
        print('generating predictions for flat model %s' % model_id)

        output_dir = config.get("output_dir", "/kaggle/working/outputs")
        if not output_dir.endswith(model_id):
            target_dir = os.path.join(output_dir, model_id)
        else:
            target_dir = output_dir

        weights_file = os.path.join(target_dir, f"{model_id}.weights.h5")
        predictions_file = os.path.join(target_dir, f"{model_id}_model")

        data_file = config['processed_h5']
        with h5py.File(data_file, 'r') as f:
            test_size = len(f['test_data/labels'])
        test_steps = int(np.ceil(test_size / batch_size))
    else:
        num_mon_sites = config['num_mon_sites']
        num_mon_inst_test = config['num_mon_inst_test']
        num_unmon_sites_test = config['num_unmon_sites_test']
        predictions_dir = config['predictions_dir']

        print('generating predictions for %s %s model' % (model_name, sub_model_name))
        weights_dir = config.get('data_dir', '.')
        weights_file = os.path.join(weights_dir, f"{model_name}_{sub_model_name}.weights.h5")
        predictions_file = os.path.join(predictions_dir, sub_model_name + "_model")

        test_size = num_mon_sites * num_mon_inst_test + num_unmon_sites_test
        test_steps = test_size // batch_size

    # Crucial: Load best weights before predicting
    if os.path.exists(weights_file):
        print(f'---> Loading best weights from {weights_file} before prediction...')
        model.load_weights(weights_file)

    test_time_start = time.time()
    predictions = model.predict(
        data_generator.generate(config, 'test_data', mixture_num),
        steps=test_steps if (is_flat or test_size % batch_size == 0) else test_steps + 1,
        verbose=0)
    test_time_end = time.time()

    if is_flat:
        np.save(file=predictions_file, arr=predictions)
    else:
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        np.save(file=predictions_file, arr=predictions)

    print('Total test time: %f' % (test_time_end - test_time_start))


# --- SỬ DỤNG ARGPARSE ĐỂ ĐỌC CONFIG-DRIVEN ---
parser = argparse.ArgumentParser(description="Train Var-CNN using a specified config file.")
parser.add_argument("--config", type=str, required=True, help="Path to config JSON file (e.g., config_high.json)")
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    config = json.load(config_file)
    if config['model_name'] == 'df':
        update_config(config, {'mixture': [['dir']], 'batch_size': 128})

is_flat = "sequence_dataset" in config

if is_flat:
    # Setup H5 and other variables
    batch_size = config['batch_size']
    model_name = config['model_name']
    epochs = config['var_cnn_max_epochs'] if model_name == 'var-cnn' else config['df_epochs']

    # Train 1 model
    if model_name == 'var-cnn':
        model, callbacks = var_cnn.get_model(config)
    else:
        raise ValueError("Deprecated model.")

    train_and_val(config, model, callbacks)
    predict(config, model)

    # Save copy of the configuration run
    model_id = config.get("model_id", "default_model")
    output_dir = config.get("output_dir", "/kaggle/working/outputs")
    if not output_dir.endswith(model_id):
        target_dir = os.path.join(output_dir, model_id)
    else:
        target_dir = output_dir
    os.makedirs(target_dir, exist_ok=True)

    config_out_path = os.path.join(target_dir, f"{model_id}.config.json")
    with open(config_out_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved config copy to {config_out_path}")

    # Evaluate
    print('evaluating model on test data...')
    evaluate.main(config)
else:
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
    epochs = config['var_cnn_max_epochs'] if model_name == 'var-cnn' \
        else config['df_epochs']
    is_valid_mixture(mixture)

    data_file = config.get('processed_h5')
    if not data_file:
        data_file = '%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites, num_mon_inst,
                                          num_unmon_sites_train, num_unmon_sites_test)

    if not os.path.exists(data_file):
        print("[!] Warning: H5 data file not found at %s. Please run convert_data.py first." % data_file)

    for mixture_num, inner_comb in enumerate(mixture):
        sub_model_name = '_'.join(inner_comb)
        
        if model_name == 'var-cnn':
            model, callbacks = var_cnn.get_model(config, mixture_num, sub_model_name)
        else:
            raise ValueError("Model 'df' requires df.py.")

        train_and_val(config, model, callbacks, mixture_num, sub_model_name)
        predict(config, model, mixture_num, sub_model_name)

    print('evaluating mixture on test data...')
    evaluate.main(config)
