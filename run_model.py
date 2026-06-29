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


def train_and_val(config, model, callbacks):
    """Train and validate model."""
    batch_size = config['batch_size']
    model_name = config['model_name']
    epochs = config['var_cnn_max_epochs'] if model_name == 'var-cnn' else config['df_epochs']

    model_id = config.get("model_id", "default_model")
    print('training flat model %s' % model_id)

    data_file = config['processed_h5']
    with h5py.File(data_file, 'r') as f:
        if "train_indices_file" in config:
            indices_path = config["train_indices_file"]
            if os.path.exists(indices_path):
                train_size = len(np.load(indices_path))
            else:
                raise FileNotFoundError(f"train_indices_file not found at {indices_path}")
        else:
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

    # Resume/Load weights logic
    if os.path.exists(weights_file):
        print(f'---> Found existing weights file, making backup copy and resuming training...')
        try:
            shutil.copy(weights_file, weights_file + ".bak")
        except Exception:
            pass
        model.load_weights(weights_file)

    if callbacks is None:
        callbacks = []
    # Avoid duplicate ModelCheckpoint callback
    from tensorflow.keras.callbacks import ModelCheckpoint
    has_checkpoint = any(isinstance(c, ModelCheckpoint) for c in callbacks)
    if not has_checkpoint:
        checkpoint = ModelCheckpoint(weights_file, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max', save_weights_only=True)
        callbacks.append(checkpoint)

    train_time_start = time.time()
    model.fit(
        data_generator.generate(config, 'training_data'),
        steps_per_epoch=train_steps,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=data_generator.generate(config, 'validation_data'),
        validation_steps=val_steps,
        shuffle=False)
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))


def predict(config, model):
    """Compute and save final predictions on test set."""
    batch_size = config['batch_size']
    model_name = config['model_name']

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

    # Crucial: Load best weights before predicting
    if os.path.exists(weights_file):
        print(f'---> Loading best weights from {weights_file} before prediction...')
        model.load_weights(weights_file)

    test_time_start = time.time()
    predictions = model.predict(
        data_generator.generate(config, 'test_data'),
        steps=test_steps,
        verbose=0)
    test_time_end = time.time()

    np.save(file=predictions_file, arr=predictions)
    print('Total test time: %f' % (test_time_end - test_time_start))


# --- SỬ DỤNG ARGPARSE ĐỂ ĐỌC CONFIG-DRIVEN ---
parser = argparse.ArgumentParser(description="Train Var-CNN using a specified config file.")
parser.add_argument("--config", type=str, required=True, help="Path to config JSON file (e.g., config_high.json)")
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    config = json.load(config_file)

batch_size = config['batch_size']
model_name = config['model_name']

if model_name == 'var-cnn':
    model, callbacks = var_cnn.get_model(config)
elif model_name == 'df':
    model, callbacks = df.get_model(config)
else:
    raise ValueError(f"Unknown model name: {model_name}")

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
