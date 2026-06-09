from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import shutil
import time
import argparse
import numpy as np
import h5py

# Mute tensorflow warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import df
import evaluate
import data_generator


def train_and_val(config, model, callbacks):
    """Train and validate Deep Fingerprinting model."""
    batch_size = config['batch_size']
    model_id = config.get("model_id", "df_model")
    epochs = config.get('df_epochs', 150)

    print(f'Training Deep Fingerprinting model: {model_id}')

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

    # Resume/Load weights logic if weights file exists
    if os.path.exists(weights_file):
        print(f'---> Found existing weights file, making backup copy and resuming training...')
        try:
            shutil.copy(weights_file, weights_file + ".bak")
        except Exception as e:
            print(f"Warning: could not backup weights file: {e}")
        try:
            model.load_weights(weights_file)
        except Exception as e:
            print(f"Warning: could not load existing weights: {e}. Starting training from scratch.")

    if callbacks is None:
        callbacks = []

    train_time_start = time.time()
    model.fit(
        data_generator.generate(config, 'training_data'),
        steps_per_epoch=train_steps,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=data_generator.generate(config, 'validation_data'),
        validation_steps=val_steps,
        shuffle=False
    )
    train_time_end = time.time()

    print('Total training time: %f seconds' % (train_time_end - train_time_start))

    # Save final epoch weights
    os.makedirs(target_dir, exist_ok=True)
    model.save_weights(weights_file)
    print(f"Saved final epoch weights to {weights_file}")


def predict(config, model):
    """Compute and save final predictions on test set."""
    batch_size = config['batch_size']
    model_id = config.get("model_id", "df_model")

    print(f'Generating predictions for Deep Fingerprinting model: {model_id}')

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

    # Load best weights before predicting
    if os.path.exists(weights_file):
        print(f'---> Loading best weights from {weights_file} before prediction...')
        model.load_weights(weights_file)
    else:
        print(f'WARNING: weights file not found at {weights_file}, predicting with current weights.')

    test_time_start = time.time()
    predictions = model.predict(
        data_generator.generate(config, 'test_data'),
        steps=test_steps,
        verbose=0
    )
    test_time_end = time.time()

    # Save predictions
    os.makedirs(target_dir, exist_ok=True)
    np.save(file=predictions_file, arr=predictions)
    print('Total test time: %f seconds' % (test_time_end - test_time_start))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Deep Fingerprinting model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file (e.g., configs/df/df_cw_dir.json)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Sanity checks on model name
    if config.get('model_name') != 'df':
        print(f"Warning: expected model_name 'df', got '{config.get('model_name')}'. Forcing to 'df'.")
        config['model_name'] = 'df'

    # Build model & callbacks
    model, callbacks = df.get_model(config)

    # Train and predict
    train_and_val(config, model, callbacks)
    predict(config, model)

    # Save copy of the configuration run
    model_id = config.get("model_id", "df_model")
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
    print('Evaluating Deep Fingerprinting model on test data...')
    evaluate.main(config)


if __name__ == '__main__':
    main()
