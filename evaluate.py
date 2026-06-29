from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import h5py
import os

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
except ImportError:
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)
    def precision_score(y_true, y_pred, **kwargs):
        return 0.0
    def recall_score(y_true, y_pred, **kwargs):
        return 0.0
    def f1_score(y_true, y_pred, **kwargs):
        return 0.0
    def roc_auc_score(y_true, y_score, **kwargs):
        return 0.0


def find_accuracy(model_predictions, conf_thresh, actual_labels=None,
                  num_mon_sites=None, num_mon_inst_test=None,
                  num_unmon_sites_test=None, num_unmon_sites=None):
    """Compute TPR and FPR based on softmax output predictions."""

    # Calculates output classes (classes with the highest probability)
    actual_labels_idx = np.argmax(actual_labels, axis=1)
    
    is_binary = (model_predictions.shape[1] == 2)
    thresh_model_labels = np.zeros(len(model_predictions))

    if is_binary:
        # Class 0: Monitored
        # Class 1: Unmonitored
        # Predict monitored (0) if monitored prob >= conf_thresh, else unmonitored (1)
        for inst_num, softmax in enumerate(model_predictions):
            if softmax[0] >= conf_thresh:
                thresh_model_labels[inst_num] = 0
            else:
                thresh_model_labels[inst_num] = 1

        two_class_true_pos = 0
        multi_class_true_pos = 0  # In binary case, this is the same as two_class_true_pos
        false_pos = 0

        for inst_num, inst_label in enumerate(actual_labels_idx):
            if inst_label == 1:  # Supposed to be unmon
                if thresh_model_labels[inst_num] == 0:  # Classified as mon
                    false_pos += 1
            else:  # Supposed to be mon (0)
                if thresh_model_labels[inst_num] == 0:  # Classified as mon
                    two_class_true_pos += 1
                    multi_class_true_pos += 1

        actual_mon_samples = np.sum(actual_labels_idx == 0)
        actual_unmon_samples = np.sum(actual_labels_idx == 1)
    else:
        # Standard multi-class (101 classes) open-world or closed-world
        for inst_num, softmax in enumerate(model_predictions):
            predicted_class = np.argmax(softmax)
            if predicted_class < num_mon_sites and \
                    softmax[predicted_class] < conf_thresh:
                thresh_model_labels[inst_num] = num_mon_sites
            else:
                thresh_model_labels[inst_num] = predicted_class

        # Computes TPR and FPR
        two_class_true_pos = 0  # Mon correctly classified as any mon site
        multi_class_true_pos = 0  # Mon correctly classified as specific mon site
        false_pos = 0  # Unmon incorrectly classified as mon site

        for inst_num, inst_label in enumerate(actual_labels_idx):
            if inst_label == num_mon_sites:  # Supposed to be unmon site
                if thresh_model_labels[inst_num] < num_mon_sites:
                    false_pos += 1
            else:  # Supposed to be mon site
                if thresh_model_labels[inst_num] < num_mon_sites:
                    two_class_true_pos += 1
                if thresh_model_labels[inst_num] == inst_label:
                    multi_class_true_pos += 1

        actual_mon_samples = np.sum(actual_labels_idx < num_mon_sites)
        actual_unmon_samples = np.sum(actual_labels_idx == num_mon_sites)

    denominator = actual_mon_samples if actual_mon_samples > 0 else (num_mon_sites * num_mon_inst_test)
    if denominator > 0:
        two_class_tpr = two_class_true_pos / denominator * 100
        multi_class_tpr = multi_class_true_pos / denominator * 100
    else:
        two_class_tpr = 0.0
        multi_class_tpr = 0.0
        
    two_class_tpr_str = '%.2f' % two_class_tpr + '%'
    multi_class_tpr_str = '%.2f' % multi_class_tpr + '%'

    if num_unmon_sites == 0:  # closed-world
        fpr_str = '0.00%'
    else:
        unmon_denominator = actual_unmon_samples if actual_unmon_samples > 0 else num_unmon_sites_test
        if unmon_denominator > 0:
            fpr = false_pos / unmon_denominator * 100
        else:
            fpr = 0.0
        fpr_str = '%.2f' % fpr + '%'

    return two_class_tpr_str, multi_class_tpr_str, fpr_str


def log_cw(results, sub_model_name, softmax, **parameters):
    print('%s model:' % sub_model_name)
    two_class_tpr, multi_class_tpr, fpr = find_accuracy(
        softmax, 0., **parameters)
    print('\t raw accuracy (TPR): %s' % multi_class_tpr)
    results['%s_acc' % sub_model_name] = multi_class_tpr

    actual_labels = parameters.get('actual_labels')
    if actual_labels is not None:
        actual_labels_idx = np.argmax(actual_labels, axis=1)
        predicted_labels = np.argmax(softmax, axis=1)

        acc_sk = accuracy_score(actual_labels_idx, predicted_labels)
        prec = precision_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
        rec = recall_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
        f1 = f1_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)

        print('\t accuracy: %.2f%%' % (acc_sk * 100))
        print('\t precision (macro): %.4f' % prec)
        print('\t recall (macro): %.4f' % rec)
        print('\t f1-score (macro): %.4f' % f1)

        results['%s_accuracy_sk' % sub_model_name] = float(acc_sk)
        results['%s_precision' % sub_model_name] = float(prec)
        results['%s_recall' % sub_model_name] = float(rec)
        results['%s_f1_score' % sub_model_name] = float(f1)

        # Detailed binary metrics
        if softmax.shape[1] == 2:
            prec_mon = precision_score(actual_labels_idx, predicted_labels, pos_label=0, average='binary', zero_division=0)
            rec_mon = recall_score(actual_labels_idx, predicted_labels, pos_label=0, average='binary', zero_division=0)
            f1_mon = f1_score(actual_labels_idx, predicted_labels, pos_label=0, average='binary', zero_division=0)

            prec_unmon = precision_score(actual_labels_idx, predicted_labels, pos_label=1, average='binary', zero_division=0)
            rec_unmon = recall_score(actual_labels_idx, predicted_labels, pos_label=1, average='binary', zero_division=0)
            f1_unmon = f1_score(actual_labels_idx, predicted_labels, pos_label=1, average='binary', zero_division=0)

            print('\t precision (monitored): %.4f' % prec_mon)
            print('\t recall (monitored): %.4f' % rec_mon)
            print('\t f1-score (monitored): %.4f' % f1_mon)
            print('\t precision (unmonitored): %.4f' % prec_unmon)
            print('\t recall (unmonitored): %.4f' % rec_unmon)
            print('\t f1-score (unmonitored): %.4f' % f1_unmon)

            results['%s_precision_mon' % sub_model_name] = float(prec_mon)
            results['%s_recall_mon' % sub_model_name] = float(rec_mon)
            results['%s_f1_score_mon' % sub_model_name] = float(f1_mon)
            results['%s_precision_unmon' % sub_model_name] = float(prec_unmon)
            results['%s_recall_unmon' % sub_model_name] = float(rec_unmon)
            results['%s_f1_score_unmon' % sub_model_name] = float(f1_unmon)


def log_ow(results, sub_model_name, softmax, **parameters):
    print('%s model:' % sub_model_name)
    
    # Calculate ROC AUC
    actual_labels = parameters.get('actual_labels')
    if actual_labels is not None:
        actual_labels_idx = np.argmax(actual_labels, axis=1)
        num_mon_sites = parameters.get('num_mon_sites', 100)
        y_true = (actual_labels_idx < num_mon_sites).astype(np.float32)
        
        if softmax.shape[1] == 2:
            y_score = softmax[:, 0]  # monitored prob
        else:
            y_score = 1.0 - softmax[:, num_mon_sites]  # monitored prob is 1 - unmonitored prob
            
        try:
            auc_val = roc_auc_score(y_true, y_score)
            print('\t ROC AUC (Monitored vs Unmonitored): %.4f' % auc_val)
            results['%s_auc' % sub_model_name] = float(auc_val)
        except Exception as e:
            print('\t Could not calculate ROC AUC: %s' % e)

    for conf_thresh in np.arange(0, 1.01, 0.1):
        two_class_tpr, multi_class_tpr, fpr = find_accuracy(
            softmax, conf_thresh, **parameters)
        print('\t conf: %f' % conf_thresh)
        print('\t \t two-class TPR: %s' % two_class_tpr)
        print('\t \t multi-class TPR: %s' % multi_class_tpr)
        print('\t \t FPR: %s' % fpr)

        prefix = '%s_%f' % (sub_model_name, conf_thresh)
        results['%s_two_TPR' % prefix] = two_class_tpr
        results['%s_multi_TPR' % prefix] = multi_class_tpr
        results['%s_FPR' % prefix] = fpr


def log_setting(setting, predictions, results, **parameters):
    print(setting + '-world results')
    for sub_model_name, softmax in predictions.items():
        if setting == 'closed':
            log_cw(results, sub_model_name, softmax, **parameters)
        elif setting == 'open':
            log_ow(results, sub_model_name, softmax, **parameters)


def main(config):
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    data_dir = config.get('data_dir', '.')
    data_file = config.get('processed_h5')
    if not data_file:
        data_file = '%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                          num_mon_inst, num_unmon_sites_train,
                                          num_unmon_sites_test)

    with h5py.File(data_file, 'r') as f:
        test_labels = f['test_data/labels'][:]

    scenario = config.get("scenario", "closed_world")
    is_cw = (scenario == "closed_world") or (test_labels.shape[1] == 100) or (num_unmon_sites == 0)

    if not is_cw:
        # Convert 101-class labels to 2-class binary labels
        class_indices = np.argmax(test_labels, axis=1)
        binary_labels = np.zeros((len(test_labels), 2), dtype=np.float32)
        binary_labels[class_indices < num_mon_sites, 0] = 1.0
        binary_labels[class_indices == num_mon_sites, 1] = 1.0
        test_labels = binary_labels

    model_id = config.get("model_id", "default_model")
    output_dir = config.get("output_dir", "/kaggle/working/outputs")
    if not output_dir.endswith(model_id):
        target_dir = os.path.join(output_dir, model_id)
    else:
        target_dir = output_dir

    predictions_path = os.path.join(target_dir, f"{model_id}_model.npy")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions not found at {predictions_path}")

    softmax = np.load(predictions_path)

    results = {}
    parameters = {
        'actual_labels': test_labels,
        'num_mon_sites': num_mon_sites,
        'num_mon_inst_test': num_mon_inst_test,
        'num_unmon_sites_test': num_unmon_sites_test,
        'num_unmon_sites': num_unmon_sites
    }

    if is_cw:
        print("\n" + "="*50)
        print("CLOSED-WORLD RESULTS")
        print("="*50)
        log_cw(results, model_id, softmax, **parameters)
    else:
        print("\n" + "="*50)
        print("OPEN-WORLD BINARY METRICS (Threshold = 0.5)")
        print("="*50)
        log_cw(results, model_id, softmax, **parameters)
        
        print("\n" + "="*50)
        print("OPEN-WORLD BINARY METRICS (Threshold Analysis)")
        print("="*50)
        log_ow(results, model_id, softmax, **parameters)

    result_path = os.path.join(target_dir, f"{model_id}_result.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    print(f"Saved evaluation results to {result_path}")


if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Evaluate single models or ensembles.")
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--ensemble', nargs='+', help='Paths to multiple configuration JSON files to evaluate as an ensemble')
    args = parser.parse_args()

    if args.ensemble:
        configs = []
        for path in args.ensemble:
            with open(path, 'r', encoding='utf-8') as f:
                configs.append(json.load(f))
        
        first_config = configs[0]
        scenario = first_config.get("scenario", "closed_world")
        processed_h5 = first_config["processed_h5"]
        num_mon_sites = first_config["num_mon_sites"]
        num_mon_inst_test = first_config["num_mon_inst_test"]
        num_unmon_sites_test = first_config["num_unmon_sites_test"]
        num_unmon_sites = first_config["num_unmon_sites"]
        
        with h5py.File(processed_h5, 'r') as f:
            test_labels = f['test_data/labels'][:]
            
        is_cw = (scenario == "closed_world") or (test_labels.shape[1] == 100) or (num_unmon_sites == 0)
        
        if not is_cw:
            class_indices = np.argmax(test_labels, axis=1)
            binary_labels = np.zeros((len(test_labels), 2), dtype=np.float32)
            binary_labels[class_indices < num_mon_sites, 0] = 1.0
            binary_labels[class_indices == num_mon_sites, 1] = 1.0
            test_labels = binary_labels
            
        ensemble_softmax = None
        loaded_count = 0
        
        for cfg in configs:
            model_id = cfg["model_id"]
            output_dir = cfg.get("output_dir", "/kaggle/working/outputs")
            if not output_dir.endswith(model_id):
                target_dir = os.path.join(output_dir, model_id)
            else:
                target_dir = output_dir
                
            pred_file = os.path.join(target_dir, f"{model_id}_model.npy")
            if not os.path.exists(pred_file):
                pred_file_fallback = os.path.join(target_dir, f"{model_id}_predictions.npy")
                if os.path.exists(pred_file_fallback):
                    pred_file = pred_file_fallback
                else:
                    print(f"[!] Warning: Prediction file not found at {pred_file}")
                    continue
                
            softmax = np.load(pred_file)
            if ensemble_softmax is None:
                ensemble_softmax = np.zeros_like(softmax)
            ensemble_softmax += softmax
            loaded_count += 1
            
        if loaded_count == 0:
            print("[!] Error: No prediction files could be loaded.")
            sys.exit(1)
            
        ensemble_softmax = ensemble_softmax / loaded_count
        
        results = {}
        parameters = {
            'actual_labels': test_labels,
            'num_mon_sites': num_mon_sites,
            'num_mon_inst_test': num_mon_inst_test,
            'num_unmon_sites_test': num_unmon_sites_test,
            'num_unmon_sites': num_unmon_sites
        }
        
        if is_cw:
            print("\n" + "="*50)
            print("CLOSED-WORLD ENSEMBLE RESULTS")
            print("="*50)
            log_cw(results, "ensemble", ensemble_softmax, **parameters)
        else:
            print("\n" + "="*50)
            print("OPEN-WORLD BINARY ENSEMBLE METRICS (Threshold = 0.5)")
            print("="*50)
            log_cw(results, "ensemble", ensemble_softmax, **parameters)
            
            print("\n" + "="*50)
            print("OPEN-WORLD BINARY ENSEMBLE METRICS (Threshold Analysis)")
            print("="*50)
            log_ow(results, "ensemble", ensemble_softmax, **parameters)
            
        out_dir = first_config.get("output_dir", "/kaggle/working/outputs")
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, "ensemble_result.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        print(f"\nSaved ensemble evaluation results to {result_path}")
        
    elif args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        main(config)
    else:
        config_file_path = 'config.json'
        if not os.path.exists(config_file_path):
            print(f"[!] Error: {config_file_path} not found. Please provide --config or --ensemble arguments.")
            sys.exit(1)
        with open(config_file_path) as config_file:
            config = json.load(config_file)
        main(config)
