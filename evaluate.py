from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import h5py
import os

# Nạp an toàn sklearn nếu có trong môi trường Kaggle/Colab
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
except ImportError:
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)
    def precision_score(y_true, y_pred, **kwargs):
        return 0.0
    def recall_score(y_true, y_pred, **kwargs):
        return 0.0
    def f1_score(y_true, y_pred, **kwargs):
        return 0.0


def find_accuracy(model_predictions, conf_thresh, actual_labels,
                  num_mon_sites, num_mon_inst_test,
                  num_unmon_sites_test, num_unmon_sites):
    """Compute TPR and FPR based on softmax output predictions."""

    actual_labels_idx = np.argmax(actual_labels, axis=1)

    # Changes predictions according to confidence threshold
    thresh_model_labels = np.zeros(len(model_predictions))
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

    denominator = num_mon_sites * num_mon_inst_test
    if denominator > 0:
        two_class_tpr = two_class_true_pos / denominator * 100
        multi_class_tpr = multi_class_true_pos / denominator * 100
    else:
        two_class_tpr = 0.0
        multi_class_tpr = 0.0
        
    two_class_tpr = '%.2f' % two_class_tpr + '%'
    multi_class_tpr = '%.2f' % multi_class_tpr + '%'

    if num_unmon_sites == 0:  # closed-world
        fpr = '0.00%'
    else:
        if num_unmon_sites_test > 0:
            fpr = false_pos / num_unmon_sites_test * 100
        else:
            fpr = 0.0
        fpr = '%.2f' % fpr + '%'

    return two_class_tpr, multi_class_tpr, fpr


def log_cw(results, sub_model_name, softmax, **parameters):
    print('%s model:' % sub_model_name)
    two_class_tpr, multi_class_tpr, fpr = find_accuracy(
        softmax, 0.,
        actual_labels=parameters['actual_labels'],
        num_mon_sites=parameters['num_mon_sites'],
        num_mon_inst_test=parameters['num_mon_inst_test'],
        num_unmon_sites_test=parameters['num_unmon_sites_test'],
        num_unmon_sites=parameters['num_unmon_sites']
    )
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


def log_ow(results, sub_model_name, softmax, **parameters):
    print('%s model:' % sub_model_name)
    for conf_thresh in np.arange(0, 1.01, 0.1):
        two_class_tpr, multi_class_tpr, fpr = find_accuracy(
            softmax, conf_thresh,
            actual_labels=parameters['actual_labels'],
            num_mon_sites=parameters['num_mon_sites'],
            num_mon_inst_test=parameters['num_mon_inst_test'],
            num_unmon_sites_test=parameters['num_unmon_sites_test'],
            num_unmon_sites=parameters['num_unmon_sites']
        )
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


def evaluate_closed_world(softmax, test_labels, target_dir, model_id):
    """Computes closed world evaluation metrics: accuracy, precision, recall, f1."""
    actual_labels_idx = np.argmax(test_labels, axis=1)
    predicted_labels = np.argmax(softmax, axis=1)

    results = {}
    acc = accuracy_score(actual_labels_idx, predicted_labels)
    prec = precision_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
    rec = recall_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)

    print("\n" + "="*50)
    print("CLOSED-WORLD EVALUATION RESULTS")
    print("="*50)
    print('Accuracy: %.2f%%' % (acc * 100))
    print('Precision (macro): %.4f' % prec)
    print('Recall (macro): %.4f' % rec)
    print('F1-Score (macro): %.4f' % f1)

    results['accuracy'] = float(acc)
    results['precision'] = float(prec)
    results['recall'] = float(rec)
    results['f1_score'] = float(f1)

    result_path = os.path.join(target_dir, f"{model_id}_result.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    print(f"Saved evaluation results to {result_path}")
    return results


def evaluate_open_world(softmax, test_labels, num_mon_sites, num_mon_inst_test, num_unmon_sites_test, num_unmon_sites, target_dir, model_id):
    """Computes open world evaluation metrics: threshold sweep for TPR, FPR, and confidence reject."""
    results = {}
    
    # 1. Đo các chỉ số CW trên tập OW trước (đánh giá không có ngưỡng chặn)
    actual_labels_idx = np.argmax(test_labels, axis=1)
    predicted_labels = np.argmax(softmax, axis=1)
    
    acc = accuracy_score(actual_labels_idx, predicted_labels)
    prec = precision_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
    rec = recall_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(actual_labels_idx, predicted_labels, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("OPEN-WORLD: CLOSED-WORLD METRICS (Evaluation without Threshold)")
    print("="*50)
    print('Accuracy: %.2f%%' % (acc * 100))
    print('Precision (macro): %.4f' % prec)
    print('Recall (macro): %.4f' % rec)
    print('F1-Score (macro): %.4f' % f1)
    
    results['cw_accuracy'] = float(acc)
    results['cw_precision'] = float(prec)
    results['cw_recall'] = float(rec)
    results['cw_f1_score'] = float(f1)
    
    # 2. Quét Threshold đo đạc tính chất Open-world
    print("\n" + "="*50)
    print("OPEN-WORLD METRICS (Threshold Analysis)")
    print("="*50)
    
    for conf_thresh in np.arange(0, 1.01, 0.1):
        two_class_tpr, multi_class_tpr, fpr = find_accuracy(
            softmax, conf_thresh, actual_labels=test_labels,
            num_mon_sites=num_mon_sites, num_mon_inst_test=num_mon_inst_test,
            num_unmon_sites_test=num_unmon_sites_test, num_unmon_sites=num_unmon_sites
        )
        print('conf: %f' % conf_thresh)
        print('\t two-class TPR: %s' % two_class_tpr)
        print('\t multi-class TPR: %s' % multi_class_tpr)
        print('\t FPR: %s' % fpr)
        
        prefix = '%s_%f' % (model_id, conf_thresh)
        results['%s_two_TPR' % prefix] = two_class_tpr
        results['%s_multi_TPR' % prefix] = multi_class_tpr
        results['%s_FPR' % prefix] = fpr

    result_path = os.path.join(target_dir, f"{model_id}_result.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    print(f"Saved evaluation results to {result_path}")
    return results


def main(config):
    is_flat = "sequence_dataset" in config
    
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    data_dir = config['data_dir']
    data_file = config.get('processed_h5')
    if not data_file:
        data_file = '%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                          num_mon_inst, num_unmon_sites_train,
                                          num_unmon_sites_test)

    with h5py.File(data_file, 'r') as f:
        test_labels = f['test_data/labels'][:]

    if is_flat:
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
        
        scenario = config.get("scenario", "closed_world")
        is_cw = (scenario == "closed_world") or (test_labels.shape[1] == 100) or (num_unmon_sites == 0)
        
        if is_cw:
            evaluate_closed_world(softmax, test_labels, target_dir, model_id)
        else:
            evaluate_open_world(softmax, test_labels, num_mon_sites, num_mon_inst_test, num_unmon_sites_test, num_unmon_sites, target_dir, model_id)
    else:
        # Fallback chạy theo mixture (ensembles) cũ
        predictions_dir = config['predictions_dir']
        mixture = config['mixture']

        predictions = {}
        ensemble_softmax = None
        for inner_comb in mixture:
            sub_model_name = '_'.join(inner_comb)
            softmax_file = '%s%s_model.npy' % (predictions_dir, sub_model_name)
            if not os.path.exists(softmax_file):
                print(f"[!] Warning: prediction file {softmax_file} not found.")
                continue
            softmax = np.load(softmax_file)
            if ensemble_softmax is None:
                ensemble_softmax = np.zeros_like(softmax)
            predictions[sub_model_name] = softmax

        parameters = {'actual_labels': test_labels,
                      'num_mon_sites': num_mon_sites,
                      'num_mon_inst_test': num_mon_inst_test,
                      'num_unmon_sites_test': num_unmon_sites_test,
                      'num_unmon_sites': num_unmon_sites}

        if predictions and ensemble_softmax is not None:
            for softmax in predictions.values():
                ensemble_softmax += softmax
            ensemble_softmax = ensemble_softmax / len(predictions)
            if len(predictions) > 1:
                predictions['ensemble'] = ensemble_softmax

        results = {}
        if num_unmon_sites == 0:  # Closed-world
            print("\n" + "="*50)
            print("CLOSED-WORLD RESULTS (Unmonitored = 0)")
            print("="*50)
            log_setting('closed', predictions, results, **parameters)
        else:  # Open-world
            print("\n" + "="*50)
            print("CLOSED-WORLD METRICS (Evaluation without Threshold)")
            print("="*50)
            log_setting('closed', predictions, results, **parameters)
            
            print("\n" + "="*50)
            print("OPEN-WORLD METRICS (Threshold Analysis)")
            print("="*50)
            log_setting('open', predictions, results, **parameters)

        with open('job_result.json', 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
    main(config)
