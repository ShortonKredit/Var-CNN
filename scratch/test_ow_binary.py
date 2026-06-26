import os
import json
import numpy as np
import h5py
import subprocess
import sys

# Paths
scratch_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scratch_dir)

dummy_cw_h5 = os.path.join(scratch_dir, "dummy_cw.h5")
dummy_ow_h5 = os.path.join(scratch_dir, "dummy_ow.h5")

cw_config_path = os.path.join(scratch_dir, "temp_cw_config.json")
ow_config_path = os.path.join(scratch_dir, "temp_ow_config.json")

outputs_dir = os.path.join(scratch_dir, "outputs")
cw_output_dir = os.path.join(outputs_dir, "temp_cw")
ow_output_dir = os.path.join(outputs_dir, "temp_ow")

os.makedirs(outputs_dir, exist_ok=True)

# 1. Create dummy Closed-World dataset (100 classes)
print("1. Creating dummy Closed-World H5 dataset...")
with h5py.File(dummy_cw_h5, "w") as f:
    for split, size in [("training_data", 6), ("validation_data", 2), ("test_data", 2)]:
        grp = f.create_group(split)
        
        # Labels: one-hot encoded (shape: N x 100)
        labels = np.zeros((size, 100), dtype=np.float32)
        for i in range(size):
            labels[i, i % 100] = 1.0
        grp.create_dataset("labels", data=labels)
        
        # Sequences (shape: N x 5000 x 1)
        dir_seq = np.random.choice([-1.0, 0.0, 1.0], size=(size, 5000, 1)).astype(np.float32)
        grp.create_dataset("dir_seq", data=dir_seq)
        
        # Metadata (shape: N x 7)
        metadata = np.random.randn(size, 7).astype(np.float32)
        grp.create_dataset("metadata", data=metadata)

# 2. Create dummy Open-World dataset (101 classes)
print("2. Creating dummy Open-World H5 dataset...")
with h5py.File(dummy_ow_h5, "w") as f:
    for split, size in [("training_data", 6), ("validation_data", 2), ("test_data", 4)]:
        grp = f.create_group(split)
        
        # Labels: one-hot encoded (shape: N x 101)
        # Class indices < 100 are monitored, index 100 is unmonitored
        labels = np.zeros((size, 101), dtype=np.float32)
        for i in range(size):
            # Make sure we have some monitored and some unmonitored samples
            if split == "test_data":
                # For test, let 2 be monitored, 2 be unmonitored
                class_idx = (i % 100) if i < 2 else 100
            else:
                class_idx = (i % 100) if i % 2 == 0 else 100
            labels[i, class_idx] = 1.0
        grp.create_dataset("labels", data=labels)
        
        # Sequences (shape: N x 5000 x 1)
        dir_seq = np.random.choice([-1.0, 0.0, 1.0], size=(size, 5000, 1)).astype(np.float32)
        grp.create_dataset("dir_seq", data=dir_seq)
        
        # Metadata (shape: N x 7)
        metadata = np.random.randn(size, 7).astype(np.float32)
        grp.create_dataset("metadata", data=metadata)

# 3. Create CW Config and run training to generate base weights
print("3. Generating CW config and training CW model...")
cw_config = {
    "scenario": "closed_world",
    "processed_h5": dummy_cw_h5,
    "num_classes": 100,
    "num_mon_sites": 100,
    "num_mon_inst_train": 1,
    "num_mon_inst_test": 1,
    "num_unmon_sites_train": 0,
    "num_unmon_sites_test": 0,
    "batch_size": 2,
    "seq_length": 5000,
    "model_name": "var-cnn",
    "df_epochs": 1,
    "var_cnn_max_epochs": 1,
    "var_cnn_base_patience": 2,
    "dir_dilations": True,
    "time_dilations": True,
    "output_dir": cw_output_dir,
    "model_id": "temp_cw",
    "sequence_dataset": "dir_seq",
    "sequence_input_name": "dir_input",
    "sequence_model_suffix": "dir",
    "metadata_dataset": "metadata",
    "metadata_type": "metadata",
    "wfmeta_k": 7
}

with open(cw_config_path, "w", encoding="utf-8") as f:
    json.dump(cw_config, f, indent=4)

cw_weights_file = os.path.join(cw_output_dir, "temp_cw.weights.h5")

# Run CW training
cmd_cw = [sys.executable, os.path.join(project_dir, "run_model.py"), "--config", cw_config_path]
res_cw = subprocess.run(cmd_cw, capture_output=True, text=True)
if res_cw.returncode != 0:
    print("CW Training failed!")
    print(res_cw.stdout)
    print(res_cw.stderr)
    sys.exit(res_cw.returncode)
print("  CW Training completed successfully. Weights generated at:", cw_weights_file)

# 4. Create OW Config referencing the CW weights file
print("4. Generating OW config...")
ow_config = {
    "scenario": "open_world",
    "processed_h5": dummy_ow_h5,
    "num_classes": 101,
    "num_mon_sites": 100,
    "num_mon_inst_train": 1,
    "num_mon_inst_test": 1,
    "num_unmon_sites_train": 1,
    "num_unmon_sites_test": 1,
    "batch_size": 2,
    "seq_length": 5000,
    "model_name": "var-cnn",
    "df_epochs": 1,
    "var_cnn_max_epochs": 1,
    "var_cnn_base_patience": 2,
    "dir_dilations": True,
    "time_dilations": True,
    "output_dir": ow_output_dir,
    "model_id": "temp_ow",
    "sequence_dataset": "dir_seq",
    "sequence_input_name": "dir_input",
    "sequence_model_suffix": "dir",
    "metadata_dataset": "metadata",
    "metadata_type": "metadata",
    "wfmeta_k": 7,
    "pretrained_closed_world_weights": cw_weights_file
}

with open(ow_config_path, "w", encoding="utf-8") as f:
    json.dump(ow_config, f, indent=4)

# 5. Run OW training end-to-end (loads CW weights, replaces head, trains final layer)
print("5. Executing run_model.py for binary Open-World training and evaluation...")
cmd_ow = [sys.executable, os.path.join(project_dir, "run_model.py"), "--config", ow_config_path]
res_ow = subprocess.run(cmd_ow, capture_output=True, text=True)

print("--- OPEN-WORLD PROCESS STDOUT ---")
print(res_ow.stdout)
print("--- OPEN-WORLD PROCESS STDERR ---")
print(res_ow.stderr)

if res_ow.returncode != 0:
    print("OW Training failed!")
    sys.exit(res_ow.returncode)

# 6. Verify OW output metrics and JSON reports
print("6. Verifying outputs...")
target_ow_dir = ow_output_dir
result_json_path = os.path.join(target_ow_dir, "temp_ow_result.json")

if os.path.exists(result_json_path):
    print(f"  [PASS] Found results JSON at {result_json_path}")
    with open(result_json_path, "r") as f:
        metrics = json.load(f)
    print("  Saved metrics JSON keys:", list(metrics.keys()))
    print("  Sample metrics values:")
    for k in ["temp_ow_acc", "temp_ow_accuracy_sk", "temp_ow_precision_mon", "temp_ow_recall_mon", "temp_ow_f1_score_mon"]:
        if k in metrics:
            print(f"    {k}: {metrics[k]}")
        else:
            print(f"    [FAIL] Missing key: {k}")
else:
    print(f"  [FAIL] Missing results JSON at {result_json_path}")
    sys.exit(1)

print("\nBinary Open-World Integration Test completed successfully!")
