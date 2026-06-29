import os
import json
import numpy as np
import h5py
import subprocess
import sys

# Paths
scratch_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scratch_dir)

dummy_ow_h5 = os.path.join(scratch_dir, "dummy_ow.h5")
scratch_config_path = os.path.join(scratch_dir, "temp_scratch_config.json")

outputs_dir = os.path.join(scratch_dir, "outputs")
scratch_output_dir = os.path.join(outputs_dir, "temp_ow_scratch")

os.makedirs(outputs_dir, exist_ok=True)

# 1. Create dummy Open-World dataset (101 classes) if not already present
print("1. Creating dummy Open-World H5 dataset...")
with h5py.File(dummy_ow_h5, "w") as f:
    for split, size in [("training_data", 6), ("validation_data", 2), ("test_data", 4)]:
        grp = f.create_group(split)
        
        # Labels: one-hot encoded (shape: N x 101)
        # Class indices < 100 are monitored, index 100 is unmonitored
        labels = np.zeros((size, 101), dtype=np.float32)
        for i in range(size):
            if split == "test_data":
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

# 2. Create OW Scratch configuration (NO pretrained weights key)
print("2. Generating OW scratch configuration...")
scratch_config = {
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
    "output_dir": scratch_output_dir,
    "model_id": "temp_ow_scratch",
    "sequence_dataset": "dir_seq",
    "sequence_input_name": "dir_input",
    "sequence_model_suffix": "dir",
    "metadata_dataset": "metadata",
    "metadata_type": "metadata",
    "wfmeta_k": 7
}

with open(scratch_config_path, "w", encoding="utf-8") as f:
    json.dump(scratch_config, f, indent=4)

# 3. Run OW scratch training
print("3. Executing run_model.py for training from scratch...")
cmd_scratch = [sys.executable, os.path.join(project_dir, "run_model.py"), "--config", scratch_config_path]
res_scratch = subprocess.run(cmd_scratch, capture_output=True, text=True)

print("--- SCRATCH PROCESS STDOUT ---")
print(res_scratch.stdout)
print("--- SCRATCH PROCESS STDERR ---")
print(res_scratch.stderr)

if res_scratch.returncode != 0:
    print("Scratch Training failed!")
    sys.exit(res_scratch.returncode)

# 4. Verify outputs
print("4. Verifying outputs...")
target_ow_dir = scratch_output_dir
result_json_path = os.path.join(target_ow_dir, "temp_ow_scratch_result.json")

if os.path.exists(result_json_path):
    print(f"  [PASS] Found results JSON at {result_json_path}")
    with open(result_json_path, "r") as f:
        metrics = json.load(f)
    print("  Saved metrics JSON keys:", list(metrics.keys()))
    print("  Sample metrics values:")
    for k in ["temp_ow_scratch_acc", "temp_ow_scratch_accuracy_sk", "temp_ow_scratch_precision_mon", "temp_ow_scratch_recall_mon", "temp_ow_scratch_f1_score_mon"]:
        if k in metrics:
            print(f"    {k}: {metrics[k]}")
        else:
            print(f"    [FAIL] Missing key: {k}")
else:
    print(f"  [FAIL] Missing results JSON at {result_json_path}")
    sys.exit(1)

print("\nOpen-World Scratch Training Test completed successfully!")
